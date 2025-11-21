"""
Figure detection component for granular PageIndex node generation.

This module provides functionality to detect figures in PDF documents using
PyMuPDF's native image extraction and LLM-based analysis for descriptions.
"""

import logging
import re
import os
import json
import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import pymupdf
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types


class BoundingBox(BaseModel):
    """Bounding box coordinates as percentages of page dimensions."""
    x_min: float = Field(..., description="Left edge as percentage (0-100)")
    y_min: float = Field(..., description="Top edge as percentage (0-100)")
    x_max: float = Field(..., description="Right edge as percentage (0-100)")
    y_max: float = Field(..., description="Bottom edge as percentage (0-100)")


class DetectedFigure(BaseModel):
    """Structured output for a detected figure from vision API."""
    figure_number: str = Field(..., description="Figure number (e.g., 'Figure 1', 'Fig. 2', 'Figure S1')")
    caption: str = Field(default="", description="Complete caption text associated with the figure")
    figure_type: str = Field(..., description="Type of figure (e.g., 'line plot', 'bar chart', 'SEM image', 'schematic diagram', 'photograph')")
    bbox: Optional[BoundingBox] = Field(None, description="Approximate bounding box coordinates")


class FigureDetectionResponse(BaseModel):
    """Response containing all detected figures on a page."""
    figures: List[DetectedFigure] = Field(default_factory=list, description="List of all figures detected on this page")


class FigureTypeResponse(BaseModel):
    """Response for figure type classification."""
    figure_type: str = Field(..., description="Type of figure (e.g., 'line plot', 'bar chart', 'photograph')")


class FigureNode(BaseModel):
    """Data structure representing a detected figure with context."""
    figure_number: str = Field(..., description="Figure number (e.g., 'Figure 3')")
    caption: str = Field(..., description="Full caption text")
    page: int = Field(..., description="Page number (1-indexed)")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box if available")
    figure_type: str = Field(..., description="Type of figure (e.g., 'SEM image', 'line plot', 'schematic')")
    context: str = Field(..., description="Surrounding paragraphs")
    summary: str = Field(..., description="LLM-generated summary of figure content")


class FigureDetector:
    """
    Detects figures in PDF documents using PyMuPDF's native image extraction.
    
    This class extracts images directly from the PDF structure and uses LLM
    to analyze figure content and generate descriptions. This is much more
    efficient than vision-based detection.
    
    Includes caching to avoid redundant API calls for the same images.
    """
    
    def __init__(self, llm_client, pdf_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the FigureDetector.
        
        Args:
            llm_client: LLM client instance for generating descriptions
            pdf_path: Path to the PDF file or BytesIO stream
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.pdf_path = pdf_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Gemini vision client for image analysis (optional)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.vision_client = genai.Client(api_key=api_key)
        else:
            self.vision_client = None
            self.logger.warning("GEMINI_API_KEY not found - vision analysis will be limited")
        
        # Open PDF document
        if isinstance(pdf_path, BytesIO):
            self.doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str):
            self.doc = pymupdf.open(pdf_path)
        else:
            raise ValueError("pdf_path must be a file path string or BytesIO stream")
        
        # Initialize caches for detection results
        self._image_type_cache = {}  # Cache for image type analysis
        self._summary_cache = {}  # Cache for figure summaries
        self.logger.debug("Initialized caching for figure detection")
    
    def _extract_images_from_page(self, page_num: int) -> List[dict]:
        """
        Extract all images from a PDF page using PyMuPDF.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of image dictionaries with metadata
        """
        page = self.doc[page_num]
        images = []
        
        # Get list of images on the page
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]  # Image xref number
                
                # Get image bbox on page
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue
                
                # Use the first rectangle (images can appear multiple times)
                rect = img_rects[0]
                
                # Convert to percentage coordinates
                page_rect = page.rect
                bbox = BoundingBox(
                    x_min=(rect.x0 / page_rect.width) * 100,
                    y_min=(rect.y0 / page_rect.height) * 100,
                    x_max=(rect.x1 / page_rect.width) * 100,
                    y_max=(rect.y1 / page_rect.height) * 100
                )
                
                # Extract image data
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Filter out very small images (likely decorative)
                width, height = pil_image.size
                if width < 50 or height < 50:
                    continue
                
                images.append({
                    'xref': xref,
                    'bbox': bbox,
                    'image': pil_image,
                    'ext': image_ext,
                    'size': (width, height)
                })
                
            except Exception as e:
                self.logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                continue
        
        return images
    
    def _find_caption_for_image(self, page_num: int, bbox: BoundingBox, page_text: str) -> Tuple[str, str]:
        """
        Find the caption for an image by searching nearby text with error handling.
        
        Args:
            page_num: Page number (1-indexed)
            bbox: Image bounding box
            page_text: Full text of the page
            
        Returns:
            Tuple of (figure_number, caption) - uses placeholder if not found
        """
        try:
            # Look for figure captions in the text
            # Common patterns: "Figure 1:", "Fig. 2:", "Figure S1.", etc.
            caption_patterns = [
                r'(Figure\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
                r'(Fig\.?\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
                r'(FIG\.?\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
            ]
            
            for pattern in caption_patterns:
                try:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        figure_number = match.group(1).strip()
                        caption_text = match.group(2).strip()
                        
                        # Clean up caption (remove extra whitespace)
                        caption_text = ' '.join(caption_text.split())
                        
                        if figure_number and caption_text:
                            return figure_number, caption_text
                except Exception as e:
                    self.logger.debug(f"Error matching pattern {pattern}: {e}")
                    continue
            
            # No caption found - use placeholder
            self.logger.debug(f"No caption found for image on page {page_num}, using placeholder")
            return f"Figure (Page {page_num})", ""
            
        except Exception as e:
            self.logger.warning(f"Error finding caption for image on page {page_num}: {e}")
            return f"Figure (Page {page_num})", ""
    

    def detect_figures(self, page_range: Tuple[int, int], page_texts: List[Tuple[str, int]], batch_size: int = 5, use_vision_fallback: bool = True) -> List[FigureNode]:
        """
        Detect all figures within a page range using PyMuPDF image extraction with batch processing.
        
        If no embedded images are found and use_vision_fallback=True, falls back to vision-based
        detection on rendered pages (useful for vector graphics).
        
        Args:
            page_range: (start_page, end_page) tuple (1-indexed, inclusive)
            page_texts: List of (page_text, token_count) tuples for context extraction
            batch_size: Number of images to process in a single batch for type analysis and summary generation
            use_vision_fallback: If True, use vision API on rendered pages when no images found
            
        Returns:
            List of FigureNode objects with metadata
        """
        start_page, end_page = page_range
        figures = []
        
        self.logger.info(f"Extracting figures from pages {start_page} to {end_page} (batch_size={batch_size})")
        
        # First pass: Extract all images and captions
        image_data_list = []
        
        for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
            try:
                # Extract images from page using PyMuPDF
                images = self._extract_images_from_page(page_num)
                
                self.logger.info(f"Found {len(images)} embedded image(s) on page {page_num + 1}")
                
                # Get page text for caption extraction
                page_text = page_texts[page_num][0] if page_num < len(page_texts) else ""
                
                # Process each image
                for img_idx, img_data in enumerate(images):
                    try:
                        # Find caption for this image (with error handling)
                        try:
                            figure_number, caption = self._find_caption_for_image(
                                page_num + 1,
                                img_data['bbox'],
                                page_text
                            )
                        except Exception as e:
                            self.logger.warning(f"Error finding caption for image {img_idx} on page {page_num + 1}: {e}")
                            figure_number = f"Figure (Page {page_num + 1})"
                            caption = ""
                        
                        # Extract context around figure (with error handling)
                        try:
                            context = self.extract_figure_context(
                                page_num + 1,
                                img_data['bbox'],
                                page_texts
                            )
                        except Exception as e:
                            self.logger.warning(f"Error extracting context for image {img_idx} on page {page_num + 1}: {e}")
                            context = ""
                        
                        # Store image data for batch processing
                        image_data_list.append({
                            'image': img_data['image'],
                            'bbox': img_data['bbox'],
                            'page': page_num + 1,
                            'figure_number': figure_number,
                            'caption': caption,
                            'context': context
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing image {img_idx} on page {page_num + 1}: {e}")
                        continue
                    
            except Exception as e:
                self.logger.error(f"Error extracting figures from page {page_num + 1}: {e}")
                continue
        
        # Second pass: Batch process images for type analysis and summary generation
        self.logger.info(f"Processing {len(image_data_list)} images in batches of {batch_size}")
        
        for batch_start in range(0, len(image_data_list), batch_size):
            batch_end = min(batch_start + batch_size, len(image_data_list))
            batch = image_data_list[batch_start:batch_end]
            
            self.logger.debug(f"Processing batch {batch_start // batch_size + 1}/{(len(image_data_list) + batch_size - 1) // batch_size}")
            
            # Process batch
            batch_results = self._process_figure_batch(batch)
            
            # Create figure nodes from batch results
            for img_data, result in zip(batch, batch_results):
                try:
                    figure_node = FigureNode(
                        figure_number=img_data['figure_number'],
                        caption=img_data['caption'] if img_data['caption'] else "No caption available",
                        page=img_data['page'],
                        bbox=img_data['bbox'],
                        figure_type=result['figure_type'],
                        context=img_data['context'],
                        summary=result['summary']
                    )
                    figures.append(figure_node)
                    
                    self.logger.debug(f"  - {img_data['figure_number']}: {result['figure_type']}")
                except Exception as e:
                    self.logger.error(f"Error creating FigureNode for {img_data['figure_number']}: {e}")
                    continue
        
        self.logger.info(f"Total figures extracted from embedded images: {len(figures)}")
        
        # Fallback: If no figures found and vision API available, try vision-based detection
        if len(figures) == 0 and use_vision_fallback and self.vision_client:
            self.logger.info("No embedded images found - trying vision-based detection on rendered pages...")
            vision_figures = self._detect_figures_with_vision(page_range, page_texts)
            figures.extend(vision_figures)
            self.logger.info(f"Total figures detected with vision: {len(figures)}")
        
        return figures
    
    def _process_figure_batch(self, batch: List[dict]) -> List[dict]:
        """
        Process a batch of figures for type analysis and summary generation.
        
        Args:
            batch: List of image data dictionaries
            
        Returns:
            List of result dictionaries with 'figure_type' and 'summary' keys
        """
        results = []
        
        for img_data in batch:
            try:
                # Analyze image type (with retry)
                try:
                    figure_type = "image"
                    if self.vision_client:
                        figure_type = self._analyze_image_type(img_data['image'])
                except Exception as e:
                    self.logger.warning(f"Error analyzing image type for {img_data['figure_number']}: {e}")
                    figure_type = "image"
                
                # Generate summary (with retry and fallback)
                try:
                    summary = self._generate_figure_summary_from_image(
                        img_data['image'],
                        img_data['figure_number'],
                        img_data['caption'],
                        figure_type,
                        img_data['context']
                    )
                except Exception as e:
                    self.logger.error(f"Error generating summary for {img_data['figure_number']}: {e}")
                    # Use caption as fallback, or placeholder if no caption
                    summary = img_data['caption'] if img_data['caption'] else f"{img_data['figure_number']} ({figure_type})"
                
                results.append({
                    'figure_type': figure_type,
                    'summary': summary
                })
                
            except Exception as e:
                self.logger.error(f"Error processing figure in batch: {e}")
                # Add fallback result
                results.append({
                    'figure_type': 'image',
                    'summary': img_data.get('caption', 'Figure')
                })
        
        return results
    
    def _analyze_image_type(self, image: Image.Image, max_retries: int = 3) -> str:
        """
        Analyze an image to determine its type using vision API with retry logic and caching.
        
        Args:
            image: PIL Image object
            max_retries: Maximum number of retry attempts
            
        Returns:
            Type of figure (e.g., 'line plot', 'bar chart', 'photograph')
        """
        if not self.vision_client:
            self.logger.debug("Vision client not available, using default type")
            return "image"
        
        # Create cache key from image hash
        import hashlib
        image_bytes = image.tobytes()
        cache_key = hashlib.md5(image_bytes).hexdigest()
        
        # Check cache
        if cache_key in self._image_type_cache:
            self.logger.debug("Using cached image type")
            return self._image_type_cache[cache_key]
        
        prompt = """Analyze this image and identify what type of figure it is.
            
Respond with ONLY ONE of these types:
- line plot
- bar chart
- scatter plot
- pie chart
- histogram
- heatmap
- diagram
- schematic
- flowchart
- photograph
- microscopy image
- table
- map
- other

Provide only the type, nothing else."""

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.vision_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=FigureTypeResponse.model_json_schema()
                    )
                )
                
                # Parse and validate using Pydantic
                type_response = FigureTypeResponse.model_validate_json(response.text)
                figure_type = type_response.figure_type.strip().lower()
                
                if figure_type:
                    # Cache the result
                    self._image_type_cache[cache_key] = figure_type
                    return figure_type
                else:
                    self.logger.warning("Empty response from vision API for image type")
                    return "image"
                
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                self.logger.warning(f"Error analyzing image type (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to analyze image type after {max_retries} attempts, using fallback")
                    return "image"
        
        return "image"
    
    def _generate_figure_summary_from_image(
        self,
        image: Image.Image,
        figure_number: str,
        caption: str,
        figure_type: str,
        context: str,
        max_retries: int = 3
    ) -> str:
        """
        Generate a summary of the figure using vision API and LLM with retry logic and caching.
        
        Args:
            image: PIL Image of the figure
            figure_number: Figure number
            caption: Figure caption
            figure_type: Type of figure
            context: Surrounding text context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Summary string
        """
        # Create cache key from image hash and caption
        import hashlib
        image_bytes = image.tobytes()
        cache_key = hashlib.md5(f"{image_bytes.hex()[:32]}_{caption}_{figure_type}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self._summary_cache:
            self.logger.debug(f"Using cached summary for {figure_number}")
            return self._summary_cache[cache_key]
        
        # Try vision API first with retry logic
        if self.vision_client:
            prompt = f"""Analyze this figure and generate a concise summary (2-3 sentences).

Figure: {figure_number}
Type: {figure_type}
Caption: {caption}

Context from paper:
{context[:800]}

Summary should describe:
1. What the figure shows
2. Key findings or data presented
3. Relevance to the research

Provide only the summary, no additional text."""

            for attempt in range(max_retries):
                try:
                    response = self.vision_client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=[image, prompt]
                    )
                    
                    summary = response.text.strip()
                    if summary:
                        # Cache the result
                        self._summary_cache[cache_key] = summary
                        return summary
                    
                except Exception as e:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Vision API failed for summary (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        self.logger.debug(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.warning("Vision API failed after retries, falling back to text-only summary")
        
        # Fallback: Generate summary from caption and context only
        if caption:
            try:
                prompt = f"""Generate a concise summary (2-3 sentences) of this figure based on its caption and context.

Figure: {figure_number}
Type: {figure_type}
Caption: {caption}

Context:
{context[:1000]}

Summary should describe what the figure shows and its relevance.
Provide only the summary, no additional text."""

                # Use the LLM client with retry
                if hasattr(self.llm_client, 'provider') and self.llm_client.provider == 'gemini':
                    model = 'gemini-2.5-flash-lite'
                else:
                    model = getattr(self.llm_client, 'default_model', 'gpt-4o-mini')
                
                for attempt in range(max_retries):
                    try:
                        response = self.llm_client.chat_completion(
                            model=model,
                            prompt=prompt,
                            temperature=0
                        )
                        
                        if response and response != "Error":
                            summary = response.strip()
                            # Cache the result
                            self._summary_cache[cache_key] = summary
                            return summary
                        
                    except Exception as e:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"LLM failed for caption-based summary (attempt {attempt + 1}/{max_retries}): {e}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error generating caption-based summary: {e}")
        
        # Final fallback: use caption or placeholder
        if caption:
            self.logger.debug(f"Using caption as summary for {figure_number}")
            return caption
        else:
            self.logger.warning(f"No caption available for {figure_number}, using placeholder")
            return f"{figure_number} ({figure_type})"
    
    def extract_figure_context(self, page_num: int, bbox: Optional[BoundingBox], 
                               page_texts: List[Tuple[str, int]]) -> str:
        """
        Extract surrounding text context for a figure.
        
        Extracts ±2 paragraphs around the figure location for context.
        Handles edge cases where figures are at page boundaries.
        
        Args:
            page_num: Page number (1-indexed) where figure is located
            bbox: BoundingBox object (optional, with coordinates as percentages)
            page_texts: List of (page_text, token_count) tuples
            
        Returns:
            Contextual text around the figure
        """
        try:
            # Get text from current page
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= len(page_texts):
                self.logger.warning(f"Page index {page_num} out of range")
                return ""
            
            current_page_text = page_texts[page_idx][0]
            
            # Split into paragraphs (handle both \n\n and single \n)
            paragraphs = []
            for p in current_page_text.split('\n\n'):
                p = p.strip()
                if p:
                    paragraphs.append(p)
            
            # If no double-newline paragraphs, try single newlines
            if len(paragraphs) <= 1:
                paragraphs = [p.strip() for p in current_page_text.split('\n') if p.strip()]
            
            if not paragraphs:
                self.logger.warning(f"No paragraphs found on page {page_num}")
                return current_page_text[:1000]  # Return first 1000 chars as fallback
            
            # Determine figure position on page (top, middle, bottom)
            figure_position = 'middle'
            if bbox:
                y_center = (bbox.y_min + bbox.y_max) / 2
                if y_center < 33:
                    figure_position = 'top'
                elif y_center > 67:
                    figure_position = 'bottom'
            
            # Find paragraphs that mention the figure
            figure_mentions = []
            for i, para in enumerate(paragraphs):
                # Look for figure references in the paragraph
                if re.search(r'\b[Ff]ig(?:ure)?\.?\s*\d+', para):
                    figure_mentions.append(i)
            
            # Extract context paragraphs
            context_paragraphs = []
            
            if figure_mentions:
                # Take paragraphs around the first mention
                mention_idx = figure_mentions[0]
                start_idx = max(0, mention_idx - 2)
                end_idx = min(len(paragraphs), mention_idx + 3)
                context_paragraphs = paragraphs[start_idx:end_idx]
            else:
                # No explicit mention found, use position-based heuristic
                if figure_position == 'top':
                    # Figure at top, take first few paragraphs
                    context_paragraphs = paragraphs[:min(5, len(paragraphs))]
                elif figure_position == 'bottom':
                    # Figure at bottom, take last few paragraphs
                    context_paragraphs = paragraphs[max(0, len(paragraphs)-5):]
                else:
                    # Figure in middle, take middle paragraphs
                    if len(paragraphs) > 4:
                        mid = len(paragraphs) // 2
                        context_paragraphs = paragraphs[max(0, mid-2):min(len(paragraphs), mid+3)]
                    else:
                        context_paragraphs = paragraphs
            
            context_text = '\n\n'.join(context_paragraphs)
            
            # Handle edge case: figure at page boundaries
            # Add context from previous page if figure is at top or context is short
            if (figure_position == 'top' or len(context_text) < 300) and page_idx > 0:
                prev_page_text = page_texts[page_idx - 1][0]
                prev_paragraphs = [p.strip() for p in prev_page_text.split('\n\n') if p.strip()]
                if not prev_paragraphs:
                    prev_paragraphs = [p.strip() for p in prev_page_text.split('\n') if p.strip()]
                
                if prev_paragraphs:
                    # Add last 1-2 paragraphs from previous page
                    num_prev = min(2, len(prev_paragraphs))
                    prev_context = '\n\n'.join(prev_paragraphs[-num_prev:])
                    context_text = prev_context + '\n\n' + context_text
                    self.logger.debug(f"Added context from previous page for figure on page {page_num}")
            
            # Add context from next page if figure is at bottom or context is short
            if (figure_position == 'bottom' or len(context_text) < 300) and page_idx < len(page_texts) - 1:
                next_page_text = page_texts[page_idx + 1][0]
                next_paragraphs = [p.strip() for p in next_page_text.split('\n\n') if p.strip()]
                if not next_paragraphs:
                    next_paragraphs = [p.strip() for p in next_page_text.split('\n') if p.strip()]
                
                if next_paragraphs:
                    # Add first 1-2 paragraphs from next page
                    num_next = min(2, len(next_paragraphs))
                    next_context = '\n\n'.join(next_paragraphs[:num_next])
                    context_text = context_text + '\n\n' + next_context
                    self.logger.debug(f"Added context from next page for figure on page {page_num}")
            
            # Limit context length to reasonable size
            if len(context_text) > 2000:
                context_text = context_text[:2000] + "..."
            
            return context_text
            
        except Exception as e:
            self.logger.error(f"Error extracting context for figure on page {page_num}: {e}")
            # Return partial page text as fallback
            try:
                page_idx = page_num - 1
                if 0 <= page_idx < len(page_texts):
                    return page_texts[page_idx][0][:1000]
            except:
                pass
            return ""
    

    def _detect_figures_with_vision(self, page_range: Tuple[int, int], page_texts: List[Tuple[str, int]], max_retries: int = 3) -> List[FigureNode]:
        """
        Detect figures by rendering pages and using vision API.
        
        This is a fallback method for PDFs with vector graphics instead of embedded images.
        
        Args:
            page_range: (start_page, end_page) tuple (1-indexed, inclusive)
            page_texts: List of (page_text, token_count) tuples
            max_retries: Maximum retry attempts for API calls
            
        Returns:
            List of FigureNode objects
        """
        start_page, end_page = page_range
        figures = []
        
        for page_num in range(start_page - 1, end_page):
            try:
                # Render page as image
                page = self.doc[page_num]
                mat = pymupdf.Matrix(2.0, 2.0)  # 2x resolution
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                from io import BytesIO
                img_data = pix.tobytes("png")
                page_image = Image.open(BytesIO(img_data))
                
                # Get page text
                page_text = page_texts[page_num][0] if page_num < len(page_texts) else ""
                
                # Detect figures on this page using vision API
                detected_figures = self._detect_figures_on_rendered_page(
                    page_image, 
                    page_num + 1, 
                    page_text,
                    max_retries
                )
                
                if detected_figures:
                    self.logger.info(f"Vision API detected {len(detected_figures)} figure(s) on page {page_num + 1}")
                    
                    # Process each detected figure
                    for fig_data in detected_figures:
                        try:
                            # Extract context
                            context = self.extract_figure_context(
                                page_num + 1,
                                fig_data.get('bbox'),
                                page_texts
                            )
                            
                            # Create figure node
                            figure_node = FigureNode(
                                figure_number=fig_data.get('figure_number', f"Figure (Page {page_num + 1})"),
                                caption=fig_data.get('caption', ''),
                                page=page_num + 1,
                                bbox=fig_data.get('bbox'),
                                figure_type=fig_data.get('figure_type', 'diagram'),
                                context=context,
                                summary=fig_data.get('summary', fig_data.get('caption', ''))
                            )
                            figures.append(figure_node)
                            
                        except Exception as e:
                            self.logger.warning(f"Error creating figure node: {e}")
                            continue
                
            except Exception as e:
                self.logger.error(f"Error detecting figures on page {page_num + 1}: {e}")
                continue
        
        return figures
    
    def _detect_figures_on_rendered_page(self, page_image: Image.Image, page_num: int, page_text: str, max_retries: int = 3) -> List[dict]:
        """
        Detect figures on a rendered page image using vision API.
        
        Args:
            page_image: PIL Image of the page
            page_num: Page number (1-indexed)
            page_text: Text content of the page
            max_retries: Maximum retry attempts
            
        Returns:
            List of figure dictionaries
        """
        if not self.vision_client:
            return []
        
        # Detection prompt
        detection_prompt = """Analyze this page and detect all figures, charts, diagrams, plots, and illustrations.

For each figure found, provide:
1. Figure number (e.g., "Figure 1", "Fig. 2") - extract from caption if visible
2. Caption text (if visible on the page)
3. Approximate bounding box as percentages (x_min, y_min, x_max, y_max where 0-100)
4. Type of figure (e.g., "line plot", "bar chart", "diagram", "schematic", "photograph")

If no figures are found, respond with an empty list.

Format your response as JSON:
{
  "figures": [
    {
      "figure_number": "Figure 1",
      "caption": "Illustration of...",
      "bbox": {"x_min": 10, "y_min": 20, "x_max": 90, "y_max": 50},
      "figure_type": "diagram"
    }
  ]
}

Respond ONLY with valid JSON, no additional text."""

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.vision_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[page_image, detection_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=FigureDetectionResponse.model_json_schema()
                    )
                )
                
                # Parse response
                detection_response = FigureDetectionResponse.model_validate_json(response.text)
                
                detected_figures = []
                for fig in detection_response.figures:
                    fig_dict = {
                        'figure_number': fig.figure_number,
                        'caption': fig.caption,
                        'bbox': fig.bbox,
                        'figure_type': fig.figure_type
                    }
                    
                    # Generate summary using vision API
                    try:
                        summary = self._generate_figure_summary_from_image(
                            page_image,
                            fig.figure_number,
                            fig.caption,
                            fig.figure_type,
                            page_text[:1000]  # Use first 1000 chars as context
                        )
                        fig_dict['summary'] = summary
                    except Exception as e:
                        self.logger.warning(f"Error generating summary for {fig.figure_number}: {e}")
                        fig_dict['summary'] = fig.caption if fig.caption else f"{fig.figure_number} ({fig.figure_type})"
                    
                    detected_figures.append(fig_dict)
                
                return detected_figures
                
            except Exception as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Vision detection failed on page {page_num} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Vision detection failed after {max_retries} attempts")
                    return []
        
        return []

    def __del__(self):
        """Clean up PDF document on deletion."""
        try:
            if hasattr(self, 'doc') and self.doc is not None:
                self.doc.close()
        except:
            pass  # Ignore errors during cleanup
