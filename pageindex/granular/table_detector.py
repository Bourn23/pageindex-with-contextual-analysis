"""
Table detection component for granular PageIndex node generation.

This module provides functionality to detect tables in PDF documents using
vision-based analysis with Gemini models. Uses a smart two-phase approach:
1. Detection phase: Fast model (gemini-2.5-flash-lite) for table detection
2. Extraction phase: Fast model for structure extraction and summary generation
"""

import logging
import re
import os
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


class DetectedTable(BaseModel):
    """Structured output for a detected table from vision API."""
    table_number: str = Field(..., description="Table number (e.g., 'Table 1', 'Table S2')")
    caption: str = Field(..., description="Complete caption text associated with the table")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    key_values: dict = Field(default_factory=dict, description="Important data points from the table")
    bbox: Optional[BoundingBox] = Field(None, description="Approximate bounding box coordinates")


class TableDetectionResponse(BaseModel):
    """Response containing all detected tables on a page."""
    tables: List[DetectedTable] = Field(default_factory=list, description="List of all tables detected on this page")


class TableStructure(BaseModel):
    """Structured output for table structure extraction."""
    headers: List[str] = Field(default_factory=list, description="Column headers from left to right")
    num_rows: int = Field(default=0, description="Approximate number of rows")
    key_values: dict = Field(default_factory=dict, description="Important data points with context")


class TableNode(BaseModel):
    """Data structure representing a detected table with context."""
    table_number: str = Field(..., description="Table number (e.g., 'Table 2')")
    caption: str = Field(..., description="Full caption text")
    page: int = Field(..., description="Page number (1-indexed)")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box if available")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    key_values: dict = Field(default_factory=dict, description="Important data points")
    context: str = Field(..., description="Surrounding paragraphs")
    summary: str = Field(..., description="LLM-generated summary of table content")


class TableDetector:
    """
    Detects tables in PDF documents using vision-based analysis.
    
    Uses a smart two-phase approach:
    - Detection: Fast model (gemini-2.5-flash-lite) to detect tables
    - Extraction: Fast model for structure extraction and summary
    
    Includes caching to avoid redundant API calls for the same pages.
    """
    
    def __init__(self, llm_client, pdf_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the TableDetector.
        
        Args:
            llm_client: LLM client instance for generating descriptions
            pdf_path: Path to the PDF file or BytesIO stream
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.pdf_path = pdf_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Gemini vision client
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.vision_client = genai.Client(api_key=api_key)
        else:
            self.vision_client = None
            self.logger.warning("GEMINI_API_KEY not found - table detection will be limited")
        
        # Open PDF document
        if isinstance(pdf_path, BytesIO):
            self.doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str):
            self.doc = pymupdf.open(pdf_path)
        else:
            raise ValueError("pdf_path must be a file path string or BytesIO stream")
        
        # Model configuration for smart detection
        self.detection_model = "gemini-2.5-flash-lite"  # Fast model for detection
        self.extraction_model = "gemini-2.5-flash-lite"  # Fast model for extraction
        
        # Initialize caches for detection results
        self._detection_cache = {}  # Cache for table detection per page
        self._summary_cache = {}  # Cache for table summaries
        self.logger.debug("Initialized caching for table detection")

    def _render_page_as_image(self, page_num: int) -> Image.Image:
        """
        Render a PDF page as an image for vision analysis.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            PIL Image of the page
        """
        page = self.doc[page_num]
        
        # Render page at 2x resolution for better table detection
        mat = pymupdf.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(BytesIO(img_data))
        
        return image
    
    def _find_caption_for_table(self, page_num: int, bbox: Optional[BoundingBox], page_text: str) -> Tuple[str, str]:
        """
        Find the caption for a table by searching nearby text with error handling.
        
        Args:
            page_num: Page number (1-indexed)
            bbox: Table bounding box (optional)
            page_text: Full text of the page
            
        Returns:
            Tuple of (table_number, caption) - uses placeholder if not found
        """
        try:
            # Look for table captions in the text
            # Common patterns: "Table 1:", "Table 2.", "TABLE 1:", etc.
            caption_patterns = [
                r'(Table\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
                r'(TABLE\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
                r'(Tab\.?\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)',
            ]
            
            for pattern in caption_patterns:
                try:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE)
                    for match in matches:
                        table_number = match.group(1).strip()
                        caption_text = match.group(2).strip()
                        
                        # Clean up caption (remove extra whitespace)
                        caption_text = ' '.join(caption_text.split())
                        
                        if table_number and caption_text:
                            return table_number, caption_text
                except Exception as e:
                    self.logger.debug(f"Error matching pattern {pattern}: {e}")
                    continue
            
            # No caption found - use placeholder
            self.logger.debug(f"No caption found for table on page {page_num}, using placeholder")
            return f"Table (Page {page_num})", ""
            
        except Exception as e:
            self.logger.warning(f"Error finding caption for table on page {page_num}: {e}")
            return f"Table (Page {page_num})", ""
    
    def detect_tables(self, page_range: Tuple[int, int], page_texts: List[Tuple[str, int]], batch_size: int = 3) -> List[TableNode]:
        """
        Detect all tables within a page range using vision-based analysis with batch processing.
        
        Uses gemini-2.5-flash-lite for fast detection. Processes multiple pages in batches
        to reduce API calls.
        
        Args:
            page_range: (start_page, end_page) tuple (1-indexed, inclusive)
            page_texts: List of (page_text, token_count) tuples for context extraction
            batch_size: Number of pages to process in a single batch
            
        Returns:
            List of TableNode objects with metadata
        """
        if not self.vision_client:
            self.logger.warning("Vision client not available - skipping table detection")
            return []
        
        start_page, end_page = page_range
        tables = []
        
        self.logger.info(f"Detecting tables from pages {start_page} to {end_page} (batch_size={batch_size})")
        
        # Process pages in batches
        page_nums = list(range(start_page - 1, end_page))
        
        for batch_start in range(0, len(page_nums), batch_size):
            batch_end = min(batch_start + batch_size, len(page_nums))
            batch_pages = page_nums[batch_start:batch_end]
            
            self.logger.debug(f"Processing batch {batch_start // batch_size + 1}/{(len(page_nums) + batch_size - 1) // batch_size}")
            
            # Detect tables in batch
            batch_tables = self._detect_tables_batch(batch_pages, page_texts)
            
            # Process each detected table
            for table_data, page_image, page_num in batch_tables:
                try:
                    
                    # Extract context around table (with error handling)
                    try:
                        context = self.extract_table_context(
                            page_num,
                            table_data.bbox,
                            page_texts
                        )
                    except Exception as e:
                        self.logger.warning(f"Error extracting context for table on page {page_num}: {e}")
                        context = ""
                    
                    # Generate summary using extraction model (with retry and fallback)
                    try:
                        summary = self._generate_table_summary(
                            page_image,
                            table_data,
                            context
                        )
                    except Exception as e:
                        self.logger.error(f"Error generating summary for table on page {page_num}: {e}")
                        # Use caption as fallback
                        summary = table_data.caption if table_data.caption else f"{table_data.table_number}"
                    
                    # Create table node
                    try:
                        table_node = TableNode(
                            table_number=table_data.table_number,
                            caption=table_data.caption if table_data.caption else "No caption available",
                            page=page_num,
                            bbox=table_data.bbox,
                            headers=table_data.headers,
                            key_values=table_data.key_values,
                            context=context,
                            summary=summary
                        )
                        tables.append(table_node)
                        
                        self.logger.debug(f"  - {table_data.table_number}")
                    except Exception as e:
                        self.logger.error(f"Error creating TableNode on page {page_num}: {e}")
                        continue
                    
                except Exception as e:
                    self.logger.warning(f"Error processing table: {e}")
                    continue
        
        self.logger.info(f"Total tables detected: {len(tables)}")
        return tables
    
    def _detect_tables_batch(self, page_nums: List[int], page_texts: List[Tuple[str, int]]) -> List[Tuple[DetectedTable, Image.Image, int]]:
        """
        Detect tables on a batch of pages.
        
        Args:
            page_nums: List of page numbers (0-indexed)
            page_texts: List of (page_text, token_count) tuples
            
        Returns:
            List of tuples (DetectedTable, page_image, page_num_1indexed)
        """
        results = []
        
        for page_num in page_nums:
            try:
                # Render page as image
                page_image = self._render_page_as_image(page_num)
                
                # Get page text for caption extraction
                page_text = page_texts[page_num][0] if page_num < len(page_texts) else ""
                
                # Detect tables on this page using fast model
                detected_tables = self._detect_tables_on_page(page_image, page_num + 1, page_text)
                
                if detected_tables:
                    self.logger.info(f"Found {len(detected_tables)} table(s) on page {page_num + 1}")
                    
                    # Add to results with page image and page number
                    for table_data in detected_tables:
                        results.append((table_data, page_image, page_num + 1))
                
            except Exception as e:
                self.logger.error(f"Error detecting tables on page {page_num + 1}: {e}")
                continue
        
        return results

    def _detect_tables_on_page(self, page_image: Image.Image, page_num: int, page_text: str, max_retries: int = 3) -> List[DetectedTable]:
        """
        Detect tables on a single page using vision API with retry logic and caching.
        
        Uses gemini-2.5-flash-lite for fast detection with exponential backoff on failures.
        
        Args:
            page_image: PIL Image of the page
            page_num: Page number (1-indexed)
            page_text: Text content of the page
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of DetectedTable objects
        """
        # Check cache first
        if page_num in self._detection_cache:
            self.logger.debug(f"Using cached table detection for page {page_num}")
            return self._detection_cache[page_num]
        
        # Detection prompt - focused on finding tables quickly
        detection_prompt = """Analyze this page and detect all tables present.

For each table found, provide:
1. Table number (e.g., "Table 1", "Table 2", "TABLE I") - extract from caption if visible
2. Caption text (if visible on the page)
3. Approximate bounding box as percentages (x_min, y_min, x_max, y_max where 0-100)
4. Column headers (list the main column headers)
5. Key values (extract 2-3 important data points as key-value pairs)

If no tables are found, respond with an empty list.

Format your response as JSON:
{
  "tables": [
    {
      "table_number": "Table 1",
      "caption": "Performance metrics...",
      "bbox": {"x_min": 10, "y_min": 20, "x_max": 90, "y_max": 50},
      "headers": ["Method", "Accuracy", "Speed"],
      "key_values": {"Best Method": "Method A", "Highest Accuracy": "95.2%"}
    }
  ]
}

Respond ONLY with valid JSON, no additional text."""

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                # Call vision API with detection model (fast) and JSON schema enforcement
                response = self.vision_client.models.generate_content(
                    model=self.detection_model,
                    contents=[page_image, detection_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=TableDetectionResponse.model_json_schema()
                    )
                )
                
                # Parse and validate response using Pydantic
                detection_response = TableDetectionResponse.model_validate_json(response.text)
                result = {"tables": [table.model_dump() for table in detection_response.tables]}
                
                detected_tables = []
                for table_data in result.get("tables", []):
                    try:
                        # Extract table number and caption
                        table_number = table_data.get("table_number", f"Table (Page {page_num})")
                        caption = table_data.get("caption", "")
                        
                        # If caption not found in vision response, try to find it in text
                        if not caption:
                            try:
                                bbox_data = table_data.get("bbox")
                                bbox = None
                                if bbox_data:
                                    bbox = BoundingBox(**bbox_data)
                                _, caption = self._find_caption_for_table(page_num, bbox, page_text)
                            except Exception as e:
                                self.logger.debug(f"Error finding caption from text: {e}")
                        
                        # Parse bounding box
                        bbox = None
                        if "bbox" in table_data:
                            try:
                                bbox = BoundingBox(**table_data["bbox"])
                            except Exception as e:
                                self.logger.warning(f"Invalid bbox for table on page {page_num}: {e}")
                        
                        # Extract headers and key values (with error handling)
                        try:
                            headers = table_data.get("headers", [])
                            if not isinstance(headers, list):
                                headers = []
                        except Exception as e:
                            self.logger.debug(f"Error extracting headers: {e}")
                            headers = []
                        
                        try:
                            key_values = table_data.get("key_values", {})
                            if not isinstance(key_values, dict):
                                key_values = {}
                        except Exception as e:
                            self.logger.debug(f"Error extracting key values: {e}")
                            key_values = {}
                        
                        detected_table = DetectedTable(
                            table_number=table_number,
                            caption=caption if caption else "No caption available",
                            headers=headers,
                            key_values=key_values,
                            bbox=bbox
                        )
                        detected_tables.append(detected_table)
                        
                    except Exception as e:
                        self.logger.warning(f"Error parsing table data on page {page_num}: {e}")
                        continue
                
                # Cache the result
                self._detection_cache[page_num] = detected_tables
                return detected_tables
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response for page {page_num} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.warning("Vision API failed after retries, falling back to text-based detection")
                    return self._fallback_text_based_detection(page_text, page_num)
                    
            except Exception as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Error in table detection on page {page_num} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Table detection failed after {max_retries} attempts, falling back to text-based detection")
                    return self._fallback_text_based_detection(page_text, page_num)
        
        # Should not reach here, but return empty list as final fallback
        return []
    
    def _fallback_text_based_detection(self, page_text: str, page_num: int) -> List[DetectedTable]:
        """
        Fallback method to detect tables from text when vision API fails.
        
        Args:
            page_text: Text content of the page
            page_num: Page number (1-indexed)
            
        Returns:
            List of DetectedTable objects (basic detection only)
        """
        detected_tables = []
        
        # Look for table captions in text
        caption_patterns = [
            r'(Table\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+)',
            r'(TABLE\s+\d+[a-zA-Z]?)[:\.]?\s*([^\n]+)',
        ]
        
        for pattern in caption_patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE)
            for match in matches:
                table_number = match.group(1).strip()
                caption = match.group(2).strip()
                
                detected_table = DetectedTable(
                    table_number=table_number,
                    caption=caption,
                    headers=[],
                    key_values={},
                    bbox=None
                )
                detected_tables.append(detected_table)
        
        return detected_tables

    def extract_table_structure(self, page_image: Image.Image, table_bbox: Optional[BoundingBox]) -> dict:
        """
        Extract detailed table structure (headers, rows, key values).
        
        Uses gemini-2.5-flash-lite for structure extraction.
        
        Args:
            page_image: PIL Image of the page
            table_bbox: Bounding box of the table (optional)
            
        Returns:
            Dictionary with structured table data
        """
        if not self.vision_client:
            return {"headers": [], "rows": [], "key_values": {}}
        
        try:
            # Crop image to table region if bbox available
            if table_bbox:
                width, height = page_image.size
                crop_box = (
                    int(table_bbox.x_min * width / 100),
                    int(table_bbox.y_min * height / 100),
                    int(table_bbox.x_max * width / 100),
                    int(table_bbox.y_max * height / 100)
                )
                table_image = page_image.crop(crop_box)
            else:
                table_image = page_image
            
            # Extraction prompt - focused on structure
            extraction_prompt = """Extract the structure of this table.

Provide:
1. Column headers (list all column headers from left to right)
2. Number of rows (approximate)
3. Key data points (extract 3-5 important values with their row/column context)

Format as JSON:
{
  "headers": ["Column 1", "Column 2", ...],
  "num_rows": 10,
  "key_values": {
    "Row 1, Column 2": "value",
    "Maximum value": "95.2%",
    ...
  }
}

Respond ONLY with valid JSON."""

            response = self.vision_client.models.generate_content(
                model=self.extraction_model,
                contents=[table_image, extraction_prompt],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=TableStructure.model_json_schema()
                )
            )
            
            # Parse and validate using Pydantic
            table_structure = TableStructure.model_validate_json(response.text)
            return table_structure.model_dump()
            
        except Exception as e:
            self.logger.warning(f"Error extracting table structure: {e}")
            return {"headers": [], "rows": [], "key_values": {}}
    
    def _generate_table_summary(self, page_image: Image.Image, table_data: DetectedTable, context: str, max_retries: int = 3) -> str:
        """
        Generate a summary of the table using vision API with retry logic and caching.
        
        Uses gemini-2.5-flash-lite for summary generation with exponential backoff.
        
        Args:
            page_image: PIL Image of the page
            table_data: DetectedTable object
            context: Surrounding text context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Summary string
        """
        # Create cache key from table number and caption
        import hashlib
        cache_key = hashlib.md5(f"{table_data.table_number}_{table_data.caption}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self._summary_cache:
            self.logger.debug(f"Using cached summary for {table_data.table_number}")
            return self._summary_cache[cache_key]
        
        # Try vision-based summary with retry
        for attempt in range(max_retries):
            try:
                # Crop to table region if bbox available
                if table_data.bbox:
                    try:
                        width, height = page_image.size
                        crop_box = (
                            int(table_data.bbox.x_min * width / 100),
                            int(table_data.bbox.y_min * height / 100),
                            int(table_data.bbox.x_max * width / 100),
                            int(table_data.bbox.y_max * height / 100)
                        )
                        table_image = page_image.crop(crop_box)
                    except Exception as e:
                        self.logger.warning(f"Error cropping table image: {e}, using full page")
                        table_image = page_image
                else:
                    table_image = page_image
                
                # Summary prompt
                summary_prompt = f"""Analyze this table and generate a concise summary (2-3 sentences).

Table: {table_data.table_number}
Caption: {table_data.caption}
Headers: {', '.join(table_data.headers) if table_data.headers else 'Not extracted'}

Context from paper:
{context[:800]}

Summary should describe:
1. What data the table presents
2. Key findings or trends in the data
3. Relevance to the research

Provide only the summary, no additional text."""

                response = self.vision_client.models.generate_content(
                    model=self.extraction_model,
                    contents=[table_image, summary_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0
                    )
                )
                
                summary = response.text.strip()
                if summary:
                    # Cache the result
                    self._summary_cache[cache_key] = summary
                    return summary
                
            except Exception as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Error generating table summary (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.warning("Vision API failed after retries, falling back to caption-based summary")
        
        # Fallback to caption-based summary
        return self._generate_caption_based_summary(table_data, context)
    
    def _generate_caption_based_summary(self, table_data: DetectedTable, context: str, max_retries: int = 3) -> str:
        """
        Generate summary from caption and context when vision API fails, with retry logic.
        
        Args:
            table_data: DetectedTable object
            context: Surrounding text context
            max_retries: Maximum number of retry attempts
            
        Returns:
            Summary string
        """
        # If no caption, return simple placeholder
        if not table_data.caption or table_data.caption == "No caption available":
            self.logger.debug(f"No caption for {table_data.table_number}, using placeholder")
            return f"{table_data.table_number}: Data table"
        
        # Try LLM-based summary with retry
        prompt = f"""Generate a concise summary (2-3 sentences) of this table based on its caption and context.

Table: {table_data.table_number}
Caption: {table_data.caption}
Headers: {', '.join(table_data.headers) if table_data.headers else 'Not available'}

Context:
{context[:1000]}

Summary should describe what the table presents and its relevance.
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
                self.logger.warning(f"Error generating caption-based summary (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
        
        # Final fallback: use caption directly
        self.logger.debug(f"Using caption as summary for {table_data.table_number}")
        return table_data.caption

    def extract_table_context(self, page_num: int, bbox: Optional[BoundingBox], 
                             page_texts: List[Tuple[str, int]]) -> str:
        """
        Extract surrounding text context for a table.
        
        Extracts ±2 paragraphs around the table location for context.
        Handles edge cases where tables are at page boundaries or span multiple pages.
        
        Args:
            page_num: Page number (1-indexed) where table is located
            bbox: BoundingBox object (optional, with coordinates as percentages)
            page_texts: List of (page_text, token_count) tuples
            
        Returns:
            Contextual text around the table
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
            
            # Determine table position on page (top, middle, bottom)
            table_position = 'middle'
            if bbox:
                y_center = (bbox.y_min + bbox.y_max) / 2
                if y_center < 33:
                    table_position = 'top'
                elif y_center > 67:
                    table_position = 'bottom'
            
            # Find paragraphs that mention the table
            table_mentions = []
            for i, para in enumerate(paragraphs):
                # Look for table references in the paragraph
                if re.search(r'\b[Tt]able\.?\s*\d+', para):
                    table_mentions.append(i)
            
            # Extract context paragraphs
            context_paragraphs = []
            
            if table_mentions:
                # Take paragraphs around the first mention
                mention_idx = table_mentions[0]
                start_idx = max(0, mention_idx - 2)
                end_idx = min(len(paragraphs), mention_idx + 3)
                context_paragraphs = paragraphs[start_idx:end_idx]
            else:
                # No explicit mention found, use position-based heuristic
                if table_position == 'top':
                    # Table at top, take first few paragraphs
                    context_paragraphs = paragraphs[:min(5, len(paragraphs))]
                elif table_position == 'bottom':
                    # Table at bottom, take last few paragraphs
                    context_paragraphs = paragraphs[max(0, len(paragraphs)-5):]
                else:
                    # Table in middle, take middle paragraphs
                    if len(paragraphs) > 4:
                        mid = len(paragraphs) // 2
                        context_paragraphs = paragraphs[max(0, mid-2):min(len(paragraphs), mid+3)]
                    else:
                        context_paragraphs = paragraphs
            
            context_text = '\n\n'.join(context_paragraphs)
            
            # Handle edge case: table at page boundaries
            # Add context from previous page if table is at top or context is short
            if (table_position == 'top' or len(context_text) < 300) and page_idx > 0:
                prev_page_text = page_texts[page_idx - 1][0]
                prev_paragraphs = [p.strip() for p in prev_page_text.split('\n\n') if p.strip()]
                if not prev_paragraphs:
                    prev_paragraphs = [p.strip() for p in prev_page_text.split('\n') if p.strip()]
                
                if prev_paragraphs:
                    # Add last 1-2 paragraphs from previous page
                    num_prev = min(2, len(prev_paragraphs))
                    prev_context = '\n\n'.join(prev_paragraphs[-num_prev:])
                    context_text = prev_context + '\n\n' + context_text
                    self.logger.debug(f"Added context from previous page for table on page {page_num}")
            
            # Add context from next page if table is at bottom or context is short
            if (table_position == 'bottom' or len(context_text) < 300) and page_idx < len(page_texts) - 1:
                next_page_text = page_texts[page_idx + 1][0]
                next_paragraphs = [p.strip() for p in next_page_text.split('\n\n') if p.strip()]
                if not next_paragraphs:
                    next_paragraphs = [p.strip() for p in next_page_text.split('\n') if p.strip()]
                
                if next_paragraphs:
                    # Add first 1-2 paragraphs from next page
                    num_next = min(2, len(next_paragraphs))
                    next_context = '\n\n'.join(next_paragraphs[:num_next])
                    context_text = context_text + '\n\n' + next_context
                    self.logger.debug(f"Added context from next page for table on page {page_num}")
            
            # Limit context length to reasonable size
            if len(context_text) > 2000:
                context_text = context_text[:2000] + "..."
            
            return context_text
            
        except Exception as e:
            self.logger.error(f"Error extracting context for table on page {page_num}: {e}")
            # Return partial page text as fallback
            try:
                page_idx = page_num - 1
                if 0 <= page_idx < len(page_texts):
                    return page_texts[page_idx][0][:1000]
            except:
                pass
            return ""
    
    def __del__(self):
        """Clean up PDF document on deletion."""
        try:
            if hasattr(self, 'doc') and self.doc is not None:
                self.doc.close()
        except:
            pass  # Ignore errors during cleanup
