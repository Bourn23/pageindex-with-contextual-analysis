"""
Semantic analysis component for granular PageIndex node generation.

This module provides functionality to analyze section content and identify
semantic boundaries for subdivision into coherent sub-sections.
"""

import logging
import json
import re
import time
import os
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class SemanticUnit(BaseModel):
    """Data structure representing a semantic sub-section."""
    title: str = Field(..., description="Descriptive title for the semantic unit")
    start_paragraph: int = Field(..., description="Starting paragraph index (0-indexed)")
    end_paragraph: int = Field(..., description="Ending paragraph index (0-indexed, inclusive)")
    start_page: int = Field(..., description="Starting page number (1-indexed)")
    end_page: int = Field(..., description="Ending page number (1-indexed)")
    semantic_type: str = Field(..., description="Type of semantic unit (e.g., 'motivation', 'methodology', 'results')")
    summary: str = Field(default="", description="LLM-generated summary of the semantic unit")


class SemanticUnitData(BaseModel):
    """Individual semantic unit data from LLM response."""
    title: str = Field(..., description="Descriptive title for the semantic unit")
    start_paragraph: int = Field(..., description="Starting paragraph index (0-indexed)")
    end_paragraph: int = Field(..., description="Ending paragraph index (0-indexed, inclusive)")
    semantic_type: str = Field(..., description="Type of semantic unit")
    summary: str = Field(default="", description="Brief summary of the semantic unit")


class SemanticAnalysisResponse(BaseModel):
    """Response containing all semantic units identified in a section."""
    semantic_units: List[SemanticUnitData] = Field(default_factory=list, description="List of semantic units identified")


class KeywordData(BaseModel):
    """Individual keyword/concept extracted from text."""
    term: str = Field(..., description="The keyword or key concept")
    context: str = Field(..., description="Brief context or definition (1 sentence)")
    relevance: str = Field(..., description="Why this keyword is important to the section")


class KeywordExtractionResponse(BaseModel):
    """Response containing keywords extracted from a section."""
    keywords: List[KeywordData] = Field(default_factory=list, description="List of keywords/concepts identified")


class SemanticAnalyzer:
    """
    Analyzes section content and identifies semantic boundaries for subdivision.
    
    Uses LLM to intelligently identify coherent sub-topics within sections
    based on the section type (Introduction, Methods, Results, Discussion, etc.).
    """
    
    def __init__(self, llm_client, logger: Optional[logging.Logger] = None):
        """
        Initialize the SemanticAnalyzer.
        
        Args:
            llm_client: LLM client instance for semantic analysis
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Gemini client for structured output
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            self.logger.warning("GEMINI_API_KEY not found - will use fallback LLM client")
        
        # Section-type-aware prompt templates
        self.section_prompts = self._initialize_section_prompts()
        
        # Keyword extraction prompt
        self.keyword_prompt = self._initialize_keyword_prompt()
    
    def _initialize_keyword_prompt(self) -> str:
        """
        Initialize the keyword extraction prompt template.
        
        Returns:
            Keyword extraction prompt string
        """
        return """Extract key terms and concepts from this text section.

Your task is to identify the most important keywords, technical terms, and concepts that:
1. Are central to understanding this section
2. Would be useful for information retrieval (RAG queries)
3. Represent domain-specific terminology or important ideas

For each keyword/concept, provide:
1. term: The keyword or key phrase (2-5 words max)
2. context: A brief 1-sentence explanation or definition
3. relevance: Why this term is important to this section (1 sentence)

Section Title: {section_title}

Section Text:
{section_text}

Respond with JSON in this format:
{{
  "keywords": [
    {{
      "term": "composite polymer electrolyte",
      "context": "A material combining polymer matrix with ceramic fillers for battery applications",
      "relevance": "Core material being studied in this research"
    }},
    {{
      "term": "ionic conductivity",
      "context": "Measure of how well ions move through a material",
      "relevance": "Primary performance metric being measured"
    }}
  ]
}}

IMPORTANT:
- Extract 5-15 keywords depending on section length
- Focus on technical terms, methodologies, materials, and key concepts
- Avoid generic words like "study", "research", "paper"
- Keep terms concise and specific
- Respond ONLY with valid JSON, no additional text."""
    
    def _initialize_section_prompts(self) -> Dict[str, str]:
        """
        Initialize section-type-aware prompt templates.
        
        Returns:
            Dictionary mapping section types to their analysis prompts
        """
        return {
            'introduction': """Analyze this Introduction section and identify semantic sub-sections.

Common semantic units in an Introduction:
- Motivation: Why is this research important?
- Background/Literature Review: What has been done before?
- Research Gap: What is missing in current knowledge?
- Contribution/Objectives: What does this paper contribute?

For each semantic unit you identify, provide:
1. title: A descriptive title, ensure to include sufficient information for future RAG queries (retrieval augmented generation) while maintaining brevity (e.g., "Motivation and Background")
2. start_paragraph: Index of the first paragraph (0-indexed)
3. end_paragraph: Index of the last paragraph (0-indexed, inclusive)
4. semantic_type: One of: motivation, background, literature_review, research_gap, contribution, objectives
5. summary: A 1-2 sentence summary of what this unit covers

Section text with paragraph indices:
{text_with_indices}

Respond with JSON in this format:
{{
  "semantic_units": [
    {{
      "title": "Motivation and Background",
      "start_paragraph": 0,
      "end_paragraph": 2,
      "semantic_type": "motivation",
      "summary": "Discusses the importance of..."
    }}
  ]
}}

IMPORTANT:
- Paragraphs must be contiguous (no gaps)
- Each paragraph should belong to exactly one unit
- If the section is too short (< 3 paragraphs), return an empty list
- Respond ONLY with valid JSON, no additional text.""",

            'methods': """Analyze this Methods/Experimental section and identify semantic sub-sections.

Common semantic units in Methods:
- Materials: What materials/samples were used?
- Sample Preparation: How were samples prepared?
- Characterization Techniques: What measurement/analysis methods were used?
- Analysis Procedures: How was data analyzed?
- Experimental Setup: How was the experiment configured?

For each semantic unit you identify, provide:
1. title: A descriptive title (e.g., "Sample Preparation")
2. start_paragraph: Index of the first paragraph (0-indexed)
3. end_paragraph: Index of the last paragraph (0-indexed, inclusive)
4. semantic_type: One of: materials, sample_preparation, characterization, analysis, experimental_setup, procedure
5. summary: A 1-2 sentence summary of what this unit covers

Section text with paragraph indices:
{text_with_indices}

Respond with JSON in this format:
{{
  "semantic_units": [
    {{
      "title": "Sample Preparation",
      "start_paragraph": 0,
      "end_paragraph": 1,
      "semantic_type": "sample_preparation",
      "summary": "Describes how samples were prepared..."
    }}
  ]
}}

IMPORTANT:
- Paragraphs must be contiguous (no gaps)
- Each paragraph should belong to exactly one unit
- If the section is too short (< 3 paragraphs), return an empty list
- Respond ONLY with valid JSON, no additional text.""",

            'results': """Analyze this Results section and identify semantic sub-sections.

Common semantic units in Results:
- Individual Experiments: Each distinct experiment or analysis
- Specific Findings: Groups of related findings
- Data Presentations: Sections focused on specific datasets or measurements

For each semantic unit you identify, provide:
1. title: A descriptive title with enough information for future RAG queries
2. start_paragraph: Index of the first paragraph (0-indexed)
3. end_paragraph: Index of the last paragraph (0-indexed, inclusive)
4. semantic_type: One of: experiment, finding, measurement, observation, analysis
5. summary: A 1-2 sentence summary of what this unit covers

Section text with paragraph indices:
{text_with_indices}

Respond with JSON in this format:
{{
  "semantic_units": [
    {{
      "title": "Conductivity Measurements",
      "start_paragraph": 0,
      "end_paragraph": 2,
      "semantic_type": "measurement",
      "summary": "Presents conductivity data for..."
    }}
  ]
}}

IMPORTANT:
- Paragraphs must be contiguous (no gaps)
- Each paragraph should belong to exactly one unit
- If the section is too short (< 3 paragraphs), return an empty list
- Respond ONLY with valid JSON, no additional text.""",

            'discussion': """Analyze this Discussion section and identify semantic sub-sections.

Common semantic units in Discussion:
- Interpretation: Explaining what the results mean
- Comparison with Literature: How results compare to previous work
- Implications: What the findings mean for the field
- Limitations: Acknowledging constraints or limitations
- Future Work: Suggestions for future research

For each semantic unit you identify, provide:
1. title: A descriptive title (e.g., "Interpretation of Results")
2. start_paragraph: Index of the first paragraph (0-indexed)
3. end_paragraph: Index of the last paragraph (0-indexed, inclusive)
4. semantic_type: One of: interpretation, comparison, implications, limitations, future_work
5. summary: A 1-2 sentence summary of what this unit covers

Section text with paragraph indices:
{text_with_indices}

Respond with JSON in this format:
{{
  "semantic_units": [
    {{
      "title": "Interpretation of Results",
      "start_paragraph": 0,
      "end_paragraph": 1,
      "semantic_type": "interpretation",
      "summary": "Explains the meaning of..."
    }}
  ]
}}

IMPORTANT:
- Paragraphs must be contiguous (no gaps)
- Each paragraph should belong to exactly one unit
- If the section is too short (< 3 paragraphs), return an empty list
- Respond ONLY with valid JSON, no additional text.""",

            'default': """Analyze this section and identify semantic sub-sections based on topic shifts.

For each semantic unit you identify, provide:
1. title: A descriptive title
2. start_paragraph: Index of the first paragraph (0-indexed)
3. end_paragraph: Index of the last paragraph (0-indexed, inclusive)
4. semantic_type: A brief descriptor of the content type
5. summary: A 1-2 sentence summary of what this unit covers

Section text with paragraph indices:
{text_with_indices}

Respond with JSON in this format:
{{
  "semantic_units": [
    {{
      "title": "Descriptive Title",
      "start_paragraph": 0,
      "end_paragraph": 2,
      "semantic_type": "content_type",
      "summary": "Brief summary..."
    }}
  ]
}}

IMPORTANT:
- Paragraphs must be contiguous (no gaps)
- Each paragraph should belong to exactly one unit
- If the section is too short (< 3 paragraphs), return an empty list
- Respond ONLY with valid JSON, no additional text."""
        }

    
    def _detect_section_type(self, section_title: str, section_text: str = "") -> str:
        """
        Detect the type of section using LLM analysis.
        
        This method uses LLM to intelligently classify section titles, which is more
        robust than keyword matching. It handles:
        - Variations in terminology across different papers
        - Compound section titles (e.g., "Results and Discussion")
        - Domain-specific section names
        - Numbered sections without clear keywords
        
        Args:
            section_title: Title of the section
            section_text: Optional text content for better classification (not currently used)
            
        Returns:
            Section type string (introduction, methods, results, discussion, or default)
        """
        try:
            # Create prompt for section type detection
            prompt = f"""Classify this section title into ONE of these categories:

Categories:
- introduction: Introduces topic, provides background, motivation, literature review, or states objectives/contributions
- methods: Describes experimental procedures, materials, techniques, methodology, or approach
- results: Presents findings, data, measurements, observations, or experimental outcomes
- discussion: Interprets results, compares with literature, discusses implications, limitations, or conclusions

Section Title: "{section_title}"

Rules:
- If the title contains multiple aspects (e.g., "Results and Discussion"), choose the PRIMARY focus
- Common section titles:
  * "Introduction", "Background" → introduction
  * "Methods", "Experimental", "Materials and Methods", "Methodology" → methods
  * "Results", "Findings", "Observations", "Results and Discussion" → results
  * "Discussion", "Conclusion", "Implications" → discussion
- If unclear or doesn't fit any category, use "default"

Respond with ONLY ONE WORD: introduction, methods, results, discussion, or default

Classification:"""

            # Determine model to use
            if hasattr(self.llm_client, 'provider') and self.llm_client.provider == 'gemini':
                model = 'gemini-2.5-flash-lite'
            else:
                model = getattr(self.llm_client, 'default_model', 'gpt-4o-mini')
            
            # Call LLM
            response = self.llm_client.chat_completion(
                model=model,
                prompt=prompt,
                temperature=0
            )
            
            if response and response != "Error":
                section_type = response.strip().lower()
                
                # Validate response
                valid_types = ['introduction', 'methods', 'results', 'discussion', 'default']
                if section_type in valid_types:
                    return section_type
            
            # Fallback to default
            self.logger.debug(f"Could not classify section '{section_title}', using default")
            return 'default'
            
        except Exception as e:
            self.logger.warning(f"Error detecting section type for '{section_title}': {e}")
            return 'default'
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Full text to split
            
        Returns:
            List of paragraph strings
        """
        # Split by double newlines first
        paragraphs = []
        for p in text.split('\n\n'):
            p = p.strip()
            if p:
                paragraphs.append(p)
        
        # If no double-newline paragraphs, try single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        return paragraphs
    
    def _create_text_with_indices(self, paragraphs: List[str]) -> str:
        """
        Create text with paragraph indices for LLM analysis.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            Text with paragraph indices
        """
        indexed_text = []
        for i, para in enumerate(paragraphs):
            indexed_text.append(f"[Paragraph {i}]\n{para}\n")
        
        return '\n'.join(indexed_text)
    
    def _map_paragraphs_to_pages(self, paragraphs: List[str], full_text: str, 
                                 start_page: int, end_page: int, 
                                 page_texts: List[Tuple[str, int]]) -> List[int]:
        """
        Map each paragraph to its page number.
        
        Args:
            paragraphs: List of paragraph strings
            full_text: Full section text
            start_page: Starting page of section (1-indexed)
            end_page: Ending page of section (1-indexed)
            page_texts: List of (page_text, token_count) tuples
            
        Returns:
            List of page numbers (1-indexed) for each paragraph
        """
        paragraph_pages = []
        
        # Build cumulative text for each page in the section
        page_cumulative_text = []
        cumulative = ""
        for page_num in range(start_page - 1, end_page):
            if page_num < len(page_texts):
                cumulative += page_texts[page_num][0]
                page_cumulative_text.append((page_num + 1, cumulative))
        
        # For each paragraph, find which page it belongs to
        for para in paragraphs:
            # Find the paragraph in the cumulative text
            para_position = full_text.find(para)
            
            if para_position == -1:
                # Fallback: assign to start page
                paragraph_pages.append(start_page)
                continue
            
            # Find which page this position corresponds to
            assigned_page = start_page
            for page_num, cum_text in page_cumulative_text:
                if len(cum_text) >= para_position:
                    assigned_page = page_num
                    break
            
            paragraph_pages.append(assigned_page)
        
        return paragraph_pages
    
    def analyze_section(self, section_node: dict, page_texts: List[Tuple[str, int]], 
                       min_pages: float = 0.5) -> List[SemanticUnit]:
        """
        Analyze a section and identify semantic sub-units.
        
        Args:
            section_node: Existing section node from PageIndex with 'title', 'text', 
                         'start_index', 'end_index' fields
            page_texts: List of (page_text, token_count) tuples for page mapping
            min_pages: Minimum section length (in pages) to attempt subdivision
            
        Returns:
            List of SemanticUnit objects representing sub-sections
        """
        try:
            # Extract section information
            section_title = section_node.get('title', '')
            full_text = section_node.get('text', '')
            start_page = section_node.get('start_index', 1)
            end_page = section_node.get('end_index', 1)
            
            # Check if section is too short
            section_length = end_page - start_page + 1
            if section_length < min_pages:
                self.logger.debug(f"Section '{section_title}' is too short ({section_length} pages) - skipping subdivision")
                return []
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(full_text)
            
            if len(paragraphs) < 3:
                self.logger.debug(f"Section '{section_title}' has too few paragraphs ({len(paragraphs)}) - skipping subdivision")
                return []
            
            self.logger.info(f"Analyzing section '{section_title}' ({len(paragraphs)} paragraphs, {section_length} pages)")
            
            # Detect section type
            section_type = self._detect_section_type(section_title)
            self.logger.debug(f"Detected section type: {section_type}")
            
            # Create text with paragraph indices
            text_with_indices = self._create_text_with_indices(paragraphs)
            
            # Get appropriate prompt template
            prompt_template = self.section_prompts.get(section_type, self.section_prompts['default'])
            prompt = prompt_template.format(text_with_indices=text_with_indices)
            
            # Call LLM for semantic analysis
            semantic_units = self._call_llm_for_analysis(prompt, section_title)
            
            if not semantic_units:
                self.logger.debug(f"No semantic units identified for section '{section_title}'")
                return []
            
            # Map paragraphs to pages
            paragraph_pages = self._map_paragraphs_to_pages(
                paragraphs, full_text, start_page, end_page, page_texts
            )
            
            # Convert to SemanticUnit objects with page ranges
            semantic_unit_objects = []
            for unit_data in semantic_units:
                try:
                    start_para = unit_data['start_paragraph']
                    end_para = unit_data['end_paragraph']
                    
                    # Validate paragraph indices
                    if start_para < 0 or end_para >= len(paragraphs) or start_para > end_para:
                        self.logger.warning(f"Invalid paragraph range [{start_para}, {end_para}] for unit '{unit_data.get('title', 'Unknown')}'")
                        continue
                    
                    # Get page range for this unit
                    unit_start_page = paragraph_pages[start_para]
                    unit_end_page = paragraph_pages[end_para]
                    
                    semantic_unit = SemanticUnit(
                        title=unit_data['title'],
                        start_paragraph=start_para,
                        end_paragraph=end_para,
                        start_page=unit_start_page,
                        end_page=unit_end_page,
                        semantic_type=unit_data.get('semantic_type', 'general'),
                        summary=unit_data.get('summary', '')
                    )
                    semantic_unit_objects.append(semantic_unit)
                    
                except Exception as e:
                    self.logger.warning(f"Error creating SemanticUnit: {e}")
                    continue
            
            # Validate non-overlapping boundaries
            validated_units = self._validate_boundaries(semantic_unit_objects, len(paragraphs))
            
            self.logger.info(f"Identified {len(validated_units)} semantic units in section '{section_title}'")
            
            return validated_units
            
        except Exception as e:
            self.logger.error(f"Error analyzing section: {e}")
            return []
    
    def _call_llm_for_analysis(self, prompt: str, section_title: str, max_retries: int = 3) -> List[dict]:
        """
        Call LLM to perform semantic analysis with structured output.
        
        With Gemini's structured output, the response is guaranteed to be valid JSON
        matching the schema, so no cleanup or fallback parsing is needed.
        
        Args:
            prompt: Analysis prompt
            section_title: Title of the section being analyzed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of semantic unit dictionaries
        """
        if not self.gemini_client:
            self.logger.error("Gemini client not available - GEMINI_API_KEY not set")
            return []
        
        return self._call_gemini_with_schema(prompt, section_title, max_retries)
    
    def _call_gemini_with_schema(self, prompt: str, section_title: str, max_retries: int = 3) -> List[dict]:
        """
        Call Gemini with JSON schema enforcement for structured output.
        
        With structured output, the response is guaranteed to be valid JSON
        matching the schema - no cleanup needed!
        
        Args:
            prompt: Analysis prompt
            section_title: Title of the section being analyzed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of semantic unit dictionaries
        """
        model = 'gemini-2.5-flash-lite'
        
        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                # Call Gemini with JSON schema enforcement
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_json_schema=SemanticAnalysisResponse.model_json_schema()
                    )
                )
                
                # With structured output, response.text is guaranteed to be valid JSON!
                analysis_response = SemanticAnalysisResponse.model_validate_json(response.text)
                
                # Convert to list of dicts
                semantic_units = [unit.model_dump() for unit in analysis_response.semantic_units]
                
                if semantic_units:
                    return semantic_units
                else:
                    self.logger.debug(f"No semantic units found in response for '{section_title}'")
                    return []
                
            except Exception as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Error calling Gemini for analysis (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    self.logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Gemini analysis failed after {max_retries} attempts")
                    return []
        
        return []
    
    def _parse_llm_response(self, response: str) -> List[dict]:
        """
        Parse LLM response to extract semantic units with robust error handling.
        
        Args:
            response: LLM response text
            
        Returns:
            List of semantic unit dictionaries
        """
        try:
            # Extract JSON from response
            json_text = response.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith('```'):
                try:
                    # Find the first { and last }
                    start_idx = json_text.find('{')
                    end_idx = json_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_text = json_text[start_idx:end_idx+1]
                    else:
                        self.logger.warning("Could not extract JSON from markdown code block")
                        return []
                except Exception as e:
                    self.logger.warning(f"Error extracting JSON from markdown: {e}")
                    return []
            
            # Parse JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response: {e}")
                self.logger.debug(f"Response text: {response[:500]}")
                return []
            
            # Extract semantic units
            try:
                semantic_units = data.get('semantic_units', [])
                
                if not isinstance(semantic_units, list):
                    self.logger.warning(f"semantic_units is not a list, got {type(semantic_units)}")
                    return []
                
                # Validate each semantic unit has required fields
                validated_units = []
                for unit in semantic_units:
                    try:
                        if not isinstance(unit, dict):
                            self.logger.debug(f"Skipping non-dict semantic unit: {unit}")
                            continue
                        
                        # Check required fields
                        required_fields = ['title', 'start_paragraph', 'end_paragraph', 'semantic_type']
                        if not all(field in unit for field in required_fields):
                            self.logger.debug(f"Semantic unit missing required fields: {unit}")
                            continue
                        
                        validated_units.append(unit)
                    except Exception as e:
                        self.logger.debug(f"Error validating semantic unit: {e}")
                        continue
                
                return validated_units
                
            except Exception as e:
                self.logger.warning(f"Error extracting semantic_units from data: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _validate_boundaries(self, semantic_units: List[SemanticUnit], 
                            total_paragraphs: int) -> List[SemanticUnit]:
        """
        Validate that semantic unit boundaries are non-overlapping and contiguous.
        Handles invalid boundaries by merging overlapping units.
        
        Args:
            semantic_units: List of SemanticUnit objects
            total_paragraphs: Total number of paragraphs in the section
            
        Returns:
            Validated list of SemanticUnit objects
        """
        if not semantic_units:
            return []
        
        try:
            # Sort by start paragraph
            sorted_units = sorted(semantic_units, key=lambda u: u.start_paragraph)
            
            validated = []
            last_end = -1
            
            for unit in sorted_units:
                try:
                    # Check for overlap with previous unit
                    if unit.start_paragraph <= last_end:
                        self.logger.warning(f"Overlapping unit detected: '{unit.title}' starts at {unit.start_paragraph}, previous ended at {last_end}")
                        
                        # Try to merge with previous unit if overlap is significant
                        if validated and unit.start_paragraph < last_end - 1:
                            prev_unit = validated[-1]
                            self.logger.info(f"Merging overlapping units: '{prev_unit.title}' and '{unit.title}'")
                            
                            # Extend previous unit to include this one
                            prev_unit.end_paragraph = max(prev_unit.end_paragraph, unit.end_paragraph)
                            prev_unit.end_page = max(prev_unit.end_page, unit.end_page)
                            prev_unit.title = f"{prev_unit.title} and {unit.title}"
                            prev_unit.summary = f"{prev_unit.summary} {unit.summary}"
                            
                            last_end = prev_unit.end_paragraph
                            continue
                        else:
                            # Adjust start to avoid overlap
                            unit.start_paragraph = last_end + 1
                            
                            # If adjustment makes unit invalid, skip it
                            if unit.start_paragraph > unit.end_paragraph:
                                self.logger.warning(f"Skipping unit '{unit.title}' due to overlap")
                                continue
                    
                    # Check for gaps (optional - we allow gaps)
                    if unit.start_paragraph > last_end + 1:
                        self.logger.debug(f"Gap detected before unit '{unit.title}': paragraphs {last_end + 1} to {unit.start_paragraph - 1}")
                    
                    # Validate end paragraph
                    if unit.end_paragraph >= total_paragraphs:
                        self.logger.warning(f"Unit '{unit.title}' end paragraph {unit.end_paragraph} exceeds total {total_paragraphs}")
                        unit.end_paragraph = total_paragraphs - 1
                    
                    # Validate start paragraph is within bounds
                    if unit.start_paragraph < 0:
                        self.logger.warning(f"Unit '{unit.title}' start paragraph {unit.start_paragraph} is negative, adjusting to 0")
                        unit.start_paragraph = 0
                    
                    # Final check that unit is valid
                    if unit.start_paragraph <= unit.end_paragraph:
                        validated.append(unit)
                        last_end = unit.end_paragraph
                    else:
                        self.logger.warning(f"Skipping invalid unit '{unit.title}': start {unit.start_paragraph} > end {unit.end_paragraph}")
                    
                except Exception as e:
                    self.logger.warning(f"Error validating unit '{unit.title}': {e}")
                    continue
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error in boundary validation: {e}")
            return []
    
    def identify_boundaries(self, text: str, section_type: str) -> List[int]:
        """
        Identify paragraph indices that represent semantic boundaries.
        
        This is a simplified interface that returns just the boundary indices.
        
        Args:
            text: Section text
            section_type: Type of section (e.g., "Introduction", "Methods")
            
        Returns:
            List of paragraph indices marking boundaries
        """
        try:
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(text)
            
            if len(paragraphs) < 3:
                return []
            
            # Create a minimal section node for analysis
            section_node = {
                'title': section_type,
                'text': text,
                'start_index': 1,
                'end_index': 1
            }
            
            # Analyze section (without page mapping)
            semantic_units = self.analyze_section(section_node, [(text, 0)], min_pages=0)
            
            # Extract boundary indices
            boundaries = []
            for unit in semantic_units:
                boundaries.append(unit.start_paragraph)
            
            # Add end boundary
            if semantic_units:
                boundaries.append(semantic_units[-1].end_paragraph + 1)
            
            return sorted(set(boundaries))
            
        except Exception as e:
            self.logger.error(f"Error identifying boundaries: {e}")
            return []
    
    def extract_keywords(self, section_node: dict, max_retries: int = 3) -> List[dict]:
        """
        Extract keywords and key concepts from a section.
        
        This method uses Gemini with structured output to extract important
        keywords, technical terms, and concepts from a section for the
        "keywords" granularity level.
        
        Args:
            section_node: Section node with 'title' and 'text' fields
            max_retries: Maximum retry attempts
            
        Returns:
            List of keyword dictionaries with 'term', 'context', 'relevance'
        """
        if not self.gemini_client:
            self.logger.error("Gemini client not available - GEMINI_API_KEY not set")
            return []
        
        try:
            section_title = section_node.get('title', 'Unknown Section')
            section_text = section_node.get('text', '')
            
            if not section_text.strip():
                self.logger.debug(f"Empty text for section '{section_title}' - skipping keyword extraction")
                return []
            
            # Truncate very long sections to avoid token limits
            max_chars = 15000
            if len(section_text) > max_chars:
                section_text = section_text[:max_chars] + "\n\n[Text truncated for keyword extraction]"
            
            # Format prompt
            prompt = self.keyword_prompt.format(
                section_title=section_title,
                section_text=section_text
            )
            
            self.logger.info(f"Extracting keywords from section '{section_title}'")
            
            # Call Gemini with structured output
            model = 'gemini-2.5-flash-lite'
            
            for attempt in range(max_retries):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type="application/json",
                            response_json_schema=KeywordExtractionResponse.model_json_schema()
                        )
                    )
                    
                    # Parse structured response
                    keyword_response = KeywordExtractionResponse.model_validate_json(response.text)
                    keywords = [kw.model_dump() for kw in keyword_response.keywords]
                    
                    self.logger.info(f"Extracted {len(keywords)} keywords from '{section_title}'")
                    return keywords
                    
                except Exception as e:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Error extracting keywords (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Keyword extraction failed after {max_retries} attempts")
                        return []
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in keyword extraction: {e}")
            return []
    
    def create_keyword_nodes(self, keywords: List[dict], section_node: dict) -> List[dict]:
        """
        Create node structures from extracted keywords.
        
        Args:
            keywords: List of keyword dictionaries
            section_node: Parent section node (the immediate parent semantic unit)
            
        Returns:
            List of keyword node dictionaries
        """
        nodes = []
        
        for kw in keywords:
            try:
                # Create a rich text field that includes both context and parent info
                text_parts = [
                    f"**Keyword:** {kw['term']}",
                    f"**Context:** {kw['context']}",
                    f"**Relevance:** {kw['relevance']}",
                ]
                
                # Add parent section info for context
                if section_node.get('title'):
                    text_parts.append(f"\n**From Section:** {section_node['title']}")
                
                # Use parent's text directly - this should be the immediate parent's text
                # NOT the grandparent's text
                parent_text = section_node.get('text', '')
                if parent_text:
                    text_parts.append(f"\n**Parent Section Text:**\n{parent_text}")
                
                # Use parent's text directly - this preserves the immediate parent's context
                parent_text = section_node.get('text', '')
                
                node = {
                    'title': kw['term'],
                    'start_index': section_node.get('start_index', 1),
                    'end_index': section_node.get('end_index', 1),
                    'text': parent_text,  # Use parent's text directly
                    'summary': kw['context'],
                    'node_type': 'keyword',
                    '_text_locked': True,  # Flag to prevent text extraction
                    'metadata': {
                        'term': kw['term'],
                        'context': kw['context'],
                        'relevance': kw['relevance'],
                        'parent_title': section_node.get('title', 'Unknown'),
                        'parent_node_type': section_node.get('node_type', 'section')
                    },
                    'nodes': []
                }
                
                nodes.append(node)
                
            except Exception as e:
                self.logger.warning(f"Error creating keyword node: {e}")
                continue
        
        return nodes
    
    def create_nodes_from_semantic_units(self, semantic_units: List[SemanticUnit], 
                                        section_node: dict, 
                                        page_texts: List[Tuple[str, int]]) -> List[dict]:
        """
        Create node structures from SemanticUnit objects.
        
        Ensures unique titles by appending numbers to duplicates (e.g., "Background", "Background (2)").
        
        Args:
            semantic_units: List of SemanticUnit objects
            section_node: Parent section node
            page_texts: List of (page_text, token_count) tuples
            
        Returns:
            List of node dictionaries ready to be inserted into tree
        """
        nodes = []
        title_counts = {}  # Track title occurrences
        
        full_text = section_node.get('text', '')
        paragraphs = self._split_into_paragraphs(full_text)
        
        for unit in semantic_units:
            try:
                # Extract text for this semantic unit
                unit_paragraphs = paragraphs[unit.start_paragraph:unit.end_paragraph + 1]
                unit_text = '\n\n'.join(unit_paragraphs)
                
                # Make title unique if duplicate
                original_title = unit.title
                unique_title = original_title
                
                if original_title in title_counts:
                    title_counts[original_title] += 1
                    unique_title = f"{original_title} ({title_counts[original_title]})"
                    self.logger.debug(f"Duplicate title '{original_title}' renamed to '{unique_title}'")
                else:
                    title_counts[original_title] = 1
                
                # Create node structure
                node = {
                    'title': unique_title,
                    'start_index': unit.start_page,
                    'end_index': unit.end_page,
                    'text': unit_text,
                    'summary': unit.summary,
                    'node_type': 'semantic_unit',
                    'metadata': {
                        'semantic_type': unit.semantic_type,
                        'start_paragraph': unit.start_paragraph,
                        'end_paragraph': unit.end_paragraph,
                        'original_title': original_title  # Keep original for reference
                    },
                    'nodes': []
                }
                
                nodes.append(node)
                
            except Exception as e:
                self.logger.warning(f"Error creating node for semantic unit '{unit.title}': {e}")
                continue
        
        return nodes
