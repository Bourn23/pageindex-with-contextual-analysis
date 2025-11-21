"""
Demo script for FigureDetector - shows how to detect figures in a PDF.

This script demonstrates:
1. How to initialize the FigureDetector
2. How to detect figures in a page range
3. What data is returned (figure nodes with metadata)
4. How the caching and batch processing work
"""

import os
import sys
import logging
from pathlib import Path

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.figure_detector import FigureDetector, FigureNode
from pageindex.llm_client import get_llm_client
import pymupdf

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_figure_detection(pdf_path: str):
    """
    Demonstrate figure detection on a PDF.
    
    Args:
        pdf_path: Path to PDF file
    """
    logger.info("=" * 80)
    logger.info("FIGURE DETECTOR DEMO")
    logger.info("=" * 80)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Check if GEMINI_API_KEY is set (required for vision analysis)
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set - vision analysis will be limited")
        logger.info("Set GEMINI_API_KEY in .env file for full functionality")
    
    try:
        # Step 1: Initialize LLM client
        logger.info("\n1. Initializing LLM client...")
        llm_client = get_llm_client()
        logger.info(f"   ✓ LLM client initialized (provider: {llm_client.provider})")
        
        # Step 2: Initialize FigureDetector
        logger.info("\n2. Initializing FigureDetector...")
        detector = FigureDetector(
            llm_client=llm_client,
            pdf_path=pdf_path,
            logger=logger
        )
        logger.info("   ✓ FigureDetector initialized")
        logger.info(f"   - PDF has {len(detector.doc)} pages")
        
        # Step 3: Extract page texts (needed for context extraction)
        logger.info("\n3. Extracting page texts...")
        page_texts = []
        for page_num in range(len(detector.doc)):
            page = detector.doc[page_num]
            text = page.get_text()
            page_texts.append((text, len(text.split())))
        logger.info(f"   ✓ Extracted text from {len(page_texts)} pages")
        
        # Step 4: Detect figures in a page range
        # Let's detect figures in the first 5 pages
        start_page = 1
        end_page = min(5, len(detector.doc))
        
        logger.info(f"\n4. Detecting figures in pages {start_page}-{end_page}...")
        logger.info("   (This may take a moment - using vision API for analysis)")
        logger.info("   Note: If no embedded images found, will use vision-based detection on rendered pages")
        
        figures = detector.detect_figures(
            page_range=(start_page, end_page),
            page_texts=page_texts,
            batch_size=5,  # Process up to 5 images at once
            use_vision_fallback=True  # Enable vision-based detection for vector graphics
        )
        
        logger.info(f"   ✓ Detection complete: Found {len(figures)} figure(s)")
        
        # Step 5: Display results
        if figures:
            logger.info("\n5. Figure Details:")
            logger.info("=" * 80)
            
            for i, fig in enumerate(figures, 1):
                logger.info(f"\n   Figure {i}:")
                logger.info(f"   - Number: {fig.figure_number}")
                logger.info(f"   - Page: {fig.page}")
                logger.info(f"   - Type: {fig.figure_type}")
                logger.info(f"   - Caption: {fig.caption[:100]}..." if len(fig.caption) > 100 else f"   - Caption: {fig.caption}")
                
                if fig.bbox:
                    logger.info(f"   - Bounding Box: ({fig.bbox.x_min:.1f}, {fig.bbox.y_min:.1f}) to ({fig.bbox.x_max:.1f}, {fig.bbox.y_max:.1f})")
                
                logger.info(f"   - Summary: {fig.summary[:150]}..." if len(fig.summary) > 150 else f"   - Summary: {fig.summary}")
                
                if fig.context:
                    logger.info(f"   - Context (first 100 chars): {fig.context[:100]}...")
        else:
            logger.info("\n5. No figures detected in the specified page range")
        
        # Step 6: Demonstrate caching
        logger.info("\n6. Testing cache (re-detecting same pages)...")
        figures_cached = detector.detect_figures(
            page_range=(start_page, end_page),
            page_texts=page_texts,
            batch_size=5
        )
        logger.info(f"   ✓ Cache working: Found {len(figures_cached)} figure(s) (should be instant)")
        
        # Step 7: Show what data structure is returned
        logger.info("\n7. Data Structure Example:")
        if figures:
            logger.info("   FigureNode fields:")
            logger.info(f"   - figure_number: str = '{figures[0].figure_number}'")
            logger.info(f"   - caption: str = '{figures[0].caption[:50]}...'")
            logger.info(f"   - page: int = {figures[0].page}")
            logger.info(f"   - bbox: Optional[BoundingBox] = {figures[0].bbox}")
            logger.info(f"   - figure_type: str = '{figures[0].figure_type}'")
            logger.info(f"   - context: str = (length: {len(figures[0].context)} chars)")
            logger.info(f"   - summary: str = (length: {len(figures[0].summary)} chars)")
        
        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'detector' in locals():
            del detector


def main():
    """Main entry point."""
    # Example: Use a test PDF
    # You can change this to any PDF path
    test_pdf = "tests/pdfs/earthmover.pdf"  # Change this to your PDF
    
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    
    logger.info(f"Using PDF: {test_pdf}")
    demo_figure_detection(test_pdf)


if __name__ == "__main__":
    main()
