"""
Demo script for TableDetector - shows how to detect tables in a PDF.

This script demonstrates:
1. How to initialize the TableDetector
2. How to detect tables in a page range
3. What data is returned (table nodes with metadata)
4. How the caching and batch processing work
"""

import os
import sys
import logging
from pathlib import Path

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.table_detector import TableDetector, TableNode
from pageindex.llm_client import get_llm_client
import pymupdf

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_table_detection(pdf_path: str):
    """
    Demonstrate table detection on a PDF.
    
    Args:
        pdf_path: Path to PDF file
    """
    logger.info("=" * 80)
    logger.info("TABLE DETECTOR DEMO")
    logger.info("=" * 80)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Check if GEMINI_API_KEY is set (required for vision analysis)
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set - table detection requires vision API")
        logger.info("Set GEMINI_API_KEY in .env file")
        return
    
    try:
        # Step 1: Initialize LLM client
        logger.info("\n1. Initializing LLM client...")
        llm_client = get_llm_client()
        logger.info(f"   ✓ LLM client initialized (provider: {llm_client.provider})")
        
        # Step 2: Initialize TableDetector
        logger.info("\n2. Initializing TableDetector...")
        detector = TableDetector(
            llm_client=llm_client,
            pdf_path=pdf_path,
            logger=logger
        )
        logger.info("   ✓ TableDetector initialized")
        logger.info(f"   - PDF has {len(detector.doc)} pages")
        logger.info(f"   - Detection model: {detector.detection_model}")
        logger.info(f"   - Extraction model: {detector.extraction_model}")
        
        # Step 3: Extract page texts (needed for context extraction)
        logger.info("\n3. Extracting page texts...")
        page_texts = []
        for page_num in range(len(detector.doc)):
            page = detector.doc[page_num]
            text = page.get_text()
            page_texts.append((text, len(text.split())))
        logger.info(f"   ✓ Extracted text from {len(page_texts)} pages")
        
        # Step 4: Detect tables in a page range
        # Let's detect tables in the first 5 pages
        start_page = 1
        end_page = min(5, len(detector.doc))
        
        logger.info(f"\n4. Detecting tables in pages {start_page}-{end_page}...")
        logger.info("   (This may take a moment - using vision API for analysis)")
        
        tables = detector.detect_tables(
            page_range=(start_page, end_page),
            page_texts=page_texts,
            batch_size=3  # Process up to 3 pages at once
        )
        
        logger.info(f"   ✓ Detection complete: Found {len(tables)} table(s)")
        
        # Step 5: Display results
        if tables:
            logger.info("\n5. Table Details:")
            logger.info("=" * 80)
            
            for i, table in enumerate(tables, 1):
                logger.info(f"\n   Table {i}:")
                logger.info(f"   - Number: {table.table_number}")
                logger.info(f"   - Page: {table.page}")
                logger.info(f"   - Caption: {table.caption[:100]}..." if len(table.caption) > 100 else f"   - Caption: {table.caption}")
                
                if table.bbox:
                    logger.info(f"   - Bounding Box: ({table.bbox.x_min:.1f}, {table.bbox.y_min:.1f}) to ({table.bbox.x_max:.1f}, {table.bbox.y_max:.1f})")
                
                if table.headers:
                    logger.info(f"   - Headers: {', '.join(table.headers[:5])}" + ("..." if len(table.headers) > 5 else ""))
                
                if table.key_values:
                    logger.info(f"   - Key Values ({len(table.key_values)} items):")
                    for key, value in list(table.key_values.items())[:3]:
                        logger.info(f"     • {key}: {value}")
                    if len(table.key_values) > 3:
                        logger.info(f"     ... and {len(table.key_values) - 3} more")
                
                logger.info(f"   - Summary: {table.summary[:150]}..." if len(table.summary) > 150 else f"   - Summary: {table.summary}")
                
                if table.context:
                    logger.info(f"   - Context (first 100 chars): {table.context[:100]}...")
        else:
            logger.info("\n5. No tables detected in the specified page range")
        
        # Step 6: Demonstrate caching
        logger.info("\n6. Testing cache (re-detecting same pages)...")
        tables_cached = detector.detect_tables(
            page_range=(start_page, end_page),
            page_texts=page_texts,
            batch_size=3
        )
        logger.info(f"   ✓ Cache working: Found {len(tables_cached)} table(s) (should be instant)")
        
        # Step 7: Show what data structure is returned
        logger.info("\n7. Data Structure Example:")
        if tables:
            logger.info("   TableNode fields:")
            logger.info(f"   - table_number: str = '{tables[0].table_number}'")
            logger.info(f"   - caption: str = '{tables[0].caption[:50]}...'")
            logger.info(f"   - page: int = {tables[0].page}")
            logger.info(f"   - bbox: Optional[BoundingBox] = {tables[0].bbox}")
            logger.info(f"   - headers: List[str] = {tables[0].headers}")
            logger.info(f"   - key_values: dict = {len(tables[0].key_values)} items")
            logger.info(f"   - context: str = (length: {len(tables[0].context)} chars)")
            logger.info(f"   - summary: str = (length: {len(tables[0].summary)} chars)")
        
        # Step 8: Demonstrate table structure extraction (optional)
        if tables and tables[0].bbox:
            logger.info("\n8. Extracting detailed table structure...")
            try:
                # Render the page as image
                page_image = detector._render_page_as_image(tables[0].page - 1)
                
                # Extract structure
                structure = detector.extract_table_structure(page_image, tables[0].bbox)
                
                logger.info(f"   ✓ Structure extracted:")
                logger.info(f"   - Headers: {structure.get('headers', [])}")
                logger.info(f"   - Rows: ~{structure.get('num_rows', 0)}")
                logger.info(f"   - Key values: {len(structure.get('key_values', {}))} items")
            except Exception as e:
                logger.warning(f"   Could not extract structure: {e}")
        
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
    demo_table_detection(test_pdf)


if __name__ == "__main__":
    main()
