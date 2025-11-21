"""
Debug script for FigureDetector - detailed logging to diagnose issues.
"""

import os
import sys
import logging
from pathlib import Path

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.figure_detector import FigureDetector
from pageindex.llm_client import get_llm_client
import pymupdf

# Set up very detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def debug_figure_detection(pdf_path: str):
    """
    Debug figure detection with extensive logging.
    """
    logger.info("=" * 80)
    logger.info("FIGURE DETECTOR DEBUG MODE")
    logger.info("=" * 80)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set!")
        return
    else:
        logger.info(f"GEMINI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize LLM client
        logger.info("\n1. Initializing LLM client...")
        llm_client = get_llm_client()
        logger.info(f"   ✓ LLM client: {llm_client}")
        logger.info(f"   - Provider: {llm_client.provider}")
        
        # Initialize FigureDetector
        logger.info("\n2. Initializing FigureDetector...")
        detector = FigureDetector(
            llm_client=llm_client,
            pdf_path=pdf_path,
            logger=logger
        )
        logger.info(f"   ✓ FigureDetector initialized")
        logger.info(f"   - PDF pages: {len(detector.doc)}")
        logger.info(f"   - Vision client: {detector.vision_client}")
        
        # Extract page texts
        logger.info("\n3. Extracting page texts...")
        page_texts = []
        for page_num in range(len(detector.doc)):
            page = detector.doc[page_num]
            text = page.get_text()
            page_texts.append((text, len(text.split())))
            logger.debug(f"   Page {page_num + 1}: {len(text)} chars, {len(text.split())} words")
        
        # Test image extraction on first few pages
        logger.info("\n4. Testing image extraction on each page...")
        for page_num in range(min(5, len(detector.doc))):
            logger.info(f"\n   === PAGE {page_num + 1} ===")
            
            # Extract images
            images = detector._extract_images_from_page(page_num)
            logger.info(f"   Found {len(images)} image(s)")
            
            if images:
                for i, img_data in enumerate(images):
                    logger.info(f"\n   Image {i + 1}:")
                    logger.info(f"   - Size: {img_data['size']}")
                    logger.info(f"   - Format: {img_data['ext']}")
                    logger.info(f"   - BBox: ({img_data['bbox'].x_min:.1f}, {img_data['bbox'].y_min:.1f}) to ({img_data['bbox'].x_max:.1f}, {img_data['bbox'].y_max:.1f})")
                    
                    # Try to find caption
                    page_text = page_texts[page_num][0]
                    logger.info(f"   - Page text length: {len(page_text)} chars")
                    logger.info(f"   - Page text preview: {page_text[:200]}...")
                    
                    try:
                        figure_number, caption = detector._find_caption_for_image(
                            page_num + 1,
                            img_data['bbox'],
                            page_text
                        )
                        logger.info(f"   - Caption found: {figure_number}")
                        logger.info(f"   - Caption text: {caption[:100]}..." if len(caption) > 100 else f"   - Caption text: {caption}")
                    except Exception as e:
                        logger.error(f"   - Error finding caption: {e}", exc_info=True)
                    
                    # Try to analyze image type
                    if detector.vision_client:
                        try:
                            logger.info(f"   - Analyzing image type...")
                            figure_type = detector._analyze_image_type(img_data['image'])
                            logger.info(f"   - Type detected: {figure_type}")
                        except Exception as e:
                            logger.error(f"   - Error analyzing type: {e}", exc_info=True)
            else:
                logger.info("   No images found on this page")
                
                # Check if there are any images at all
                page = detector.doc[page_num]
                image_list = page.get_images(full=True)
                logger.info(f"   Raw image count from PyMuPDF: {len(image_list)}")
                
                if image_list:
                    logger.info("   Images exist but were filtered out:")
                    for img_idx, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = detector.doc.extract_image(xref)
                            from PIL import Image
                            from io import BytesIO
                            pil_image = Image.open(BytesIO(base_image["image"]))
                            width, height = pil_image.size
                            logger.info(f"   - Image {img_idx + 1}: {width}x{height} (filtered: {width < 50 or height < 50})")
                        except Exception as e:
                            logger.warning(f"   - Error checking image {img_idx + 1}: {e}")
        
        # Now try full detection
        logger.info("\n5. Running full detection on pages 1-5...")
        start_page = 1
        end_page = min(5, len(detector.doc))
        
        figures = detector.detect_figures(
            page_range=(start_page, end_page),
            page_texts=page_texts,
            batch_size=5
        )
        
        logger.info(f"\n   ✓ Detection complete: Found {len(figures)} figure(s)")
        
        if figures:
            logger.info("\n6. Figure Details:")
            for i, fig in enumerate(figures, 1):
                logger.info(f"\n   Figure {i}:")
                logger.info(f"   - Number: {fig.figure_number}")
                logger.info(f"   - Page: {fig.page}")
                logger.info(f"   - Type: {fig.figure_type}")
                logger.info(f"   - Caption: {fig.caption}")
                logger.info(f"   - Summary: {fig.summary}")
        else:
            logger.warning("\n6. NO FIGURES DETECTED!")
            logger.info("\n   Possible reasons:")
            logger.info("   1. Images are too small (< 50x50 pixels)")
            logger.info("   2. No images in the PDF (text-only)")
            logger.info("   3. Images are embedded differently")
            logger.info("   4. Vision API failed to detect them")
        
        logger.info("\n" + "=" * 80)
        logger.info("DEBUG COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)
    finally:
        if 'detector' in locals():
            del detector


def main():
    """Main entry point."""
    test_pdf = "tests/pdfs/earthmover.pdf"
    
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    
    logger.info(f"Debugging PDF: {test_pdf}")
    debug_figure_detection(test_pdf)


if __name__ == "__main__":
    main()
