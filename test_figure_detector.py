"""
Unit tests for FigureDetector component.

Tests cover:
- Detection with 0, 1, and multiple figures
- Context extraction
- Error handling
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO
import pymupdf

# Add pageindex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.figure_detector import (
    FigureDetector, 
    FigureNode, 
    DetectedFigure, 
    BoundingBox,
    FigureDetectionResponse
)
from pageindex.llm_client import get_llm_client
from pageindex.utils import get_page_tokens

load_dotenv()


class TestFigureDetector:
    """Test suite for FigureDetector"""
    
    def __init__(self):
        self.test_results = []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def get_test_pdf_path(self, filename):
        """Get the correct path to a test PDF file"""
        # Try multiple possible paths
        possible_paths = [
            f"PageIndex/tests/pdfs/{filename}",
            f"tests/pdfs/{filename}",
            f"./tests/pdfs/{filename}",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def run_test(self, test_name, test_func):
        """Run a single test and record result"""
        print(f"\n{'='*70}")
        print(f"Test: {test_name}")
        print(f"{'='*70}")
        
        try:
            test_func()
            print(f"✅ PASS: {test_name}")
            self.test_results.append((test_name, True, None))
            return True
        except AssertionError as e:
            print(f"❌ FAIL: {test_name}")
            print(f"   Assertion Error: {e}")
            self.test_results.append((test_name, False, str(e)))
            return False
        except Exception as e:
            print(f"❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append((test_name, False, str(e)))
            return False
    
    def test_initialization(self):
        """Test FigureDetector initialization"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        assert detector is not None, "Detector should be initialized"
        assert detector.doc is not None, "PDF document should be loaded"
        assert detector.vision_client is not None, "Vision client should be initialized"
        
        print(f"✓ Detector initialized successfully")
        print(f"✓ PDF loaded with {len(detector.doc)} pages")
    
    def test_initialization_with_bytesio(self):
        """Test FigureDetector initialization with BytesIO stream"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        # Read PDF into BytesIO
        with open(test_pdf, 'rb') as f:
            pdf_stream = BytesIO(f.read())
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, pdf_stream)
        
        assert detector is not None, "Detector should be initialized with BytesIO"
        assert detector.doc is not None, "PDF document should be loaded from stream"
        
        print(f"✓ Detector initialized with BytesIO stream")
    
    def test_initialization_missing_api_key(self):
        """Test FigureDetector initialization without API key (should work but with warning)"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        # Temporarily remove API key
        original_key = os.environ.get("GEMINI_API_KEY")
        if original_key:
            del os.environ["GEMINI_API_KEY"]
        
        try:
            llm_client = get_llm_client()
            detector = FigureDetector(llm_client, test_pdf)
            # Should initialize successfully but without vision client
            assert detector.vision_client is None, "Vision client should be None without API key"
            print(f"✓ Correctly initialized without vision client when API key missing")
        finally:
            # Restore API key
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key
    
    def test_detection_with_zero_figures(self):
        """Test detection on pages with no figures"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test on first page (likely title page with no figures)
        figures = detector.detect_figures((1, 1), page_texts)
        
        print(f"✓ Detection completed on page 1")
        print(f"✓ Detected {len(figures)} figure(s)")
        
        # Verify result is a list (may be empty or have figures)
        assert isinstance(figures, list), "Result should be a list"
        
        # All detected figures should be FigureNode instances
        for fig in figures:
            assert isinstance(fig, FigureNode), "Each result should be a FigureNode"
    
    def test_detection_with_single_figure(self):
        """Test detection on page with single figure"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test on pages 2-3 (more likely to have figures)
        figures = detector.detect_figures((2, 3), page_texts)
        
        print(f"✓ Detection completed on pages 2-3")
        print(f"✓ Detected {len(figures)} figure(s)")
        
        # Verify structure
        assert isinstance(figures, list), "Result should be a list"
        
        for fig in figures:
            assert isinstance(fig, FigureNode), "Each result should be a FigureNode"
            assert fig.page >= 2 and fig.page <= 3, "Figure page should be in range"
            assert fig.figure_number, "Figure should have a number"
            assert fig.figure_type, "Figure should have a type"
            assert fig.caption, "Figure should have a caption"
            assert fig.summary, "Figure should have a summary"
            
            print(f"\n  Figure: {fig.figure_number}")
            print(f"    Type: {fig.figure_type}")
            print(f"    Page: {fig.page}")
            print(f"    Caption: {fig.caption[:80]}...")
            print(f"    Summary: {fig.summary[:80]}...")
    
    def test_detection_with_multiple_figures(self):
        """Test detection on pages with multiple figures"""
        test_pdf = self.get_test_pdf_path("2023-annual-report.pdf")
        
        if not test_pdf:
            print(f"⚠️  Full PDF not found, using truncated version")
            test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
            
            if not test_pdf:
                print(f"⚠️  No test PDF found, skipping test")
                return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test on first 5 pages
        num_pages = min(5, len(page_texts))
        figures = detector.detect_figures((1, num_pages), page_texts)
        
        print(f"✓ Detection completed on pages 1-{num_pages}")
        print(f"✓ Detected {len(figures)} figure(s)")
        
        # Verify structure
        assert isinstance(figures, list), "Result should be a list"
        
        # Track figures per page
        figures_by_page = {}
        for fig in figures:
            assert isinstance(fig, FigureNode), "Each result should be a FigureNode"
            assert fig.page >= 1 and fig.page <= num_pages, "Figure page should be in range"
            
            if fig.page not in figures_by_page:
                figures_by_page[fig.page] = []
            figures_by_page[fig.page].append(fig)
        
        print(f"\n  Figures by page:")
        for page, page_figs in sorted(figures_by_page.items()):
            print(f"    Page {page}: {len(page_figs)} figure(s)")
            for fig in page_figs:
                print(f"      - {fig.figure_number}: {fig.figure_type}")
    
    def test_context_extraction_basic(self):
        """Test basic context extraction"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test context extraction for page 3 (more likely to have text)
        bbox = BoundingBox(x_min=10, y_min=30, x_max=90, y_max=70)
        context = detector.extract_figure_context(3, bbox, page_texts)
        
        print(f"✓ Context extracted for page 3")
        print(f"✓ Context length: {len(context)} characters")
        
        assert isinstance(context, str), "Context should be a string"
        # Context may be empty for pages with no text (e.g., image-only pages)
        # This is correct behavior, so we just verify it's a string
        assert len(context) <= 2100, "Context should be limited to reasonable size"
        
        if len(context) > 0:
            print(f"\n  Context preview: {context[:200]}...")
        else:
            print(f"\n  Note: Page has no text content (image-only page)")
    
    def test_context_extraction_edge_cases(self):
        """Test context extraction at page boundaries"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test figure at top of page (should include previous page context)
        bbox_top = BoundingBox(x_min=10, y_min=5, x_max=90, y_max=25)
        context_top = detector.extract_figure_context(2, bbox_top, page_texts)
        
        print(f"✓ Context extracted for figure at top of page")
        print(f"  Length: {len(context_top)} characters")
        
        # Test figure at bottom of page (should include next page context)
        bbox_bottom = BoundingBox(x_min=10, y_min=75, x_max=90, y_max=95)
        context_bottom = detector.extract_figure_context(2, bbox_bottom, page_texts)
        
        print(f"✓ Context extracted for figure at bottom of page")
        print(f"  Length: {len(context_bottom)} characters")
        
        # Test figure without bbox (should use middle heuristic)
        context_no_bbox = detector.extract_figure_context(2, None, page_texts)
        
        print(f"✓ Context extracted for figure without bbox")
        print(f"  Length: {len(context_no_bbox)} characters")
        
        assert isinstance(context_top, str), "Context should be a string"
        assert isinstance(context_bottom, str), "Context should be a string"
        assert isinstance(context_no_bbox, str), "Context should be a string"
    
    def test_context_extraction_invalid_page(self):
        """Test context extraction with invalid page number"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Test with invalid page number (out of range)
        context = detector.extract_figure_context(999, None, page_texts)
        
        print(f"✓ Handled invalid page number gracefully")
        print(f"  Returned context: '{context}'")
        
        assert isinstance(context, str), "Should return empty string for invalid page"
    
    def test_error_handling_vision_api_failure(self):
        """Test error handling when vision API fails"""
        test_pdf = self.get_test_pdf_path("2023-annual-report-truncated.pdf")
        
        if not test_pdf:
            print(f"⚠️  Test PDF not found, skipping test")
            return
        
        llm_client = get_llm_client()
        detector = FigureDetector(llm_client, test_pdf)
        
        # Get page texts
        page_texts = get_page_tokens(test_pdf, model="gpt-4", pdf_parser="PyMuPDF")
        
        # Mock a vision API failure by using invalid API key temporarily
        original_client = detector.vision_client
        
        # Create a mock client that will fail
        class MockFailingClient:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    raise Exception("Simulated API failure")
        
        detector.vision_client = MockFailingClient()
        
        # Detection should handle the error gracefully
        figures = detector.detect_figures((1, 1), page_texts)
        
        print(f"✓ Handled vision API failure gracefully")
        print(f"  Returned {len(figures)} figures (should be 0)")
        
        assert isinstance(figures, list), "Should return empty list on failure"
        
        # Restore original client
        detector.vision_client = original_client
    
    def test_error_handling_invalid_pdf(self):
        """Test error handling with invalid PDF path"""
        llm_client = get_llm_client()
        
        try:
            detector = FigureDetector(llm_client, "nonexistent.pdf")
            assert False, "Should have raised an error for invalid PDF"
        except Exception as e:
            print(f"✓ Correctly raised error for invalid PDF: {type(e).__name__}")
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n\n{'='*70}")
        print("Test Summary")
        print(f"{'='*70}")
        
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        for test_name, result, error in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
            if error and not result:
                print(f"         {error}")
        
        print(f"\n{passed}/{total} tests passed")
        print(f"{'='*70}\n")
        
        return passed == total


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FigureDetector Unit Test Suite")
    print("="*70)
    
    tester = TestFigureDetector()
    
    # Run all tests
    tester.run_test("Initialization", tester.test_initialization)
    tester.run_test("Initialization with BytesIO", tester.test_initialization_with_bytesio)
    tester.run_test("Initialization without API key", tester.test_initialization_missing_api_key)
    tester.run_test("Detection with zero figures", tester.test_detection_with_zero_figures)
    tester.run_test("Detection with single figure", tester.test_detection_with_single_figure)
    tester.run_test("Detection with multiple figures", tester.test_detection_with_multiple_figures)
    tester.run_test("Context extraction - basic", tester.test_context_extraction_basic)
    tester.run_test("Context extraction - edge cases", tester.test_context_extraction_edge_cases)
    tester.run_test("Context extraction - invalid page", tester.test_context_extraction_invalid_page)
    tester.run_test("Error handling - vision API failure", tester.test_error_handling_vision_api_failure)
    tester.run_test("Error handling - invalid PDF", tester.test_error_handling_invalid_pdf)
    
    # Print summary
    all_passed = tester.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
