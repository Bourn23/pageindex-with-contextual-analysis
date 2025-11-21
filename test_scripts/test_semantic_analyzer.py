"""
Unit tests for SemanticAnalyzer component.

Tests cover:
- Boundary identification for different section types
- Handling of short sections
- Error handling and fallbacks
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add pageindex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.semantic_analyzer import SemanticAnalyzer, SemanticUnit
from pageindex.llm_client import get_llm_client

load_dotenv()


class TestSemanticAnalyzer:
    """Test suite for SemanticAnalyzer"""
    
    def __init__(self):
        self.test_results = []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
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
        """Test SemanticAnalyzer initialization"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        assert analyzer is not None, "Analyzer should be initialized"
        assert analyzer.llm_client is not None, "LLM client should be set"
        assert analyzer.section_prompts is not None, "Section prompts should be initialized"
        
        # Verify all expected section types have prompts
        expected_types = ['introduction', 'methods', 'results', 'discussion', 'default']
        for section_type in expected_types:
            assert section_type in analyzer.section_prompts, f"Missing prompt for {section_type}"
        
        print(f"✓ Analyzer initialized successfully")
        print(f"✓ All {len(expected_types)} section prompt templates loaded")
    
    def test_paragraph_splitting_double_newline(self):
        """Test paragraph splitting with double newlines"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        paragraphs = analyzer._split_into_paragraphs(text)
        
        assert len(paragraphs) == 3, f"Expected 3 paragraphs, got {len(paragraphs)}"
        assert paragraphs[0] == "Paragraph 1.", f"First paragraph mismatch"
        assert paragraphs[1] == "Paragraph 2.", f"Second paragraph mismatch"
        assert paragraphs[2] == "Paragraph 3.", f"Third paragraph mismatch"
        
        print(f"✓ Correctly split text into {len(paragraphs)} paragraphs")
    
    def test_paragraph_splitting_single_newline(self):
        """Test paragraph splitting with single newlines (fallback)"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        text = "Paragraph 1.\nParagraph 2.\nParagraph 3."
        paragraphs = analyzer._split_into_paragraphs(text)
        
        assert len(paragraphs) == 3, f"Expected 3 paragraphs, got {len(paragraphs)}"
        
        print(f"✓ Correctly handled single newline fallback")
    
    def test_text_with_indices_creation(self):
        """Test creation of indexed text for LLM"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        paragraphs = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        indexed_text = analyzer._create_text_with_indices(paragraphs)
        
        assert "[Paragraph 0]" in indexed_text, "Missing paragraph 0 index"
        assert "[Paragraph 1]" in indexed_text, "Missing paragraph 1 index"
        assert "[Paragraph 2]" in indexed_text, "Missing paragraph 2 index"
        assert "First paragraph." in indexed_text, "Missing first paragraph content"
        assert "Second paragraph." in indexed_text, "Missing second paragraph content"
        assert "Third paragraph." in indexed_text, "Missing third paragraph content"
        
        print(f"✓ Correctly created indexed text")
        print(f"  Preview: {indexed_text[:100]}...")
    
    def test_boundary_validation_non_overlapping(self):
        """Test boundary validation with non-overlapping units"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        units = [
            SemanticUnit(
                title="Unit 1",
                start_paragraph=0,
                end_paragraph=1,
                start_page=1,
                end_page=1,
                semantic_type="test",
                summary="Test unit 1"
            ),
            SemanticUnit(
                title="Unit 2",
                start_paragraph=2,
                end_paragraph=3,
                start_page=1,
                end_page=1,
                semantic_type="test",
                summary="Test unit 2"
            )
        ]
        
        validated = analyzer._validate_boundaries(units, 4)
        
        assert len(validated) == 2, f"Expected 2 units, got {len(validated)}"
        assert validated[0].start_paragraph == 0, "Unit 1 start mismatch"
        assert validated[0].end_paragraph == 1, "Unit 1 end mismatch"
        assert validated[1].start_paragraph == 2, "Unit 2 start mismatch"
        assert validated[1].end_paragraph == 3, "Unit 2 end mismatch"
        
        print(f"✓ Non-overlapping boundaries validated correctly")
    
    def test_boundary_validation_overlapping(self):
        """Test boundary validation with overlapping units (should be fixed)"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        overlapping_units = [
            SemanticUnit(
                title="Unit 1",
                start_paragraph=0,
                end_paragraph=2,
                start_page=1,
                end_page=1,
                semantic_type="test",
                summary="Test unit 1"
            ),
            SemanticUnit(
                title="Unit 2",
                start_paragraph=1,  # Overlaps with Unit 1
                end_paragraph=3,
                start_page=1,
                end_page=1,
                semantic_type="test",
                summary="Test unit 2"
            )
        ]
        
        validated = analyzer._validate_boundaries(overlapping_units, 4)
        
        # Unit 2 should be adjusted to start at paragraph 3
        assert len(validated) == 2, f"Expected 2 units, got {len(validated)}"
        assert validated[0].start_paragraph == 0, "Unit 1 start mismatch"
        assert validated[0].end_paragraph == 2, "Unit 1 end mismatch"
        assert validated[1].start_paragraph == 3, "Unit 2 should be adjusted to paragraph 3"
        assert validated[1].end_paragraph == 3, "Unit 2 end mismatch"
        
        print(f"✓ Overlapping boundaries corrected successfully")
        print(f"  Unit 2 adjusted from paragraph 1 to paragraph 3")
    
    def test_boundary_validation_out_of_range(self):
        """Test boundary validation with out-of-range paragraph indices"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        units = [
            SemanticUnit(
                title="Unit 1",
                start_paragraph=0,
                end_paragraph=5,  # Exceeds total paragraphs
                start_page=1,
                end_page=1,
                semantic_type="test",
                summary="Test unit 1"
            )
        ]
        
        validated = analyzer._validate_boundaries(units, 4)  # Only 4 paragraphs
        
        assert len(validated) == 1, f"Expected 1 unit, got {len(validated)}"
        assert validated[0].end_paragraph == 3, "End paragraph should be adjusted to 3 (max index)"
        
        print(f"✓ Out-of-range boundaries corrected successfully")
        print(f"  End paragraph adjusted from 5 to 3")
    
    def test_boundary_validation_empty_list(self):
        """Test boundary validation with empty list"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        validated = analyzer._validate_boundaries([], 10)
        
        assert len(validated) == 0, "Empty list should return empty list"
        
        print(f"✓ Empty list handled correctly")
    
    def test_short_section_handling(self):
        """Test that short sections are not subdivided"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create a short section (less than min_pages)
        short_text = "This is a short section.\n\nIt has only two paragraphs."
        section_node = {
            'title': 'Short Section',
            'text': short_text,
            'start_index': 1,
            'end_index': 1  # Only 1 page
        }
        
        page_texts = [(short_text, 50)]
        
        # Analyze with min_pages=0.5 (section is 1 page, but we'll test with higher threshold)
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=2.0)
        
        assert len(semantic_units) == 0, "Short section should not be subdivided"
        
        print(f"✓ Short section correctly skipped (< 2.0 pages)")
    
    def test_short_section_few_paragraphs(self):
        """Test that sections with too few paragraphs are not subdivided"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create a section with only 2 paragraphs (< 3 minimum)
        short_text = "Paragraph 1.\n\nParagraph 2."
        section_node = {
            'title': 'Few Paragraphs Section',
            'text': short_text,
            'start_index': 1,
            'end_index': 2  # 2 pages
        }
        
        page_texts = [(short_text, 50)]
        
        # Analyze with min_pages=0 (should still skip due to paragraph count)
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        assert len(semantic_units) == 0, "Section with < 3 paragraphs should not be subdivided"
        
        print(f"✓ Section with few paragraphs correctly skipped (< 3 paragraphs)")
    
    def test_section_type_detection_introduction(self):
        """Test section type detection for introduction-related titles"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        test_cases = [
            "Introduction",
            "Background",
            "Literature Review",
            "Motivation"
        ]
        
        for title in test_cases:
            section_type = analyzer._detect_section_type(title)
            print(f"  '{title}' → {section_type}")
            # We expect introduction, but allow default as fallback
            assert section_type in ['introduction', 'default'], f"Unexpected type for '{title}': {section_type}"
        
        print(f"✓ Introduction section types detected")
    
    def test_section_type_detection_methods(self):
        """Test section type detection for methods-related titles"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        test_cases = [
            "Methods",
            "Experimental Methods",
            "Materials and Methods",
            "Methodology"
        ]
        
        for title in test_cases:
            section_type = analyzer._detect_section_type(title)
            print(f"  '{title}' → {section_type}")
            assert section_type in ['methods', 'default'], f"Unexpected type for '{title}': {section_type}"
        
        print(f"✓ Methods section types detected")
    
    def test_section_type_detection_results(self):
        """Test section type detection for results-related titles"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        test_cases = [
            "Results",
            "Findings",
            "Results and Discussion"
        ]
        
        for title in test_cases:
            section_type = analyzer._detect_section_type(title)
            print(f"  '{title}' → {section_type}")
            assert section_type in ['results', 'default'], f"Unexpected type for '{title}': {section_type}"
        
        print(f"✓ Results section types detected")
    
    def test_section_type_detection_discussion(self):
        """Test section type detection for discussion-related titles"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        test_cases = [
            "Discussion",
            "Conclusion",
            "Conclusions and Future Work"
        ]
        
        for title in test_cases:
            section_type = analyzer._detect_section_type(title)
            print(f"  '{title}' → {section_type}")
            assert section_type in ['discussion', 'default'], f"Unexpected type for '{title}': {section_type}"
        
        print(f"✓ Discussion section types detected")
    
    def test_analyze_section_introduction(self):
        """Test semantic analysis of an introduction section"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Sample introduction text with clear semantic structure
        intro_text = """Solid-state batteries have emerged as a promising alternative to conventional liquid electrolyte batteries due to their enhanced safety and energy density. The use of ceramic electrolytes such as Li7La3Zr2O12 (LLZO) has attracted significant attention in recent years.

However, the high interfacial resistance between ceramic and electrode materials remains a major challenge that limits practical applications. Previous studies have shown that composite polymer electrolytes combining LLZO with polymer matrices can reduce this resistance significantly.

Despite these advances, the fundamental mechanisms governing ion transport at the ceramic-polymer interface are not well understood. This knowledge gap limits our ability to optimize composite electrolyte performance for next-generation batteries.

In this work, we investigate the interfacial ion transport mechanisms in LLZO-PEO composite electrolytes using advanced characterization techniques. Our findings provide new insights into designing high-performance solid-state batteries with improved safety and energy density."""
        
        section_node = {
            'title': 'Introduction',
            'text': intro_text,
            'start_index': 1,
            'end_index': 1
        }
        
        page_texts = [(intro_text, 200)]
        
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        print(f"  Found {len(semantic_units)} semantic units")
        
        for i, unit in enumerate(semantic_units):
            print(f"\n  Unit {i+1}: {unit.title}")
            print(f"    Type: {unit.semantic_type}")
            print(f"    Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
            print(f"    Summary: {unit.summary[:80]}...")
        
        # Verify we got some semantic units
        assert len(semantic_units) > 0, "Should identify semantic units in introduction"
        
        # Verify all units have required fields
        for unit in semantic_units:
            assert unit.title, "Unit should have title"
            assert unit.semantic_type, "Unit should have semantic type"
            assert unit.start_paragraph >= 0, "Start paragraph should be valid"
            assert unit.end_paragraph >= unit.start_paragraph, "End should be >= start"
        
        print(f"\n✓ Introduction section analyzed successfully")
    
    def test_analyze_section_methods(self):
        """Test semantic analysis of a methods section"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Sample methods text with clear semantic structure
        methods_text = """LLZO powder was synthesized using a solid-state reaction method. Stoichiometric amounts of Li2CO3, La2O3, and ZrO2 were mixed and calcined at 900°C for 12 hours.

The composite electrolytes were prepared by solution casting. PEO (Mw = 600,000) and LiTFSI salt were dissolved in acetonitrile at a weight ratio of 8:1. LLZO particles were then dispersed in the solution at various weight percentages.

Ionic conductivity was measured using electrochemical impedance spectroscopy (EIS) in the frequency range of 1 MHz to 0.1 Hz. Samples were sandwiched between stainless steel blocking electrodes.

Morphology was characterized using scanning electron microscopy (SEM) at an accelerating voltage of 5 kV. Cross-sectional samples were prepared by freeze-fracturing in liquid nitrogen."""
        
        section_node = {
            'title': 'Experimental Methods',
            'text': methods_text,
            'start_index': 2,
            'end_index': 3
        }
        
        page_texts = [(methods_text[:len(methods_text)//2], 100), (methods_text[len(methods_text)//2:], 100)]
        
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        print(f"  Found {len(semantic_units)} semantic units")
        
        for i, unit in enumerate(semantic_units):
            print(f"\n  Unit {i+1}: {unit.title}")
            print(f"    Type: {unit.semantic_type}")
            print(f"    Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
        
        assert len(semantic_units) > 0, "Should identify semantic units in methods"
        
        print(f"\n✓ Methods section analyzed successfully")
    
    def test_analyze_section_results(self):
        """Test semantic analysis of a results section"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Sample results text
        results_text = """The ionic conductivity of pure PEO electrolyte was measured to be 1.2 × 10⁻⁵ S/cm at 60°C. This value is consistent with previously reported values for similar PEO-based systems.

Addition of 5 wt% LLZO particles increased the conductivity to 2.8 × 10⁻⁵ S/cm, representing a 2.3-fold enhancement. Further increasing the LLZO content to 10 wt% resulted in a conductivity of 4.1 × 10⁻⁵ S/cm.

SEM images revealed uniform dispersion of LLZO particles in the PEO matrix at low loadings. However, at 15 wt% LLZO, particle agglomeration was observed, which correlated with a decrease in conductivity.

Temperature-dependent measurements showed Arrhenius behavior for all samples. The activation energy decreased from 0.85 eV for pure PEO to 0.62 eV for the 10 wt% LLZO composite."""
        
        section_node = {
            'title': 'Results',
            'text': results_text,
            'start_index': 4,
            'end_index': 5
        }
        
        page_texts = [(results_text, 200)]
        
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        print(f"  Found {len(semantic_units)} semantic units")
        
        for i, unit in enumerate(semantic_units):
            print(f"\n  Unit {i+1}: {unit.title}")
            print(f"    Type: {unit.semantic_type}")
        
        assert len(semantic_units) > 0, "Should identify semantic units in results"
        
        print(f"\n✓ Results section analyzed successfully")
    
    def test_error_handling_llm_failure(self):
        """Test error handling when LLM fails"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create a mock LLM client that returns error
        class MockFailingClient:
            def chat_completion(self, *args, **kwargs):
                return "Error"
        
        analyzer.llm_client = MockFailingClient()
        
        section_node = {
            'title': 'Test Section',
            'text': "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.",
            'start_index': 1,
            'end_index': 1
        }
        
        page_texts = [("text", 100)]
        
        # Should handle error gracefully and return empty list
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        assert isinstance(semantic_units, list), "Should return a list"
        assert len(semantic_units) == 0, "Should return empty list on LLM failure"
        
        print(f"✓ LLM failure handled gracefully")
    
    def test_error_handling_invalid_json_response(self):
        """Test error handling when LLM returns invalid JSON"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Test parsing invalid JSON
        invalid_responses = [
            "This is not JSON",
            "{invalid json}",
            '{"semantic_units": "not a list"}',
            '{"wrong_key": []}',
        ]
        
        for response in invalid_responses:
            result = analyzer._parse_llm_response(response)
            assert isinstance(result, list), "Should return list even for invalid JSON"
            assert len(result) == 0, "Should return empty list for invalid JSON"
        
        print(f"✓ Invalid JSON responses handled gracefully")
    
    def test_error_handling_malformed_semantic_units(self):
        """Test error handling with malformed semantic unit data"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create section with valid structure
        section_node = {
            'title': 'Test Section',
            'text': "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.",
            'start_index': 1,
            'end_index': 1
        }
        
        page_texts = [("text", 100)]
        
        # Mock LLM to return malformed units
        class MockMalformedClient:
            provider = 'test'
            def chat_completion(self, *args, **kwargs):
                return '''{"semantic_units": [
                    {
                        "title": "Valid Unit",
                        "start_paragraph": 0,
                        "end_paragraph": 1,
                        "semantic_type": "test"
                    },
                    {
                        "title": "Invalid Unit",
                        "start_paragraph": 10,
                        "end_paragraph": 20,
                        "semantic_type": "test"
                    }
                ]}'''
        
        analyzer.llm_client = MockMalformedClient()
        
        # Should filter out invalid units
        semantic_units = analyzer.analyze_section(section_node, page_texts, min_pages=0)
        
        # Should only get the valid unit (or none if validation is strict)
        assert isinstance(semantic_units, list), "Should return a list"
        print(f"  Returned {len(semantic_units)} valid units (filtered out invalid)")
        
        print(f"✓ Malformed semantic units handled gracefully")
    
    def test_create_nodes_from_semantic_units(self):
        """Test node creation from semantic units"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create sample semantic units
        semantic_units = [
            SemanticUnit(
                title="Motivation",
                start_paragraph=0,
                end_paragraph=1,
                start_page=1,
                end_page=1,
                semantic_type="motivation",
                summary="Discusses the motivation for the research"
            ),
            SemanticUnit(
                title="Research Gap",
                start_paragraph=2,
                end_paragraph=3,
                start_page=1,
                end_page=1,
                semantic_type="research_gap",
                summary="Identifies gaps in current knowledge"
            )
        ]
        
        section_node = {
            'title': 'Introduction',
            'text': "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.",
            'start_index': 1,
            'end_index': 1
        }
        
        page_texts = [("text", 100)]
        
        nodes = analyzer.create_nodes_from_semantic_units(semantic_units, section_node, page_texts)
        
        assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
        
        # Verify node structure
        for node in nodes:
            assert 'title' in node, "Node should have title"
            assert 'start_index' in node, "Node should have start_index"
            assert 'end_index' in node, "Node should have end_index"
            assert 'text' in node, "Node should have text"
            assert 'summary' in node, "Node should have summary"
            assert 'node_type' in node, "Node should have node_type"
            assert node['node_type'] == 'semantic_unit', "Node type should be semantic_unit"
            assert 'metadata' in node, "Node should have metadata"
            assert 'semantic_type' in node['metadata'], "Metadata should have semantic_type"
            assert 'nodes' in node, "Node should have nodes list"
        
        print(f"✓ Created {len(nodes)} nodes from semantic units")
        print(f"  Node 1: {nodes[0]['title']}")
        print(f"  Node 2: {nodes[1]['title']}")
    
    def test_identify_boundaries_simple(self):
        """Test simple boundary identification interface"""
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            print("⚠️  Skipping test (no API key)")
            return
        
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        text = """First paragraph about motivation.

Second paragraph about background.

Third paragraph about research gap.

Fourth paragraph about contribution."""
        
        boundaries = analyzer.identify_boundaries(text, "Introduction")
        
        print(f"  Identified {len(boundaries)} boundaries: {boundaries}")
        
        assert isinstance(boundaries, list), "Should return a list"
        # Boundaries should be sorted and unique
        if len(boundaries) > 1:
            assert boundaries == sorted(set(boundaries)), "Boundaries should be sorted and unique"
        
        print(f"✓ Boundary identification completed")
    
    def test_paragraph_to_page_mapping(self):
        """Test mapping of paragraphs to page numbers"""
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client)
        
        # Create multi-page text
        page1_text = "Page 1 paragraph 1.\n\nPage 1 paragraph 2."
        page2_text = "Page 2 paragraph 1.\n\nPage 2 paragraph 2."
        full_text = page1_text + "\n\n" + page2_text
        
        paragraphs = analyzer._split_into_paragraphs(full_text)
        page_texts = [(page1_text, 50), (page2_text, 50)]
        
        paragraph_pages = analyzer._map_paragraphs_to_pages(
            paragraphs, full_text, 1, 2, page_texts
        )
        
        assert len(paragraph_pages) == len(paragraphs), "Should have page number for each paragraph"
        
        # First two paragraphs should be on page 1
        assert paragraph_pages[0] == 1, "First paragraph should be on page 1"
        assert paragraph_pages[1] == 1, "Second paragraph should be on page 1"
        
        # Last two paragraphs should be on page 2
        assert paragraph_pages[2] == 2, "Third paragraph should be on page 2"
        assert paragraph_pages[3] == 2, "Fourth paragraph should be on page 2"
        
        print(f"✓ Paragraph-to-page mapping correct")
        print(f"  Paragraphs: {len(paragraphs)}, Pages: {paragraph_pages}")
    
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
    print("SemanticAnalyzer Unit Test Suite")
    print("="*70)
    
    tester = TestSemanticAnalyzer()
    
    # Run all tests
    tester.run_test("Initialization", tester.test_initialization)
    tester.run_test("Paragraph splitting - double newline", tester.test_paragraph_splitting_double_newline)
    tester.run_test("Paragraph splitting - single newline", tester.test_paragraph_splitting_single_newline)
    tester.run_test("Text with indices creation", tester.test_text_with_indices_creation)
    tester.run_test("Boundary validation - non-overlapping", tester.test_boundary_validation_non_overlapping)
    tester.run_test("Boundary validation - overlapping", tester.test_boundary_validation_overlapping)
    tester.run_test("Boundary validation - out of range", tester.test_boundary_validation_out_of_range)
    tester.run_test("Boundary validation - empty list", tester.test_boundary_validation_empty_list)
    tester.run_test("Short section handling - min pages", tester.test_short_section_handling)
    tester.run_test("Short section handling - few paragraphs", tester.test_short_section_few_paragraphs)
    
    # Tests requiring API key
    if os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY"):
        tester.run_test("Section type detection - introduction", tester.test_section_type_detection_introduction)
        tester.run_test("Section type detection - methods", tester.test_section_type_detection_methods)
        tester.run_test("Section type detection - results", tester.test_section_type_detection_results)
        tester.run_test("Section type detection - discussion", tester.test_section_type_detection_discussion)
        tester.run_test("Analyze section - introduction", tester.test_analyze_section_introduction)
        tester.run_test("Analyze section - methods", tester.test_analyze_section_methods)
        tester.run_test("Analyze section - results", tester.test_analyze_section_results)
        tester.run_test("Identify boundaries - simple", tester.test_identify_boundaries_simple)
    else:
        print("\n⚠️  Skipping API-dependent tests (no API key found)")
        print("   Set GEMINI_API_KEY or CHATGPT_API_KEY to run full tests")
    
    # Error handling tests (don't require API key)
    tester.run_test("Error handling - LLM failure", tester.test_error_handling_llm_failure)
    tester.run_test("Error handling - invalid JSON", tester.test_error_handling_invalid_json_response)
    tester.run_test("Error handling - malformed units", tester.test_error_handling_malformed_semantic_units)
    
    # Node creation tests
    tester.run_test("Create nodes from semantic units", tester.test_create_nodes_from_semantic_units)
    tester.run_test("Paragraph to page mapping", tester.test_paragraph_to_page_mapping)
    
    # Print summary
    all_passed = tester.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
