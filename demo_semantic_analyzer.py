"""
Demo script for SemanticAnalyzer - shows how to analyze sections and identify semantic boundaries.

This script demonstrates:
1. How to initialize the SemanticAnalyzer
2. How to analyze a section and identify semantic sub-units
3. What data is returned (semantic units with boundaries)
4. How section-type-aware analysis works
"""

import os
import sys
import logging
from pathlib import Path

# Add PageIndex to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex.granular.semantic_analyzer import SemanticAnalyzer, SemanticUnit
from pageindex.llm_client import get_llm_client

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample section texts for demonstration
SAMPLE_INTRODUCTION = """
Solid-state batteries have emerged as a promising alternative to conventional lithium-ion batteries due to their enhanced safety and energy density. The use of solid electrolytes eliminates the flammability concerns associated with liquid electrolytes, making them attractive for electric vehicle applications.

Previous research has extensively studied ceramic electrolytes such as Li7La3Zr2O12 (LLZO) garnet-type materials, which exhibit high ionic conductivity at room temperature. However, these materials face challenges in terms of interfacial resistance when coupled with lithium metal anodes. Several studies have attempted to address this issue through surface modifications and composite approaches.

Despite these advances, a significant gap remains in understanding the interfacial dynamics between ceramic particles and polymer matrices in composite electrolytes. The mechanisms governing lithium-ion transport across these interfaces are not well understood, limiting the optimization of composite electrolyte performance.

This work aims to characterize the interfacial lithium-ion exchange process in ceramic-polymer composite electrolytes using solid-state NMR spectroscopy. We demonstrate that interfacial interactions play a crucial role in determining the overall ionic conductivity and propose design principles for optimizing composite electrolyte architectures.
"""

SAMPLE_METHODS = """
Composite electrolytes were prepared by mixing Li7La3Zr2O12 (LLZO) nanoparticles with polyethylene oxide (PEO) and lithium bis(trifluoromethanesulfonyl)imide (LiTFSI) salt. LLZO nanoparticles were synthesized using a sol-gel method and calcined at 900°C for 12 hours.

The composite films were fabricated by dissolving PEO and LiTFSI in acetonitrile, followed by the addition of LLZO nanoparticles. The mixture was stirred for 24 hours and cast onto Teflon plates. After solvent evaporation, the films were dried under vacuum at 60°C for 48 hours.

Solid-state NMR measurements were performed on a Bruker Avance III 400 MHz spectrometer equipped with a 4 mm magic-angle spinning (MAS) probe. 7Li NMR spectra were acquired at various temperatures ranging from 25°C to 80°C with a spinning rate of 10 kHz.

Ionic conductivity was measured using electrochemical impedance spectroscopy (EIS) with blocking electrodes. The samples were sandwiched between stainless steel electrodes and measured over a frequency range of 1 Hz to 1 MHz at temperatures from 25°C to 80°C.
"""

SAMPLE_RESULTS = """
The 7Li NMR spectra of the composite electrolytes revealed two distinct peaks at 0 ppm and -1.2 ppm, corresponding to lithium ions in the polymer phase and at the ceramic-polymer interface, respectively. The relative intensity of the interfacial peak increased with LLZO content, indicating enhanced interfacial interactions.

Temperature-dependent NMR measurements showed that the interfacial lithium signal exhibited faster relaxation times compared to the bulk polymer signal. This suggests higher mobility of lithium ions at the interface, which is consistent with the formation of a space-charge layer.

Ionic conductivity measurements demonstrated a non-monotonic dependence on LLZO content. The conductivity increased from 1.2 × 10⁻⁵ S/cm for pure PEO to a maximum of 3.8 × 10⁻⁴ S/cm at 20 wt% LLZO, then decreased at higher loadings. This behavior correlates with the NMR observations of interfacial dynamics.

Activation energy analysis revealed that composites with optimal LLZO content exhibited lower activation energies (0.45 eV) compared to pure PEO (0.65 eV), indicating that interfacial pathways facilitate lithium-ion transport.
"""


def demo_semantic_analysis():
    """Demonstrate semantic analysis on sample sections."""
    logger.info("=" * 80)
    logger.info("SEMANTIC ANALYZER DEMO")
    logger.info("=" * 80)
    
    # Check if GEMINI_API_KEY is set (required for structured output)
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set - semantic analysis requires Gemini API")
        logger.info("Set GEMINI_API_KEY in .env file")
        return
    
    try:
        # Step 1: Initialize LLM client
        logger.info("\n1. Initializing LLM client...")
        llm_client = get_llm_client()
        logger.info(f"   ✓ LLM client initialized (provider: {llm_client.provider})")
        
        # Step 2: Initialize SemanticAnalyzer
        logger.info("\n2. Initializing SemanticAnalyzer...")
        analyzer = SemanticAnalyzer(
            llm_client=llm_client,
            logger=logger
        )
        logger.info("   ✓ SemanticAnalyzer initialized")
        logger.info(f"   - Available section types: {list(analyzer.section_prompts.keys())}")
        
        # Step 3: Analyze Introduction section
        logger.info("\n3. Analyzing INTRODUCTION section...")
        logger.info("   (This uses section-type-aware prompts)")
        
        intro_node = {
            'title': 'Introduction',
            'text': SAMPLE_INTRODUCTION,
            'start_index': 1,
            'end_index': 2
        }
        
        # Create dummy page_texts (for page mapping)
        page_texts = [(SAMPLE_INTRODUCTION, len(SAMPLE_INTRODUCTION.split()))]
        
        intro_units = analyzer.analyze_section(
            section_node=intro_node,
            page_texts=page_texts,
            min_pages=0  # Allow analysis even for short sections in demo
        )
        
        logger.info(f"   ✓ Found {len(intro_units)} semantic units")
        
        if intro_units:
            logger.info("\n   Introduction Semantic Units:")
            for i, unit in enumerate(intro_units, 1):
                logger.info(f"\n   Unit {i}: {unit.title}")
                logger.info(f"   - Type: {unit.semantic_type}")
                logger.info(f"   - Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
                logger.info(f"   - Pages: {unit.start_page} to {unit.end_page}")
                logger.info(f"   - Summary: {unit.summary}")
        
        # Step 4: Analyze Methods section
        logger.info("\n4. Analyzing METHODS section...")
        
        methods_node = {
            'title': 'Experimental Methods',
            'text': SAMPLE_METHODS,
            'start_index': 3,
            'end_index': 4
        }
        
        page_texts = [(SAMPLE_METHODS, len(SAMPLE_METHODS.split()))]
        
        methods_units = analyzer.analyze_section(
            section_node=methods_node,
            page_texts=page_texts,
            min_pages=0
        )
        
        logger.info(f"   ✓ Found {len(methods_units)} semantic units")
        
        if methods_units:
            logger.info("\n   Methods Semantic Units:")
            for i, unit in enumerate(methods_units, 1):
                logger.info(f"\n   Unit {i}: {unit.title}")
                logger.info(f"   - Type: {unit.semantic_type}")
                logger.info(f"   - Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
                logger.info(f"   - Summary: {unit.summary[:100]}..." if len(unit.summary) > 100 else f"   - Summary: {unit.summary}")
        
        # Step 5: Analyze Results section
        logger.info("\n5. Analyzing RESULTS section...")
        
        results_node = {
            'title': 'Results and Discussion',
            'text': SAMPLE_RESULTS,
            'start_index': 5,
            'end_index': 6
        }
        
        page_texts = [(SAMPLE_RESULTS, len(SAMPLE_RESULTS.split()))]
        
        results_units = analyzer.analyze_section(
            section_node=results_node,
            page_texts=page_texts,
            min_pages=0
        )
        
        logger.info(f"   ✓ Found {len(results_units)} semantic units")
        
        if results_units:
            logger.info("\n   Results Semantic Units:")
            for i, unit in enumerate(results_units, 1):
                logger.info(f"\n   Unit {i}: {unit.title}")
                logger.info(f"   - Type: {unit.semantic_type}")
                logger.info(f"   - Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
                logger.info(f"   - Summary: {unit.summary[:100]}..." if len(unit.summary) > 100 else f"   - Summary: {unit.summary}")
        
        # Step 6: Show data structure
        logger.info("\n6. Data Structure Example:")
        if intro_units:
            unit = intro_units[0]
            logger.info("   SemanticUnit fields:")
            logger.info(f"   - title: str = '{unit.title}'")
            logger.info(f"   - start_paragraph: int = {unit.start_paragraph}")
            logger.info(f"   - end_paragraph: int = {unit.end_paragraph}")
            logger.info(f"   - start_page: int = {unit.start_page}")
            logger.info(f"   - end_page: int = {unit.end_page}")
            logger.info(f"   - semantic_type: str = '{unit.semantic_type}'")
            logger.info(f"   - summary: str = '{unit.summary[:50]}...'")
        
        # Step 7: Demonstrate section type detection
        logger.info("\n7. Section Type Detection:")
        test_titles = [
            "Introduction",
            "Materials and Methods",
            "Experimental Procedures",
            "Results and Discussion",
            "Conclusions",
            "3.2 Sample Preparation"
        ]
        
        for title in test_titles:
            detected_type = analyzer._detect_section_type(title)
            logger.info(f"   '{title}' → {detected_type}")
        
        # Step 8: Show how to create nodes from semantic units
        logger.info("\n8. Creating nodes from semantic units...")
        if intro_units:
            nodes = analyzer.create_nodes_from_semantic_units(
                semantic_units=intro_units,
                section_node=intro_node,
                page_texts=[(SAMPLE_INTRODUCTION, len(SAMPLE_INTRODUCTION.split()))]
            )
            
            logger.info(f"   ✓ Created {len(nodes)} node(s)")
            if nodes:
                logger.info(f"   Example node structure:")
                logger.info(f"   - title: {nodes[0]['title']}")
                logger.info(f"   - start_index: {nodes[0]['start_index']}")
                logger.info(f"   - end_index: {nodes[0]['end_index']}")
                logger.info(f"   - node_type: {nodes[0]['node_type']}")
                logger.info(f"   - metadata: {nodes[0]['metadata']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DEMO COMPLETE")
        logger.info("=" * 80)
        logger.info("\nKey Takeaways:")
        logger.info("1. SemanticAnalyzer uses section-type-aware prompts")
        logger.info("2. It identifies coherent sub-topics within sections")
        logger.info("3. Returns SemanticUnit objects with paragraph and page ranges")
        logger.info("4. Can create node structures ready for tree integration")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)


def demo_boundary_identification():
    """Demonstrate the simpler boundary identification interface."""
    logger.info("\n" + "=" * 80)
    logger.info("BOUNDARY IDENTIFICATION DEMO")
    logger.info("=" * 80)
    
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set")
        return
    
    try:
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client=llm_client, logger=logger)
        
        logger.info("\nIdentifying boundaries in Introduction section...")
        boundaries = analyzer.identify_boundaries(
            text=SAMPLE_INTRODUCTION,
            section_type="Introduction"
        )
        
        logger.info(f"✓ Found {len(boundaries)} boundaries at paragraphs: {boundaries}")
        
        # Show what text is at each boundary
        paragraphs = analyzer._split_into_paragraphs(SAMPLE_INTRODUCTION)
        logger.info("\nBoundary locations:")
        for boundary_idx in boundaries:
            if boundary_idx < len(paragraphs):
                logger.info(f"\n  Paragraph {boundary_idx}:")
                logger.info(f"  {paragraphs[boundary_idx][:100]}...")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def demo_with_real_pdf(pdf_path: str):
    """Demonstrate semantic analysis on a real PDF."""
    logger.info("\n" + "=" * 80)
    logger.info("SEMANTIC ANALYZER - REAL PDF DEMO")
    logger.info("=" * 80)
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set")
        return
    
    try:
        import pymupdf
        
        # Open PDF
        logger.info(f"\n1. Opening PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        logger.info(f"   ✓ PDF has {len(doc)} pages")
        
        # Extract text from pages
        logger.info("\n2. Extracting text from pages...")
        page_texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            page_texts.append((text, len(text.split())))
        logger.info(f"   ✓ Extracted text from {len(page_texts)} pages")
        
        # Initialize analyzer
        logger.info("\n3. Initializing SemanticAnalyzer...")
        llm_client = get_llm_client()
        analyzer = SemanticAnalyzer(llm_client=llm_client, logger=logger)
        logger.info("   ✓ SemanticAnalyzer initialized")
        
        # Extract ABSTRACT section (page 1)
        logger.info("\n4. Analyzing ABSTRACT section...")
        abstract_text = page_texts[0][0]
        # Find abstract section
        import re
        abstract_match = re.search(r'ABSTRACT\s+(.*?)(?=\n[A-Z]{2,}|\Z)', abstract_text, re.DOTALL)
        if abstract_match:
            abstract_content = abstract_match.group(1).strip()
            
            abstract_node = {
                'title': 'Abstract',
                'text': abstract_content,
                'start_index': 1,
                'end_index': 1
            }
            
            abstract_units = analyzer.analyze_section(
                section_node=abstract_node,
                page_texts=page_texts[:1],
                min_pages=0
            )
            
            logger.info(f"   ✓ Found {len(abstract_units)} semantic units in Abstract")
            
            if abstract_units:
                for i, unit in enumerate(abstract_units, 1):
                    logger.info(f"\n   Unit {i}: {unit.title}")
                    logger.info(f"   - Type: {unit.semantic_type}")
                    logger.info(f"   - Summary: {unit.summary[:100]}..." if len(unit.summary) > 100 else f"   - Summary: {unit.summary}")
        else:
            logger.info("   Abstract section not found")
        
        # Extract INTRODUCTION section (page 1)
        logger.info("\n5. Analyzing INTRODUCTION section...")
        intro_match = re.search(r'INTRODUCTION\s+(.*?)(?=\n[A-Z]{2,}|\Z)', abstract_text, re.DOTALL)
        if intro_match:
            intro_content = intro_match.group(1).strip()
            
            intro_node = {
                'title': 'Introduction',
                'text': intro_content,
                'start_index': 1,
                'end_index': 1
            }
            
            intro_units = analyzer.analyze_section(
                section_node=intro_node,
                page_texts=page_texts[:1],
                min_pages=0
            )
            
            logger.info(f"   ✓ Found {len(intro_units)} semantic units in Introduction")
            
            if intro_units:
                for i, unit in enumerate(intro_units, 1):
                    logger.info(f"\n   Unit {i}: {unit.title}")
                    logger.info(f"   - Type: {unit.semantic_type}")
                    logger.info(f"   - Summary: {unit.summary[:100]}..." if len(unit.summary) > 100 else f"   - Summary: {unit.summary}")
        else:
            logger.info("   Introduction section not found")
        
        # Analyze full page 2 as a section
        logger.info("\n6. Analyzing full Page 2 content...")
        page2_text = page_texts[1][0]
        
        page2_node = {
            'title': 'Page 2 Content',
            'text': page2_text,
            'start_index': 2,
            'end_index': 2
        }
        
        page2_units = analyzer.analyze_section(
            section_node=page2_node,
            page_texts=page_texts[1:2],
            min_pages=0
        )
        
        logger.info(f"   ✓ Found {len(page2_units)} semantic units in Page 2")
        
        if page2_units:
            for i, unit in enumerate(page2_units, 1):
                logger.info(f"\n   Unit {i}: {unit.title}")
                logger.info(f"   - Type: {unit.semantic_type}")
                logger.info(f"   - Paragraphs: {unit.start_paragraph} to {unit.end_paragraph}")
                logger.info(f"   - Summary: {unit.summary[:100]}..." if len(unit.summary) > 100 else f"   - Summary: {unit.summary}")
        
        logger.info("\n" + "=" * 80)
        logger.info("REAL PDF DEMO COMPLETE")
        logger.info("=" * 80)
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error during real PDF demo: {e}", exc_info=True)


def main():
    """Main entry point."""
    # Run sample text demos
    demo_semantic_analysis()
    demo_boundary_identification()
    
    # Run real PDF demo if file exists
    test_pdf = "tests/pdfs/earthmover.pdf"
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
    
    if os.path.exists(test_pdf):
        demo_with_real_pdf(test_pdf)
    else:
        logger.info(f"\nSkipping real PDF demo - {test_pdf} not found")
        logger.info(f"Run with: python demo_semantic_analyzer.py path/to/pdf")


if __name__ == "__main__":
    main()
