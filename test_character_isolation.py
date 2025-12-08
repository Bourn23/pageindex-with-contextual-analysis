#!/usr/bin/env python3
"""Systematically test which characters cause the Gemini API to hang."""
import asyncio
import logging
from pageindex.llm_client import get_llm_client
from pageindex.granular.semantic_analyzer import SemanticAnalyzer

logging.basicConfig(level=logging.WARNING)

async def test_text(text, description, timeout=20):
    """Test a text variant with timeout."""
    node = {'title': 'Test', 'text': text}
    
    llm_client = get_llm_client(model='gemini-2.5-flash-lite')
    analyzer = SemanticAnalyzer(llm_client, logging.getLogger())
    
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"Length: {len(text)} chars")
    print(f"First 100 chars: {repr(text[:100])}")
    
    try:
        keywords = await asyncio.wait_for(
            asyncio.to_thread(analyzer.extract_keywords, node),
            timeout=timeout
        )
        print(f"✓ SUCCESS - {len(keywords)} keywords extracted")
        return True
    except asyncio.TimeoutError:
        print(f"✗ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {str(e)[:100]}")
        return False

async def main():
    # Original problematic text
    original = """The above discussion reveals various reasons accounting for the observations of changes in the ionic conductivities of composite electrolytes. Solid-state NMR is capable of clearly identifying the contributing factors, including ion mobility, ion transport pathways, and active ion concentration, and following their changes. This helps pinpoint why certain strategies for ionic conductivity enhancement fail and others work but for reasons other than apparent explanations. For instance, the observed increase in ionic conductivity for LLZO (20 wt %)-PEO (LiTFSI) is merely due to extra Li ions from decomposed LLZO, which has not been identified and discussed before. This may explain a number of observations in reported studies that large ionic conductivity was observed for Li-containing fillers compared to that for fillers without Li. 21-27 The results also infer that composites with a large content of ceramic particles in polymers are not likely to produce synergies for ionic conductivity enhancement, as the polymer-ceramic interfaces play a very small role in ion conduction.

Despite the fact that high content of LLZO does not lead to high ionic conductivity, it increases the stability of the composite when used in solid-state batteries with Li metal as the anode. Symmetric cells made of Li metal electrodes and PEO or composite electrolytes are cycled with a constant current of 15  $\mu$ A/cm², and the current direction is switched every 60 min. The cell voltage profile as a function of cycling time is presented in Figure S6. The cell voltage increases over time for all electrolytes, indicating the increase of impedance induced by electrochemical instability. The voltage of PEO reached 10 V after 366 cycles. With the addition of LLZO, the stability improved significantly. LLZO (5 wt %)—PEO (LiTFSI) was cycled for 743 times before arriving at 10 V, and LLZO (50 wt %)—PEO (LiTFSI) lasted for 920 cycles before arriving at 5 V."""
    
    print("="*70)
    print("SYSTEMATIC CHARACTER ISOLATION TEST")
    print("="*70)
    
    # Test 1: Original (we know this times out)
    await test_text(original, "1. ORIGINAL TEXT (baseline - expect timeout)")
    
    # Test 2: Just the sentence with $\mu$
    latex_sentence = "Symmetric cells made of Li metal electrodes and PEO or composite electrolytes are cycled with a constant current of 15  $\mu$ A/cm², and the current direction is switched every 60 min."
    await test_text(latex_sentence, "2. Just the sentence with $\\mu$")
    
    # Test 3: Just $\mu$ in isolation
    just_latex = "The value is 15 $\mu$ A/cm²."
    await test_text(just_latex, "3. Minimal text with $\\mu$")
    
    # Test 4: Multiple $\mu$ instances
    multiple_latex = "First: 15 $\mu$ A. Second: 20 $\mu$ V. Third: 30 $\mu$ m."
    await test_text(multiple_latex, "4. Multiple $\\mu$ instances")
    
    # Test 5: Just em-dashes
    just_dashes = "LLZO (5 wt %)—PEO (LiTFSI) was cycled for 743 times—this is significant—before arriving at 10 V."
    await test_text(just_dashes, "5. Text with em-dashes (—)")
    
    # Test 6: Combination of $\mu$ and em-dash in same text
    combo = "The current is 15 $\mu$ A—this is important—for the test."
    await test_text(combo, "6. Combination: $\\mu$ AND em-dash")
    
    # Test 7: Original with ONLY $\mu$ replaced
    only_mu_fixed = original.replace('$\\mu$', 'μ')
    await test_text(only_mu_fixed, "7. Original with $\\mu$ → μ")
    
    # Test 8: Original with ONLY em-dash replaced
    only_dash_fixed = original.replace('—', '-')
    await test_text(only_dash_fixed, "8. Original with — → -")
    
    # Test 9: Original with ONLY $ removed
    only_dollar_removed = original.replace('$', '')
    await test_text(only_dollar_removed, "9. Original with $ removed")
    
    # Test 10: Check if it's the superscript ²
    superscript_test = "The area is 15 A/cm² and the volume is 20 m³."
    await test_text(superscript_test, "10. Text with superscripts (² ³)")
    
    # Test 11: Check if it's the specific context around $\mu$
    context_test = "cycled with a constant current of 15  $\mu$ A/cm², and the current"
    await test_text(context_test, "11. Exact context around $\\mu$")
    
    # Test 12: Check spacing around $\mu$
    spacing_test1 = "15 $\mu$ A"
    spacing_test2 = "15  $\mu$ A"  # double space
    spacing_test3 = "15$\mu$A"  # no space
    await test_text(spacing_test1, "12a. Single space: '15 $\\mu$ A'")
    await test_text(spacing_test2, "12b. Double space: '15  $\\mu$ A'")
    await test_text(spacing_test3, "12c. No space: '15$\\mu$A'")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == '__main__':
    asyncio.run(main())
