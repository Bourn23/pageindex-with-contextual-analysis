#!/usr/bin/env python3
"""Test the problematic node that keeps timing out."""
import asyncio
import logging
from pageindex.llm_client import get_llm_client
from pageindex.granular.semantic_analyzer import SemanticAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

async def test():
    text = """The above discussion reveals various reasons accounting for the observations of changes in the ionic conductivities of composite electrolytes. Solid-state NMR is capable of clearly identifying the contributing factors, including ion mobility, ion transport pathways, and active ion concentration, and following their changes. This helps pinpoint why certain strategies for ionic conductivity enhancement fail and others work but for reasons other than apparent explanations. For instance, the observed increase in ionic conductivity for LLZO (20 wt %)-PEO (LiTFSI) is merely due to extra Li ions from decomposed LLZO, which has not been identified and discussed before. This may explain a number of observations in reported studies that large ionic conductivity was observed for Li-containing fillers compared to that for fillers without Li. 21-27 The results also infer that composites with a large content of ceramic particles in polymers are not likely to produce synergies for ionic conductivity enhancement, as the polymer-ceramic interfaces play a very small role in ion conduction.

Despite the fact that high content of LLZO does not lead to high ionic conductivity, it increases the stability of the composite when used in solid-state batteries with Li metal as the anode. Symmetric cells made of Li metal electrodes and PEO or composite electrolytes are cycled with a constant current of 15  $\mu$ A/cm², and the current direction is switched every 60 min. The cell voltage profile as a function of cycling time is presented in Figure S6. The cell voltage increases over time for all electrolytes, indicating the increase of impedance induced by electrochemical instability. The voltage of PEO reached 10 V after 366 cycles. With the addition of LLZO, the stability improved significantly. LLZO (5 wt %)—PEO (LiTFSI) was cycled for 743 times before arriving at 10 V, and LLZO (50 wt %)—PEO (LiTFSI) lasted for 920 cycles before arriving at 5 V."""
    
    node = {
        'title': 'Correlation of NMR Findings with Ionic Conductivity and Stability',
        'text': text
    }
    
    print("Testing keyword extraction on problematic node...")
    print(f"Text length: {len(text)} chars")
    print(f"Has LaTeX: {'$' in text}")
    print(f"Has special chars: {any(ord(c) > 127 for c in text)}")
    
    llm_client = get_llm_client(model='gemini-2.5-flash-lite')
    analyzer = SemanticAnalyzer(llm_client)
    
    print("\nAttempting keyword extraction with 30s timeout...")
    try:
        keywords = await asyncio.wait_for(
            asyncio.to_thread(analyzer.extract_keywords, node),
            timeout=30.0
        )
        print(f"✓ Success! Extracted {len(keywords)} keywords")
        for kw in keywords[:3]:
            print(f"  - {kw.get('term')}: {kw.get('context')[:50]}...")
    except asyncio.TimeoutError:
        print("✗ Timeout after 30 seconds")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == '__main__':
    asyncio.run(test())
