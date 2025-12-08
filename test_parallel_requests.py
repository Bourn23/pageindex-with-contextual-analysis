#!/usr/bin/env python3
"""Test if parallel requests cause the timeout."""
import asyncio
import logging
from pageindex.llm_client import get_llm_client
from pageindex.granular.semantic_analyzer import SemanticAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# The problematic text
PROBLEMATIC_TEXT = """The above discussion reveals various reasons accounting for the observations of changes in the ionic conductivities of composite electrolytes. Solid-state NMR is capable of clearly identifying the contributing factors, including ion mobility, ion transport pathways, and active ion concentration, and following their changes. This helps pinpoint why certain strategies for ionic conductivity enhancement fail and others work but for reasons other than apparent explanations. For instance, the observed increase in ionic conductivity for LLZO (20 wt %)-PEO (LiTFSI) is merely due to extra Li ions from decomposed LLZO, which has not been identified and discussed before. This may explain a number of observations in reported studies that large ionic conductivity was observed for Li-containing fillers compared to that for fillers without Li. 21-27 The results also infer that composites with a large content of ceramic particles in polymers are not likely to produce synergies for ionic conductivity enhancement, as the polymer-ceramic interfaces play a very small role in ion conduction.

Despite the fact that high content of LLZO does not lead to high ionic conductivity, it increases the stability of the composite when used in solid-state batteries with Li metal as the anode. Symmetric cells made of Li metal electrodes and PEO or composite electrolytes are cycled with a constant current of 15  $\mu$ A/cm², and the current direction is switched every 60 min. The cell voltage profile as a function of cycling time is presented in Figure S6. The cell voltage increases over time for all electrolytes, indicating the increase of impedance induced by electrochemical instability. The voltage of PEO reached 10 V after 366 cycles. With the addition of LLZO, the stability improved significantly. LLZO (5 wt %)—PEO (LiTFSI) was cycled for 743 times before arriving at 10 V, and LLZO (50 wt %)—PEO (LiTFSI) lasted for 920 cycles before arriving at 5 V."""

# Different texts for other parallel requests
OTHER_TEXTS = [
    "This is test text number 1 about battery technology and lithium ion transport.",
    "This is test text number 2 discussing polymer electrolytes and their properties.",
    "This is test text number 3 covering solid-state NMR spectroscopy techniques.",
    "This is test text number 4 about electrochemical impedance spectroscopy methods.",
    "This is test text number 5 regarding ionic conductivity measurements.",
    "This is test text number 6 on composite materials and their applications.",
]

async def extract_keywords_with_timeout(node, analyzer, node_id, timeout=30):
    """Extract keywords with timeout and logging."""
    try:
        print(f"[Node {node_id}] Starting extraction: '{node['title'][:50]}'")
        keywords = await asyncio.wait_for(
            asyncio.to_thread(analyzer.extract_keywords, node),
            timeout=timeout
        )
        print(f"[Node {node_id}] ✓ Success: {len(keywords)} keywords")
        return (node_id, True, len(keywords))
    except asyncio.TimeoutError:
        print(f"[Node {node_id}] ✗ TIMEOUT after {timeout}s")
        return (node_id, False, 0)
    except Exception as e:
        print(f"[Node {node_id}] ✗ Error: {type(e).__name__}")
        return (node_id, False, 0)

async def test_parallel(num_parallel, problematic_position):
    """
    Test with N parallel requests, with the problematic text at a specific position.
    
    Args:
        num_parallel: Total number of parallel requests
        problematic_position: Position (0-indexed) where to place the problematic text
    """
    print(f"\n{'='*70}")
    print(f"TEST: {num_parallel} parallel requests, problematic at position {problematic_position}")
    print(f"{'='*70}")
    
    llm_client = get_llm_client(model='gemini-2.5-flash-lite')
    analyzer = SemanticAnalyzer(llm_client, logging.getLogger())
    
    # Create nodes
    nodes = []
    for i in range(num_parallel):
        if i == problematic_position:
            node = {
                'title': f'Node {i}: PROBLEMATIC TEXT',
                'text': PROBLEMATIC_TEXT
            }
        else:
            node = {
                'title': f'Node {i}: Test text',
                'text': OTHER_TEXTS[i % len(OTHER_TEXTS)]
            }
        nodes.append(node)
    
    # Run in parallel
    tasks = [extract_keywords_with_timeout(node, analyzer, i) for i, node in enumerate(nodes)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successes = sum(1 for r in results if not isinstance(r, Exception) and r[1])
    timeouts = sum(1 for r in results if not isinstance(r, Exception) and not r[1])
    
    print(f"\nResults: {successes} success, {timeouts} timeout")
    
    # Check if problematic node timed out
    prob_result = results[problematic_position]
    if not isinstance(prob_result, Exception):
        if prob_result[1]:
            print(f"✓ Problematic node SUCCEEDED")
        else:
            print(f"✗ Problematic node TIMED OUT")
    
    return successes, timeouts

async def main():
    print("="*70)
    print("PARALLEL REQUEST TIMEOUT INVESTIGATION")
    print("="*70)
    
    # Test 1: Sequential (no parallelism)
    print("\n" + "="*70)
    print("BASELINE: Sequential requests (no parallelism)")
    print("="*70)
    await test_parallel(1, 0)
    
    # Test 2: 3 parallel requests
    await test_parallel(3, 1)  # Problematic in middle
    
    # Test 3: 5 parallel requests
    await test_parallel(5, 2)  # Problematic in middle
    
    # Test 4: 7 parallel requests (like in production)
    await test_parallel(7, 6)  # Problematic at end (like in production)
    
    # Test 5: 7 parallel requests, problematic at start
    await test_parallel(7, 0)  # Problematic at start
    
    # Test 6: 7 parallel requests, problematic in middle
    await test_parallel(7, 3)  # Problematic in middle
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    asyncio.run(main())
