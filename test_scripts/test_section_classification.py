"""
Test LLM-based section classification with challenging titles.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pageindex.granular.semantic_analyzer import SemanticAnalyzer
from pageindex.llm_client import get_llm_client


def test_challenging_section_titles():
    """Test section classification with real-world challenging titles."""
    
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
        print("⚠️  Skipping test - no API key found")
        return
    
    print("Testing LLM-based section classification with challenging titles...\n")
    
    # Initialize
    llm_client = get_llm_client()
    analyzer = SemanticAnalyzer(llm_client)
    
    # Real-world challenging section titles
    test_cases = [
        # Clear cases
        ("1. Introduction", "introduction"),
        ("2. Experimental Section", "methods"),
        ("3. Results", "results"),
        ("4. Discussion and Conclusions", "discussion"),
        
        # Ambiguous or compound titles
        ("Synthesis and Characterization", "methods"),
        ("Structural Analysis", "results"),
        ("Theoretical Background", "introduction"),
        ("Computational Methods", "methods"),
        
        # Domain-specific titles
        ("Sample Preparation", "methods"),
        ("Electrochemical Measurements", "methods"),
        ("Morphological Characterization", "results"),
        ("Performance Evaluation", "results"),
        
        # Less common formats
        ("Related Work", "introduction"),
        ("Approach", "methods"),
        ("Observations", "results"),
        ("Implications", "discussion"),
        
        # Numbered sections without clear keywords
        ("3.1 Conductivity Studies", "results"),
        ("2.2 Fabrication Process", "methods"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for title, expected in test_cases:
        result = analyzer._detect_section_type(title)
        
        if result == expected:
            print(f"✓ '{title}' → {result}")
            correct += 1
        else:
            print(f"✗ '{title}' → {result} (expected: {expected})")
    
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"{'='*60}")
    
    if correct >= total * 0.8:  # 80% threshold
        print("\n✅ Section classification is working well!")
    else:
        print("\n⚠️  Section classification needs improvement")


if __name__ == "__main__":
    try:
        test_challenging_section_titles()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
