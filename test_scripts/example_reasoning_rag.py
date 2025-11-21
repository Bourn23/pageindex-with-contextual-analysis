#!/usr/bin/env python3
"""
Example: Reasoning-based RAG with PageIndex

Demonstrates the two-step RAG process:
1. Tree Search with Reasoning
2. Answer Generation
"""

import asyncio
import json
from pageindex_rag import PageIndexRAG, print_wrapped


async def main():
    """Run example reasoning-based RAG."""
    
    # Load structure
    structure_path = "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_structure_patched.json"
    
    print("=" * 80)
    print("PageIndex Reasoning-Based RAG Example")
    print("=" * 80)
    
    # Initialize LLM client
    print("\n📡 Initializing LLM client...")
    from pageindex.llm_client import get_llm_client
    llm_client = get_llm_client()
    
    # Initialize RAG
    print(f"📚 Loading document structure...")
    rag = PageIndexRAG(structure_path, llm_client=llm_client)
    
    print(f"   Document: {rag.doc_name}")
    print(f"   Total nodes: {len(rag.node_map)}")
    
    # Example queries
    queries = [
        "How was LLZO synthesized?",
        "What are the main findings about ionic conductivity?",
        "What characterization techniques were used?"
    ]
    
    for i, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"\n🔍 Query {i}: {query}")
        print("=" * 80)
        
        # Run RAG pipeline
        result = await rag.answer_question(query, model="gpt-4o-mini", verbose=True)
        
        # Print summary
        print("\n" + "-" * 80)
        print("📊 Summary:")
        print(f"   Retrieved {len(result['retrieved_nodes'])} nodes")
        print(f"   Context length: {len(result['context'])} characters")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
