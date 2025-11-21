#!/usr/bin/env python3
"""
Demo: Reasoning-Based RAG with PageIndex

This script demonstrates the complete workflow:
1. Load PageIndex structure
2. Perform reasoning-based tree search
3. Generate answer from retrieved context

No vectors, no chunking - just LLM reasoning over document structure.
"""

import asyncio
import json
from pathlib import Path


async def demo_reasoning_rag():
    """Demonstrate reasoning-based RAG."""
    
    print("=" * 80)
    print("PageIndex: Reasoning-Based RAG Demo")
    print("=" * 80)
    print()
    print("This demo shows how LLM reasoning replaces vector similarity search.")
    print("The LLM navigates the document structure like a human expert would.")
    print()
    
    # Import
    from pageindex_rag import PageIndexRAG, print_wrapped
    from pageindex.llm_client import get_llm_client
    
    # Setup
    structure_path = "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_structure_patched.json"
    
    if not Path(structure_path).exists():
        print(f"❌ Structure file not found: {structure_path}")
        print("   Please run extraction first:")
        print("   python run_pageindex.py --pdf_path document.pdf --granularity medium")
        return
    
    print("📡 Step 0: Initialize LLM client")
    print("-" * 80)
    llm_client = get_llm_client()
    print("✓ LLM client ready")
    print()
    
    print("📚 Step 1: Load document structure")
    print("-" * 80)
    rag = PageIndexRAG(structure_path, llm_client=llm_client)
    print(f"✓ Document: {rag.doc_name}")
    print(f"✓ Total nodes: {len(rag.node_map)}")
    print()
    
    # Example query
    query = "How was LLZO synthesized?"
    
    print("❓ Question:")
    print("-" * 80)
    print(f"   {query}")
    print()
    
    print("🔍 Step 2: Reasoning-based tree search")
    print("-" * 80)
    print("The LLM examines the document structure and reasons about")
    print("which nodes are likely to contain the answer...")
    print()
    
    search_result = await rag.tree_search(query)
    
    print("💭 LLM Reasoning Process:")
    print_wrapped(search_result['thinking'], width=76)
    print()
    
    print(f"📄 Retrieved Nodes: {len(search_result['node_list'])}")
    for node_id in search_result['node_list']:
        node = rag.node_map.get(node_id)
        if node:
            title = node.get('title', 'Unknown')
            pages = f"p.{node.get('start_index', '?')}"
            semantic_type = node.get('metadata', {}).get('semantic_type', 'N/A')
            print(f"   • {title}")
            print(f"     └─ {pages}, type: {semantic_type}")
    print()
    
    print("📝 Step 3: Extract context from retrieved nodes")
    print("-" * 80)
    context = rag.get_context_from_nodes(search_result['node_list'])
    print(f"✓ Context length: {len(context)} characters")
    print()
    print("Context preview:")
    print_wrapped(context[:300] + "...", width=76)
    print()
    
    print("💡 Step 4: Generate answer from context")
    print("-" * 80)
    answer = await rag.generate_answer(query, context)
    print("✅ Answer:")
    print()
    print_wrapped(answer, width=76)
    print()
    
    print("=" * 80)
    print("🎯 Key Takeaways")
    print("=" * 80)
    print()
    print("✓ No vector database needed")
    print("✓ No chunking required")
    print("✓ Transparent reasoning process")
    print("✓ Human-like document navigation")
    print("✓ Hierarchical context preserved")
    print()
    
    print("=" * 80)
    print("📊 Comparison with Vector RAG")
    print("=" * 80)
    print()
    print("Vector RAG:")
    print("  1. Embed query → 2. Search vectors → 3. Get top-k chunks")
    print("  ❌ Black box similarity")
    print("  ❌ No reasoning")
    print("  ❌ Requires vector DB")
    print()
    print("Reasoning RAG:")
    print("  1. LLM examines structure → 2. Reasons about relevance → 3. Retrieves nodes")
    print("  ✓ Transparent reasoning")
    print("  ✓ Human-like navigation")
    print("  ✓ No infrastructure needed")
    print()
    
    print("=" * 80)
    print("🚀 Try it yourself!")
    print("=" * 80)
    print()
    print("from pageindex_rag import PageIndexRAG")
    print("from pageindex.llm_client import get_llm_client")
    print()
    print("llm_client = get_llm_client()")
    print("rag = PageIndexRAG('structure.json', llm_client=llm_client)")
    print()
    print("result = await rag.answer_question('Your question here', verbose=True)")
    print("print(result['reasoning'])  # See the LLM's thought process")
    print("print(result['answer'])     # Get the answer")
    print()


if __name__ == '__main__':
    asyncio.run(demo_reasoning_rag())
