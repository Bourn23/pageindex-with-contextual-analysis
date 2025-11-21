import asyncio
from pageindex_rag import PageIndexRAG
from pageindex.llm_client import get_llm_client


async def main():
    """Test reasoning-based RAG."""
    
    # Initialize
    print("Initializing RAG system...")
    llm_client = get_llm_client(provider='gemini')
    
    print(f"Provider: {llm_client.provider}")
    print(f"Default model: {llm_client.default_model or 'auto-detect'}\n")
    
    rag = PageIndexRAG(
        'results/adefowoke-ojokoh-et-al-2009-automated-document-metadata-extraction_structure.json',
        llm_client=llm_client
    )
    
    print(f"Document: {rag.doc_name}")
    print(f"Total nodes: {len(rag.node_map)}\n")
    
    # Ask question - LLM reasons over document structure
    # Model will be auto-detected as gemini-2.0-flash-exp for Gemini provider
    result = await rag.answer_question("What's the contribution of this paper?", verbose=True)
    
    print(f"\n{'='*80}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved {len(result['retrieved_nodes'])} nodes")


if __name__ == '__main__':
    asyncio.run(main())