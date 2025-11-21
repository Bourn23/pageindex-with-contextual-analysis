#!/usr/bin/env python3
"""
PageIndex RAG: Reasoning-based retrieval without vectors.

Implements the two-step RAG process:
1. Tree Search with Reasoning - LLM navigates the document structure
2. Answer Generation - LLM generates answer from retrieved context
"""

import json
import asyncio
import os
from typing import List, Dict, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class TreeSearchResponse(BaseModel):
    """Structured response for tree search with reasoning."""
    thinking: str = Field(..., description="The reasoning process for identifying relevant nodes")
    node_list: List[str] = Field(..., description="List of node IDs that are relevant to the question")


class PageIndexRAG:
    """
    Reasoning-based RAG system using PageIndex structure.
    
    No vectors, no chunking - just reasoning over document structure.
    """
    
    def __init__(self, structure_path: str, llm_client=None):
        """
        Initialize PageIndex RAG.
        
        Args:
            structure_path: Path to PageIndex structure JSON
            llm_client: LLM client for reasoning (optional, can be set later)
        """
        self.structure_path = Path(structure_path)
        self.llm_client = llm_client
        
        # Load structure
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.doc_name = data.get('doc_name', 'Unknown')
        self.structure = data.get('structure', [])
        
        # Create node mapping for quick lookup
        self.node_map = self._create_node_mapping(self.structure)
        
        # Create simplified tree (without text) for reasoning
        self.tree_for_reasoning = self._remove_text_fields(self.structure)
    
    def _create_node_mapping(self, nodes: List[Dict], node_map: Optional[Dict] = None) -> Dict:
        """Create mapping from node_id to node."""
        if node_map is None:
            node_map = {}
        
        for node in nodes:
            node_id = node.get('node_id')
            if node_id:
                node_map[node_id] = node
            
            # Recursively process children
            if node.get('nodes'):
                self._create_node_mapping(node['nodes'], node_map)
        
        return node_map
    
    def _remove_text_fields(self, nodes: List[Dict]) -> List[Dict]:
        """Remove text fields for reasoning (keep structure and summaries)."""
        cleaned = []
        
        for node in nodes:
            cleaned_node = {
                'node_id': node.get('node_id', ''),
                'title': node.get('title', ''),
                'summary': node.get('summary', ''),
                'start_index': node.get('start_index'),
                'end_index': node.get('end_index'),
                'node_type': node.get('node_type', 'section')
            }
            
            # Add semantic type if available
            if node.get('metadata', {}).get('semantic_type'):
                cleaned_node['semantic_type'] = node['metadata']['semantic_type']
            
            # Recursively process children
            if node.get('nodes'):
                cleaned_node['nodes'] = self._remove_text_fields(node['nodes'])
            
            cleaned.append(cleaned_node)
        
        return cleaned
    
    async def tree_search(self, query: str, model: str = None) -> Dict:
        """
        Step 1: Reasoning-based tree search.
        
        LLM navigates the document structure to find relevant nodes.
        
        Args:
            query: User question
            model: LLM model to use
            
        Returns:
            Dict with 'thinking' (reasoning process) and 'node_list' (relevant node IDs)
        """
        if not self.llm_client:
            raise ValueError("LLM client not set. Please provide llm_client in constructor.")
        
        # Use client's default model if not specified
        if model is None:
            model = getattr(self.llm_client, 'default_model', None)
            if model is None:
                # Auto-detect based on provider
                if getattr(self.llm_client, 'provider', 'openai') == 'gemini':
                    model = 'gemini-2.5-flash'
                else:
                    model = 'gpt-4o-mini'
        
        search_prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(self.tree_for_reasoning, indent=2)}

Guidelines:
- Consider the hierarchical structure - parent nodes provide context
- Look for semantic types that match the query (e.g., "methods" for "how" questions)
- Include nodes whose summaries are relevant to the question
- Be selective - only include nodes that likely contain the answer

Provide your reasoning process and the list of relevant node IDs.
"""
        
        # Use structured output for Gemini
        if getattr(self.llm_client, 'provider', 'openai') == 'gemini':
            result = await self._tree_search_gemini_structured(search_prompt, model)
        else:
            # Fallback to JSON parsing for OpenAI
            result = await self._tree_search_openai(search_prompt, model)
        
        return result
    
    async def _tree_search_gemini_structured(self, prompt: str, model: str) -> Dict:
        """Tree search using Gemini's structured output."""
        from google import genai
        from google.genai import types
        
        # Get Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        gemini_client = genai.Client(api_key=api_key)
        
        try:
            # Call Gemini with JSON schema enforcement
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=TreeSearchResponse.model_json_schema()
                )
            )
            
            # With structured output, response.text is guaranteed to be valid JSON!
            search_response = TreeSearchResponse.model_validate_json(response.text)
            
            return {
                'thinking': search_response.thinking,
                'node_list': search_response.node_list
            }
        except Exception as e:
            raise ValueError(f"Gemini structured output failed: {e}")
    
    async def _tree_search_openai(self, prompt: str, model: str) -> Dict:
        """Tree search using OpenAI (with JSON parsing fallback)."""
        full_prompt = prompt + """

Please reply in the following JSON format:
{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}

Directly return the final JSON structure. Do not output anything else.
"""
        
        response = await self.llm_client.chat_completion_async(
            model=model,
            prompt=full_prompt,
            temperature=0
        )
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {response}")
    
    def get_context_from_nodes(self, node_ids: List[str]) -> str:
        """
        Extract text content from retrieved nodes.
        
        Args:
            node_ids: List of node IDs to retrieve
            
        Returns:
            Combined text content from all nodes
        """
        context_parts = []
        
        for node_id in node_ids:
            node = self.node_map.get(node_id)
            if node:
                text = node.get('text', '')
                if text:
                    title = node.get('title', 'Unknown')
                    pages = f"p.{node.get('start_index', '?')}"
                    context_parts.append(f"[{title} ({pages})]\n{text}")
        
        return "\n\n".join(context_parts)
    
    async def generate_answer(self, query: str, context: str, model: str = None) -> str:
        """
        Step 2: Generate answer from retrieved context.
        
        Args:
            query: User question
            context: Retrieved context from tree search
            model: LLM model to use
            
        Returns:
            Generated answer
        """
        if not self.llm_client:
            raise ValueError("LLM client not set. Please provide llm_client in constructor.")
        
        # Use client's default model if not specified
        if model is None:
            model = getattr(self.llm_client, 'default_model', None)
            if model is None:
                # Auto-detect based on provider
                if getattr(self.llm_client, 'provider', 'openai') == 'gemini':
                    model = 'gemini-2.5-flash'
                else:
                    model = 'gpt-4o-mini'
        
        answer_prompt = f"""Answer the question based on the context:

Question: {query}

Context:
{context}

Provide a clear, concise answer based only on the context provided.
If the context doesn't contain enough information, say so.
"""
        
        answer = await self.llm_client.chat_completion_async(
            model=model,
            prompt=answer_prompt,
            temperature=0
        )
        
        return answer
    
    async def answer_question(self, query: str, model: str = None, 
                            verbose: bool = False) -> Dict:
        """
        Complete RAG pipeline: tree search + answer generation.
        
        Args:
            query: User question
            model: LLM model to use
            verbose: Print intermediate steps
            
        Returns:
            Dict with 'question', 'answer', 'reasoning', 'retrieved_nodes', 'context'
        """
        # Step 1: Tree search with reasoning
        if verbose:
            print("🔍 Step 1: Reasoning-based tree search...")
        
        search_result = await self.tree_search(query, model=model)
        
        if verbose:
            print(f"\n💭 Reasoning Process:")
            print(search_result['thinking'])
            print(f"\n📄 Retrieved {len(search_result['node_list'])} nodes:")
            for node_id in search_result['node_list']:
                node = self.node_map.get(node_id)
                if node:
                    print(f"  - {node.get('title', 'Unknown')} (p.{node.get('start_index', '?')})")
        
        # Step 2: Extract context
        context = self.get_context_from_nodes(search_result['node_list'])
        
        if verbose:
            print(f"\n📝 Context length: {len(context)} characters")
        
        # Step 3: Generate answer
        if verbose:
            print("\n💡 Step 2: Generating answer...")
        
        answer = await self.generate_answer(query, context, model=model)
        
        if verbose:
            print(f"\n✅ Answer:\n{answer}")
        
        return {
            'question': query,
            'answer': answer,
            'reasoning': search_result['thinking'],
            'retrieved_nodes': [
                {
                    'node_id': node_id,
                    'title': self.node_map[node_id].get('title', 'Unknown'),
                    'pages': [
                        self.node_map[node_id].get('start_index'),
                        self.node_map[node_id].get('end_index')
                    ]
                }
                for node_id in search_result['node_list']
                if node_id in self.node_map
            ],
            'context': context
        }
    
    def search_by_semantic_type(self, semantic_type: str) -> List[Dict]:
        """
        Filter nodes by semantic type (e.g., 'sample_preparation', 'results').
        
        Args:
            semantic_type: Semantic type to filter by
            
        Returns:
            List of matching nodes
        """
        matching_nodes = []
        
        for node_id, node in self.node_map.items():
            node_semantic_type = node.get('metadata', {}).get('semantic_type', '')
            if node_semantic_type == semantic_type:
                matching_nodes.append({
                    'node_id': node_id,
                    'title': node.get('title', 'Unknown'),
                    'summary': node.get('summary', ''),
                    'pages': [node.get('start_index'), node.get('end_index')]
                })
        
        return matching_nodes


# Utility functions for printing
def print_wrapped(text: str, width: int = 80):
    """Print text with word wrapping."""
    import textwrap
    wrapped = textwrap.fill(text, width=width)
    print(wrapped)


def print_tree(nodes: List[Dict], indent: int = 0, max_depth: int = 3):
    """Print tree structure in a readable format."""
    if indent > max_depth:
        return
    
    for node in nodes:
        title = node.get('title', 'Unknown')
        node_id = node.get('node_id', '')
        pages = f"p.{node.get('start_index', '?')}"
        
        print("  " * indent + f"├─ {title} ({node_id}, {pages})")
        
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1, max_depth)


# Example usage
async def main():
    """Example usage of PageIndex RAG."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pageindex_rag.py <structure.json> [query]")
        sys.exit(1)
    
    structure_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What are the main findings?"
    
    # Initialize LLM client (you need to implement this based on your LLM)
    from pageindex.llm_client import get_llm_client
    llm_client = get_llm_client()
    
    # Initialize RAG
    rag = PageIndexRAG(structure_path, llm_client=llm_client)
    
    print(f"📚 Document: {rag.doc_name}")
    print(f"📊 Total nodes: {len(rag.node_map)}")
    print(f"\n❓ Question: {query}\n")
    print("=" * 80)
    
    # Answer question with verbose output
    result = await rag.answer_question(query, verbose=True)
    
    print("\n" + "=" * 80)
    print("\n📋 Full Result:")
    print(json.dumps({
        'question': result['question'],
        'answer': result['answer'],
        'retrieved_nodes': result['retrieved_nodes']
    }, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
