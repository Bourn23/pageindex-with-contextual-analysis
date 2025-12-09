#!/usr/bin/env python3
"""
Visualize keyword semantic clusters using LLM embeddings.
Helps identify semantically similar keywords that may be labeled differently.

Usage:
    python visualize_keyword_clusters.py <structure_json> [options]

Examples:
    python visualize_keyword_clusters.py results/paper_keywords_structure.json
    python visualize_keyword_clusters.py results/paper_keywords_structure.json --provider gemini
    python visualize_keyword_clusters.py results/paper_keywords_structure.json --min-similarity 0.7
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()


def extract_keywords_from_structure(structure: list) -> List[Dict]:
    """Extract all keyword nodes from the structure."""
    keywords = []
    
    def traverse(nodes):
        for node in nodes:
            if node.get('node_type') == 'keyword':
                keywords.append({
                    'term': node.get('metadata', {}).get('term') or node.get('title'),
                    'context': node.get('summary') or node.get('metadata', {}).get('context', ''),
                    'relevance': node.get('metadata', {}).get('relevance', ''),
                    'parent': node.get('metadata', {}).get('parent_title', 'Unknown'),
                    'pages': f"{node.get('start_index')}-{node.get('end_index')}",
                    'node_id': node.get('node_id'),
                    'text': node.get('text', '')[:200]  # First 200 chars
                })
            
            if 'nodes' in node and node['nodes']:
                traverse(node['nodes'])
    
    traverse(structure)
    return keywords


def get_gemini_embeddings(texts: List[str], model_name: str = "models/text-embedding-004", batch_size: int = 100) -> np.ndarray:
    """Generate embeddings using Gemini's API."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    client = genai.Client(api_key=api_key)
    embedding_config = types.EmbedContentConfig(
        output_dimensionality=768,  # Using 768 for efficiency
        task_type='clustering'
    )
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating embeddings"):
        batch = texts[i:i + batch_size]
        try:
            result = client.models.embed_content(
                model=model_name,
                contents=batch,
                config=embedding_config
            )
            
            # Normalize embeddings
            for embedding in result.embeddings:
                embedding_np = np.array(embedding.values)
                normed_embedding = embedding_np / np.linalg.norm(embedding_np)
                all_embeddings.append(normed_embedding)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Add zero vectors for failed batch
            all_embeddings.extend([np.zeros(768) for _ in batch])
    
    return np.array(all_embeddings)


def get_openai_embeddings(texts: List[str], model_name: str = "text-embedding-3-small", batch_size: int = 100) -> np.ndarray:
    """Generate embeddings using OpenAI's API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('CHATGPT_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY or CHATGPT_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating embeddings"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=model_name,
                input=batch
            )
            
            for item in response.data:
                embedding_np = np.array(item.embedding)
                normed_embedding = embedding_np / np.linalg.norm(embedding_np)
                all_embeddings.append(normed_embedding)
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Add zero vectors for failed batch
            dim = 1536 if 'ada' in model_name else 1536
            all_embeddings.extend([np.zeros(dim) for _ in batch])
    
    return np.array(all_embeddings)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    # Normalize embeddings (should already be normalized, but just in case)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)
    return similarity


def find_clusters(keywords: List[Dict], similarity_matrix: np.ndarray, threshold: float = 0.75) -> List[List[int]]:
    """Find clusters of similar keywords using similarity threshold."""
    n = len(keywords)
    visited = set()
    clusters = []
    
    for i in range(n):
        if i in visited:
            continue
        
        # Find all keywords similar to this one
        cluster = [i]
        visited.add(i)
        
        for j in range(i + 1, n):
            if j not in visited and similarity_matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        
        if len(cluster) > 1:  # Only keep clusters with multiple items
            clusters.append(cluster)
    
    return clusters


def generate_cluster_labels(clusters: List[List[int]], keywords: List[Dict], provider: str = 'gemini') -> List[str]:
    """Generate semantic labels for clusters using LLM."""
    try:
        # Import the existing LLM client
        from pageindex.llm_client import get_llm_client
    except ImportError as e:
        print(f"Warning: Could not import LLM client ({e}). Using default labels.")
        return [f"Cluster {i+1}" for i in range(len(clusters))]
    
    labels = []
    client = get_llm_client(provider=provider)
    
    print(f"Generating semantic labels for {len(clusters)} clusters...")
    
    for cluster_idx, cluster in enumerate(tqdm(clusters, desc="Labeling clusters")):
        cluster_keywords = [keywords[i] for i in cluster]
        
        # Prepare prompt with keyword terms and contexts
        keyword_list = "\n".join([
            f"- {kw['term']}: {kw['context']}"
            for kw in cluster_keywords  # Limit to first 10 for prompt size
        ])
        
        prompt = f"""You are analyzing a cluster of semantically related keywords from a research document.

Keywords in this cluster:
{keyword_list}

Task: Generate a concise, descriptive label (2-5 words) that captures the main concept or theme connecting these keywords.

Requirements:
- Be specific and technical when appropriate
- Use domain terminology if relevant
- Keep it short (2-5 words maximum)
- Make it descriptive and meaningful

Return ONLY the label, nothing else."""

        try:
            # Use the client's chat_completion method
            model = client.default_model or 'gemini-2.5-flash-lite'
            response = client.chat_completion(
                model=model,
                prompt=prompt,
                temperature=0.3
            )
            
            label = response.strip()
            # Clean up the label
            label = label.strip('"').strip("'").strip()
            labels.append(label)
            
        except Exception as e:
            print(f"Error generating label for cluster {cluster_idx + 1}: {e}")
            labels.append(f"Cluster {cluster_idx + 1}")
    
    return labels


def generate_html_visualization(keywords: List[Dict], embeddings: np.ndarray, 
                                similarity_matrix: np.ndarray, output_path: str,
                                doc_name: str = "Document", provider: str = 'gemini',
                                generate_labels: bool = True):
    """Generate interactive HTML visualization of keyword clusters."""
    
    # Find clusters at different thresholds
    clusters_high = find_clusters(keywords, similarity_matrix, threshold=0.85)
    clusters_medium = find_clusters(keywords, similarity_matrix, threshold=0.75)
    clusters_low = find_clusters(keywords, similarity_matrix, threshold=0.65)
    
    # Generate semantic labels for clusters
    if generate_labels:
        labels_high = generate_cluster_labels(clusters_high, keywords, provider)
        labels_medium = generate_cluster_labels(clusters_medium, keywords, provider)
        labels_low = generate_cluster_labels(clusters_low, keywords, provider)
    else:
        labels_high = [f"Cluster {i+1}" for i in range(len(clusters_high))]
        labels_medium = [f"Cluster {i+1}" for i in range(len(clusters_medium))]
        labels_low = [f"Cluster {i+1}" for i in range(len(clusters_low))]
    
    # Prepare data for JavaScript
    keywords_data = []
    for i, kw in enumerate(keywords):
        keywords_data.append({
            'id': i,
            'term': kw['term'],
            'context': kw['context'],
            'parent': kw['parent'],
            'pages': kw['pages'],
            'node_id': kw['node_id']
        })
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Keyword Clusters - {doc_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        
        .controls {{
            background: white;
            padding: 1rem 2rem;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .control-group label {{
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        .control-group select, .control-group input {{
            padding: 0.4rem 0.8rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        .stats {{
            margin-left: auto;
            display: flex;
            gap: 1rem;
        }}
        
        .stat-item {{
            background: #e8eaf6;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}
        
        .container {{
            display: flex;
            height: calc(100vh - 180px);
        }}
        
        .sidebar {{
            width: 350px;
            background: white;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            padding: 1rem;
        }}
        
        .content {{
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
        }}
        
        .cluster-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #8bc34a;
        }}
        
        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .cluster-badge {{
            background: #8bc34a;
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9rem;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .cluster-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
        }}
        
        .keyword-item {{
            background: #f9f9f9;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 3px solid #8bc34a;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .keyword-item:hover {{
            background: #f1f8e9;
            transform: translateX(2px);
        }}
        
        .keyword-term {{
            font-weight: 600;
            color: #2e7d32;
            margin-bottom: 0.3rem;
        }}
        
        .keyword-context {{
            font-size: 0.85rem;
            color: #666;
            line-height: 1.4;
            margin-bottom: 0.3rem;
        }}
        
        .keyword-meta {{
            font-size: 0.75rem;
            color: #999;
        }}
        
        .similarity-matrix {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .matrix-cell {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin: 1px;
            cursor: pointer;
        }}
        
        .search-box {{
            width: 100%;
            padding: 0.6rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #8bc34a;
        }}
        
        .keyword-list-item {{
            padding: 0.5rem;
            margin: 0.3rem 0;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        
        .keyword-list-item:hover {{
            background: #f0f0f0;
        }}
        
        .keyword-list-item.selected {{
            background: #8bc34a;
            color: white;
            font-weight: 500;
        }}
        
        .detail-panel {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .similar-keywords {{
            margin-top: 1rem;
        }}
        
        .similar-item {{
            background: #f9f9f9;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 3px solid #ffc107;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .similar-item:hover {{
            background: #fff9e6;
            transform: translateX(2px);
        }}
        
        .similarity-score {{
            display: inline-block;
            background: #ffc107;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: #999;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Keyword Semantic Clusters</h1>
        <div class="subtitle">{doc_name}</div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Similarity Threshold:</label>
            <select id="threshold-select" onchange="updateClusters()">
                <option value="0.85">High (0.85+)</option>
                <option value="0.75" selected>Medium (0.75+)</option>
                <option value="0.65">Low (0.65+)</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>View Mode:</label>
            <select id="view-mode" onchange="updateView()">
                <option value="clusters" selected>Clusters</option>
                <option value="explore">Explore Keywords</option>
            </select>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <strong>Total Keywords:</strong> <span id="total-keywords">{len(keywords)}</span>
            </div>
            <div class="stat-item">
                <strong>Clusters:</strong> <span id="cluster-count">0</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar" id="sidebar">
            <input type="text" class="search-box" id="search-box" placeholder="Search keywords...">
            <div id="keyword-list"></div>
        </div>
        
        <div class="content" id="content">
            <div class="empty-state">
                Select a view mode to begin exploring keyword clusters
            </div>
        </div>
    </div>
    
    <script>
        const keywords = {json.dumps(keywords_data, ensure_ascii=False)};
        const similarityMatrix = {similarity_matrix.tolist()};
        
        const clusters = {{
            high: {json.dumps(clusters_high)},
            medium: {json.dumps(clusters_medium)},
            low: {json.dumps(clusters_low)}
        }};
        
        const clusterLabels = {{
            high: {json.dumps(labels_high, ensure_ascii=False)},
            medium: {json.dumps(labels_medium, ensure_ascii=False)},
            low: {json.dumps(labels_low, ensure_ascii=False)}
        }};
        
        let selectedKeyword = null;
        
        function updateClusters() {{
            const threshold = document.getElementById('threshold-select').value;
            const viewMode = document.getElementById('view-mode').value;
            
            if (viewMode === 'clusters') {{
                showClusters(threshold);
            }}
        }}
        
        function updateView() {{
            const viewMode = document.getElementById('view-mode').value;
            
            if (viewMode === 'clusters') {{
                const threshold = document.getElementById('threshold-select').value;
                showClusters(threshold);
            }} else {{
                showExploreMode();
            }}
        }}
        
        function showClusters(threshold) {{
            const thresholdKey = threshold === '0.85' ? 'high' : threshold === '0.75' ? 'medium' : 'low';
            const clusterList = clusters[thresholdKey];
            const labels = clusterLabels[thresholdKey];
            
            document.getElementById('cluster-count').textContent = clusterList.length;
            
            const content = document.getElementById('content');
            
            if (clusterList.length === 0) {{
                content.innerHTML = '<div class="empty-state">No clusters found at this threshold. Try lowering the threshold.</div>';
                return;
            }}
            
            let html = '';
            
            clusterList.forEach((cluster, idx) => {{
                const clusterKeywords = cluster.map(i => keywords[i]);
                const avgSimilarity = computeAvgSimilarity(cluster);
                const label = labels[idx] || `Cluster ${{idx + 1}}`;
                
                html += `
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <span class="cluster-badge">${{escapeHtml(label)}}</span>
                            <span class="cluster-title">${{clusterKeywords.length}} keywords</span>
                            <span style="margin-left: auto; font-size: 0.85rem; color: #666;">
                                Avg similarity: ${{avgSimilarity.toFixed(3)}}
                            </span>
                        </div>
                        
                        ${{clusterKeywords.map(kw => `
                            <div class="keyword-item" onclick="showKeywordDetail(${{kw.id}})">
                                <div class="keyword-term">${{escapeHtml(kw.term)}}</div>
                                <div class="keyword-context">${{escapeHtml(kw.context)}}</div>
                                <div class="keyword-meta">
                                    Pages ${{kw.pages}} • ${{escapeHtml(kw.parent)}}
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                `;
            }});
            
            content.innerHTML = html;
        }}
        
        function showExploreMode() {{
            // Populate keyword list
            renderKeywordList();
            
            const content = document.getElementById('content');
            content.innerHTML = '<div class="empty-state">👈 Select a keyword from the list to explore similar keywords</div>';
        }}
        
        function renderKeywordList(filter = '') {{
            const keywordList = document.getElementById('keyword-list');
            
            const filtered = filter ? 
                keywords.filter(kw => kw.term.toLowerCase().includes(filter.toLowerCase())) : 
                keywords;
            
            keywordList.innerHTML = filtered.map(kw => `
                <div class="keyword-list-item" onclick="showKeywordDetail(${{kw.id}})" data-id="${{kw.id}}">
                    ${{escapeHtml(kw.term)}}
                </div>
            `).join('');
        }}
        
        function showKeywordDetail(keywordId) {{
            selectedKeyword = keywordId;
            const keyword = keywords[keywordId];
            
            // Update selection in list
            document.querySelectorAll('.keyword-list-item').forEach(el => {{
                el.classList.remove('selected');
            }});
            document.querySelector(`[data-id="${{keywordId}}"]`)?.classList.add('selected');
            
            // Find similar keywords
            const similarities = similarityMatrix[keywordId]
                .map((sim, idx) => ({{ idx, sim }}))
                .filter(item => item.idx !== keywordId)
                .sort((a, b) => b.sim - a.sim)
                .slice(0, 10);
            
            const content = document.getElementById('content');
            content.innerHTML = `
                <div class="detail-panel">
                    <h2 style="color: #2e7d32; margin-bottom: 1rem;">🔑 ${{escapeHtml(keyword.term)}}</h2>
                    
                    <div style="background: #f9f9f9; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Context:</strong> ${{escapeHtml(keyword.context)}}
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Parent Section:</strong> ${{escapeHtml(keyword.parent)}}
                        </div>
                        <div>
                            <strong>Pages:</strong> ${{keyword.pages}}
                        </div>
                    </div>
                    
                    <div class="similar-keywords">
                        <h3 style="margin-bottom: 0.5rem;">Most Similar Keywords</h3>
                        ${{similarities.map(item => {{
                            const simKw = keywords[item.idx];
                            return `
                                <div class="similar-item" onclick="showKeywordDetail(${{item.idx}})">
                                    <div>
                                        <strong>${{escapeHtml(simKw.term)}}</strong>
                                        <span class="similarity-score">${{(item.sim * 100).toFixed(1)}}%</span>
                                    </div>
                                    <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">
                                        ${{escapeHtml(simKw.context)}}
                                    </div>
                                    <div style="font-size: 0.75rem; color: #999; margin-top: 0.3rem;">
                                        Pages ${{simKw.pages}} • ${{escapeHtml(simKw.parent)}}
                                    </div>
                                </div>
                            `;
                        }}).join('')}}
                    </div>
                </div>
            `;
        }}
        
        function computeAvgSimilarity(cluster) {{
            let sum = 0;
            let count = 0;
            
            for (let i = 0; i < cluster.length; i++) {{
                for (let j = i + 1; j < cluster.length; j++) {{
                    sum += similarityMatrix[cluster[i]][cluster[j]];
                    count++;
                }}
            }}
            
            return count > 0 ? sum / count : 0;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            showClusters('0.75');
            
            document.getElementById('search-box').addEventListener('input', (e) => {{
                renderKeywordList(e.target.value);
            }});
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize keyword semantic clusters using LLM embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('structure_json', help='Path to structure JSON file with keywords')
    parser.add_argument('--provider', choices=['gemini', 'openai'], default='gemini',
                       help='Embedding provider (default: gemini)')
    parser.add_argument('--model', help='Embedding model name (optional)')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    parser.add_argument('--min-similarity', type=float, default=0.75,
                       help='Minimum similarity threshold for clustering (default: 0.75)')
    parser.add_argument('--cache', help='Cache embeddings to file for reuse')
    parser.add_argument('--no-labels', action='store_true',
                       help='Skip LLM-generated cluster labels (faster)')
    
    args = parser.parse_args()
    
    # Load structure
    print(f"Loading structure from {args.structure_json}...")
    with open(args.structure_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        doc_name = data.get('doc_name', 'Document')
        structure = data.get('structure', [])
    else:
        doc_name = 'Document'
        structure = data
    
    # Extract keywords
    print("Extracting keywords...")
    keywords = extract_keywords_from_structure(structure)
    
    if not keywords:
        print("Error: No keywords found in structure. Make sure to process with --granularity keywords")
        return 1
    
    print(f"Found {len(keywords)} keywords")
    
    # Check for cached embeddings
    cache_path = args.cache or f"{Path(args.structure_json).stem}_embeddings.npy"
    
    if Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
    else:
        # Prepare texts for embedding (term + context for better semantic representation)
        texts = [f"{kw['term']}: {kw['context']}" for kw in keywords]
        
        # Generate embeddings
        print(f"Generating embeddings using {args.provider}...")
        if args.provider == 'gemini':
            model = args.model or "models/text-embedding-004"
            embeddings = get_gemini_embeddings(texts, model)
        else:
            model = args.model or "text-embedding-3-small"
            embeddings = get_openai_embeddings(texts, model)
        
        # Cache embeddings
        print(f"Caching embeddings to {cache_path}...")
        np.save(cache_path, embeddings)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.structure_json)
        output_path = input_path.parent / f"{input_path.stem}_clusters.html"
    
    # Generate visualization
    print("Generating HTML visualization...")
    generate_html_visualization(
        keywords, embeddings, similarity_matrix, output_path, doc_name,
        provider=args.provider, generate_labels=not args.no_labels
    )
    
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Open in browser: file://{Path(output_path).absolute()}")
    
    # Print some statistics
    clusters = find_clusters(keywords, similarity_matrix, args.min_similarity)
    print(f"\nStatistics:")
    print(f"  Total keywords: {len(keywords)}")
    print(f"  Clusters found (threshold {args.min_similarity}): {len(clusters)}")
    if clusters:
        cluster_sizes = [len(c) for c in clusters]
        print(f"  Largest cluster: {max(cluster_sizes)} keywords")
        print(f"  Average cluster size: {np.mean(cluster_sizes):.1f} keywords")
    
    return 0


if __name__ == '__main__':
    exit(main())
