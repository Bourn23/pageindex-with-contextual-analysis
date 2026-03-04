<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
  <img src="https://github.com/user-attachments/assets/46201e72-675b-43bc-bfbd-081cc6b65a1d" alt="PageIndex Banner" />
</a>

<br/>

# PageIndex: Contextual Analysis Variant
### Vectorless and Traceable RAG with Table & Figure Detection

<p align="center">
  <em>A research fork of <a href="https://github.com/VectifyAI/PageIndex">PageIndex</a> focusing on granular tree control and visual debugging.</em>
</p>

<p align="center">
  <a href="#-quick-start">🚀 Quick Start</a>&nbsp; • &nbsp;
</p>
  
</div>

---

## 🔬 Why this variant?

This repository extends the original [PageIndex](https://vectify.ai/pageindex) framework to support granular control over document topology. While the original library focuses on high-level document structuring, this variant introduces:

1.  **Variable Tree Depth:** Control node refinement levels (Coarse $\to$ Medium $\to$ Fine $\to$ Keywords).
    * Sections $\to$ Semantic Units $\to$ Fine Semantic Units $\to$ Keywords
    * Keywords are extracted from the deepest (leaf) semantic nodes for maximum specificity
2.  **Enhanced Detection:** Dedicated node types for **Tables** and **Figures**, ensuring distinct processing for non-textual elements.
3.  **Visual Debugger:** An HTML-based tree visualizer to inspect the generated document structure interactively.

---

## 🚀 Quick Start

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## Usage (Choose One)

### Option 1: Command Line (Recommended)

```bash
# Process with keywords (most detailed)
python run_pageindex.py paper.pdf --granularity keywords

# Process with fine granularity
python run_pageindex.py paper.pdf --granularity fine

# Fast processing (coarse)
python run_pageindex.py paper.pdf --granularity coarse

# With visualization
python run_pageindex.py paper.pdf --granularity keywords --visualize
```

### Option 2: Python API

```python
from pageindex import page_index_main
from pageindex.utils import ConfigLoader

config_loader = ConfigLoader()
opt = config_loader.load({
    'granularity': 'keywords',
    'if_add_node_text': 'yes',
})

result = page_index_main('paper.pdf', opt)
structure = result['structure']
```

### Option 3: Example Script

```bash
python example_keywords_usage.py paper.pdf
```

---

## Granularity Levels

| Level | Speed | Detail | Use Case |
|-------|-------|--------|----------|
| `coarse` | ⚡⚡⚡ | ⭐ | Quick overview |
| `medium` | ⚡⚡ | ⭐⭐ | Balanced |
| `fine` | ⚡ | ⭐⭐⭐ | Detailed analysis |
| `keywords` | 🐌 | ⭐⭐⭐⭐ | Maximum detail + terminology |

---

## Output

Results are saved to `results/` directory:
- `paper_keywords_structure.json` - Full structure
- `paper_keywords_structure.html` - Visualization (if `--visualize` used)

---

## Common Commands

```bash
# Keywords with all features
python run_pageindex.py paper.pdf --granularity keywords --figures --tables --visualize

# Fast processing (no figures/tables)
python run_pageindex.py paper.pdf --granularity medium --no-figures --no-tables

# Custom output location
python run_pageindex.py paper.pdf -g keywords -o my_output.json

# Help
python run_pageindex.py --help
```


Updated workflow:
```bash
New Automation Scripts
1. Batch Extraction
Script: 
batch_process_v8.sh

Purpose: Runs 
basic_extraction_md_v8.py
 on every paper in a parent folder.
Usage: ./batch_process_v8.sh "./path/to/parent_folder"
Result: Generates robust_results_v8.json inside each paper's subdirectory.
2. Batch Provenance Tracing
Script: 
batch_process_provenance.sh

Purpose: Runs 
t0_provenance_tracer.py
 on every paper that has an extraction JSON.
Usage: ./batch_process_provenance.sh "./path/to/parent_folder"
Result: Generates *_provenance.json inside each paper's subdirectory.
3. Batch Visualization
Script: 
batch_visualize_provenance.sh

Purpose: Generates interactive HTML dashboards for every paper that has a provenance JSON.
Usage: ./batch_visualize_provenance.sh "./path/to/parent_folder"
Result: Generates provenance_dashboard.html inside each paper's subdirectory.
```

For debugging please see [DEV GUIDE](DEVELOPER_GUIDE.md)

-----

## 🌲 The Core Concept (PageIndex)

> *Note: The following core logic is inherited from the original PageIndex framework.*

Traditional vector-based RAG relies on semantic *similarity*, but professional documents demand *relevance*. PageIndex builds a hierarchical tree index to simulate how human experts navigate complex documents.

[Image of decision tree structure]

**Original Features:**

  - **No Vector DB:** Retrieval via tree search reasoning.
  - **Traceability:** Every retrieval step is explainable.

-----

## 📝 Enhanced Markdown Processing

*Useful for pipelines involving `marker` or `docling`.*

```python
from pageindex import markdown_page_index

# Process markdown with enhanced table/figure detection
structure = markdown_page_index(
    markdown_path="document.md",
    opt={'extract_tables': True, 'extract_figures': True}
)
```

-----

There are several work in progress:

1. showing the keywords and nodes with similar keywords is now handled by the basic visualization script.

1.2. you can classify the keywords into semantic groups using:
python visualize_keyword_clusters.py "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_keywords_structure.json" --provider gemini 2>&1 | tail -30

2. you can now extract materials, their name, processes, and conductivity

python run_extraction.py "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_keywords_structure json" 2>&1 

2.1. you can then visualize it using the following code:
 python visualize_materials.py "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_keywords_structure_materials.json"

2.2. you can also export the data in CSV format:
python materials_to_csv.py "results/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes_keywords_structure_materials.json" --format detailed


-----

## License

This project is licensed under the **MIT License**.

  * Copyright (c) 2025 **Bourn23** (Modifications)
  * Copyright (c) 2025 **Vectify AI** (Original Work)

See the [LICENSE](LICENSE) file for details.