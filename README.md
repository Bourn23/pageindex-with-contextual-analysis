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
  <a href="#-key-research-extensions">✨ New Features</a>&nbsp; • &nbsp;
  <a href="#-quick-start">🚀 Quick Start</a>&nbsp; • &nbsp;
  <a href="https://arxiv.org/your-paper-link">📄 Related Paper</a>
</p>
  
</div>

---

## 🔬 Why this variant?

This repository extends the original [PageIndex](https://vectify.ai/pageindex) framework to support granular control over document topology. While the original library focuses on high-level document structuring, this variant introduces:

1.  **Variable Tree Depth:** Control node refinement levels (Coarse $\to$ Medium $\to$ Fine).
    * *Sections $\to$ Paragraphs $\to$ Conceptual Sentences $\to$ Keywords*
2.  **Enhanced Detection:** Dedicated node types for **Tables** and **Figures**, ensuring distinct processing for non-textual elements.
3.  **Visual Debugger:** An HTML-based tree visualizer to inspect the generated document structure interactively.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip3 install --upgrade -r requirements.txt
````

### 2\. Set API Key

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_key_here
```

### 3\. Run with Research Parameters

We have added new flags to control the tree depth and visualizer:

```bash
# Example: Run with fine-grained depth and generate visualization
python3 run_pageindex.py \
  --pdf_path docs/whitepaper.pdf \
  --tree_depth fine \
  --visualize_tree yes
```

-----

## 🌲 The Core Concept (PageIndex)

> *Note: The following core logic is inherited from the original PageIndex framework.*

Traditional vector-based RAG relies on semantic *similarity*, but professional documents demand *relevance*. PageIndex builds a hierarchical tree index to simulate how human experts navigate complex documents.

[Image of decision tree structure]

**Original Features:**

  - **No Vector DB:** Retrieval via tree search reasoning.
  - **Traceability:** Every retrieval step is explainable.

-----

## 📦 Package Usage & Configuration

\<details\>
\<summary\>\<strong\>Standard Parameters\</strong\>\</summary\>

```
--model                 OpenAI model (default: gpt-4o)
--toc-check-pages       Pages to check for ToC (default: 20)
```

\</details\>

\<details\>
\<summary\>\<strong\>✨ New Parameters (This Fork)\</strong\>\</summary\>

```
--refinement_level      Control the depth (coarse, medium, fine)
--extract_figures       Boolean to toggle figure extraction nodes
--generate_html_view    Generate the debug visualization
```

\</details\>

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

## License

This project is licensed under the **MIT License**.

  * Copyright (c) 2025 **[Your Name / GitHub Username]** (Modifications)
  * Copyright (c) 2025 **Vectify AI** (Original Work)

See the [LICENSE](LICENSE) file for details.