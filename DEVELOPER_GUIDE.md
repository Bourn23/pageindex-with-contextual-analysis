# PageIndex Developer Guide

## Architecture Overview

PageIndex transforms PDF documents into hierarchical tree structures with multiple granularity levels. This guide explains how the components interact and how to debug issues.

---

## Core Processing Pipeline

### 1. Entry Point: `page_index_main()` (page_index.py)

```
PDF Input → Parse Pages → Extract TOC → Build Tree → Apply Granular Features → Add Text → Generate Summaries → Output JSON
```

**Key Steps:**
1. **Parse PDF** (`get_page_tokens()`) - Extract text from each page
2. **TOC Detection** (`find_toc_pages()`) - Find table of contents
3. **Tree Building** (`tree_parser()`) - Create initial section hierarchy
4. **Granular Features** (if enabled) - Add semantic units, figures, tables, keywords
5. **Text Addition** (`add_node_text()`) - Populate text fields for nodes
6. **Summary Generation** (optional) - Generate summaries using LLM

---

## Granularity Levels

### Coarse (max_depth = 0)
- **What**: Top-level sections only
- **Source**: Table of contents or LLM-generated structure
- **Text**: Full page text for section's page range

### Medium (max_depth = 1)
- **What**: Sections → Semantic Units
- **Process**: `SemanticAnalyzer.analyze_section()` identifies topic shifts
- **Text**: Paragraph-level text extracted from parent section

### Fine (max_depth = 2)
- **What**: Sections → Semantic Units → Fine Semantic Units
- **Process**: Recursive semantic subdivision
- **Text**: Sub-paragraph level text

### Keywords (max_depth = 3)
- **What**: Sections → Semantic Units → Fine Semantic Units → Keywords
- **Process**: `SemanticAnalyzer.extract_keywords()` on leaf nodes
- **Text**: Immediate parent's text (NOT grandparent's)

---

## Component Interactions

### Main Flow Diagram

```
page_index.py
    ├─> tree_parser()
    │   ├─> TOC extraction
    │   ├─> Structure building
    │   └─> Granular features (if enabled)
    │       └─> integration.py
    │           ├─> apply_semantic_subdivision()
    │           │   └─> semantic_analyzer.py
    │           │       ├─> analyze_section()
    │           │       ├─> create_nodes_from_semantic_units()
    │           │       ├─> extract_keywords() [keywords level]
    │           │       └─> create_keyword_nodes()
    │           └─> detect_and_integrate_figures_tables()
    │               ├─> figure_detector.py
    │               └─> table_detector.py
    ├─> add_node_text() [CRITICAL: Can overwrite text!]
    └─> generate_summaries_for_structure()
```

---

## Critical Code Paths

### 1. Semantic Unit Creation

**File**: `pageindex/granular/semantic_analyzer.py`

```python
def create_nodes_from_semantic_units(semantic_units, section_node, page_texts):
    # Extracts paragraph-specific text from parent section
    full_text = section_node.get('text', '')
    paragraphs = self._split_into_paragraphs(full_text)
    
    for unit in semantic_units:
        # Extract ONLY the paragraphs for this semantic unit
        unit_paragraphs = paragraphs[unit.start_paragraph:unit.end_paragraph + 1]
        unit_text = '\n\n'.join(unit_paragraphs)  # ← Paragraph-level text
        
        node = {
            'text': unit_text,  # ← This is the correct, specific text
            'node_type': 'semantic_unit',
            ...
        }
```

**Key Point**: Semantic units get paragraph-specific text, NOT full page text.

---

### 2. Keyword Node Creation

**File**: `pageindex/granular/semantic_analyzer.py`

```python
def create_keyword_nodes(keywords, section_node):
    # section_node is the IMMEDIATE PARENT (fine semantic unit)
    parent_text = section_node.get('text', '')  # ← Parent's paragraph text
    
    for kw in keywords:
        node = {
            'title': kw['term'],
            'text': parent_text,  # ← Use parent's text directly
            'node_type': 'keyword',
            '_text_locked': True,  # ← Prevent overwriting
            ...
        }
```

**Key Point**: Keywords inherit their immediate parent's text, not the grandparent's.

---

### 3. Text Addition (THE GOTCHA!)

**File**: `pageindex/utils.py`

```python
def add_node_text(node, pdf_pages):
    # This function runs AFTER tree building
    # It extracts text based on page ranges
    
    node_type = node.get('node_type')
    if node_type in ['semantic_unit', 'figure', 'table', 'keyword']:
        # ← MUST skip these! They already have specialized text
        return
    
    # For other nodes, extract text from pages
    start_page = node.get('start_index')
    end_page = node.get('end_index')
    node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
```

**⚠️ CRITICAL**: This function can overwrite carefully crafted text if node types aren't skipped!

**Why This Matters**:
- Semantic units have paragraph-specific text
- Keywords have parent-specific text
- If `add_node_text()` processes them, it replaces their text with FULL PAGE TEXT
- This was the bug that caused keywords to have grandparent's text

---

## Debugging Guide

### Problem: Keywords have wrong text

**Symptoms**: Keywords show grandparent's full section text instead of parent's text

**Debug Steps**:

1. **Check keyword creation**:
```python
# In semantic_analyzer.py, add logging:
parent_text = section_node.get('text', '')
print(f"Creating keyword with parent text length: {len(parent_text)}")
```

2. **Check if text is being overwritten**:
```python
# In utils.py, add logging to add_node_text():
if node_type == 'keyword':
    print(f"Processing keyword node: {node.get('title')}")
    print(f"Current text length: {len(node.get('text', ''))}")
```

3. **Verify node_type is set**:
```python
# Keywords MUST have node_type='keyword'
# Check in create_keyword_nodes()
```

4. **Check skip list in add_node_text()**:
```python
# Ensure 'keyword' is in the skip list:
if node_type in ['semantic_unit', 'figure', 'table', 'keyword']:
```

---

### Problem: Semantic units have wrong text

**Symptoms**: Semantic units show full page text instead of paragraph-specific text

**Debug Steps**:

1. **Check paragraph extraction**:
```python
# In create_nodes_from_semantic_units():
print(f"Parent text length: {len(full_text)}")
print(f"Extracted paragraphs: {len(paragraphs)}")
print(f"Unit text length: {len(unit_text)}")
```

2. **Check if text is preserved**:
```python
# In integration.py, process_node():
if node.get('node_type') == 'semantic_unit' and node.get('text'):
    print(f"Semantic unit already has text: {len(node['text'])} chars")
```

3. **Verify skip in add_node_text()**:
```python
# Ensure semantic_unit is skipped
```

---

### Problem: Granular features not running

**Symptoms**: No semantic units, figures, tables, or keywords in output

**Debug Steps**:

1. **Check granularity setting**:
```python
# In page_index.py:
if opt.granularity in ['medium', 'fine', 'keywords']:  # ← Must include your level
```

2. **Check feature flags**:
```python
# In config or opt:
enable_semantic_subdivision: true
enable_figure_detection: true
enable_table_detection: true
```

3. **Check logs**:
```bash
# Look for these messages:
grep "Applying granular features" logs/*.json
grep "Semantic subdivision complete" logs/*.json
```

---

## Common Pitfalls

### 1. Text Overwriting
**Problem**: Carefully extracted text gets replaced with page-level text

**Solution**: Always add new node types to skip list in `add_node_text()`

**Example**:
```python
# When adding a new node type like 'citation':
if node_type in ['semantic_unit', 'figure', 'table', 'keyword', 'citation']:
    return  # Skip text extraction
```

---

### 2. Missing node_type
**Problem**: Nodes don't get special treatment because node_type isn't set

**Solution**: Always set node_type when creating nodes

**Example**:
```python
node = {
    'title': 'My Node',
    'node_type': 'my_custom_type',  # ← REQUIRED
    ...
}
```

---

### 3. Page Range Inheritance
**Problem**: Child nodes inherit parent's page range, causing wrong text extraction

**Solution**: 
- Set specific page ranges for child nodes
- OR skip child node type in `add_node_text()`
- OR use `_text_locked` flag

---

### 4. Async Processing Order
**Problem**: Text gets overwritten because async operations complete out of order

**Solution**: Understand the order:
1. Tree building (sync)
2. Granular features (async) - creates nodes with specific text
3. Text addition (sync) - CAN OVERWRITE if not careful
4. Summary generation (async)

---

## File Organization

```
pageindex/
├── page_index.py           # Main entry point, orchestrates everything
├── utils.py                # Utilities including add_node_text() ⚠️
├── llm_client.py           # LLM API wrapper
├── config.yaml             # Default configuration
└── granular/               # Granular features
    ├── integration.py      # Orchestrates granular processing
    ├── semantic_analyzer.py # Semantic subdivision & keywords
    ├── figure_detector.py  # Figure detection
    └── table_detector.py   # Table detection
```

---

## Testing Strategy

### Unit Testing
```python
# Test semantic unit creation
def test_semantic_unit_text():
    analyzer = SemanticAnalyzer(llm_client)
    section_node = {'text': 'Para 1\n\nPara 2\n\nPara 3', ...}
    units = analyzer.analyze_section(section_node, ...)
    nodes = analyzer.create_nodes_from_semantic_units(units, section_node, ...)
    
    # Verify text is paragraph-specific
    assert len(nodes[0]['text']) < len(section_node['text'])
```

### Integration Testing
```python
# Test full pipeline
result = page_index_main('test.pdf', opt)
tree = result['structure']

# Verify keywords have parent's text
keyword = find_keyword_node(tree)
parent = find_parent_of_keyword(tree, keyword)
assert keyword['text'] == parent['text']
```

### Visual Testing
```bash
# Generate visualization
python visualize_structure.py results/output.json

# Open in browser and verify:
# - Keywords show parent's text
# - Semantic units show paragraph text
# - Hierarchy is correct
```

---

## Performance Considerations

### API Call Optimization
- **Semantic analysis**: 1 call per section
- **Keyword extraction**: 1 call per leaf semantic unit
- **Figure detection**: 1 call per page (batched)
- **Table detection**: 1 call per page (batched)

**For a 10-page paper with 5 sections**:
- Coarse: ~5 calls (TOC extraction)
- Medium: ~10 calls (5 sections × 2 levels)
- Fine: ~20 calls (recursive subdivision)
- Keywords: ~40 calls (20 leaf nodes × 2)

### Caching Strategy
- Semantic analyzer caches detection results
- Figure/table detectors cache per-page results
- LLM responses are NOT cached (consider adding)

---

## Adding New Granularity Levels

### Example: Adding "Sentences" Level

1. **Update config.yaml**:
```yaml
granularity: "sentences"  # Add to valid options
```

2. **Update integration.py**:
```python
if granularity == 'sentences':
    max_depth = 4  # One more than keywords
```

3. **Add extraction logic**:
```python
# In semantic_analyzer.py
def extract_sentences(self, section_node):
    text = section_node.get('text', '')
    sentences = text.split('. ')
    return [{'text': s, 'node_type': 'sentence'} for s in sentences]
```

4. **Update utils.py**:
```python
# Add to skip list
if node_type in ['semantic_unit', 'figure', 'table', 'keyword', 'sentence']:
```

5. **Test thoroughly**!

---

## Troubleshooting Checklist

- [ ] Is `node_type` set correctly?
- [ ] Is the node type in `add_node_text()` skip list?
- [ ] Is granularity level in the check in `page_index.py`?
- [ ] Are feature flags enabled in config?
- [ ] Is `GEMINI_API_KEY` set in `.env`?
- [ ] Are logs showing the expected messages?
- [ ] Is text being extracted at the right time?
- [ ] Are async operations completing in order?

---

## Key Takeaways

1. **Text is sacred**: Once you extract specific text (paragraphs, captions), protect it from being overwritten
2. **node_type is your friend**: Use it to identify and skip nodes in post-processing
3. **Order matters**: Tree building → Granular features → Text addition → Summaries
4. **Debug with logging**: Add print statements at key points to trace text flow
5. **Visualize**: Use the HTML visualizer to verify structure and text content

---

## Getting Help

- Check `logs/` directory for detailed execution logs
- Use `visualize_structure.py` to inspect the tree
- Add debug logging at key points
- Test with small PDFs first
- Compare output across granularity levels

---

**Last Updated**: November 2025
**Maintainer**: PageIndex Development Team
