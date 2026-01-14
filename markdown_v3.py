"""
This version introduces batch processing
"""
import os
import asyncio
import json
import re
import uuid
from typing import List, Optional, Dict, Any, Tuple, Union
from itertools import groupby
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor
import random

# --- GEMINI/Pydantic Setup (Requires 'google-genai' and 'pydantic' installed) ---
from google import genai
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()
# Initialize Gemini Client (Ensure API Key is configured in environment)
client = genai.Client() # Uncomment in a real environment
# ----------------------------------------------------------------------
# Try importing Spacy, fallback to Regex if missing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ Spacy not found. Using Regex fallback for sentence splitting.")
if SPACY_AVAILABLE:
    from spacy.symbols import ORTH
    nlp.tokenizer.add_special_case("Fig.", [{ORTH: "Fig."}])
    nlp.tokenizer.add_special_case("Figs.", [{ORTH: "Figs."}])
    nlp.tokenizer.add_special_case("Eq.", [{ORTH: "Eq."}])
    nlp.tokenizer.add_special_case("Eqs.", [{ORTH: "Eqs."}])
    nlp.tokenizer.add_special_case("Tab.", [{ORTH: "Tab."}])
    nlp.tokenizer.add_special_case("Tabs.", [{ORTH: "Tabs."}])
    nlp.tokenizer.add_special_case("al.", [{ORTH: "al."}])
    nlp.tokenizer.add_special_case("Refs.", [{ORTH: "Refs."}])
    nlp.tokenizer.add_special_case("Ref.", [{ORTH: "Ref."}])
    nlp.tokenizer.add_special_case("vs.", [{ORTH: "vs."}])
    nlp.tokenizer.add_special_case("i.e.", [{ORTH: "i.e."}])
    nlp.tokenizer.add_special_case("e.g.", [{ORTH: "e.g."}])

# ==========================================
# 1. PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ==========================================

class KeywordMetadata(BaseModel):
    """Deep metadata for an extracted entity."""
    term: str = Field(description="The exact scientific term.")
    summary: str = Field(description="A brief definition.")
    relevance: str = Field(description="Why this is important in this context.")

class BatchKeywordExtraction(BaseModel):
    # Map sentence_index to a list of keywords
    results: Dict[int, List[KeywordMetadata]]
    
class KeywordNode(BaseModel):
    """Leaf Node: A specific extracted entity."""
    title: str
    start_index: int = 1
    end_index: int = 1
    text: str = Field(description="Parent sentence text.")
    summary: str
    node_type: str = "keyword"
    _text_locked: bool = True
    metadata: KeywordMetadata
    nodes: List[Any] = Field(default_factory=list)
    node_id: str

class SentenceNode(BaseModel):
    """Level 3 Node: A single atomic sentence."""
    title: str
    text: str
    line_num: int
    node_type: str = "sentence"
    nodes: List[KeywordNode]
    node_id: str

class ImageNode(BaseModel):
    """Leaf Node: An image/figure"""
    title: str = "Image"
    src: str = Field(description="File Path or URL of the image")
    text: str = Field(default="", description="Alt text provided in markdown")
    node_type: str = "image"
    node_id: str

class TableNode(BaseModel):
    """Leaf Node: A data table."""
    title: str = Field(description="The table caption")
    text: str = Field(description="The raw markdown representation of the table")
    node_type: str = "table"
    node_id: str

class SemanticGroupNode(BaseModel):
    """Level 2.5 Node: A paragraph or semantic cluster of sentences."""
    title: str
    text: str
    line_num: int
    node_type: str = "semantic_group"
    nodes: List[Union[SentenceNode, ImageNode, TableNode]]
    node_id: str

class SectionNode(BaseModel):
    """Level 2 Node: A document section."""
    title: str
    text: str
    line_num: int
    node_type: str = "section"
    nodes: List[SemanticGroupNode] # Contains Semantic Groups
    node_id: str

class DocumentRoot(BaseModel):
    """Level 1 Node: The Document."""
    title: str
    doc_name: str
    structure: List[SectionNode]

class DocumentStructure(BaseModel):
    doc_name: str
    structure: List[DocumentRoot]

# ==========================================
# 2. INTERMEDIATE OUTPUT MODELS (The LLM Interface)
# ==========================================

class SemanticGroupIndices(BaseModel):
    """LLM response for grouping: Returns IDs, not text."""
    group_title: str
    start_sentence_id: int
    end_sentence_id: int

class GroupingOutput(BaseModel):
    groups: List[SemanticGroupIndices]

class KeywordOutput(BaseModel):
    keywords: List[KeywordMetadata]

class SectionInfo(BaseModel):
    title: str = Field(description="The title of the section.")
    content: str = Field(description="The full raw text content of the section.")
    line_num: int = Field(default=1)
    id: str = Field(default="0")

class SectionSplitOutput(BaseModel):
    document_title: str
    sections: List[SectionInfo]

# ==========================================
# 3. UTILITIES & CLIENT
# ==========================================
class BlockRegistry:
    def __init__(self):
        self.blocks = {} # { "UUID_KEY": NodeObject }

    def register(self, key: str, node: Union[ImageNode, TableNode]):
        self.blocks[key] = node
        
    def get(self, key: str):
        return self.blocks.get(key)

def extract_special_blocks(text: str) -> Tuple[str, BlockRegistry]:
    """
    detects Images and Tables, creates their Nodes, adds them to a registry,
    and replaces their occurrences in the text with a placeholder __BLOCK_{ID}__.
    """
    registry = BlockRegistry()
    lines = text.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # --- 1. DETECT IMAGES ---
        # Regex: ![_page_0_Picture_5.jpeg] or ![](_page_0_Picture_5.jpeg)
        img_match = re.search(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            alt_text = img_match.group(1)
            src = img_match.group(2)
            
            block_id = f"__IMG_{get_uuid()}__"
            node = ImageNode(
                title=f"Figure: {src}",
                src=src,
                text=alt_text,
                node_id=get_uuid()
            )
            registry.register(block_id, node)
            processed_lines.append(f"\n{block_id}\n")
            i += 1
            continue

        # --- 2. DETECT TABLES ---
        # Heuristic: A line starting with '|' is likely a table row
        if stripped.startswith('|'):
            # Look backwards for a caption (up to 2 lines back, handling HTML spans)
            caption = "Untitled Table"
            caption_found = False
            
            # Check previous line for text
            if processed_lines:
                prev_line = processed_lines[-1].strip()
                # Remove HTML spans if present: <span id="..."></span>Table 1...
                clean_prev = re.sub(r'<[^>]+>', '', prev_line).strip()
                
                if clean_prev.lower().startswith('table'):
                    caption = clean_prev
                    processed_lines.pop() # Remove caption from text flow so it attaches to table
                    caption_found = True
            
            # Consume all table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            block_id = f"__TBL_{get_uuid()}__"
            node = TableNode(
                title=caption,
                text="\n".join(table_lines),
                node_id=get_uuid()
            )
            registry.register(block_id, node)
            processed_lines.append(f"\n{block_id}\n")
            # Don't increment i here, the inner while loop did it
            continue

        # Normal line
        processed_lines.append(line)
        i += 1

    return "\n".join(processed_lines), registry


def get_uuid(): return str(uuid.uuid4())[:8]

def split_text_to_sentences(text: str) -> List[str]:
    """
    Hybrid Splitter:
    1. Uses Regex to isolate __IMG__ and __TBL__ blocks.
    2. Uses Spacy (if available) to intelligently split the text between blocks.
    """
    
    # --- PASS 1: STRUCTURAL SPLIT (Isolate Blocks) ---
    # Split by the specific placeholder format. 
    # Capturing group () keeps the delimiter in the list.
    block_pattern = r'(__[A-Z]{3}_[a-f0-9-]{8}__)'
    structural_chunks = re.split(block_pattern, text)
    
    final_sentences = []

    # --- PASS 2: LINGUISTIC SPLIT (Process Text) ---
    for chunk in structural_chunks:
        clean_chunk = chunk.strip()
        if not clean_chunk: 
            continue

        # Check if this chunk is a Special Block ID
        if re.match(r'^__[A-Z]{3}_[a-f0-9-]{8}__$', clean_chunk):
            # It's a block! Keep it whole.
            final_sentences.append(clean_chunk)
        else:
            # It's actual text! Use Spacy.
            if SPACY_AVAILABLE:
                doc = nlp(clean_chunk)
                # Spacy handles abbreviations like "Fig." much better than regex
                spacy_sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                final_sentences.extend(spacy_sents)
            else:
                # Fallback Regex if Spacy isn't installed
                # (This is the robust science regex from before)
                science_pattern = (
                    r'(?<!\bFig\.)(?<!\bFigs\.)(?<!\bEq\.)(?<!\bEqs\.)'
                    r'(?<!\bTab\.)(?<!\bRef\.)(?<!\bal\.)'
                    r'(?<!\b[A-Z]\.)'
                    r'(?<=\.|\?|\!)'
                    r'\s'
                )
                regex_sents = re.split(science_pattern, clean_chunk)
                final_sentences.extend([s.strip() for s in regex_sents if s.strip()])
            
    return final_sentences

async def llm_call_async(prompt: str, json_schema: Optional[Dict[str, Any]] = None, timeout: int = 60) -> str:
    """
    Wraps the blocking Gemini call in a thread executor so it doesn't freeze the loop.
    Includes a timeout to prevent infinite hangs.
    """
    loop = asyncio.get_running_loop()

    max_retries = 3
    base_delay = 2

    # Define the blocking work
    def blocking_io():
        # --- CACHING LOGIC CAN GO HERE ---
        return client.models.generate_content(
            model="gemini-2.5-flash", # Flash is faster for this
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": json_schema,
            },
        ).text

    # Run in a thread pool with a timeout
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, blocking_io), timeout=timeout)
        except asyncio.TimeoutError:
            delay = base_delay * (2 ** attempt)
            print(f"  ⚠️ Timeout (Attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
            await asyncio.sleep(delay)
        except Exception as e:
            error_str = e

            # Check for specific "Overloaded" (503) or "Rate Limit" (429) errors
            if "503" in error_str or "429" in error_str or "Overloaded" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    # Calculate delay: 2s, 4s, 8s, 16s... + random jitter
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                    
                    print(f"  ⏳ Server Busy (503). Retrying in {sleep_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(sleep_time)
                    continue # Try loop again
            
            # If it's a different error (e.g., 400 Bad Request), or we ran out of retries, crash.
            print(f"❌ Unrecoverable LLM Error: {e}")
            raise

    raise Exception(f"Failed to get LLM response after {max_retries} retries.")

def chain(input_data: str, prompts: list[tuple[str, BaseModel]]) -> str:
    """Chain multiple LLM calls sequentially, using Pydantic schema for output."""
    result = input_data
    for i, (prompt, schema) in enumerate(prompts, 1):
        result = llm_call(f"{prompt}\nInput: {result}", schema.model_json_schema())
    return result

def parallel(task_func, inputs: list, n_workers: int = 3) -> list:
    """Process multiple inputs concurrently."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(task_func, inputs))
    return results

# ==========================================
# 3. PHASE 1: DETERMINISTIC MARKDOWN SPLITTER
# ==========================================

def parse_markdown_structure(md_text: str) -> Tuple[str, List[SectionInfo]]:
    """
    Robustly parses markdown into a Document Title and a list of Sections.
    
    Logic:
    1. Identifies all headers (lines starting with #, ##, etc.)
    2. Treats the first Level 1 header (#) as the Document Title.
    3. Splits the rest of the content into sections based on headers.
    4. Captures "Introduction" text that might appear before the first section header.
    
    Returns:
        (document_title, list_of_section_info_objects)
    """
    lines = md_text.split('\n')
    
    headers = []
    in_code_block = False
    
    # Regex handles:
    # 1. Standard Headers: ## Title
    # 2. Bolding in headers: ## **Title**
    # 3. Trailing hashes (ATX style): ## Title ##
    header_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$')

    # Pass 1: Scan for headers and code blocks
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Toggle code block state
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Skip headers if inside a code block
        if in_code_block:
            continue
        
        # Regex to match headers (e.g., "# Title", "## Section")
        match = header_pattern.match(stripped)
        if match:
            # clean the title text (remove internal bold markers if needed, strictly optional)
            title_text = match.group(2).strip()

            headers.append({
                'level': len(match.group(1)),
                'title': title_text,
                'line_num': i
            })

    # Default values
    doc_title = "Untitled Document"
    sections: List[SectionInfo] = []
    
    # Handle Case: No headers at all
    # Handle Case: No headers
    if not headers:
        # If text is very short/empty, returns empty list
        if not md_text.strip():
            return doc_title, []
        return doc_title, [SectionInfo(title="Full Text", content=md_text, line_num=1, id=str(uuid.uuid4()))]
        
    # Pass 2: Determine Document Title
    # Prioritize the first H1. If no H1 exists, use the first header found as title.
    h1_headers = [h for h in headers if h['level'] == 1]
    
    if h1_headers:
        title_header = h1_headers[0]
        doc_title = title_header['title']
    else:
        # Fallback for documents that start with ## (common in partial rips)
        title_header = headers[0] 
        doc_title = title_header['title']
    
    # Slicing Logic
    # We want to slice text between headers.
    # We add a "phantom" end header to simplify the loop.
    slice_points = [h['line_num'] for h in headers] + [len(lines)]
    
    # 1. Handle Pre-Header Content (e.g., Images, Metadata blocks)
    if headers[0]['line_num'] > 0:
        pre_text = "\n".join(lines[0:headers[0]['line_num']]).strip()
        # Filter: Only add "Introduction" if it has actual alphanumerics (ignores stray images)
        if pre_text and re.search(r'[a-zA-Z0-9]', pre_text):
            sections.append(SectionInfo(
                title="Introduction", 
                content=pre_text, 
                line_num=1,
                id=str(uuid.uuid4())
            ))

    # Iterate through headers to build sections
    for i, header in enumerate(headers):
        # Determine start and end lines for this section's content
        # Start: The line *after* the header
        start_line = header['line_num'] + 1
        
        # End: The line *before* the next header (or end of file)
        end_line = slice_points[i+1]
        
        # Extract content
        content_lines = lines[start_line:end_line]
        content_text = "\n".join(content_lines).strip()
        
        # Logic: 
        # If this is the Main Title (H1), treat its content as "Abstract"
        # If this is a Section Header (H2+), treat it as a standard section
        
        if header['level'] == 1 and header['title'] == doc_title:
            if content_text:
                sections.append(SectionInfo(
                    title="Abstract / Introduction",
                    content=content_text,
                    line_num=start_line,
                    id=get_uuid()
                ))
        else:
            # Standard Section (H2, H3, etc.)
            if content_text: 
                sections.append(SectionInfo(
                    title=header['title'],
                    content=content_text,
                    line_num=start_line,
                    id=get_uuid()
                ))

    return doc_title, sections

# ==========================================
# 3. THE REFINEMENT LOGIC
# ==========================================
async def process_group_async(
    grp: SemanticGroupIndices, 
    raw_sentences: List[str], 
    start_line_num: int,
    registry: BlockRegistry,
    is_references: bool = False
) -> Optional[SemanticGroupNode]:
    
    # 1. Slice Logic (CPU - Fast)
    start = max(0, grp.start_sentence_id)
    end = min(len(raw_sentences), grp.end_sentence_id + 1)
    group_sents = raw_sentences[start:end]
    
    if not group_sents: return None
    
    # Filter: Identify which items are real sentences vs. special blocks
    # we only want to send REAL sentences to the LLM for keyword extraction
    real_sentence_indices = []
    for idx, txt in enumerate(group_sents):
        if not registry.get(txt.strip()):
            real_sentence_indices.append(idx)
    
    full_group_text = " ".join(group_sents)

    # Dictionary to hold LLM results
    batch_data = {}

    if real_sentence_indices and not is_references:
        # 3. Prepare Prompt
        indexed_text_lines = []
        for local_idx in real_sentence_indices:
            indexed_text_lines.append(f"[{local_idx}] {group_sents[local_idx]}")

        indexed_group_text = "\n".join(indexed_text_lines)
        batch_prompt = f"""
        Context:
        {indexed_group_text}
        
        Task: Extract scientific keywords for EACH sentence index provided above.
        
        CRITICAL FORMATTING INSTRUCTIONS:
        1. The output must be a Dictionary where the KEY is the Integer Index (e.g. 0, 1, 5).
        2. NEVER use the text content (e.g. "University of Illinois") as the key.
        3. Only use the integer provided in the square brackets [] above.
        
        Return a Dict[int, List[KeywordMetadata]].
        """

        # 4. AWAIT THE NETWORK CALL (With Timeout)
        try:
            response_text = await llm_call_async(batch_prompt, BatchKeywordExtraction.model_json_schema())
            batch_data = BatchKeywordExtraction.model_validate_json(response_text).results
        except Exception as e:
            print(f"Error in group '{grp.group_title}': {e}")
            batch_data = {}

    # 5. Assemble Result (CPU - Fast)
    final_nodes = []
    
    for i, sent_text in enumerate(group_sents):
        clean_text = sent_text.strip()
        
        if "__IMG_" in clean_text or "__TBL_" in clean_text:
            print(f"  🔍 Checking placeholder: '{clean_text}'")
        
        # CHECK 1: is this a masked image or table?
        special_node = registry.get(clean_text)
        if special_node:
            print(f"  ✅ Found Special Node: {clean_text}")
            final_nodes.append(special_node)
        else:
            # CHECK 2: is this a real sentence?
            
            kw_data = batch_data.get(i, [])
        
            # Hallucination Check & Node Creation
            verified_keywords = [
                KeywordNode(
                    title=kw.term, text=sent_text, summary=kw.summary, 
                    metadata=kw, node_id=get_uuid()
                )
                for kw in kw_data if kw.term.lower() in sent_text.lower()
            ]
        
            final_nodes.append(SentenceNode(
                title=sent_text[:40] + "...",
                text=sent_text,
                line_num=start_line_num + start + i,
                nodes=verified_keywords,
                node_id=get_uuid()
            ))

    return SemanticGroupNode(
        title=grp.group_title,
        text=full_group_text,
        line_num=start_line_num + start,
        nodes=final_nodes,
        node_id=get_uuid()
    )

async def process_section_async(section_data: SectionInfo) -> SectionNode:
    print(f"  -> Started Section: {section_data.title}")

    # 1. Extract Special Blocks (Image/Tables)
    clean_contents, block_registry = extract_special_blocks(section_data.content)
    
    # 2. Deterministic Split
    raw_sentences = split_text_to_sentences(clean_contents)
    total_sentences = len(raw_sentences) # Needed for audit
    
    if not raw_sentences:
        return SectionNode(title=section_data.title, text=section_data.content, line_num=section_data.line_num, nodes=[], node_id=get_uuid())

    # 3. Grouping (Pass the cleaned text with placeholders to the LLM for grouping)
    # The LLM is usually smart enough to group "__TBL_123__" with surrounding context
    indexed_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(raw_sentences)])
    group_prompt = f"Context: {section_data.title}\nSentences:\n{indexed_text}\nTask: Group sentences."
    
    try:
        group_resp = await llm_call_async(group_prompt, GroupingOutput.model_json_schema())
        raw_groups = GroupingOutput.model_validate_json(group_resp).groups
    except Exception as e:
        print(f"Grouping failed for {section_data.title}: {e}")
        raw_groups = [SemanticGroupIndices(group_title="General", start_sentence_id=0, end_sentence_id=len(raw_sentences)-1)]

    # =========================================================
    # 🛡️ THE AUDIT & REPAIR LOGIC
    # =========================================================
    covered_indices = set()
    valid_groups = []

    for grp in raw_groups:
        # Clamp indices to valid range
        start = max(0, grp.start_sentence_id)
        end = min(total_sentences - 1, grp.end_sentence_id)
        
        if start > end: continue # Invalid range
        
        # Update the group object with clamped values to be safe
        grp.start_sentence_id = start
        grp.end_sentence_id = end
        
        valid_groups.append(grp)
        # Record coverage
        for i in range(start, end + 1):
            covered_indices.add(i)

    # Detect Gaps
    all_indices = set(range(total_sentences))
    missing_indices = sorted(list(all_indices - covered_indices))

    # Auto-Repair: Create "Recovery Groups" for gaps
    if missing_indices:
        print(f"⚠️ Warning: LLM dropped sentences {missing_indices} in '{section_data.title}'. Auto-recovering...")
        
        for k, g in groupby(enumerate(missing_indices), lambda ix: ix[0] - ix[1]):
            consecutive_missing = list(map(itemgetter(1), g))
            start_miss = consecutive_missing[0]
            end_miss = consecutive_missing[-1]
            
            valid_groups.append(SemanticGroupIndices(
                group_title="Recovered Content",
                start_sentence_id=start_miss,
                end_sentence_id=end_miss
            ))

    # Sort groups by start index to maintain document flow
    valid_groups.sort(key=lambda x: x.start_sentence_id)
    # =========================================================

    # 4. SCATTER-GATHER (The Optimization)
    # We create a list of tasks (coroutines) but don't await them yet
    is_refs = "REFERENCES" in section_data.title.upper()
    
    tasks = [
        process_group_async(grp, raw_sentences, section_data.line_num, block_registry, is_references=is_refs) 
        for grp in valid_groups
    ]
    
    # Now we fire them all at once!
    # If there are 10 groups, this takes ~3 seconds total (wait time of slowest request)
    # instead of 30 seconds (sum of all requests).
    semantic_group_nodes = await asyncio.gather(*tasks)
    
    # Filter out None results from empty groups
    valid_nodes = [n for n in semantic_group_nodes if n is not None]

    print(f"  ✓ Finished Section: {section_data.title}")
    return SectionNode(
        title=section_data.title,
        text=section_data.content,
        line_num=section_data.line_num,
        nodes=valid_nodes,
        node_id=get_uuid()
    )


# ==========================================
# Node reordering
# ==========================================
def reindex_document_tree(doc: DocumentRoot, start_from: int = 1) -> Tuple[DocumentRoot, Dict[str, str]]:
    """
    Traverses the tree and assigns:
    1. Sequential IDs (0001, 0002) to Structural Nodes (Section, Group, Sentence, Image, Table).
    2. Deduplicated IDs (kw_0001, kw_0002) to Keyword Nodes based on the term.
    """
    
    # Counter for Structure (Sections, Sentences, Images, etc.)
    struct_counter = start_from
    
    # Counter for Keywords
    kw_counter = 1
    
    # Registry for Keywords: { "term_lowercase": "kw_0001" }
    keyword_map = {}

    def get_struct_id():
        nonlocal struct_counter
        nid = f"{struct_counter:04d}"
        struct_counter += 1
        return nid

    def get_keyword_id(term: str):
        nonlocal kw_counter
        key = term.lower().strip()
        
        if key in keyword_map:
            return keyword_map[key]
        else:
            # Create new ID
            nid = f"kw_{kw_counter:04d}"
            keyword_map[key] = nid
            kw_counter += 1
            return nid

    # Traverse Depth-First
    for section in doc.structure:
        section.node_id = get_struct_id()
        
        for group in section.nodes:
            group.node_id = get_struct_id()
            
            for item in group.nodes:
                # Handle Structural Leaves (Sentence, Image, Table)
                item.node_id = get_struct_id()
                
                # If it's a Sentence, it might have children (Keywords)
                if hasattr(item, 'nodes'):
                    for keyword in item.nodes:
                        # Use the deduplicated Keyword ID logic
                        term = keyword.metadata.term
                        keyword.node_id = get_keyword_id(term)
                        
    print(f"  ✓ Re-indexed {struct_counter - 1} structural nodes.")
    print(f"  ✓ Mapped {kw_counter - 1} unique keywords.")
    
    return doc, keyword_map

# ==========================================
# 4. THE ORCHESTRATOR
# ==========================================

async def main_pipeline(md_text: str):
    print("🚀 Starting Async Semantic Parser...")
    
    # 1. Parse Structure (Fast, Synchronous)
    doc_title, sections = parse_markdown_structure(md_text)
    
    # 2. Create Tasks for all Sections
    # This runs ALL sections AND ALL groups within them concurrently
    section_tasks = [process_section_async(s) for s in sections]
    
    # 3. Execute
    processed_sections = await asyncio.gather(*section_tasks)
    
    # 4. Output
    final_doc = DocumentRoot(
        title=doc_title,
        doc_name="parsed_doc",
        structure=processed_sections
    )

    # 5. Reindex (Now returns the map too!)
    final_doc, keyword_registry = reindex_document_tree(final_doc)
    
    # Optional: You can attach the registry to the final output if you want
    # or just return the doc. 
    result = final_doc.model_dump()
    
    # Inject the global keyword map into the JSON output (Optional but useful)
    result['global_keyword_map'] = keyword_registry
    
    return final_doc.model_dump()

# ==========================================
# 5. EXECUTION
# ==========================================

async def run_main():
    import argparse
    import os
    import json # Added this import for json.dump

    parser = argparse.ArgumentParser(description='Async Semantic Parser for Markdown')
    parser.add_argument('md_path', help='Path to the input markdown file')
    parser.add_argument('--output', '-o', help='Path to the output JSON file', default=None)
    

    print("\n--- Starting Async Semantic Parser ---")
    args = parser.parse_args()
    
    if not os.path.exists(args.md_path):
        print(f"Error: File not found: {args.md_path}")
        return

    print(f"Processing: {args.md_path}")

    # Run it
    with open(args.md_path, "r", encoding="utf-8") as f:
        raw_md = f.read()

    result = await main_pipeline(raw_md)
    print("\n--- Final Structured Output ---")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default to results/md_filename_structure.json
        basename = os.path.splitext(os.path.basename(args.md_path))[0]
        output_path = f'./results/{basename}_structure.json'
        
    # Ensure results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save the result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Saved to: {output_path}")
    import os; os._exit(0)

if __name__ == '__main__':
    try:
        print("\n--- Starting Async Semantic Parser ---")
        asyncio.run(run_main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os._exit(0)