"""
This version introduces batch processing
"""
import asyncio
import json
import re
import uuid
from typing import List, Optional, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
from concurrent.futures import ThreadPoolExecutor

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

class SemanticGroupNode(BaseModel):
    """Level 2.5 Node: A paragraph or semantic cluster of sentences."""
    title: str
    text: str
    line_num: int
    node_type: str = "semantic_group"
    nodes: List[SentenceNode]
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

def get_uuid(): return str(uuid.uuid4())[:8]

def split_text_to_sentences(text: str) -> List[str]:
    """Deterministic splitting. Source of Truth."""
    if SPACY_AVAILABLE:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Robust Regex Fallback
        return [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) if s.strip()]

async def llm_call_async(prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
    """
    Wraps the blocking Gemini call in a thread executor so it doesn't freeze the loop.
    Matches the pattern in your 'apply_semantic_subdivision' snippet.
    """
    loop = asyncio.get_running_loop()
    
    # Define the blocking work
    def blocking_io():
        # --- CACHING LOGIC CAN GO HERE ---
        return client.models.generate_content(
            model="gemini-2.5-flash-lite", # Flash is faster for this
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": json_schema,
            },
        ).text

    # Run in a thread pool (non-blocking await)
    return await loop.run_in_executor(None, blocking_io)

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
    start_line_num: int
) -> Optional[SemanticGroupNode]:
    
    # 1. Slice Logic (CPU - Fast)
    start = max(0, grp.start_sentence_id)
    end = min(len(raw_sentences), grp.end_sentence_id + 1)
    group_sents = raw_sentences[start:end]
    
    if not group_sents: return None
    
    full_group_text = " ".join(group_sents)

    # 2. Prepare Prompt
    indexed_group_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(group_sents)])
    batch_prompt = f"""
    Context:
    {indexed_group_text}
    
    Task: Extract scientific keywords for EACH sentence index.
    Return a Dict[int, List[KeywordMetadata]].
    """

    # 3. AWAIT THE NETWORK CALL (The magic happens here)
    # While this waits, Python goes off to start processing the next group!
    try:
        response_text = await llm_call_async(batch_prompt, BatchKeywordExtraction.model_json_schema())
        batch_data = BatchKeywordExtraction.model_validate_json(response_text).results
    except Exception as e:
        print(f"Error in group '{grp.group_title}': {e}")
        batch_data = {}

    # 4. Assemble Result (CPU - Fast)
    sentence_nodes = []
    for i, sent_text in enumerate(group_sents):
        kw_data = batch_data.get(i, [])
        
        # Hallucination Check & Node Creation
        verified_keywords = [
            KeywordNode(
                title=kw.term, text=sent_text, summary=kw.summary, 
                metadata=kw, node_id=get_uuid()
            )
            for kw in kw_data if kw.term.lower() in sent_text.lower()
        ]
        
        sentence_nodes.append(SentenceNode(
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
        nodes=sentence_nodes,
        node_id=get_uuid()
    )

async def process_section_async(section_data: SectionInfo) -> SectionNode:
    print(f"  -> Started Section: {section_data.title}")
    
    # A. Deterministic Split
    raw_sentences = split_text_to_sentences(section_data.content)
    total_sentences = len(raw_sentences) # Needed for audit
    
    if not raw_sentences:
        return SectionNode(title=section_data.title, text=section_data.content, line_num=section_data.line_num, nodes=[], node_id=get_uuid())

    # B. Grouping (This must happen first, so we await it)
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

    # C. SCATTER-GATHER (The Optimization)
    # We create a list of tasks (coroutines) but don't await them yet
    tasks = [
        process_group_async(grp, raw_sentences, section_data.line_num) 
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
    return final_doc.model_dump()

# ==========================================
# 5. EXECUTION
# ==========================================

md_path = "./tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/short_original.md"
# md_path = "./tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes.md"

# Run it
with open(md_path, "r", encoding="utf-8") as f:
    raw_md = f.read()

result = asyncio.run(main_pipeline(raw_md))
print("\n--- Final Structured Output ---")

# save the result
with open('./results/test-results3.json', 'w') as f:
    json.dump(result, f, indent=4)