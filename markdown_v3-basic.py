import json
import re
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# --- GEMINI/Pydantic Setup (Requires 'google-genai' and 'pydantic' installed) ---
from google import genai
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()
# Initialize Gemini Client (Ensure API Key is configured in environment)
client = genai.Client() # Uncomment in a real environment
# ----------------------------------------------------------------------


# ==========================================
# 1. PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ==========================================

class KeywordMetadata(BaseModel):
    """Schema for the keyword/entity details."""
    term: str = Field(description="The exact scientific term or entity extracted.")
    summary: str = Field(description="A brief definition or description of the term.")
    relevance: str = Field(description="Explanation of why this term is relevant.")
    parent_title: Optional[str] = Field(description="The title of the parent section.")
    parent_node_type: str = Field(description="The type of the parent node.")

class KeywordNode(BaseModel):
    """Leaf Node: A specific extracted entity."""
    title: str
    start_index: int = 1
    end_index: int = 1
    text: str
    summary: str  # Required field
    node_type: str = "keyword"
    _text_locked: bool = True
    metadata: KeywordMetadata
    nodes: List[Any] = Field(default_factory=list)
    node_id: str

class SentenceNode(BaseModel):
    """Level 3 Node: A single atomic sentence."""
    title: str
    start_index: int = 1
    end_index: int = 1
    text: str
    line_num: Optional[int] = None
    node_type: str = "sentence"
    nodes: List[KeywordNode]  # Children are Keywords
    node_id: str

class SemanticGroupNode(BaseModel):
    """Level 2.5 Node: A paragraph or semantic cluster of sentences."""
    title: str = Field(description="Generated title summarizing this group.")
    start_index: int = 1
    end_index: int = 1
    text: str
    line_num: Optional[int] = None
    node_type: str = "semantic_group"
    nodes: List[SentenceNode] # Children are Sentences
    node_id: str

class SectionNode(BaseModel):
    """Level 2 Node: A document section (e.g., Results)."""
    title: str
    start_index: int = 1
    end_index: int = 1
    text: str
    line_num: int
    nodes: List[SemanticGroupNode] # <--- UPDATED: Children are Semantic Groups now
    node_id: str

class DocumentRoot(BaseModel):
    """Level 1 Node: The Document Root."""
    title: str
    start_index: int = 1
    end_index: int = 1
    text: str
    line_num: int
    nodes: List[SectionNode] # Children are Sections
    node_id: str

class DocumentStructure(BaseModel):
    doc_name: str
    structure: List[DocumentRoot]

# --- Temp Output Models for Chain Steps ---
class ParagraphSplitOutput(BaseModel):
    semantic_groups: List[Dict[str, str]]

class SentenceSplitOutput(BaseModel):
    sentences: List[str]

class KeywordListOutput(BaseModel):
    keywords: List[KeywordMetadata]
    
# ==========================================
# 2. THE AGENTIC PRIMITIVES (Updated)
# ==========================================

def get_uuid(length=4):
    import uuid
    return str(uuid.uuid4())[:length]

def llm_call(prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
    """
    Simulates or executes the Gemini API call with structured output.
    Uses a simple mock for execution without a key.
    """
    
    # --- REAL GEMINI API CALL (Uncomment for production) ---
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        },
    )
    return response.text

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
# 3. THE REFINEMENT LOGIC
# ==========================================

# Temporary Pydantic Schemas for chain steps
class SectionSplitOutput(BaseModel):
    document_title: str
    sections: List[Dict[str, Any]] # Will hold temp structure

class SentenceSplitOutput(BaseModel):
    sentences: List[str]

class KeywordListOutput(BaseModel):
    keywords: List[KeywordMetadata]

class ParagraphSplitOutput(BaseModel):
    """Used to structure the raw text into paragraphs/groups."""
    semantic_groups: List[Dict[str, str]] = Field(description="A list of objects, each containing a 'group_title' and 'raw_text'.")    


def process_single_section(section_data: Dict[str, Any]) -> SectionNode:
    """
    WORKER function: Runs a CHAIN of refinement on a single section.
    New Flow: Section -> Semantic Group -> Sentence -> Keyword
    """
    section_title = section_data['title']
    raw_text = section_data['content']
    section_line_num = section_data.get('line_num', 1)

    # --- CHAIN STEP 1: Split into Semantic Groups/Paragraphs ---
    group_split_prompt = """
    Analyze the following text from the section '{section_title}'.
    Decompose the raw text into distinct semantic groups or paragraphs based on topic shifts, even if not explicitly separated by empty lines.
    For each group, generate a concise 'group_title' that summarizes its content, and include the 'raw_text'. 
    Return a list of these groups in the ParagraphSplitOutput schema.
    """
    groups_json = llm_call(f"{group_split_prompt}\nInput: {raw_text}", ParagraphSplitOutput.model_json_schema())
    
    try:
        groups_output = ParagraphSplitOutput.model_validate_json(groups_json)
        semantic_groups_data = groups_output.semantic_groups
    except Exception as e:
        print(f"Group split error: {e}. Falling back to one large group.")
        semantic_groups_data = [{"group_title": "Section Summary", "raw_text": raw_text}]


    # --- CHAIN STEP 2: Process each Semantic Group (Mini-Parallel/Linear) ---
    section_group_nodes: List[SemanticGroupNode] = []
    
    current_line = section_line_num + 1

    for group_data in semantic_groups_data:
        group_title = group_data['group_title']
        group_text = group_data['raw_text']

        # --- Sub-Chain A: Split Group Text into Sentences ---
        sentence_split_chain = [
            (
                "Clean the text by removing all markdown formatting, tables, and footnote references. Then, split the text into atomic, standalone sentences. Return as JSON list.", 
                SentenceSplitOutput
            )
        ]
        sentences_json = chain(group_text, sentence_split_chain)
        
        try:
            sentences_output = SentenceSplitOutput.model_validate_json(sentences_json)
            sentences = sentences_output.sentences
        except:
            sentences = [group_text[:200].replace('\n', ' ') + "..."] 
        

        # --- Sub-Chain B: Process each Sentence (Keyword Extraction) ---
        group_sentence_nodes: List[SentenceNode] = []
        
        for i, sent in enumerate(sentences):
            # KEYWORD EXTRACTION (The final step)
            keyword_prompt = f"""
            Analyze the following sentence from the semantic group '{group_title}'. 
            Extract key scientific entities, materials, and concepts. 
            For each, provide a precise 'summary' definition and an explanation of its 'relevance' to the group's topic. 
            Return a list of KeywordMetadata objects.
            """
            keywords_json = llm_call(keyword_prompt, KeywordListOutput.model_json_schema())
            
            try:
                keywords_output = KeywordListOutput.model_validate_json(keywords_json)
                keywords_data = keywords_output.keywords
            except:
                keywords_data = []

            # Build Keyword Nodes
            keyword_nodes = [
                KeywordNode(
                    title=k_data.term,
                    text=sent,
                    summary=k_data.summary, 
                    metadata=KeywordMetadata(**k_data.model_dump()),
                    node_id=get_uuid()
                ) for k_data in keywords_data
            ]

            # Build Sentence Node
            group_sentence_nodes.append(SentenceNode(
                title=sent[:60].replace('\n', ' ') + "...",
                text=sent,
                line_num=current_line + i,
                nodes=keyword_nodes,
                node_id=get_uuid()
            ))

        # Build Semantic Group Node
        semantic_group_node = SemanticGroupNode(
            title=group_title,
            text=group_text,
            line_num=current_line,
            nodes=group_sentence_nodes,
            node_id=get_uuid()
        )
        section_group_nodes.append(semantic_group_node)
        current_line += len(sentences) # Update line count

    # Return the fully constructed section tree
    return SectionNode(
        title=section_title,
        text=raw_text,
        line_num=section_line_num,
        nodes=section_group_nodes,
        node_id=get_uuid()
    )

# ==========================================
# 4. THE ORCHESTRATOR
# ==========================================

def semantic_parse_pipeline(markdown_text: str) -> Dict[str, Any]:
    print("🚀 Starting Semantic Parse Pipeline (Gemini/Pydantic)...\n")

    # PHASE 1: STRUCTURAL ANALYSIS
    # Extract Document Title and Section Contents (Uses structured output)
    print("--- Phase 1: Structural Analysis ---")
    structure_prompt = "Extract the document title (Level 1 header) and all section titles (Level 2 headers) with their raw content. Return JSON format with fields 'document_title' and 'sections' (list of {'title', 'content', 'line_num', 'id'})."
    
    sections_json = llm_call(f"{structure_prompt}\nInput: {markdown_text}", SectionSplitOutput.model_json_schema())
    
    try:
        section_output = SectionSplitOutput.model_validate_json(sections_json)
        doc_title = section_output.document_title
        sections = section_output.sections
    except Exception as e:
        print(f"Error parsing section split: {e}. Using fallback.")
        doc_title = "Document Parse Error"
        sections = [{"title": "Content Fallback", "content": markdown_text, "line_num": 1}]
    
    print(f"Document Title: {doc_title}")
    print(f"Found {len(sections)} sections. Dispatching to workers...")

    # PHASE 2: PARALLEL REFINEMENT
    print("\n--- Phase 2: Parallel Refinement Chains ---")
    processed_sections = parallel(process_single_section, sections)

    # PHASE 3: AGGREGATION & FINAL VALIDATION
    print("\n--- Phase 3: Aggregation & Validation ---")
    
    # Construct the final Pydantic model for validation
    document_root = DocumentRoot(
        title=doc_title,
        text=markdown_text,
        line_num=1,
        nodes=processed_sections,
        node_id=get_uuid()
    )

    final_tree = DocumentStructure(
        doc_name=doc_title.lower().replace(' ', '_'),
        structure=[document_root]
    )
    
    # Return the dictionary representation of the validated model
    return final_tree.model_dump()

# ==========================================
# 5. EXECUTION
# ==========================================

raw_markdown = "./tests/markdowns/New Insights into the Compositional Dependence of Li-Ion Transport in polymer-ceramic composite electrolytes/short_original.md"

# Run it
result = semantic_parse_pipeline(raw_markdown)
print("\n--- Final Structured Output ---")

# save the result
with open('./results/test-results.json', 'w') as f:
    json.dump(result, f, indent=4)