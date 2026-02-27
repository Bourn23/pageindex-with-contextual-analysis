import os
import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid

load_dotenv()

# Configuration
EXTRACTION_MODEL = "gemini-2.5-flash"
EVALUATION_MODEL = "gemini-3.1-pro-preview"
CONCURRENCY_LIMIT = 5

class SectionInfo:
    def __init__(self, title: str, content: str, line_num: int, end_line_num: int, id: str):
        self.title = title
        self.content = content
        self.line_num = line_num
        self.end_line_num = end_line_num
        self.id = id

class MarkdownContextParser:
    def parse_structure(self, md_text: str) -> Tuple[str, List[SectionInfo]]:
        """Parses headers to build document sections."""
        lines = md_text.split('\n')
        headers = []
        in_code_block = False
        header_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$')

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block: continue
            
            match = header_pattern.match(line.strip())
            if match:
                headers.append({'level': len(match.group(1)), 'title': match.group(2).strip(), 'line_num': i})

        doc_title = "Untitled Document"
        sections: List[SectionInfo] = []

        def create_section(title, start, end):
            content = "\n".join(lines[start:end]).strip()
            if content:
                return SectionInfo(title=title, content=content, line_num=start, end_line_num=end, id=str(uuid.uuid4()))
            return None

        if not headers:
            if not md_text.strip(): return doc_title, []
            return doc_title, [create_section("Full Text", 0, len(lines))]

        h1 = next((h for h in headers if h['level'] == 1), headers[0])
        doc_title = h1['title']
        
        slice_points = [h['line_num'] for h in headers] + [len(lines)]

        if headers[0]['line_num'] > 0:
            intro = create_section("Introduction", 0, headers[0]['line_num'])
            if intro: sections.append(intro)

        for i, header in enumerate(headers):
            start = header['line_num'] + 1
            end = slice_points[i+1]
            title = header['title']
            sec = create_section(title, start, end)
            if sec: sections.append(sec)

        return doc_title, sections

async def get_gemini_response(client, model_name, prompt):
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return None

async def extract_processing_methods_full(client, md_content):
    prompt = f"""
Extract all processing methods used to synthesize or prepare the samples/materials described in the following research paper.
List them concisely, including key parameters like temperatures, times, solvents, if mentioned.

Paper Content:
{md_content}

Extracted Processing Methods:
"""
    return await get_gemini_response(client, EXTRACTION_MODEL, prompt)

async def extract_processing_methods_from_chunk(client, chunk_title, chunk_content, sem):
    async with sem:
        prompt = f"""
Extract any processing methods (synthesis, preparation, sintering, etc.) mentioned in this document section.
Include key parameters like temperatures, times, solvents, if mentioned.
If no processing methods are mentioned, simply reply 'None'.

Section Title: {chunk_title}
Section Content:
{chunk_content}

Extracted Processing Methods:
"""
        return await get_gemini_response(client, EXTRACTION_MODEL, prompt)

async def evaluate_results(client, md_content, method1_res, method2_res):
    prompt = f"""
You are an expert evaluator for material science data extraction.
I have two sets of 'processing methods' extracted from a research paper using different strategies.
- Method 1: Extracted by passing the entire document at once.
- Method 2: Extracted by breaking the document into sections, extracting from each, and aggregating.

Your task is to evaluate which method performed better and why.
Consider:
1. Missing information: Did one method capture details the other missed?
2. Accuracy: Are the extracted methods actually present and correctly described?
3. Redundancy: Is one method significantly more verbose or repetitive?
4. Overall quality: Which summary is more useful for a database?

Original Paper Content (Truncated if necessary):
{md_content[:30000] if len(md_content) > 30000 else md_content}

Method 1 Results:
{method1_res}

Method 2 Results:
{method2_res}

Provide your detailed evaluation and a final score (1-10) for each method.
"""
    return await get_gemini_response(client, EVALUATION_MODEL, prompt)

async def process_paper(client, paper_dir, sem):
    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        print(f"No MD file found in {paper_dir}")
        return None

    md_path = md_files[0]
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print(f"Processing: {paper_dir.name}")

    # Method 1: Entire Document
    res1 = await extract_processing_methods_full(client, md_content)

    # Method 2: Chunk-based (Commented out per user request)
    # parser = MarkdownContextParser()
    # _, sections = parser.parse_structure(md_content)
    # 
    # chunk_tasks = []
    # for sec in sections:
    #     chunk_tasks.append(extract_processing_methods_from_chunk(client, sec.title, sec.content, sem))
    # 
    # chunk_results = await asyncio.gather(*chunk_tasks)
    # # Aggregate and deduplicate (roughly)
    # aggregated_res2 = "\n".join([r for r in chunk_results if r and r.strip().lower() != 'none'])
    aggregated_res2 = "Skipped (Method 2 commented out)"

    # Evaluation (Commented out per user request)
    # evaluation = await evaluate_results(client, md_content, res1, aggregated_res2)
    evaluation = "Skipped (Evaluation commented out)"

    result = {
        "paper": paper_dir.name,
        "method1": res1,
        "method2": aggregated_res2,
        "evaluation": evaluation
    }

    output_path = paper_dir / "processing_extraction_comparison.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", default="/Users/bourn23/Downloads/general/PageIndex/output/deb_downloaded_papers")
    parser.add_argument("--sample_path", help="Process only this specific paper path")
    args = parser.parse_args()
    
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), http_options={'api_version': 'v1alpha'})
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    target_path = Path(args.target_dir)
    if args.sample_path:
        paper_dirs = [Path(args.sample_path)]
    else:
        paper_dirs = [d for d in target_path.iterdir() if d.is_dir()]

    tasks = []
    for paper_dir in paper_dirs:
        tasks.append(process_paper(client, paper_dir, sem))

    await asyncio.gather(*tasks)
    print("\nProcessing complete.")

if __name__ == "__main__":
    asyncio.run(main())
