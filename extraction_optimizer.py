import os
import json
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import re


load_dotenv()

# --- Configuration ---
GROUND_TRUTH_CSV = "OBELiX/data/processed.csv"
PDF_DIR = "fetched_papers/obelix_pdf"
RESULTS_DIR = "fetched_papers/obelix_parsed_v5"
TARGET_SCRIPT = "basic_extraction_auto_improved.py"

# --- Prompts ---
HOLISTIC_ANALYSIS_PROMPT = """
You are a Senior Data Extraction Engineer analyzing extraction pipeline performance.

**TASK**: determine where the missing information is coming from in my data extraction pipeline.

**INPUTS**:
1. **Paper PDF**: Attached
2. **Ground Truth Data** (what SHOULD have been extracted):
{ground_truth_table}

3. **Actually Extracted Data**:
{extracted_table}

**YOUR ANALYSIS SHOULD**:
1. **Identify Patterns**: Don't just list individual misses. Look for:
   - Are we missing data from specific table types? (e.g., Arrhenius plots, composition-sweep tables)
   - Are certain units or formats consistently missed? (e.g., "10^-3 S/cm", "mS/cm", "1000/T")
   - Are we failing on specific notation styles? (e.g., "Li₁.₃Al₀.₃Ti₁.₇(PO₄)₃", subscript handling)
   - Are we missing data from figures vs tables vs text?
   - Are complex stoichiometry expressions being skipped? (e.g., "Li₇₋ₓLa₃Zr₂₋ₓTaₓO₁₂")

2. **Root Cause**: For each pattern, explain WHY the current extraction likely fails.

3. **Generic Solutions**: Propose **broadly applicable** improvements, not fixes for individual cases.
   - Example: "Add support for Arrhenius plot detection by looking for '1000/T' axis labels"
   - Example: "Extend unit normalization to handle milli-prefix variations (mS/cm → S/cm)"
   - Example: "Improve subscript/superscript Unicode parsing in composition strings"

**OUTPUT** (JSON):
{{
  "patterns_missed": [
    {{
      "pattern": "Description of systematic issue",
      "examples": ["Comp1 @ Value1", "Comp2 @ Value2"],
      "source": "Where is this missing data(s) come from?",
      "root_cause": "Why this happens",
      "prevalence": "How common (rare/occasional/frequent)"
    }}
  ],
  "generic_suggestions": [
    "Actionable improvement that addresses a pattern",
    "Another improvement"
  ]
}}

**IMPORTANT**: Focus on **patterns**, not individual data points. We want fixes that improve extraction across many papers.
"""

FRAMEWORK_IMPROVER_PROMPT = """
You are a Lead Python Engineer specializing in LLM-based extraction pipelines.

We've analyzed extraction failures across multiple papers and identified **systematic patterns** that need fixing.

**YOUR TASK**: 
Rewrite the `basic_extraction_auto_improved.py` script to address these patterns.

**AGGREGATED PATTERNS & SUGGESTIONS**:
{aggregated_improvements}

**CONSTRAINTS**:
- **Preserve Structure**: Keep the overall architecture. Modify prompts, regexes, parsing logic, and helper functions.
- **Generic Solutions**: Ensure fixes work broadly, not just for specific papers.
- **Backward Compatibility**: Don't break existing successful extractions.
- **Error Handling**: Wrap all parsing in try/except blocks.

**OUTPUT**: 
The FULL, UPDATED Python script code for `basic_extraction_auto_improved.py`.
"""

def parse_ground_truth_from_log(log_path: Path) -> Dict:
    """
    Extract ground truth data and DOI from pipeline log file.
    
    Parses the section between "Found DOI:" and "--- Starting Validation ---"
    to extract GT entries in format: GT-XXX: Composition | Conductivity S/cm
    
    Args:
        log_path: Path to the pipeline log file
    
    Returns:
        Dict with 'doi' and 'ground_truth' (list of GT entries)
    """
    try:
        log_content = log_path.read_text()
    except Exception as e:
        print(f"   ⚠️  Failed to read log: {e}")
        return {'doi': None, 'ground_truth': []}
    
    # Extract DOI
    doi_match = re.search(r'Found DOI:\s*(\S+)', log_content)
    doi = doi_match.group(1) if doi_match else None
    
    if not doi:
        print(f"   ⚠️  No DOI found in log: {log_path.name}")
        return {'doi': None, 'ground_truth': []}
    
    # Extract GT entries between "Found DOI" and "--- Starting Validation ---"
    gt_section_pattern = r'Found DOI:.*?\n(.*?)--- Starting Validation ---'
    gt_section_match = re.search(gt_section_pattern, log_content, re.DOTALL)
    
    if not gt_section_match:
        print(f"   ⚠️  No GT section found in log: {log_path.name}")
        return {'doi': doi, 'ground_truth': []}
    
    gt_section = gt_section_match.group(1)
    
    # Parse individual GT entries
    # Pattern: GT-XXX: Composition | Conductivity S/cm
    gt_pattern = r'GT-(\d+):\s*([^\|]+)\s*\|\s*([\d.e+-]+)\s*S/cm'
    
    ground_truth = []
    for match in re.finditer(gt_pattern, gt_section):
        gt_id = match.group(1)
        composition = match.group(2).strip()
        conductivity = float(match.group(3))
        
        ground_truth.append({
            'gt_id': f"GT-{gt_id}",
            'composition': composition,
            'conductivity': conductivity,
            'doi': doi
        })
    
    print(f"   ✓ Parsed {len(ground_truth)} GT entries from log (DOI: {doi})")
    return {'doi': doi, 'ground_truth': ground_truth}


def get_ground_truth_for_paper(paper_name: str, results_dir: Path) -> List[Dict]:
    """
    Get ground truth for a paper by finding and parsing its log file.
    
    Args:
        paper_name: Paper identifier (without extension)
        results_dir: Directory containing extraction results and logs
    
    Returns:
        List of ground truth dictionaries
    """
    # Find the log file matching this paper
    # Pattern: {paper_name}_v5_pipeline_*.log
    log_pattern = f"{paper_name}_v5_pipeline_*.log"
    log_files = list(results_dir.glob(log_pattern))
    
    if not log_files:
        print(f"   ⚠️  No log file found for: {paper_name}")
        return []
    
    if len(log_files) > 1:
        print(f"   ⚠️  Multiple log files found for {paper_name}, using most recent")
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    log_path = log_files[0]
    result = parse_ground_truth_from_log(log_path)
    
    return result['ground_truth']


class ExtractionOptimizer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.results_dir = Path(RESULTS_DIR)
        
    async def analyze_paper_holistically(
        self, 
        paper_name: str, 
        ground_truth: List[Dict],
        extracted_data: List[Dict], 
        pdf_path: Path
    ) -> Optional[Dict]:
        """
        Analyzes the entire extraction result for a paper to find systematic patterns.
        """
        print(f"\n📄 Analyzing: {paper_name}")
        print(f"   GT entries: {len(ground_truth)}")
        print(f"   Extracted: {len(extracted_data)}")

        # Format ground truth as table
        gt_table = "\n".join([
            f"- {gt['composition']} | {gt['conductivity']} S/cm @ Room Temp"
            for gt in ground_truth
        ])
        
        # Format extracted data as table
        ext_table = "\n".join([
            f"- {m.get('raw_composition', 'N/A')} | {m.get('raw_conductivity', 'N/A')} {m.get('raw_conductivity_unit', '')} @ {m.get('temperature_k', 'N/A')} K"
            for m in extracted_data
        ]) or "No data extracted."
        
        prompt = HOLISTIC_ANALYSIS_PROMPT.format(
            ground_truth_table=gt_table,
            extracted_table=ext_table
        )
        
        try:
            pdf_bytes = pdf_path.read_bytes()

            print("TEXT PROMPT: ", prompt)
            
            response = await self.client.aio.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part(text=prompt),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="application/pdf",
                            data=pdf_bytes
                        )
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            analysis = json.loads(response.text)
            print(f"   ✓ Found {len(analysis.get('patterns_missed', []))} patterns")
            return analysis
            
        except Exception as e:
            print(f"   ✗ Analysis failed: {e}")
            return None

    def aggregate_analyses(self, all_analyses: List[Dict]) -> Dict:
        """
        Combines multiple paper analyses into aggregated patterns and suggestions.
        """
        all_patterns = []
        all_suggestions = []
        
        for analysis in all_analyses:
            all_patterns.extend(analysis.get('patterns_missed', []))
            all_suggestions.extend(analysis.get('generic_suggestions', []))
        
        # Deduplicate suggestions (simple set-based for now)
        unique_suggestions = list(set(all_suggestions))
        
        # Group similar patterns (simplified - in production, use embeddings/clustering)
        pattern_groups = {}
        for pattern in all_patterns:
            key = pattern['pattern'][:50]  # Simple grouping by first 50 chars
            if key not in pattern_groups:
                pattern_groups[key] = []
            pattern_groups[key].append(pattern)
        
        return {
            'pattern_groups': pattern_groups,
            'unique_suggestions': unique_suggestions,
            'total_papers_analyzed': len(all_analyses)
        }

    async def run(self, max_papers: int = 10):
        print("🚀 Starting Holistic Extraction Optimizer...")
        
        # 1. Identify papers with validation reports
        results_path = Path(RESULTS_DIR)
        report_files = list(results_path.glob("*_validation_report.json"))[:max_papers]
        
        print(f"Found {len(report_files)} validation reports (analyzing {max_papers} max)")
        
        # 2. Analyze each paper holistically
        all_analyses = []
        
        for report_file in report_files:
            paper_name = report_file.name.replace("_v5_extracted_validation_report.json", "")
            
            # Load extracted data
            try:
                extracted_file = results_path / f"{paper_name}_v5_extracted.json"
                with open(extracted_file) as f:
                    ext_data = json.load(f)
                    extracted_measurements = ext_data.get("measurements", [])
            except Exception as e:
                print(f"⚠️  Skipping {paper_name}: {e}")
                continue
            
            # Load PDF
            pdf_path = Path(PDF_DIR) / f"{paper_name}.pdf"
            print(f"   📄 PDF path: {pdf_path}")
            if not pdf_path.exists():
                print(f"⚠️  PDF not found: {paper_name}")
                continue
            
            # Get ground truth for this paper
            ground_truth = get_ground_truth_for_paper(paper_name, self.results_dir)
            if not ground_truth:
                print(f"⚠️  No ground truth found: {paper_name}")
                continue
            
            # Analyze holistically
            analysis = await self.analyze_paper_holistically(
                paper_name, 
                ground_truth,
                extracted_measurements, 
                pdf_path
            )
            
            # save to the file;
            # save to the file;
            output_path = Path(RESULTS_DIR) / f"{paper_name}_extraction_evaluation.txt"
            with open(output_path, "w") as f:
                f.write(json.dumps(analysis, indent=2) if analysis else "{}")
            print(f">>> WROTE TO FILE {output_path}")
            if analysis:
                all_analyses.append(analysis)
        
        if not all_analyses:
            print("❌ No analyses completed. Exiting.")
            return
        
        # 3. Aggregate findings
        print("\n📊 Aggregating findings across papers...")
        aggregated = self.aggregate_analyses(all_analyses)
        
        print(f"\n{'='*60}")
        print(f"AGGREGATED PATTERNS ({len(aggregated['pattern_groups'])} unique):")
        print(f"{'='*60}")
        for key, patterns in list(aggregated['pattern_groups'].items())[:5]:
            print(f"\n{patterns[0]['pattern']}")
            print(f"  Prevalence: {patterns[0].get('prevalence', 'unknown')}")
            print(f"  Occurrences: {len(patterns)} paper(s)")
        
        print(f"\n{'='*60}")
        print(f"GENERIC SUGGESTIONS ({len(aggregated['unique_suggestions'])}):")
        print(f"{'='*60}")
        for i, suggestion in enumerate(aggregated['unique_suggestions'], 1):
            print(f"{i}. {suggestion}")
        
        # 4. Rewrite extraction script
        if not aggregated['unique_suggestions']:
            print("\n⚠️  No suggestions generated. Skipping code rewrite.")
            return
        
        print("\n✍️  Generating improved extraction script...")
        
        improvements_text = "\n\n".join([
            f"**Pattern {i}**: {list(aggregated['pattern_groups'].values())[i-1][0]['pattern']}\n"
            f"**Solution**: {sugg}"
            for i, sugg in enumerate(aggregated['unique_suggestions'][:10], 1)
        ])
        
        try:
            current_code = Path(TARGET_SCRIPT).read_text()
            
            prompt = FRAMEWORK_IMPROVER_PROMPT.format(
                aggregated_improvements=improvements_text
            )
            
            response = await self.client.aio.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[
                    prompt,
                    f"\n\n**CURRENT CODE**:\n```python\n{current_code}\n```"
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="text/x-python",
                    temperature=1.0
                )
            )
            
            new_code = response.text
            
            # Validation
            if "def main" in new_code and "import" in new_code:
                output_path = Path(TARGET_SCRIPT).with_suffix('.improved.py')
                output_path.write_text(new_code)
                print(f"✅ Improved script written to: {output_path}")
            else:
                print("❌ Generated code looks invalid. Not writing.")
                print(f"Preview:\n{new_code[:500]}...")
                
        except Exception as e:
            print(f"❌ Code generation failed: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Holistic Extraction Optimizer")
    parser.add_argument("--max-papers", type=int, default=10, help="Max papers to analyze")
    args = parser.parse_args()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Missing GEMINI_API_KEY environment variable")
        return
    
    optimizer = ExtractionOptimizer(api_key)
    await optimizer.run(max_papers=args.max_papers)

if __name__ == "__main__":
    asyncio.run(main())