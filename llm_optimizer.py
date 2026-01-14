import asyncio
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY")
EVAL_MODEL = "gemini-2.0-flash-exp"  # Or your preferred model
MAX_CONCURRENCY = 10

# --- SCHEMAS ---

# 1. The "Judge" Schema (Is it a match?)
class MatchDecision(BaseModel):
    match_found: bool = Field(..., description="True if the candidate is chemically identical to Ground Truth.")
    matched_index: Optional[int] = Field(None, description="The '_index' of the matching candidate.")
    reasoning: str = Field(..., description="Why it matches (e.g. 'Li26... is 4x unit cell of Li6.5...').")

# 2. The "Forensic" Schema (Why did it fail?)
class ForensicAnalysis(BaseModel):
    failure_type: str = Field(..., description="MISSING_DATA, UNIT_ERROR, STOICHIOMETRY_ERROR, or HALLUCINATION")
    explanation: str = Field(..., description="Detailed explanation of the discrepancy.")
    correction_instruction: str = Field(..., description="A specific instruction for the LLM to fix this in the future.")

# --- THE UNIFIED EVALUATOR ---

class UnifiedEvaluator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _safe_llm_call(self, prompt, schema_cls):
        """Generic safe executor for both evaluation stages"""
        async with self.semaphore:
            try:
                loop = asyncio.get_running_loop()
                # Run the blocking API call in a thread
                response = await loop.run_in_executor(None, lambda: self.client.models.generate_content(
                    model=EVAL_MODEL,
                    contents=[types.Content(parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema_cls.model_json_schema()
                    )
                ))
                return schema_cls.model_validate_json(response.text)
            except Exception as e:
                # print(f"LLM Error: {e}") # specific error logging can go here
                return None

    def _get_candidates(self, gt_cond, materials_list, mode="strict"):
        """
        Unified filter.
        mode='strict': Value must be within log10(0.3) (~2x). Used for benchmarking.
        mode='relaxed': Value within 10x OR name similarity. Used for forensics.
        """
        candidates = []
        try:
            gt_val = float(gt_cond)
            gt_log = np.log10(gt_val)
        except:
            return []

        for mat in materials_list:
            try:
                ext_val = mat.get('_norm_cond')
                # Strict Mode (Benchmarking)
                if mode == "strict":
                    if ext_val and abs(gt_log - np.log10(float(ext_val))) <= 0.3:
                        candidates.append(mat)
                
                # Relaxed Mode (Forensics)
                elif mode == "relaxed":
                    # Condition A: Value is vaguely close (0.1x to 10x)
                    val_close = False
                    if ext_val:
                        val_close = abs(gt_log - np.log10(float(ext_val))) <= 1.0
                    
                    # Condition B: Name contains key elements
                    name_close = False
                    # (Simple heuristic: check if first 4 chars of GT composition exist in extracted name)
                    # You can make this smarter later
                    if mat.get('canonical_formula') and str(gt_cond)[:3] in str(mat.get('canonical_formula')): 
                        name_close = True
                    
                    if val_close or name_close:
                        candidates.append(mat)
            except:
                continue
        return candidates

    async def _process_row(self, gt_row, all_materials):
        """
        The Core Logic:
        1. Try Strict Match.
        2. If Found -> Return Success.
        3. If Missing -> Run Forensics -> Return Diagnostic Data.
        """
        gt_id = gt_row.get('ID', 'unknown')
        gt_comp = gt_row['Composition']
        gt_cond = gt_row['Ionic conductivity (S cm-1)']

        # --- PHASE 1: BENCHMARKING (Strict) ---
        strict_candidates = self._get_candidates(gt_cond, all_materials, mode="strict")
        
        # Only call LLM if we have value-matched candidates
        if strict_candidates:
            # Format Prompt
            cand_str = "\n".join([
                f"ID {c.get('_index')}: {c.get('canonical_formula')} ({c.get('_norm_cond')} S/cm)" 
                for c in strict_candidates
            ])
            
            prompt = f"""
            Compare Ground Truth vs Extracted Candidates.
            GT: {gt_comp}, {gt_cond} S/cm
            Candidates:
            {cand_str}
            
            Return JSON indicating if a chemical match exists (handling stoichiometry/doping).
            """
            
            decision = await self._safe_llm_call(prompt, MatchDecision)

            if decision and decision.match_found:
                # SUCCESS CASE
                return {
                    "gt_id": gt_id,
                    "status": "FOUND",
                    "match_index": decision.matched_index,
                    "reason": decision.reasoning,
                    "optimization_data": None # No optimization needed
                }

        # --- PHASE 2: FORENSICS (Diagnosis) ---
        # If we reached here, we missed the match. Let's find out why.
        
        # Get broader candidates (blind spots, unit errors, etc.)
        relaxed_candidates = self._get_candidates(gt_cond, all_materials, mode="relaxed")
        
        if not relaxed_candidates:
             return {
                "gt_id": gt_id,
                "status": "MISSING_NO_CANDIDATE",
                "reason": "Extraction pipeline completely missed this section of the paper.",
                "optimization_data": {
                    "type": "RECALL_FAILURE",
                    "instruction": "Improve Table/Figure parsing coverage. Data point completely absent."
                }
            }
        
        # Pick the "closest" failure to analyze
        # Priority: Closest Value -> First in list
        best_failure = min(relaxed_candidates, key=lambda x: abs(np.log10(x.get('_norm_cond', 1e-9)) - np.log10(float(gt_cond))))
        
        forensic_prompt = f"""
        You are diagnosing an AI Extraction failure.
        
        GROUND TRUTH: {gt_comp}, {gt_cond} S/cm
        
        NEAREST EXTRACTED CANDIDATE (FAILED MATCH):
        - Formula: {best_failure.get('canonical_formula')}
        - Value: {best_failure.get('_norm_cond')} S/cm
        - Raw Text: "{best_failure.get('material_description', '')}"
        
        Analyze why this was not a strict match. 
        - Is it a Unit Error? (e.g. 10^-3 vs 10^-6)
        - Is it a Stoichiometry drift? (e.g. x=0.1 vs x=0.2)
        - Is it a Hallucination?
        
        Provide a specific 'correction_instruction' for the extractor.
        """
        
        analysis = await self._safe_llm_call(forensic_prompt, ForensicAnalysis)
        
        if not analysis:
            return {"gt_id": gt_id, "status": "ERROR_DURING_FORENSICS", "optimization_data": None}

        # FAILURE CASE WITH FEEDBACK
        return {
            "gt_id": gt_id,
            "status": "MISSING",
            "reason": analysis.explanation,
            "optimization_data": {
                "gt_context": f"{gt_comp} @ {gt_cond}",
                "extracted_context": f"{best_failure.get('canonical_formula')} @ {best_failure.get('_norm_cond')}",
                "error_type": analysis.failure_type,
                "instruction": analysis.correction_instruction
            }
        }

    async def run_pipeline(self, gt_csv, ext_json, target_doi=None):
        # Load Data
        df = pd.read_csv(gt_csv)
        if target_doi:
            df = df[df['DOI'] == target_doi]
        with open(ext_json) as f:
            materials = json.load(f).get('materials', [])
            
        print(f"Processing {len(df)} Ground Truth points...")
        
        tasks = [self._process_row(row, materials) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        
        # --- AGGREGATION ---
        
        # 1. Benchmark Report (CSV)
        report_df = pd.DataFrame([{
            k: v for k, v in r.items() if k != 'optimization_data'
        } for r in results])
        report_df.to_csv("benchmark_report.csv", index=False)
        
        # 2. Optimization Feed (JSON) - ONLY the useful failures
        opt_feed = [r['optimization_data'] for r in results if r.get('optimization_data')]
        
        with open("optimization_feed.json", "w") as f:
            json.dump(opt_feed, f, indent=2)
            
        print(f"\nDone.")
        print(f"Matched: {len(report_df[report_df['status']=='FOUND'])} / {len(df)}")
        print(f"Optimization examples generated: {len(opt_feed)}")

# --- USAGE ---
if __name__ == "__main__":
    evaluator = UnifiedEvaluator(API_KEY)
    asyncio.run(evaluator.run_pipeline(
        gt_csv="OBELiX/data/processed.csv", 
        ext_json="results/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites_structure_materials.json",
        target_doi="10.1039/c7ta08581h"
    ))