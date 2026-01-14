import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
load_dotenv()
# --- CONFIGURATION ---
EVAL_MODEL = "gemini-3-flash-preview" 
API_KEY = os.getenv("API_KEY")        
SCIENTIFIC_THINKING_LEVEL = "medium" 

# --- SCHEMAS (Structured Output) ---

class MatchDecision(BaseModel):
    match_found: bool = Field(..., description="True if one of the candidates matches the Ground Truth chemical composition.")
    matched_index: Optional[int] = Field(None, description="The '_index' of the candidate that matches. Null if no match.")
    confidence: str = Field(..., description="high, medium, or low")
    reasoning: str = Field(..., description="Explain why the formula matches (e.g., 'Li26... is simply 4x unit cell of Li6.5...').")
    error_type: Optional[str] = Field(None, description="If no match: 'missing_data', 'value_mismatch', or 'hallucination'.")

# --- THE EVALUATOR CLASS ---

class LLMEvaluator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.semaphore = asyncio.Semaphore(8) # Max concurrent LLM calls
        
        # Reuse your safe executor pattern
        self._global_executor = None 

    async def _safe_llm_call_async(self, func, *args, **kwargs):
        """Your provided safe executor (simplified for this context)"""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=60
            )
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            return None

    def _get_candidates_by_value(self, gt_cond, materials_list, tolerance_log10=0.3):
        """
        Pure Python Shortlister:
        Finds extracted materials with similar conductivity to the Ground Truth.
        tolerance_log10=0.3 means roughly +/- 2x value (e.g. 1.0 vs 2.0 is okay).
        """
        candidates = []
        try:
            gt_val = float(gt_cond)
            gt_log = np.log10(gt_val)
        except:
            return []

        for mat in materials_list:
            try:
                # Assuming your extraction saved normalized values in '_norm_cond'
                ext_val = mat.get('_norm_cond') # normalized conductivity
                if ext_val is None:
                    continue
                
                ext_log = np.log10(float(ext_val))
                
                if abs(gt_log - ext_log) <= tolerance_log10:
                    candidates.append(mat)
            except:
                continue
        return candidates

    async def _judge_single_point(self, gt_row, all_extracted_materials):
        """
        1. Shortlists candidates by value.
        2. Asks LLM to verify chemical composition match.
        """
        gt_id = gt_row.get('ID', 'unknown')
        gt_comp = gt_row['Composition']
        gt_cond = gt_row['Ionic conductivity (S cm-1)']
        
        # 1. Shortlist (The "Value Filter")
        candidates = self._get_candidates_by_value(gt_cond, all_extracted_materials)
        
        if not candidates:
            return {
                "GT_ID": gt_id,
                "Status": "MISSING (No Value Match)",
                "Reason": "No extracted data found with similar conductivity."
            }

        # 2. Build Prompt for the Judge
        candidates_str = ""
        for mat in candidates:
            candidates_str += f"""
            - Candidate ID: {mat.get('_index')}
              Name: {mat.get('electrolyte_name', {}).get('full_name')}
              Formula (Canonical): {mat.get('canonical_formula', 'N/A')}
              Conductivity: {mat.get('_norm_cond')} S/cm
              Source Text: "{mat.get('material_description', '')}"
            """

        prompt = f"""
        Act as a strict Scientific Data Validator. 
        Determine if any of the Extracted Candidates represent the SAME chemical material as the Ground Truth, regardless of stoichiometry scaling or formatting.

        GROUND TRUTH:
        - Composition: {gt_comp}
        - Ionic Conductivity: {gt_cond} S/cm

        EXTRACTED CANDIDATES (Shortlisted by conductivity similarity):
        {candidates_str}

        RULES:
        1. **Stoichiometry Scaling:** Li26P2Si2S20Br4 (Unit Cell Z=4) IS A MATCH for Li6.5P0.5Si0.5S5Br (Formula Unit).
        2. **Formatting:** Li_{{6.5}} is a match for Li6.5.
        3. **Doping:** Verify x-values match if explicit (e.g., x=0.3 must match x=0.3).
        4. If multiple candidates match chemically, pick the one with the closest conductivity.
        
        Return JSON.
        """

        # 3. Call LLM
        async with self.semaphore:
            response = await self._safe_llm_call_async(
                self.client.models.generate_content,
                model=EVAL_MODEL,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=MatchDecision.model_json_schema()
                )
            )

        if not response:
            return {"GT_ID": gt_id, "Status": "ERROR", "Reason": "LLM Timeout"}

        # 4. Parse Result
        try:
            decision = MatchDecision.model_validate_json(response.text)
            
            if decision.match_found:
                # Find the actual candidate object to return details
                matched_cand = next((c for c in candidates if c.get('_index') == decision.matched_index), None)
                return {
                    "GT_ID": gt_id,
                    "Status": "FOUND",
                    "Match_Index": decision.matched_index,
                    "Ext_Comp": matched_cand.get('canonical_formula') if matched_cand else "Unknown",
                    "Ext_Cond": matched_cand.get('_norm_cond') if matched_cand else 0,
                    "LLM_Reasoning": decision.reasoning
                }
            else:
                return {
                    "GT_ID": gt_id,
                    "Status": "MISSING",
                    "Reason": decision.reasoning,
                    "Error_Type": decision.error_type
                }
        except Exception as e:
            return {"GT_ID": gt_id, "Status": "ERROR", "Reason": str(e)}

    async def run_evaluation(self, gt_csv_path, extraction_json_path, target_doi=None):
        """Main entry point"""
        # Load Data
        gt_df = pd.read_csv(gt_csv_path)
        with open(extraction_json_path, 'r') as f:
            ext_data = json.load(f)
            
        all_materials = ext_data.get('materials', [])
        
        # Filter GT to only this DOI (assuming 1 file = 1 DOI for now)
        # You can expand this logic
        if target_doi:
            gt_df = gt_df[gt_df['DOI'] == target_doi] 

        print(gt_df[['Composition', 'Ionic conductivity (S cm-1)']])

        tasks = []
        print(f"Starting Evaluation on {len(gt_df)} Ground Truth Points...")
        
        for _, row in gt_df.iterrows():
            tasks.append(self._judge_single_point(row, all_materials))
            
        results = await asyncio.gather(*tasks)
        
        # Stats
        found = sum(1 for r in results if r['Status'] == 'FOUND')
        total = len(results)
        print(f"\nEvaluation Complete: {found}/{total} Matched ({found/total:.1%})")
        
        # Save Report
        pd.DataFrame(results).to_csv("llm_evaluation_report.csv", index=False)
        print("Detailed report saved to llm_evaluation_report.csv")

# --- USAGE ---
if __name__ == "__main__":
    evaluator = LLMEvaluator(api_key=API_KEY)
    asyncio.run(evaluator.run_evaluation(
        gt_csv_path="OBELiX/data/processed.csv", 
        extraction_json_path="results/Effect of Si substitution on the structural and transport properties of superionic Li-argyrodites_structure_materials.json",
        target_doi="10.1039/c7ta08581h"
    ))