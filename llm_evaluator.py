import asyncio
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

# --- CONFIGURATION ---
EVAL_MODEL = "gemini-2.5-pro"
API_KEY = os.getenv("API_KEY")

# --- SCHEMAS ---
## --- SIMPLIFIED SCHEMAS ---
class MaterialMatch(BaseModel):
    gt_index: int = Field(..., description="Index from ground truth")

    # forcing model to reason before answering
    reasoning: str = Field(..., description="Step-by-step logic. 1. Normalize GT composition. 2. Normalize Extracted composition. 3. Compare. 4. Check conductivity % diff.")

    extracted_index: Optional[int] = Field(None, description="Index in extracted list (0-N), null if no match")
    confidence: str = Field(..., description="high, medium, or low")
    
    # The verdict comes last, after the reasoning is complete.
    is_match: bool = Field(..., description="True only if composition is chemically equivalent AND conductivity is within tolerance.")

class LLMOutput(BaseModel):
    matches: List[MaterialMatch]
    summary: str

class EvaluationReport(BaseModel):
    """Full report including computed statistics"""
    matches: List[MaterialMatch]
    summary: str
    total_ground_truth: int
    total_extracted: int
    correctly_extracted: int
    missing: int
    accuracy_percent: float

# --- Physical Constraints ---
class ConductivityValidator:
    """Validate and compare conductivity values"""
    
    @staticmethod
    def normalize_conductivity(value: float, unit: str) -> Optional[float]:
        """Convert all conductivity to S/cm"""
        if value is None:
            return None
        
        unit = unit.strip().lower()
        
        # Already in S/cm or S cm-1
        if unit in ['s/cm', 's cm-1', 's/cm']:
            return value
        
        # mS/cm to S/cm
        if unit in ['ms/cm', 'ms cm-1']:
            return value / 1000.0
        
        # S/m to S/cm
        if unit in ['s/m', 's m-1']:
            return value / 100.0
        
        # Invalid units - likely not conductivity
        if any(x in unit for x in ['ev', 'hz', 's-1', 's⁻¹', 'ln(', 'log']):
            return None
        
        return None  # Unknown unit
    
    @staticmethod
    def conductivity_match(cond1: float, cond2: float, max_ratio: float = 3.0) -> Tuple[bool, float]:
        """
        Check if conductivities match within tolerance
        Returns (is_match, ratio)
        """
        if cond1 is None or cond2 is None or cond1 <= 0 or cond2 <= 0:
            return False, float('inf')
        
        ratio = max(cond1, cond2) / min(cond1, cond2)
        return ratio <= max_ratio, ratio


# --- THE PURE LLM EVALUATOR ---

class PureLLMEvaluator:
    def __init__(self, api_key: str, model: str = EVAL_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(5)
        self.cond_validator = ConductivityValidator()

    async def _safe_llm_call(self, prompt: str, schema: dict):
        """Execute LLM call with error handling"""
        async with self.semaphore:
            try:
                loop = asyncio.get_running_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.model,
                            contents=[types.Content(parts=[types.Part(text=prompt)])],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_json_schema=schema,
                                temperature=0.0  # Deterministic matching
                            )
                        )
                    ),
                    timeout=120
                )
                return response.text if response and hasattr(response, 'text') else None
            except Exception as e:
                print(f"LLM call failed: {e}")
                return None

    def _format_ground_truth(self, gt_data: pd.DataFrame) -> str:
        """Format ground truth data for prompt"""
        entries = []
        for idx, row in gt_data.iterrows():
            comp = row.get('Composition', 'N/A')
            cond = row.get('Ionic conductivity (S cm-1)', 'N/A')
            entries.append(f"GT-{idx}: {comp} | {cond} S/cm")
        return "\n".join(entries)

    def sanitize_measurements(self, measurements: list) -> list:
        valid_data = []

        # Define valid conductivity units
        valid_units = ['S/cm', 'S cm-1', 'S/cm2', 'S cm-2', 'S/m', 'S m-1', 'mS cm-1', 'mS/cm']

        for entry in measurements:
            unit = entry.get('raw_conductivity_unit', '').strip()
            temp = entry.get('normalized_temperature_c')

            if unit not in valid_units or temp is None:
                if any(x in unit for x in ['eV', 'Hz', '%', 's-1', 'arbitrary']):
                    continue

            if temp is not None:
                if temp < 20 or temp > 30:
                    continue

            # 3. LABEL CHECK: Discard explicit "Activation Energy" entries
            # (Sometimes the unit is missing but the label is clear)
            label = entry.get('raw_composition', '').lower()
            if 'activation' in label or 'disorder' in label or 'ea' in label:
                continue

            valid_data.append(entry)

        return valid_data

    def _format_extracted(self, extracted_data: list) -> str:
        """Format extracted data for prompt"""
        entries = []
        valid_data = self.sanitize_measurements(extracted_data)
        for idx, material in enumerate(valid_data):

            # 1. Filter out invalid entries BEFORE formatting
            
            
            # Handle different possible field names in your extraction
            # 1. Try to find the Name
            name = material.get('canonical_formula') or \
                   material.get('raw_composition') or \
                   "Unknown_Material"
            
            # 2. Try to find the Conductivity (Numeric)
            # Checking the keys from your actual JSON output
            cond = material.get('normalized_conductivity') or \
                   material.get('raw_conductivity') or \
                   "N/A"
            # 3. Try to find Temperature
            temp = material.get('normalized_temperature_c') or \
                   material.get('raw_temperature') or \
                   "N/A"
            
            # 4. Units check (helper for the LLM)
            unit = material.get('raw_conductivity_unit', '')

            entries.append(f"EXT-{idx}: {name} | {cond} ({unit}) @ {temp}C")
        entries = "\n".join(entries)
        # print("ENTRIES ", entries)
        # print('--------------\n')
        return entries

    async def evaluate_document(
        self,
        ground_truth: pd.DataFrame,
        extracted_data: list,
        context: str = ""
    ) -> EvaluationReport:
        """Pure LLM-based evaluation"""
        
        gt_formatted = self._format_ground_truth(ground_truth)
        ext_formatted = self._format_extracted(extracted_data)
        
        prompt = f"""Your task is to compare and match the Ground Truth (GT) against the Extracted Data (EXT).

        CONTEXT: {context}

        --- GROUND TRUTH ---
{gt_formatted}

        --- EXTRACTED DATA ---
{ext_formatted}

        --- RULES ---
        1. STOICHIOMETRY: Li24P4S20I4 is IDENTICAL to Li6PS5I (Divide by 4). Li6.6P0.4Ge0.6S5I is the same as Li26.4P1.6Ge2.4S20I4.
        2. UNITS: 18.4 mS/cm = 0.0184 S/cm. 1.3e-6 S/cm = 0.0000013 S/cm.
        3. IGNORE LABELS: "Cold-pressed" or "Sintered" in the name does not break the composition match, but prefer the entry with the closest conductivity.
        """

        print("Evaluation Prompt: ", prompt)

        response_text = await self._safe_llm_call(
            prompt,
            LLMOutput.model_json_schema()
        )
        
        if not response_text:
            # try again
            response_text = await self._safe_llm_call(
                prompt,
                LLMOutput.model_json_schema()
            )
        
        try:
            llm_result = LLMOutput.model_validate_json(response_text)

            # Perform calculations
            matches_list = llm_result.matches
            total_gt = len(ground_truth)
            total_ext = len(ext_formatted.split("\n"))
            correct_count = sum(1 for m in matches_list if m.is_match)
            missing_count = total_gt - correct_count
            accuracy = (correct_count / total_gt) * 100
            


            # Validate the math is correct
            return EvaluationReport(
                matches=matches_list,
                summary=llm_result.summary,
                total_ground_truth=total_gt,
                total_extracted=total_ext,
                correctly_extracted=correct_count,
                missing=missing_count,
                accuracy_percent=accuracy
            )
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response_text[:500]}")
            return EvaluationReport(
                matches=[],
                total_ground_truth=len(ground_truth),
                total_extracted=len(ext_formatted.split("\n")),
                correctly_extracted=0,
                missing=len(ground_truth),
                accuracy_percent=0.0,
                summary=f"Parsing failed: {str(e)}"
            )

    async def evaluate_multiple_documents(
        self,
        evaluations: List[dict]
    ) -> pd.DataFrame:
        """Evaluate multiple documents in parallel"""
        tasks = []
        for eval_spec in evaluations:
            task = self.evaluate_document(
                ground_truth=eval_spec['ground_truth'],
                extracted_data=eval_spec['extracted'],
                context=eval_spec.get('context', f"DOI: {eval_spec.get('doc_id', 'Unknown')}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        report_rows = []
        for eval_spec, result in zip(evaluations, results):
            doc_id = eval_spec.get('doc_id', 'Unknown')
            report_rows.append({
                'Document_ID': doc_id,
                'Total_Ground_Truth': result.total_ground_truth,
                'Total_Extracted': result.total_extracted,
                'Correctly_Extracted': result.correctly_extracted,
                'Missing': result.missing,
                'Accuracy_%': result.accuracy_percent,
                'Summary': result.summary
            })
            
            # Save detailed matches
            matches_df = pd.DataFrame([m.model_dump() for m in result.matches])
            safe_id = doc_id.replace('/', '_').replace('.', '_')
            matches_df.to_csv(f"eval_details_{safe_id}.csv", index=False)
            print(f"✓ Saved details for {doc_id}")
        
        summary_df = pd.DataFrame(report_rows)
        return summary_df


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pure LLM evaluation of material extraction')
    parser.add_argument('--extraction', '-e', required=True, help='Extracted materials JSON')
    parser.add_argument('--dataset', '-d', required=True, help='Ground truth CSV')
    parser.add_argument('--doi', '-i', required=True, help='DOI of the paper')
    parser.add_argument('--output', '-o', default='evaluation_report.csv', help='Output path')
    
    args = parser.parse_args()
    
    # Load data
    gt_df = pd.read_csv(args.dataset)
    gt_df = gt_df[gt_df['DOI'] == args.doi]
    
    with open(args.extraction, 'r') as f:
        extraction = json.load(f)

    
    
    extracted_materials = extraction.get('measurements', [])
    
    print(f"\n📊 Starting evaluation for {args.doi}")
    print(f"   Ground truth entries: {len(gt_df)}")
    print(f"   Extracted entries: {len(extracted_materials)}")
    
    # Run evaluation
    evaluator = PureLLMEvaluator(api_key=API_KEY)
    result = await evaluator.evaluate_document(
        ground_truth=gt_df,
        extracted_data=extracted_materials,
        context=f"DOI: {args.doi}"
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {args.doi}")
    print(f"{'='*70}")
    print(f"Ground Truth Entries:    {result.total_ground_truth}")
    print(f"Extracted Entries:       {result.total_extracted}")
    print(f"✓ Correctly Extracted:   {result.correctly_extracted}")
    print(f"✗ Missing:               {result.missing}")
    print(f"📈 Accuracy:             {result.accuracy_percent:.1f}%")
    print(f"\n{result.summary}")
    print(f"{'='*70}\n")
    
    # Save detailed results
    matches_df = pd.DataFrame([m.model_dump() for m in result.matches])
    matches_df.to_csv(args.output, index=False)
    print(f"💾 Detailed results saved to: {args.output}\n")
    
    # Show all matches with better formatting
    print("DETAILED MATCHES:")
    print("-" * 70)
    for m in result.matches:
        status = "✅ MATCH" if m.is_match else "❌ MISS"
        
        # --- 1. LOOKUP GROUND TRUTH DATA ---
        # Use .loc[m.gt_index] to find the row in the pandas DataFrame
        # We wrap in try/except just in case the LLM hallucinates an index that doesn't exist
        try:
            gt_row = gt_df.loc[m.gt_index] 
            gt_comp = gt_row.get('Composition', 'N/A')
            gt_cond = gt_row.get('Ionic conductivity (S cm-1)', 'N/A')
        except KeyError:
            gt_comp = "Unknown GT Index"
            gt_cond = 0.0

        # Print GT line using the LOOKED UP variables, not 'm.gt_composition'
        print(f"{status} | GT-{m.gt_index}: {str(gt_comp)[:30]:30s} ({gt_cond})")

        # --- 2. LOOKUP EXTRACTED DATA ---
        if m.is_match and m.extracted_index is not None:
            try:
                # Use list indexing to find the extracted object
                ext_item = extracted_materials[m.extracted_index]
                
                # Resolve the name (handling different possible keys)
                ext_comp = ext_item.get('canonical_formula') or ext_item.get('raw_composition') or "Unknown"
                
                # Resolve the conductivity
                ext_cond = ext_item.get('normalized_conductivity')
                if ext_cond is None:
                    ext_cond = ext_item.get('raw_conductivity', 'N/A')

                # Print EXT line using LOOKED UP variables
                print(f"       → EXT-{m.extracted_index}: {str(ext_comp)[:30]:30s} ({ext_cond})")
                
            except IndexError:
                print(f"       → EXT-{m.extracted_index}: <Invalid Extraction Index>")

        # Print reasoning from the LLM object
        print(f"       Reasoning: {m.reasoning[:20]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())