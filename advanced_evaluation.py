# Evaluates the advanced pipeline
import argparse
import json
import os
import time
import pandas as pd
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 1. The Schema for the Judge
# ==============================================================================
class MatchResult(BaseModel):
    ground_truth_id: str = Field(..., description="The ID or Index from the original data.")
    extracted_match: str = Field(..., description="The matching extracted composition/value, or 'MISSING'.")
    reason: str = Field(..., description="One sentence explanation. E.g. 'Li24... simplifies to Li6... and 0.00277 S/cm is approx 2.8 mS/cm'.")
    thinking: str = Field(..., description="Your thinking process. 2-3 sentences.")
    is_match: bool = Field(..., description="True if the extracted data represents the same physical reality as the ground truth.")

class BenchmarkReport(BaseModel):
    matches: List[MatchResult]
    total_ground_truth: int
    total_matches: int

# ==============================================================================
# 2. The "Judge" Prompt
# ==============================================================================
JUDGE_PROMPT = """
You are a strict Scientific Data Auditor. 
Compare the "Ground Truth" data (from a reliable database) against the "Extracted Data" (from an AI pipeline).

Your Goal: Determine if the Extracted Data correctly captures the Ground Truth.

RULES FOR MATCHING:
1. Chemical Formulas: Normalize stoichiometries. 
   - Example: "Li24P4S20Br4" is chemically identical to "Li6PS5Br" (divide by 4).
   - Example: "Li7La3Zr2O12" is the same as "LLZO".
   
2. Units & Values: Check for unit conversions and allow for approximate matches.
   - Example: "0.00277 S/cm" is approx "2.8 mS/cm".
   - Allow for rounding differences (e.g. 0.00277 vs 0.0028 or 3.64mS/cm vs 3.7mS/cm).
   
3. Fuzzy Logic:
   - If the extraction says "RT" or "Room Temp" and Ground Truth is missing temp or implies 25C, that is acceptable.

INPUT DATA:
Ground Truth:
{ground_truth}

Extracted Data:
{extracted_data}

TASK:
For EVERY row in Ground Truth, find the corresponding row in Extracted Data.
Report if it was found and if the values match.
"""

def main():
    parser = argparse.ArgumentParser(description='Benchmark Validator: LLM-as-a-Judge')
    parser.add_argument('--ground-truth', '-gt', required=True, help='Path to Ground Truth CSV')
    parser.add_argument('--doi', '-i', required=True, help='DOI of the paper')
    parser.add_argument('--extracted', '-ex', required=True, help='Path to Extracted Results JSON')
    parser.add_argument('--model', default='gemini-2.5-flash', help='Model for judging')
    parser.add_argument('--output', '-o', default='validation_report.json', help='Output report path')
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    # use DOI to retrieve the GT
    try:
        df = pd.read_csv(args.ground_truth)
        df = df[df['DOI'] == args.doi]
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    gt_str = "\n".join([f"GT-{idx}: {row['Composition']} | {row['Ionic conductivity (S cm-1)']} S/cm" for idx, row in df.iterrows()])
    print(gt_str)
    try:
        if args.extracted.endswith('.json'):
            ex_str = json.load(open(args.extracted, 'r'))
        elif args.extracted.endswith('.csv'):
            ex_str = open(args.extracted, 'r').read()
            ex_lines = ex_str.split("\n")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    extracted_parent = Path(args.extracted).parent
    print(f"--- Starting Validation ---")
    # print(f"Comparing {len(gt_str)} GT items vs {len(ex_str)} Extracted items...") # actually update this to read the number of lines
    # compare the len of lines not characters
    gt_lines = gt_str.split("\n")
    ex_lines = ex_str['materials']


    ## Format ex_str to be like gt_str
    ex_str_repr = []
    # check for formula/name
    for idx, row in enumerate(ex_str['materials']):
        if 'canonical_formula' in row:
            formula = row['canonical_formula']
        elif 'original_name' in row:
            formula = row['original_name']
        elif 'electrolyte_name' in row:
            formula = row['electrolyte_name']['full_name']
            if row['electrolyte_name']['acronym'] is not None:
                formula += " acronym: " + row['electrolyte_name']['acronym']
            if row['electrolyte_name']['proportion'] is not None:
                formula += " proportion: " + row['electrolyte_name']['proportion']
        else:
            formula = ""

    # check for ionic conductivity
        if '_norm_cond' in row:
            ionic_conductivity = row['_norm_cond']
        elif 'ionic_conductivity_S_per_cm' in row:
            ionic_conductivity = row['ionic_conductivity_S_per_cm']
        else:
            ionic_conductivity = ""

    # check temperature measurement
        if '_norm_temp' in row:
            temperature = row['_norm_temp']
        elif 'measurement_temperature' in row:
            temperature = row['measurement_temperature']
        else:
            temperature = ""
        ex_str_repr.append(f"EX-{idx}: {formula} | {ionic_conductivity} | {temperature} ")
    
    ex_str_repr = "\n".join(ex_str_repr)
    print(ex_str_repr)
    print(f"Comparing {len(gt_lines)} GT items vs {len(ex_lines)} Extracted items...")

    
    formatted_prompt = JUDGE_PROMPT.format(ground_truth=gt_str, extracted_data=ex_str_repr)

    # 3. Call LLM Judge
    try:
        response = client.models.generate_content(
            model=args.model,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=BenchmarkReport.model_json_schema(),
                temperature=0.0
            )
        )

        if response.text:
            report = BenchmarkReport.model_validate_json(response.text)
            
            # 4. Print Summary
            print("\n" + "="*60)
            print(f"VALIDATION SCORE: {report.total_matches / report.total_ground_truth * 100:.1f}%")
            print("="*60)
            print(f"{'GT ID':<10} | {'Match Status':<15} | {'Reason'}")
            print("-" * 60)
            
            with open(args.extracted.replace('.json', '_advanced.txt'), 'w') as f:
                for match in report.matches:
                    status = "✅ MATCH" if match.is_match else "❌ MISSING/BAD"
                    print(f"{match.ground_truth_id[:10]:<10} | {status:<15} | {match.reason}")
                    f.write(f"{match.ground_truth_id[:10]:<10} | {status:<15} | {match.reason}\n")

            # Save full report
            with open(args.extracted.replace('.json', '_advanced.json'), 'w') as f:
                f.write(response.text)
            print(f"\nFull report saved to {args.extracted.replace('.json', '_advanced.json')}")

    except Exception as e:
        print(f"Validation Failed: {e}")

if __name__ == "__main__":
    main()