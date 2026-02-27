#!/usr/bin/env python3
"""
Tier-Based Analysis Pipeline for Solid-State Electrolytes
=========================================================
filters data, calculates Effective Medium Theory (EMT) baselines,
classifies materials into Tiers 1-3, and generates design rules.

Tier 1: Enhanced conductivity (σ > σ_EMT)
Tier 2: Rule-of-mixtures behavior (σ_pure < σ <= σ_EMT) 
Tier 3: Detrimental mixing (σ < σ_pure)

Usage:
    mamba activate pokeagent
    python tier_analysis_pipeline.py --input-dir output_parsed
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dotenv import load_dotenv

# Pydantic & Gemini
from pydantic import BaseModel, Field
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model constants
REPORT_MODEL = "gemini-3.1-pro-preview"

# ============================================================================
# Reference Conductivity Lookup Table
# ============================================================================
# Literature reference conductivities at ~25°C (S/cm).
# These are fallbacks; data-driven values from the dataset override when available.

REFERENCE_CONDUCTIVITIES = {
    # Polymers (polymer+salt baseline for EMT — values represent typical polymer+salt conductivity)
    "PEO":  {"rt": 1e-6, "60c": 5e-4,
             "keywords": ["peo", "polyethylene oxide"],
             "exclude":  ["llzo", "latp", "lagp", "al2o3", "sio2", "filler", "composite", "wt%", "vol%"]},
    "PVDF": {"rt": 1e-7, "60c": 1e-5,
             "keywords": ["pvdf", "polyvinylidene"],
             "exclude":  ["llzo", "latp", "lagp", "filler", "composite", "wt%", "vol%"]},
    "PAN":  {"rt": 1e-8, "60c": 1e-5,
             "keywords": ["pan", "polyacrylonitrile"],
             "exclude":  ["llzo", "latp", "lagp", "filler", "composite", "wt%", "vol%"]},
    "PPC":  {"rt": 1e-7, "60c": 1e-5,
             "keywords": ["ppc", "polypropylene carbonate"],
             "exclude":  ["llzo", "latp", "lagp", "filler", "composite", "wt%", "vol%"]},
    "PMMA": {"rt": 1e-8, "60c": 1e-6,
             "keywords": ["pmma", "polymethyl methacrylate"],
             "exclude":  ["llzo", "latp", "lagp", "filler", "composite", "wt%", "vol%"]},
    # Ceramic fillers
    "LLZO":    {"rt": 3e-4,
                "keywords": ["llzo", "li7la3zr2o12", "garnet"],
                "exclude":  ["peo", "pan", "polymer", "pvdf", "wt%"]},
    "NASICON": {"rt": 1e-4,
                "keywords": ["nasicon", "latp", "lagp"],
                "exclude":  ["peo", "pan", "polymer", "pvdf", "wt%"]},
    "Sulfide": {"rt": 1e-3,
                "keywords": ["sulfide", "li6ps5", "lgps", "ps4"],
                "exclude":  ["peo", "pan", "polymer", "pvdf", "wt%"]},
    "Oxide":   {"rt": 1e-12,
                "keywords": ["al2o3", "sio2", "tio2"],
                "exclude":  ["peo", "pan", "polymer", "pvdf", "wt%"]},
}

# ============================================================================
# Pydantic Schemas
# ============================================================================

class DesignRule(BaseModel):
    rule_id: int = Field(..., description="Sequential rule number")
    title: str = Field(..., description="Short title for the rule")
    description: str = Field(..., description="Detailed explanation with mechanistic reasoning")
    tier_focus: str = Field(..., description="Which tier does this rule primarily address? (e.g. 'Tier 1 enhancement')")
    supporting_evidence: List[str] = Field(..., description="Specific data points supporting this rule")
    confidence_level: str = Field(..., description="'high', 'medium', or 'speculative'")
    actionable_recommendation: str = Field(..., description="Concrete recommendation for experimentalists")

class TierAnalysisReport(BaseModel):
    executive_summary: str = Field(..., description="High-level summary of findings")
    tier_1_analysis: str = Field(..., description="What drives Tier 1 performance? Common features?")
    tier_2_3_analysis: str = Field(..., description="Common pitfalls leading to Tier 2/3 performance")
    design_rules: List[DesignRule] = Field(..., description="Actionable design rules")
    suggested_experiments: List[str] = Field(..., description="Suggested experiments to achieve Tier 1")
    knowledge_gaps: List[str] = Field(..., description="Missing data or unexplored areas")

# ============================================================================
# Part 1: Data Consolidation & Filtering
# ============================================================================

def load_and_consolidate(input_dir: Path) -> pd.DataFrame:
    """Recursively load all *_extracted.json files and consolidate into a DataFrame."""
    try:
        json_files = list(input_dir.glob("**/*_mapped.json"))
    except:
        json_files = list(input_dir.glob("**/*_extracted.json"))

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"files found: {len(json_files)}")
    
    all_rows = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            doc_name = data.get("doc_name", jf.stem)
            mapping_meta = data.get("mapping_metadata", {})
            mapping_model = mapping_meta.get("mapping_model", "none")
            
            for m in data.get("measurements", []):
                # Basic validation
                cond = m.get("normalized_conductivity")
                temp = m.get("normalized_temperature_c")
                
                if cond is None or temp is None:
                    continue
                    
                row = {
                    "doc_name": doc_name,
                    "file_path": str(jf),
                    "raw_composition": m.get("raw_composition", ""),
                    "canonical_formula": m.get("canonical_formula", ""),
                    "conductivity": float(cond),
                    "temperature": float(temp),
                    "processing_method": m.get("processing_method", "not reported"),
                    "processing_method_detail": m.get("processing_method_detail", ""),
                    "mapping_model": mapping_model,
                    "source": m.get("source", "unknown"),
                    "confidence": m.get("confidence", "low"),
                    "material_definitions": "; ".join(m.get("material_definitions", [])),
                    "warnings": "; ".join(m.get("warnings", [])),
                }
                all_rows.append(row)
        except Exception as e:
            logger.warning(f"Failed to process {jf}: {e}")

    df = pd.DataFrame(all_rows)
    logger.info(f"Loaded {len(df)} measurements")
    
    # 1. Filter out duplicates (identical comp, temp, cond from same paper)
    df = df.drop_duplicates(subset=["doc_name", "raw_composition", "temperature", "conductivity"])
    
    # 2. Filter cited data unless it has complete context
    # Heuristic: Cited data is useful only if we trust it. 
    # For now, we will be strict: exclude cited data to avoid noise, as requested by user.
    # "some of the information have a source that says cited_xxxx that means that this is not the data from the original paper. we would like to filter out these if not enough information is available."
    # We'll mark them, then strictly filter for the analysis.
    df["is_cited"] = df["source"].astype(str).str.lower().str.startswith("cited")
    
    # 3. Clean numeric data
    # df = df[df["conductivity"] > 0]
    df = df[df["conductivity"] <= 1.0] # Remove suspiciously high (>1 S/cm)
    
    # 4. Filter for Room Temperature (15-30C)
    df = df[df["temperature"].between(15, 30)]
    logger.info(f"Filtered to {len(df)} measurements at Room Temperature (15-30°C)")

    return df

# ============================================================================
# Part 2: Parsing & EMT Support
# ============================================================================

def parse_composition_info(row: pd.Series) -> Dict[str, Any]:
    """
    Parse raw_composition to identify:
    - Is it a composite?
    - Polymer matrix
    - Filler material
    - Filler fraction (wt% or vol%)
    """
    raw = str(row.get("raw_composition", "")).lower()
    defs = str(row.get("material_definitions", "")).lower()
    combined = raw + " " + defs
    
    info = {
        "is_composite": False,
        "polymer": None,
        "all_polymers": [],
        "is_blend": False,
        "filler": None,
        "filler_fraction": 0.0,
        "fraction_unit": None, # 'wt%' or 'vol%'
        "fraction_assumed": False,
        "material_family": "Other"
    }
    
    # Keywords for material families
    if any(k in combined for k in ["llzo", "garnet", "li7la3", "la3zr2"]):
        info["material_family"] = "Garnet"
        info["filler"] = "LLZO"
    elif any(k in combined for k in ["nasicon", "latp", "lagp", "liti2"]):
        info["material_family"] = "NASICON"
        info["filler"] = "NASICON"
    elif any(k in combined for k in ["sulfide", "ps4", "li6ps5", "lgps"]):
        info["material_family"] = "Sulfide"
        info["filler"] = "Sulfide"
    elif any(k in combined for k in ["oxide", "al2o3", "sio2", "tio2"]):
        info["material_family"] = "Oxide Filler"
        info["filler"] = "Oxide"
    
    # Polymer detection
    polymers = {
        "peo": "PEO", "pan": "PAN", "pvdf": "PVDF", "ppc": "PPC", 
        "pmma": "PMMA", "pvc": "PVC"
    }
    found_polymers = [name for k, name in polymers.items() if k in combined]
    
    if found_polymers:
        info["all_polymers"] = found_polymers
        info["polymer"] = found_polymers[0]  # Primary match for EMT
        info["is_blend"] = len(found_polymers) > 1
        if info["material_family"] == "Other":
            info["material_family"] = "Polymer"
            
    # Composite detection
    # Look for patterns like "10 wt%" or "5 vol%"
    wt_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:wt%|wt\s|weight%)", raw)
    vol_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:vol%|vol\s|volume%)", raw)
    
    if wt_match:
        info["is_composite"] = True
        info["filler_fraction"] = float(wt_match.group(1)) / 100.0
        info["fraction_unit"] = "wt%"
    elif vol_match:
        info["is_composite"] = True
        info["filler_fraction"] = float(vol_match.group(1)) / 100.0
        info["fraction_unit"] = "vol%"
    elif info["polymer"] and info["filler"] and info["material_family"] != "Polymer":
        # It has both polymer and filler but no explicit % found in title
        # Mark as composite but fraction unknown — tier will be flagged as uncertain
        info["is_composite"] = True
        info["fraction_assumed"] = True

    return info

def get_reference_conductivity(df: pd.DataFrame, material_type: str, temperature: float, tol: float = 5.0) -> Optional[float]:
    """
    Get conductivity for a 'pure' material at a specific temperature.
    
    Strategy (in priority order):
    1. Data-driven: median from matching pure-material entries in the dataset
    2. Literature table: fallback from REFERENCE_CONDUCTIVITIES
    3. None: if material_type is completely unknown
    """
    ref = REFERENCE_CONDUCTIVITIES.get(material_type)
    if ref is None:
        return None
    
    keywords = ref["keywords"]
    exclude = ref["exclude"]
    
    # Attempt data-driven lookup first
    mask = (df["temperature"].between(temperature - tol, temperature + tol) &
            (df["is_cited"] == False) &
            df["raw_composition"].apply(lambda x: any(k in str(x).lower() for k in keywords)) &
            ~df["raw_composition"].apply(lambda x: any(k in str(x).lower() for k in exclude)))
           
    subset = df[mask]
    if len(subset) > 0:
        return subset["conductivity"].median()
    
    # Fall back to literature table
    if 15 <= temperature <= 35:
        return ref.get("rt")
    elif 50 <= temperature <= 70:
        return ref.get("60c", ref.get("rt"))
    
    return ref.get("rt")  # Best-effort fallback


def batch_lookup_unknown_conductivities(df: pd.DataFrame, batch_size: int = 15) -> Dict[str, float]:
    """
    Use Gemini 2.5 Flash to estimate reference conductivities for materials
    that don't match the lookup table. Processes in batches to limit API calls.
    
    Returns:
        Dict mapping material name -> estimated conductivity (S/cm) at ~25°C
    """
    # Find composites whose polymer or filler is not in our table
    unknown_materials = set()
    for _, row in df[df["is_composite"] == True].iterrows():
        polymer = row.get("polymer")
        filler = row.get("filler")
        if polymer and polymer not in REFERENCE_CONDUCTIVITIES:
            unknown_materials.add(polymer)
        if filler and filler not in REFERENCE_CONDUCTIVITIES:
            unknown_materials.add(filler)
    
    if not unknown_materials:
        return {}
    
    logger.info(f"Found {len(unknown_materials)} unknown materials for LLM lookup: {unknown_materials}")
    
    if not genai:
        logger.warning("Gemini SDK not installed. Skipping LLM conductivity lookup.")
        return {}
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY. Skipping LLM conductivity lookup.")
        return {}
    
    client = genai.Client(api_key=api_key)
    results = {}
    unknown_list = sorted(unknown_materials)
    
    # Process in batches
    for i in range(0, len(unknown_list), batch_size):
        batch = unknown_list[i:i + batch_size]
        materials_str = "\n".join(f"- {m}" for m in batch)
        
        prompt = f"""You are a materials scientist. For each material below, estimate its 
ionic conductivity in S/cm at room temperature (~25°C). These are solid-state 
electrolyte components (polymers or ceramics). If the material is an electronic 
insulator (like Al2O3), report a very low ionic conductivity.

Materials:
{materials_str}

Respond as a JSON object mapping material name to conductivity value in S/cm.
Example: {{"PEO": 1e-6, "LLZO": 3e-4}}
Only return the JSON, nothing else."""
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            batch_results = json.loads(response.text)
            for name, sigma in batch_results.items():
                if isinstance(sigma, (int, float)) and sigma > 0:
                    results[name] = float(sigma)
                    logger.info(f"  LLM estimate: {name} = {sigma:.2e} S/cm")
        except Exception as e:
            logger.warning(f"  LLM batch lookup failed for batch {i//batch_size + 1}: {e}")
    
    return results


# ============================================================================
# Part 3: EMT Calculation & Tier Classification
# ============================================================================

def calculate_emt_and_tier(row: pd.Series, df_ref: pd.DataFrame,
                           extra_refs: Dict[str, float] = None) -> pd.Series:
    """
    Determine Tier 1/2/3 for a single measurement.
    Supports all polymer-ceramic composites (PEO, PVDF, PAN, PPC, PMMA, ...).

    For conducting fillers (σ_filler >= 1e-8):
        synergy_score = log10(σ_exp / σ_EMT)
        Tier 1: synergy_score > +0.18  (>1.5× EMT)
        Tier 3: synergy_score < -0.30  (<0.5× EMT)
        Tier 2: everything between

    For insulating fillers (σ_filler < 1e-8, e.g. SiO₂, Al₂O₃):
        Enhancement is relative to neat polymer — filler reduces crystallinity
        Tier 1: σ_exp > σ_polymer × 3  (meaningful enhancement)
        Tier 3: σ_exp < σ_polymer × 0.5  (detrimental)
        Tier 2: everything between

    When filler fraction is unknown, tier is flagged "Tier_uncertain".
    """
    res = pd.Series({
        "tier": "Unclassified",
        "sigma_emt": None,
        "sigma_polymer": None,
        "sigma_filler_used": None,
        "improvement_ratio": None,
        "synergy_score": None,
        "insulating_filler": False,
    })

    if not row["is_composite"] or not row.get("polymer"):
        return res

    polymer_type = row["polymer"]
    filler_type = row.get("filler")

    # --- Get polymer baseline conductivity ---
    sigma_poly = get_reference_conductivity(df_ref, polymer_type, row["temperature"])
    if sigma_poly is None and extra_refs:
        sigma_poly = extra_refs.get(polymer_type)
    if sigma_poly is None:
        return res  # Cannot establish baseline

    # --- Get filler conductivity ---
    sigma_filler = None
    if filler_type:
        sigma_filler = get_reference_conductivity(df_ref, filler_type, row["temperature"])
    if sigma_filler is None and filler_type and extra_refs:
        sigma_filler = extra_refs.get(filler_type)
    if sigma_filler is None:
        # Conservative fallback: typical conducting ceramic filler
        sigma_filler = 1e-4

    sigma_exp = row["conductivity"]
    phi = row["filler_fraction"]

    # --- If fraction is unknown, flag as uncertain and return early ---
    if phi == 0 or row.get("fraction_assumed", False):
        res["sigma_polymer"] = sigma_poly
        res["sigma_filler_used"] = sigma_filler
        res["tier"] = "Tier_uncertain"
        return res

    # --- Insulating filler path (σ_filler < 1e-8, e.g. SiO₂, Al₂O₃) ---
    INSULATING_THRESHOLD = 1e-8
    if sigma_filler < INSULATING_THRESHOLD:
        res["insulating_filler"] = True
        res["sigma_polymer"] = sigma_poly
        res["sigma_filler_used"] = sigma_filler
        # EMT for insulating filler is ≈ σ_poly (the φ×σ_filler term vanishes)
        # So we classify relative to neat polymer instead
        if sigma_exp > sigma_poly * 3.0:
            res["tier"] = "Tier 1"   # Meaningful crystallinity disruption / interfacial enhancement
        elif sigma_exp < sigma_poly * 0.5:
            res["tier"] = "Tier 3"   # Detrimental — agglomeration or blocking
        else:
            res["tier"] = "Tier 2"   # Marginal effect within expected scatter
        res["improvement_ratio"] = sigma_exp / sigma_poly if sigma_poly > 0 else None
        return res

    # --- Conducting filler: EMT (linear rule of mixtures) ---
    # σ_eff = (1 - φ)σ_poly + φσ_filler
    sigma_emt = (1 - phi) * sigma_poly + phi * sigma_filler

    res["sigma_emt"] = sigma_emt
    res["sigma_polymer"] = sigma_poly
    res["sigma_filler_used"] = sigma_filler
    res["improvement_ratio"] = sigma_exp / sigma_emt if sigma_emt > 0 else None

    # --- Synergy score on a consistent log scale ---
    if sigma_emt > 0 and sigma_exp > 0:
        synergy_score = np.log10(sigma_exp / sigma_emt)
        res["synergy_score"] = synergy_score

        if synergy_score > 0.18:       # >1.5× EMT
            res["tier"] = "Tier 1"
        elif synergy_score < -0.30:    # <0.5× EMT
            res["tier"] = "Tier 3"
        else:
            res["tier"] = "Tier 2"

    return res

# ============================================================================
# Part 4: Visualization
# ============================================================================

def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate analysis plots."""
    
    # 1. Tier Distribution
    tier_counts = df["tier"].value_counts()
    for label in ["Unclassified", "Tier_uncertain"]:
        if label in tier_counts:
            tier_counts = tier_counts.drop(label)
        
    plt.figure(figsize=(8, 6))
    colors = {"Tier 1": "#2ecc71", "Tier 2": "#f1c40f", "Tier 3": "#e74c3c"}
    tier_counts.reindex(["Tier 1", "Tier 2", "Tier 3"]).plot(
        kind="bar", color=[colors.get(x, "gray") for x in ["Tier 1", "Tier 2", "Tier 3"]]
    )
    plt.title("Distribution of Measurements by Performance Tier")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "1_tier_distribution.png", dpi=100)
    plt.close()
    
    # 2. Conductivity vs EMT
    plt.figure(figsize=(10, 8))
    subset = df[df["tier"].isin(["Tier 1", "Tier 2", "Tier 3"]) & df["sigma_emt"].notna()]

    for tier in ["Tier 1", "Tier 2", "Tier 3"]:
        d = subset[subset["tier"] == tier]
        plt.scatter(d["sigma_emt"], d["conductivity"],
                    label=f"{tier} (n={len(d)})", color=colors[tier], alpha=0.7)

    # Diagonal line + ±0.5-decade shaded Tier 2 band
    lims = [
        min(subset["sigma_emt"].min(), subset["conductivity"].min()) * 0.5,
        max(subset["sigma_emt"].max(), subset["conductivity"].max()) * 2
    ]
    x_line = np.array(lims)
    plt.plot(x_line, x_line, 'k--', alpha=0.7, label="EMT (Ideal Mixing, 1:1)")
    # Tier 2 zone: synergy score in [-0.30, +0.18] → factor 0.5× to 1.5× of EMT
    plt.fill_between(x_line, x_line * 0.5, x_line * 10**0.18,
                     alpha=0.10, color="#f1c40f", label="Tier 2 zone (0.5×–1.5× EMT)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Predicted Conductivity (EMT) [S/cm]")
    plt.ylabel("Measured Conductivity [S/cm]")
    plt.title("Measured vs Predicted Conductivity")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "2_emt_comparison.png", dpi=100)
    plt.close()
    
    # 3. Processing Method Impact on Tier 1
    # Check which processing methods appear most in Tier 1
    t1 = df[df["tier"] == "Tier 1"]
    if not t1.empty:
        proc_counts = t1["processing_method"].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        proc_counts.plot(kind="barh", color="#2ecc71")
        plt.title("Top Processing Methods in Tier 1 Materials")
        plt.xlabel("Count in Tier 1")
        plt.tight_layout()
        plt.savefig(output_dir / "3_processing_impact.png", dpi=100)
        plt.close()

    # 4. Pareto Front (Conductivity vs Filler Fraction)
    plot_pareto_front(df, output_dir)
    
    # 5. Feature Importance
    analyze_feature_importance(df, output_dir)

def plot_pareto_front(df: pd.DataFrame, output_dir: Path):
    """Plot Conductivity vs Filler Fraction and highlight top performers."""
    # Filter for composites with known fraction
    subset = df[
        (df["is_composite"] == True) & 
        (df["filler_fraction"] > 0) & 
        (df["conductivity"] > 0)
    ].copy()
    
    if subset.empty:
        return
        
    plt.figure(figsize=(10, 8))
    
    # Scatter all
    plt.scatter(subset["filler_fraction"] * 100, subset["conductivity"], 
                c='gray', alpha=0.4, label="All Composites")
    
    # Highlight Tier 1
    t1 = subset[subset["tier"] == "Tier 1"]
    plt.scatter(t1["filler_fraction"] * 100, t1["conductivity"], 
                c='#2ecc71', s=50, label="Tier 1", edgecolors='black')
                
    plt.yscale("log")
    plt.xlabel("Filler Fraction (wt% or vol%)")
    plt.ylabel("Ionic Conductivity (S/cm)")
    plt.title("Composite Performance: Conductivity vs Loading")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "4_pareto_front.png", dpi=100)
    plt.close()

def analyze_feature_importance(df: pd.DataFrame, output_dir: Path):
    """Train a simple Random Forest to find drivers of conductivity."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
    except ImportError:
        logger.warning("Scikit-learn not found. Skipping feature importance.")
        return

    # Prepare Data
    # Focus on Tier 1 & 2 (Composites) roughly
    target_df = df[df["conductivity"] > 0].copy()
    
    # Features
    target_df["log_cond"] = np.log10(target_df["conductivity"])
    
    # Create simple features
    X = pd.DataFrame()
    X["temperature"] = target_df["temperature"]
    X["is_composite"] = target_df["is_composite"].astype(int)
    X["filler_fraction"] = target_df["filler_fraction"].fillna(0)
    
    # One-hot encode common processing keywords
    target_df["processing_method"] = target_df["processing_method"].fillna("unknown")
    keywords = ["sinter", "cast", "hot press", "electrospin", "ball mill"]
    for k in keywords:
        X[f"proc_{k}"] = target_df["processing_method"].str.contains(k, case=False).astype(int)

    # One-hot families
    target_df["material_family"] = target_df["material_family"].fillna("Other")
    families = pd.get_dummies(target_df["material_family"], prefix="fam")
    X = pd.concat([X, families], axis=1)
    
    y = target_df["log_cond"]
    
    # Drop NaNs
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    if len(X) < 50:
        logger.warning("Not enough data for RF analysis")
        return

    # Train/test split to get honest out-of-sample R²
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    oos_r2 = rf.score(X_test, y_test)
    logger.info(f"  RF out-of-sample R² = {oos_r2:.3f} (n_test={len(X_test)})")

    # Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(X.columns))

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance for Log(Conductivity)\n(out-of-sample R² = {oos_r2:.3f})")
    plt.barh(range(top_n), importances[indices][:top_n], align="center", color="#3498db")
    plt.yticks(range(top_n), [X.columns[i] for i in indices][:top_n])
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "5_feature_importance.png", dpi=100)
    plt.close()
    
# ============================================================================
# Part 5: LLM Report Generation
# ============================================================================

def generate_llm_report(df: pd.DataFrame, output_dir: Path):
    """Use Gemini to interpret the Tier data and generate design rules."""
    if not genai:
        logger.warning("Gemini SDK not installed. Skipping LLM report.")
        return
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No API key found. Skipping LLM report.")
        return
        
    client = genai.Client(api_key=api_key)
    
    # Prepare data summary
    tier1 = df[df["tier"] == "Tier 1"].nlargest(20, "conductivity")
    tier2 = df[df["tier"] == "Tier 2"].sample(min(10, (df["tier"] == "Tier 2").sum()), random_state=42)
    tier3 = df[df["tier"] == "Tier 3"].nsmallest(10, "conductivity")

    cols = ["raw_composition", "conductivity", "sigma_emt", "synergy_score",
            "processing_method", "material_family"]

    def format_md_table(d: pd.DataFrame) -> str:
        d = d[[c for c in cols if c in d.columns]].copy()
        for num_col in ["conductivity", "sigma_emt", "synergy_score"]:
            if num_col in d.columns:
                d[num_col] = d[num_col].apply(
                    lambda v: f"{v:.2e}" if pd.notna(v) else "N/A"
                )
        try:
            return d.to_markdown(index=False)
        except ImportError:
            return d.to_string(index=False)

    prompt = f"""
You are an expert battery scientist analyzing solid-state electrolyte data.
We have classified polymer-ceramic composite electrolytes into 3 Tiers based on
deviation from Effective Medium Theory (EMT). synergy_score = log10(σ_exp / σ_EMT).

Tier 1: synergy_score > +0.18  → Conductivity >1.5× EMT (synergistic)
Tier 2: −0.30 ≤ synergy_score ≤ +0.18  → Rule-of-mixtures behavior
Tier 3: synergy_score < −0.30  → Conductivity <0.5× EMT (detrimental)

## TOP Tier 1 Materials (Success Cases, n={len(tier1)})
{format_md_table(tier1)}

## Sample Tier 2 Materials (Baseline, n={len(tier2)})
{format_md_table(tier2)}

## Tier 3 Materials (Failure Cases, n={len(tier3)})
{format_md_table(tier3)}

## Task
Analyze the contrast between Tier 1 and Tier 2 (not just Tier 1 vs Tier 3).
What distinguishes a marginal-improvement composite (Tier 2) from a truly synergistic one (Tier 1)?
Is it specific processing (e.g. nanowires vs particles)? Specific filler families?
Generate actionable design rules for achieving Tier 1 performance.
"""

    try:
        response = client.models.generate_content(
            model=REPORT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TierAnalysisReport,
                temperature=0.35 if "2.5" in REPORT_MODEL else 1.0
            )
        )
        
        data = TierAnalysisReport.model_validate_json(response.text)
        
        # Save JSON
        with open(output_dir / "tier_report.json", "w") as f:
            f.write(data.model_dump_json(indent=2))
            
        # Save Markdown
        md = f"# Tier-Based Design Rules for Solid-State Electrolytes\n\n"
        md += f"## Executive Summary\n{data.executive_summary}\n\n"
        md += f"## 🌟 Tier 1 Drivers (The 'Secret Sauce')\n{data.tier_1_analysis}\n\n"
        md += f"## ⚠️ Tier 3 Pitfalls\n{data.tier_2_3_analysis}\n\n"
        md += f"## 📐 Design Rules\n"
        for r in data.design_rules:
            md += f"### {r.rule_id}. {r.title}\n"
            md += f"**Focus:** {r.tier_focus} | **Confidence:** {r.confidence_level}\n\n"
            md += f"{r.description}\n\n"
            md += f"> **Recommendation:** {r.actionable_recommendation}\n\n"
        
        with open(output_dir / "tier_report.md", "w") as f:
            f.write(md)
            
        logger.info("Report generated successfully.")
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tier-Based Analysis Pipeline")
    parser.add_argument("--input-dir", type=str, default="output_parsed")
    parser.add_argument("--output-dir", type=str, default="tier_analysis_output")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Load
    logger.info("Step 1: Loading Data...")
    df = load_and_consolidate(input_path)
    if df.empty:
        return
        
    # 2. Parse Info
    logger.info("Step 2: Parsing Compositions...")
    parsed_infos = df.apply(parse_composition_info, axis=1)
    df = pd.concat([df, parsed_infos.apply(pd.Series)], axis=1)
    
    # Filter for non-cited if requested (Doing it here to keep 'Is Cited' flag for checking)
    df_clean = df[~df["is_cited"]].copy()
    logger.info(f"Filtered to {len(df_clean)} non-cited measurements for analysis")
    
    # 2b. LLM batch lookup for unknown materials (if any)
    logger.info("Step 2b: Looking up unknown material conductivities...")
    extra_refs = batch_lookup_unknown_conductivities(df_clean)
    if extra_refs:
        logger.info(f"  Got {len(extra_refs)} LLM-estimated reference values")
    
    # 3. EMT & Tiers
    logger.info("Step 3: Calculating Tiers...")
    tier_results = df_clean.apply(
        lambda row: calculate_emt_and_tier(row, df_clean, extra_refs), axis=1
    )
    df_clean = pd.concat([df_clean, tier_results], axis=1)
    
    # Save processed data
    try:
        df_clean.to_csv(output_path / "processed_tier_data.csv", index=False, escapechar="\\")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
    
    # 4. Plots
    logger.info("Step 4: Generating Plots...")
    generate_plots(df_clean, output_path)
    
    # 5. LLM
    if not args.skip_llm:
        logger.info("Step 5: Generating Intelligence Report...")
        generate_llm_report(df_clean, output_path)
        
    logger.info(f"Done! Results in {output_path}")

if __name__ == "__main__":
    main()
