#!/usr/bin/env python3
"""
Design Rules Analyzer for Solid-State Electrolytes
===================================================
Consolidates extracted JSON data from the extraction pipeline,
generates exploratory plots, and uses Gemini to propose design rules.

Usage:
    mamba activate pokeagent
    python design_rules_analyzer.py --input-dir fetched_papers/obelix_parsed_v5
    python design_rules_analyzer.py --input-dir fetched_papers/obelix_parsed_v5 --skip-llm
"""

import os
import re
import csv
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from dotenv import load_dotenv
from collections import Counter
from google import genai
from google.genai import types

load_dotenv()

# ============================================================================
# Pydantic Schemas for Structured LLM Output
# ============================================================================
from pydantic import BaseModel, Field


class DesignRule(BaseModel):
    """A single actionable design rule derived from the data."""
    rule_id: int = Field(..., description="Sequential rule number")
    title: str = Field(..., description="Short title for the rule (e.g. 'Nanowire morphology outperforms nanoparticles')")
    description: str = Field(..., description="Detailed explanation of the rule with mechanistic reasoning")
    supporting_evidence: List[str] = Field(
        ..., description="List of specific data points or papers that support this rule"
    )
    confidence_level: str = Field(
        ..., description="How confident: 'high' (>5 data points), 'medium' (3-5), 'speculative' (<3)"
    )
    actionable_recommendation: str = Field(
        ..., description="Concrete recommendation for experimentalists"
    )


class ProcessingComparison(BaseModel):
    """Comparison of processing methods and their impact on conductivity."""
    method: str = Field(..., description="Processing method name")
    median_conductivity_s_cm: Optional[float] = Field(None, description="Median conductivity in S/cm")
    num_data_points: int = Field(..., description="Number of data points for this method")
    key_observations: str = Field(..., description="Key observations about this method")


class DesignRulesReport(BaseModel):
    """Complete design rules report generated from extracted data."""
    executive_summary: str = Field(
        ..., description="2-3 sentence high-level summary of the most important findings"
    )
    top_compositions: List[str] = Field(
        ..., description="Top 5 compositions by ionic conductivity at room temperature, with values"
    )
    design_rules: List[DesignRule] = Field(
        ..., description="5-10 actionable design rules ordered by importance"
    )
    processing_comparisons: List[ProcessingComparison] = Field(
        ..., description="Comparison of each processing method found in the data"
    )
    knowledge_gaps: List[str] = Field(
        ..., description="Important gaps in the data / unexplored compositions or conditions"
    )
    suggested_experiments: List[str] = Field(
        ..., description="3-5 concrete experiments suggested based on trends in the data"
    )


# ============================================================================
# Part 1: JSON Consolidation
# ============================================================================

def validate_formula_stoichiometry(formula: str) -> bool:
    """
    Validate that a chemical formula has physically plausible stoichiometry.
    Returns True if plausible, False otherwise.
    """
    if not formula or formula == "null":
        return False
    
    # Extract all element-coefficient pairs
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, str(formula))
    
    if not matches:
        return False
    
    total_atoms = 0
    element_counts = {}
    
    for element, coeff_str in matches:
        coeff = float(coeff_str) if coeff_str else 1.0
        element_counts[element] = element_counts.get(element, 0) + coeff
        total_atoms += coeff
    
    # Physical plausibility checks
    if total_atoms < 3 or total_atoms > 200:
        return False
    
    for element, count in element_counts.items():
        if element == 'Li' and (count < 0.1 or count > 50):
            return False
        if element == 'O' and (count < 1 or count > 150):
            return False
        if element not in ['Li', 'O'] and (count < 0.05 or count > 30):
            return False
    
    return True


def classify_material_family(row: pd.Series) -> str:
    """Classify a measurement into a material family based on composition strings."""
    formula = str(row.get("canonical_formula", "") or "").lower()
    raw = str(row.get("raw_composition", "") or "").lower()
    combined = formula + " " + raw

    if any(k in combined for k in ["llzo", "la3zr2", "garnet", "li7la3", "li6.55ga"]):
        return "Garnet (LLZO)"
    elif any(k in combined for k in ["nasicon", "latp", "lagp", "li1+x", "liti2", "po4"]):
        return "NASICON"
    elif any(k in combined for k in ["ps4", "sbs4", "sulfide", "li6ps5", "argyrodite", "lgps"]):
        return "Sulfide"
    elif any(k in combined for k in ["peo", "pan", "polymer", "pvdf", "ppc"]):
        if any(k in combined for k in ["composite", "llzo", "filler", "vol%", "wt%", "wt ", "nanowire", "nanoparticle"]):
            return "Polymer-Ceramic Composite"
        return "Polymer"
    elif any(k in combined for k in ["al2o3", "sio2", "tio2"]):
        return "Oxide Filler"
    else:
        return "Other"


def consolidate_jsons(input_dir: Path) -> pd.DataFrame:
    """Load all *_extracted.json files and flatten into a single DataFrame."""
    json_files = sorted(input_dir.rglob("*_extracted.json"))
    if not json_files:
        print(f"❌ No *_extracted.json files found in {input_dir}")
        return pd.DataFrame()

    print(f"📂 Found {len(json_files)} extracted JSON files")

    all_rows = []
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        doc_name = data.get("doc_name", jf.stem)
        for m in data.get("measurements", []):
            row = {
                "doc_name": doc_name,
                "raw_composition": m.get("raw_composition"),
                "canonical_formula": m.get("canonical_formula"),
                "reduced_formula": m.get("reduced_formula"),
                "normalized_conductivity": m.get("normalized_conductivity"),
                "normalized_temperature_c": m.get("normalized_temperature_c"),
                "raw_conductivity": m.get("raw_conductivity"),
                "raw_conductivity_unit": m.get("raw_conductivity_unit"),
                "processing_method": m.get("processing_method"),
                "source": m.get("source"),
                "confidence": m.get("confidence"),
                "num_warnings": len(m.get("warnings", [])),
                "material_definitions": "; ".join(m.get("material_definitions", [])),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"   → {len(df)} total measurements across {len(json_files)} papers")

    # Data cleaning & Filtering
    initial = len(df)
    # 1. Basic Cleaning
    df = df.dropna(subset=["normalized_conductivity", "normalized_temperature_c"])
    
    # 2. Ionic Conductivity Filter (interpret "larger than 0 is invalid" as log10(sigma) <= 0)
    df = df[df["normalized_conductivity"] > 0]
    df = df[df["normalized_conductivity"] <= 1.0] # Remove suspiciously high (>1 S/cm)
    
    # 3. Temperature Filter (15-30°C)
    df = df[df["normalized_temperature_c"].between(15, 30)]
    
    # 4. Source Filter (Exclude cited data)
    df = df[~df["source"].astype(str).str.lower().str.startswith("cited")]
    
    # 5. Formula Validity Check
    df = df[df["canonical_formula"].apply(validate_formula_stoichiometry)]
    
    print(f"   → {len(df)} measurements after strict filtering ({initial - len(df)} removed)")

    # Impute missing processing methods from other measurements in the same paper
    df = impute_processing_methods(df)

    # Classify material families (Keyword-based)
    df["material_family"] = df.apply(classify_material_family, axis=1)

    # Classify "Other" materials with LLM
    df = classify_others_with_llm(df)

    # Extract processing keywords
    df["processing_keywords"] = df["processing_method"].apply(extract_processing_keywords)

    return df


def impute_processing_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute 'not reported' processing methods using other measurements from the same paper.
    
    Rules:
    1. Source of truth: Measurements where processing_method is known AND source is NOT 'cited_*'
    2. Target for imputation: Measurements where processing_method is 'not reported' AND source is NOT 'cited_*'
    3. Cited data is ignored (neither used as source nor updated).
    """
    if "doc_name" not in df.columns or "processing_method" not in df.columns:
        return df

    print("   🔍 Imputing missing processing methods...")
    
    # Helper to check if a method is valid
    def is_valid_method(m):
        return m and str(m).lower() not in ["none", "nan", "not reported", "null", ""]

    # Process each paper independently
    for doc in df["doc_name"].unique():
        # Mask for this paper
        paper_mask = df["doc_name"] == doc
        
        # Mask for non-cited sources (we only want to propagate the paper's OWN method)
        # Check for 'cited' in source string (case insensitive)
        non_cited_mask = ~df["source"].astype(str).str.lower().str.startswith("cited")
        
        # Combined mask for potential source-of-truth rows
        # We need rows that belong to this paper, are not citations, and have a valid method
        potential_sources = df[paper_mask & non_cited_mask]
        
        valid_methods = [m for m in potential_sources["processing_method"] if is_valid_method(m)]
        
        if not valid_methods:
            continue
            
        # Find the most common method (mode)
        # In most papers, there's one dominant synthesis method for the main results
        most_common_method = Counter(valid_methods).most_common(1)[0][0]
        
        # Identify rows to update: 
        # Same paper + Not cited + (Method is invalid/missing)
        rows_to_update = paper_mask & non_cited_mask & \
                         df["processing_method"].apply(lambda x: not is_valid_method(x))
        
        if rows_to_update.any():
            count = rows_to_update.sum()
            # print(f"      - {doc[:30]}... : Propagating '{most_common_method[:20]}...' to {count} rows")
            df.loc[rows_to_update, "processing_method"] = most_common_method

    return df


def classify_others_with_llm(df: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
    """
    Use Gemini to classify materials labeled as "Other" into more granular families.
    """
    others = df[df["material_family"] == "Other"].copy()
    if others.empty:
        return df

    # Unique materials to classify (formula + raw composition)
    # We group by these to avoid redundant calls
    unique_materials = others.groupby(["canonical_formula", "raw_composition"]).size().reset_index()
    unique_materials["material_str"] = (
        unique_materials["canonical_formula"].fillna("") + " | " + 
        unique_materials["raw_composition"].fillna("")
    ).str.strip()
    
    # Filter out empty strings if any
    unique_materials = unique_materials[unique_materials["material_str"] != "|"]
    
    if unique_materials.empty:
        return df

    print(f"   🤖 Classifying {len(unique_materials)} unique 'Other' materials using Gemini...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("      ⚠️  GEMINI_API_KEY not found. Skipping LLM classification.")
        return df


    async def process_batch(batch, semaphore, client, model, classification_map):
        batch_text = "\n".join([f"- {m}" for m in batch])
        prompt = f"""You are a materials scientist. Classify the following solid-state electrolyte materials into exactly one of these families sequentially:
- Garnet (LLZO)
- NASICON (LATP, LAGP, etc.)
- Sulfide (Argyrodite, LGPS, LPS, etc.)
- Polymer (PEO, PVDF, etc. without ceramic filler)
- Polymer-Ceramic Composite (Polymer matrix with ceramic filler)
- Lisicon
- Anti-perovskite
- Hydride
- Halide
- Perovskite (LLTO, etc.)
- Glass-ceramic (not already covered)
- Other (if it doesn't fit any above)

Materials to classify:
{batch_text}

Respond ONLY with a JSON list of family names in the EXACT same order as the input materials.
Example Output: ["Garnet (LLZO)", "NASICON", "Sulfide", "Polymer", "Polymer-Ceramic Composite", "Other"]
"""
        async with semaphore:
            try:
                # print(">> PROMPT")
                # print(prompt)
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                batch_classes = json.loads(response.text)
                
                if isinstance(batch_classes, list) and len(batch_classes) == len(batch):
                    for material, family in zip(batch, batch_classes):
                        classification_map[material] = family
                else:
                    print(f"      ⚠️  Batch size mismatch: expected {len(batch)}, got {len(batch_classes) if isinstance(batch_classes, list) else 'non-list'}")
            except Exception as e:
                print(f"      ❌ Error in LLM batch: {e}")

    async def run_classification():
        # Move client initialization inside the async loop to avoid "Event loop is closed" errors
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        
        # Map from material_str -> new_family
        classification_map = {}
        material_list = unique_materials["material_str"].tolist()
        semaphore = asyncio.Semaphore(6)
        
        tasks = []
        for i in range(0, len(material_list), batch_size):
            batch = material_list[i : i + batch_size]
            tasks.append(process_batch(batch, semaphore, client, model, classification_map))
        
        await asyncio.gather(*tasks)
        return classification_map

    classification_map = asyncio.run(run_classification())

    # Apply the classifications back to the main dataframe
    def update_family(row):
        if row["material_family"] != "Other":
            return row["material_family"]
        
        m_str = (str(row.get("canonical_formula") or "") + " | " + 
                 str(row.get("raw_composition") or "")).strip()
        
        return classification_map.get(m_str, "Other")

    df["material_family"] = df.apply(update_family, axis=1)
    
    new_counts = df["material_family"].value_counts()
    print(f"   ✅ Classification complete. New families identified: {list(new_counts.index)}")
    
    return df


def extract_processing_keywords(method_str: Optional[str]) -> str:
    """Extract key processing method categories from free-text descriptions."""
    if not method_str or pd.isna(method_str):
        return "not reported"

    method = method_str.lower()
    keywords = []

    if "solvent cast" in method or "cast" in method:
        keywords.append("solvent casting")
    if "ball-mill" in method or "ball mill" in method:
        keywords.append("ball-milling")
    if "sinter" in method:
        # Try to extract sintering temperature
        temp_match = re.search(r"sinter\w*\s+(?:at\s+)?(\d+)\s*[°]?\s*[cC]", method)
        if temp_match:
            keywords.append(f"sintering ({temp_match.group(1)}°C)")
        else:
            keywords.append("sintering")
    if "electrospin" in method:
        keywords.append("electrospinning")
    if "sol-gel" in method or "sol gel" in method:
        keywords.append("sol-gel")
    if "solid state" in method or "solid-state" in method:
        keywords.append("solid-state reaction")
    if "calcin" in method:
        temp_match = re.search(r"calcin\w*\s+(?:at\s+)?(\d+)\s*[°]?\s*[cC]", method)
        if temp_match:
            keywords.append(f"calcination ({temp_match.group(1)}°C)")
        else:
            keywords.append("calcination")
    if "hot press" in method or "hot-press" in method:
        keywords.append("hot pressing")
    if "glovebox" in method or "argon" in method or "inert" in method:
        keywords.append("inert atmosphere")

    return "; ".join(keywords) if keywords else "other"


# ============================================================================
# Part 2: Exploratory Plots
# ============================================================================

def make_plots(df: pd.DataFrame, output_dir: Path):
    """Generate all exploratory plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Generating plots in {output_dir}/")

    plot_composition_vs_conductivity(df, output_dir)
    plot_processing_comparison(df, output_dir)
    plot_arrhenius(df, output_dir)
    plot_data_quality(df, output_dir)

    print(f"   ✅ All plots saved to {output_dir}/")


def plot_composition_vs_conductivity(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: composition vs conductivity at near-RT, colored by material family."""
    # Filter to near room-temperature (15–35°C)
    rt_df = df[(df["normalized_temperature_c"] >= 15) & (df["normalized_temperature_c"] <= 35)].copy()

    if rt_df.empty:
        print("   ⚠️  No room-temperature data for composition plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    families = rt_df["material_family"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(families), 3)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, family in enumerate(sorted(families)):
        subset = rt_df[rt_df["material_family"] == family]
        ax.scatter(
            range(len(subset)),
            subset["normalized_conductivity"],
            label=f"{family} (n={len(subset)})",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=80, alpha=0.8, edgecolors="black", linewidth=0.5,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Ionic Conductivity (S/cm)", fontsize=12)
    ax.set_xlabel("Measurement Index", fontsize=12)
    ax.set_title("Composition vs. Ionic Conductivity (Room Temperature, 15–35°C)", fontsize=14)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=1e-3, color="green", linestyle="--", alpha=0.5, label="Target: 10⁻³ S/cm")

    # Add composition labels for top performers
    top_n = min(10, len(rt_df))
    top = rt_df.nlargest(top_n, "normalized_conductivity")
    for _, row in top.iterrows():
        label = str(row["raw_composition"])[:30]
        ax.annotate(
            label, xy=(0, row["normalized_conductivity"]),
            fontsize=6, alpha=0.7, rotation=15,
            textcoords="offset points", xytext=(5, 5),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "1_composition_vs_conductivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → 1_composition_vs_conductivity.png")


def plot_processing_comparison(df: pd.DataFrame, output_dir: Path):
    """Box/strip plot: conductivity distribution per processing method."""
    # Filter to near-RT
    rt_df = df[(df["normalized_temperature_c"] >= 15) & (df["normalized_temperature_c"] <= 35)].copy()

    if rt_df.empty:
        print("   ⚠️  No room-temperature data for processing plot")
        return

    # Split multi-keyword methods into separate rows for counting
    method_rows = []
    for _, row in rt_df.iterrows():
        keywords = str(row["processing_keywords"]).split("; ")
        for kw in keywords:
            method_rows.append({
                "method": kw.strip(),
                "conductivity": row["normalized_conductivity"],
                "composition": row["raw_composition"],
            })
    method_df = pd.DataFrame(method_rows)

    # Only plot methods with >= 2 data points
    method_counts = method_df["method"].value_counts()
    methods_to_plot = method_counts[method_counts >= 2].index.tolist()
    plot_df = method_df[method_df["method"].isin(methods_to_plot)]

    if plot_df.empty:
        print("   ⚠️  Not enough data for processing comparison plot")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort methods by median conductivity
    medians = plot_df.groupby("method")["conductivity"].median().sort_values(ascending=False)
    ordered_methods = medians.index.tolist()

    positions = range(len(ordered_methods))
    box_data = [plot_df[plot_df["method"] == m]["conductivity"].values for m in ordered_methods]

    bp = ax.boxplot(box_data, positions=positions, vert=True, widths=0.6,
                    patch_artist=True, showfliers=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ordered_methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, method in enumerate(ordered_methods):
        data = plot_df[plot_df["method"] == method]["conductivity"].values
        jitter = np.random.normal(0, 0.08, len(data))
        ax.scatter(i + jitter, data, alpha=0.6, s=30, color="black", zorder=5)

    ax.set_yscale("log")
    ax.set_xticks(positions)
    
    # Custom labels with sample size (n=?)
    labels_with_n = []
    for m in ordered_methods:
        n = len(plot_df[plot_df["method"] == m])
        labels_with_n.append(f"{m}\n(n={n})")
    
    ax.set_xticklabels(labels_with_n, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Ionic Conductivity (S/cm)", fontsize=12)
    ax.set_title("Processing Method vs. Ionic Conductivity (Room Temperature)", fontsize=14)
    ax.grid(True, alpha=0.3, which="both", axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "2_processing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → 2_processing_comparison.png")


def plot_arrhenius(df: pd.DataFrame, output_dir: Path):
    """Arrhenius plot: 1000/T vs log(σ) for materials with multi-temperature data."""
    # Group by composition within each paper
    df_valid = df[
        (df["normalized_temperature_c"].notna()) &
        (df["normalized_conductivity"].notna()) &
        (df["normalized_conductivity"] > 0) &
        (df["normalized_temperature_c"] > -50) &
        (df["normalized_temperature_c"] < 200)
    ].copy()

    # Find compositions that appear at multiple temperatures
    groups = df_valid.groupby(["doc_name", "raw_composition"])
    multi_temp = {name: grp for name, grp in groups if len(grp) >= 3}

    if not multi_temp:
        print("   ⚠️  Not enough multi-temperature data for Arrhenius plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(multi_temp), 10)))

    for i, ((doc, comp), grp) in enumerate(sorted(multi_temp.items(), key=lambda x: -len(x[1]))):
        if i >= 10:  # Limit to 10 series
            break

        temp_k = grp["normalized_temperature_c"] + 273.15
        inv_t = 1000.0 / temp_k
        log_sigma = np.log10(grp["normalized_conductivity"])

        label = f"{comp[:40]}" if len(comp) <= 40 else f"{comp[:37]}..."
        ax.scatter(inv_t, log_sigma, color=colors[i], s=60, alpha=0.8,
                   edgecolors="black", linewidth=0.5, label=label, zorder=5)

        # Linear fit for activation energy
        if len(grp) >= 3:
            try:
                coeffs = np.polyfit(inv_t, log_sigma, 1)
                x_fit = np.linspace(inv_t.min(), inv_t.max(), 50)
                y_fit = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, color=colors[i], linestyle="--", alpha=0.6, linewidth=1.5)

                # Calculate activation energy: slope = -Ea / (2.303 * k_B)
                # k_B = 8.617e-5 eV/K, so Ea = -slope * 2.303 * 8.617e-5 * 1000
                ea_ev = -coeffs[0] * 2.303 * 8.617e-5 * 1000
                ax.annotate(f"Ea={ea_ev:.2f} eV", xy=(inv_t.mean(), log_sigma.mean()),
                           fontsize=7, alpha=0.8, color=colors[i])
            except Exception:
                pass

    ax.set_xlabel("1000/T (K⁻¹)", fontsize=12)
    ax.set_ylabel("log₁₀(σ) (S/cm)", fontsize=12)
    ax.set_title("Arrhenius Plot — Temperature Dependence of Ionic Conductivity", fontsize=14)
    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3)

    # Add temperature reference axis on top
    ax2 = ax.twiny()
    temp_ticks_c = [20, 40, 60, 80, 100]
    temp_ticks_inv = [1000.0 / (t + 273.15) for t in temp_ticks_c]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(temp_ticks_inv)
    ax2.set_xticklabels([f"{t}°C" for t in temp_ticks_c], fontsize=9)
    ax2.set_xlabel("Temperature", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "3_arrhenius_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → 3_arrhenius_plot.png")


def plot_data_quality(df: pd.DataFrame, output_dir: Path):
    """Bar chart of measurement counts by source type and confidence level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Source distribution
    source_counts = df["source"].value_counts()
    colors_source = plt.cm.Pastel1(np.linspace(0, 0.8, len(source_counts)))
    axes[0].barh(source_counts.index, source_counts.values, color=colors_source, edgecolor="gray")
    axes[0].set_xlabel("Count", fontsize=11)
    axes[0].set_title("Measurements by Source Type", fontsize=13)
    for i, (idx, val) in enumerate(source_counts.items()):
        axes[0].text(val + 0.5, i, str(val), va="center", fontsize=10, fontweight="bold")

    # Confidence distribution
    conf_order = ["high", "medium", "low"]
    conf_counts = df["confidence"].value_counts().reindex(conf_order, fill_value=0)
    conf_colors = {"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}
    bar_colors = [conf_colors.get(c, "gray") for c in conf_counts.index]
    axes[1].bar(conf_counts.index, conf_counts.values, color=bar_colors, edgecolor="gray")
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Measurements by Confidence Level", fontsize=13)
    for i, (idx, val) in enumerate(conf_counts.items()):
        axes[1].text(i, val + 0.5, str(val), ha="center", fontsize=10, fontweight="bold")

    plt.suptitle(f"Data Quality Overview — {len(df)} Total Measurements from {df['doc_name'].nunique()} Papers",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "4_data_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → 4_data_quality.png")


# ============================================================================
# Part 3: LLM-Powered Design Rule Generation
# ============================================================================

def summarize_for_llm(df: pd.DataFrame) -> str:
    """
    Build a structured statistical summary of the dataset for the LLM prompt.
    Compresses ~9K rows / ~2MB CSV into ~5-10KB of curated insights.
    """
    sections = []
    
    # --- 1. Per-family summary ---
    rt_df = df[(df["normalized_temperature_c"] >= 15) & (df["normalized_temperature_c"] <= 35)].copy()
    
    sections.append("## Per Material-Family Summary (Room Temperature, 15-35°C)")
    sections.append("| Family | N (RT) | N (all T) | Median σ (S/cm) | Best σ (S/cm) | Best Composition | σ Range |")
    sections.append("|--------|--------|-----------|-----------------|---------------|------------------|---------|")
    
    for family in sorted(df["material_family"].unique()):
        fam_all = df[df["material_family"] == family]
        fam_rt = rt_df[rt_df["material_family"] == family]
        if fam_rt.empty:
            continue
        median_s = fam_rt["normalized_conductivity"].median()
        best_s = fam_rt["normalized_conductivity"].max()
        best_row = fam_rt.loc[fam_rt["normalized_conductivity"].idxmax()]
        best_comp = str(best_row["raw_composition"])[:60]
        lo = fam_rt["normalized_conductivity"].min()
        hi = fam_rt["normalized_conductivity"].max()
        sections.append(
            f"| {family} | {len(fam_rt)} | {len(fam_all)} | {median_s:.2e} | {best_s:.2e} | {best_comp} | {lo:.2e} – {hi:.2e} |"
        )
    
    # --- 2. Top-10 compositions at RT (high-confidence, deduplicated) ---
    sections.append("\n## Top-10 Compositions (Room Temperature, High Confidence)")
    rt_hc = rt_df[rt_df["confidence"].isin(["high", "medium"])].copy()
    if rt_hc.empty:
        rt_hc = rt_df.copy()
    # Deduplicate: keep best measurement per composition
    top = (rt_hc.sort_values("normalized_conductivity", ascending=False)
               .drop_duplicates(subset=["raw_composition"], keep="first")
               .head(10))
    for i, (_, row) in enumerate(top.iterrows(), 1):
        comp = str(row["raw_composition"])[:80]
        formula = str(row.get("canonical_formula", ""))[:60]
        sigma = row["normalized_conductivity"]
        temp = row["normalized_temperature_c"]
        proc = str(row.get("processing_method", "not reported"))[:80]
        paper = str(row["doc_name"])[:50]
        sections.append(
            f"{i}. **{comp}** (σ={sigma:.2e} S/cm at {temp:.0f}°C)\n"
            f"   Formula: {formula} | Processing: {proc} | Paper: {paper}"
        )
    
    # --- 3. Processing method comparison ---
    sections.append("\n## Processing Method Comparison (Room Temperature)")
    if not rt_df.empty:
        proc_col = "processing_keywords" if "processing_keywords" in rt_df.columns else "processing_method"
        method_rows = []
        for _, row in rt_df.iterrows():
            kws = str(row.get(proc_col, "other")).split("; ")
            for kw in kws:
                method_rows.append({"method": kw.strip(), "sigma": row["normalized_conductivity"]})
        mdf = pd.DataFrame(method_rows)
        method_stats = (mdf.groupby("method")["sigma"]
                           .agg(["median", "count", "min", "max"])
                           .sort_values("median", ascending=False))
        method_stats = method_stats[method_stats["count"] >= 2]
        
        sections.append("| Method | Median σ (S/cm) | Count | Range |")
        sections.append("|--------|-----------------|-------|-------|")
        for method, row in method_stats.iterrows():
            sections.append(
                f"| {method} | {row['median']:.2e} | {int(row['count'])} | {row['min']:.2e} – {row['max']:.2e} |"
            )
    
    # --- 4. Notable outliers ---
    sections.append("\n## Notable Outliers")
    sections.append("### Highest Conductivity (any temperature)")
    outliers_high = (df.sort_values("normalized_conductivity", ascending=False)
                       .drop_duplicates(subset=["raw_composition"], keep="first")
                       .head(5))
    for _, row in outliers_high.iterrows():
        sections.append(
            f"- {row['raw_composition']}: σ={row['normalized_conductivity']:.2e} S/cm at {row['normalized_temperature_c']:.0f}°C ({row['material_family']})"
        )
    
    sections.append("### Lowest Conductivity (RT, potential issues)")
    if not rt_df.empty:
        outliers_low = rt_df.nsmallest(5, "normalized_conductivity")
        for _, row in outliers_low.iterrows():
            sections.append(
                f"- {row['raw_composition']}: σ={row['normalized_conductivity']:.2e} S/cm ({row['material_family']})"
            )
    
    # --- 5. Data quality summary ---
    sections.append("\n## Data Quality")
    sections.append(f"- Total measurements: {len(df)} ({df['doc_name'].nunique()} papers)")
    sections.append(f"- Room temperature (15-35°C): {len(rt_df)} measurements")
    conf_counts = df["confidence"].value_counts().to_dict()
    sections.append(f"- Confidence: {conf_counts}")
    source_counts = df["source"].value_counts().to_dict()
    sections.append(f"- Sources: {source_counts}")
    sections.append(f"- Temperature range: {df['normalized_temperature_c'].min():.0f} – {df['normalized_temperature_c'].max():.0f} °C")
    
    summary = "\n".join(sections)
    return summary

def generate_design_rules(df: pd.DataFrame, output_dir: Path):
    """Use Gemini to analyze the data and generate design rules."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)
    model = "gemini-3-pro-preview"

    # Build structured summary instead of raw CSV dump
    data_summary = summarize_for_llm(df)
    
    # Summary statistics
    n_papers = df["doc_name"].nunique()
    n_measurements = len(df)
    n_compositions = df["raw_composition"].nunique()
    families = df["material_family"].value_counts().to_dict()

    prompt = f"""You are an expert materials scientist specializing in solid-state electrolytes for lithium and sodium batteries.

I have extracted ionic conductivity data from {n_papers} research papers, yielding {n_measurements} measurements 
across {n_compositions} unique compositions.

## Material Families in the Dataset
{json.dumps(families, indent=2)}

## Dataset Summary
Below is a structured statistical summary of the full dataset. It includes per-family statistics,
top compositions, processing method comparisons, and notable outliers.

{data_summary}

## Your Task
Analyze this dataset and generate design rules for solid-state electrolytes. Specifically:

1. **Top Compositions**: Identify the highest-conductivity materials at room temperature and explain what makes them effective.

2. **Design Rules**: Propose 5-10 actionable rules that an experimentalist could use to design better solid-state electrolytes. Each rule should have:
   - A clear title
   - Mechanistic explanation
   - Supporting data points from this dataset
   - A confidence level based on data support

3. **Processing Method Impact**: Compare processing methods (solvent casting, ball-milling, sintering, electrospinning, etc.) and their impact on conductivity.

4. **Knowledge Gaps**: What compositions, temperature ranges, or processing methods are underrepresented?

5. **Suggested Experiments**: Based on trends, suggest 3-5 specific experiments that could yield high-conductivity materials.

Be quantitative. Reference specific conductivity values and compositions from the data.
"""

    print(f"\n🤖 Querying {model} for design rules...")
    print(f"   Sending {len(prompt)} chars of prompt ({n_measurements} measurements summarized)")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DesignRulesReport,
                temperature=1.0,
                max_output_tokens=8192,
            ),
        )

        # Parse structured response
        result = DesignRulesReport.model_validate_json(response.text)

        # Write the formatted markdown report
        report_path = output_dir / "design_rules_output.md"
        write_report(result, report_path, n_papers, n_measurements)

        # Also save raw JSON
        raw_path = output_dir / "design_rules_raw.json"
        with open(raw_path, "w") as f:
            f.write(result.model_dump_json(indent=2))

        print(f"   ✅ Design rules saved to {report_path}")
        print(f"   ✅ Raw JSON saved to {raw_path}")

        # Print usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            print(f"   📊 Tokens: {um.prompt_token_count} in / {um.total_token_count - um.prompt_token_count} out")

    except Exception as e:
        print(f"   ❌ LLM call failed: {e}")
        # Fallback: save whatever we got
        if hasattr(e, "response"):
            fallback_path = output_dir / "design_rules_output_raw.txt"
            with open(fallback_path, "w") as f:
                f.write(str(e))
            print(f"   Fallback saved to {fallback_path}")


def write_report(report: DesignRulesReport, path: Path, n_papers: int, n_measurements: int):
    """Write the structured report as a formatted markdown file."""
    lines = []
    lines.append("# 🔬 Design Rules for Solid-State Electrolytes")
    lines.append(f"\n*Auto-generated from {n_papers} papers ({n_measurements} measurements)*\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(report.executive_summary)
    lines.append("")

    # Top Compositions
    lines.append("## 🏆 Top Compositions (Room Temperature)\n")
    for i, comp in enumerate(report.top_compositions, 1):
        lines.append(f"{i}. {comp}")
    lines.append("")

    # Design Rules
    lines.append("## 📐 Design Rules\n")
    for rule in report.design_rules:
        confidence_emoji = {"high": "🟢", "medium": "🟡", "speculative": "🔴"}.get(
            rule.confidence_level, "⚪"
        )
        lines.append(f"### Rule {rule.rule_id}: {rule.title} {confidence_emoji}\n")
        lines.append(f"**Confidence:** {rule.confidence_level}\n")
        lines.append(rule.description)
        lines.append("")
        lines.append("**Supporting Evidence:**")
        for ev in rule.supporting_evidence:
            lines.append(f"- {ev}")
        lines.append("")
        lines.append(f"> **Recommendation:** {rule.actionable_recommendation}")
        lines.append("")
        lines.append("---\n")

    # Processing Comparisons
    lines.append("## ⚙️ Processing Method Comparison\n")
    lines.append("| Method | Median σ (S/cm) | Data Points | Observations |")
    lines.append("|--------|----------------|-------------|--------------|")
    for pc in report.processing_comparisons:
        sigma_str = f"{pc.median_conductivity_s_cm:.2e}" if pc.median_conductivity_s_cm else "N/A"
        lines.append(f"| {pc.method} | {sigma_str} | {pc.num_data_points} | {pc.key_observations} |")
    lines.append("")

    # Knowledge Gaps
    lines.append("## 🔍 Knowledge Gaps\n")
    for gap in report.knowledge_gaps:
        lines.append(f"- {gap}")
    lines.append("")

    # Suggested Experiments
    lines.append("## 🧪 Suggested Experiments\n")
    for i, exp in enumerate(report.suggested_experiments, 1):
        lines.append(f"{i}. {exp}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze extracted electrolyte data and generate design rules"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing *_extracted.json files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="design_rules_plots",
        help="Output directory for plots and reports (default: design_rules_plots)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM design rule generation (plots only)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Part 1: Consolidate
    print("=" * 60)
    print("📦 Part 1: Consolidating extracted JSON data")
    print("=" * 60)
    df = consolidate_jsons(input_dir)
    if df.empty:
        return

    # Save CSV
    csv_path = output_dir / "consolidated_measurements.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\")
    print(f"   💾 Saved to {csv_path}")

    # Print summary
    print(f"\n   📊 Summary:")
    print(f"      Papers: {df['doc_name'].nunique()}")
    print(f"      Unique compositions: {df['raw_composition'].nunique()}")
    print(f"      Material families: {df['material_family'].value_counts().to_dict()}")
    print(f"      Temperature range: {df['normalized_temperature_c'].min():.0f} – {df['normalized_temperature_c'].max():.0f} °C")
    print(f"      Conductivity range: {df['normalized_conductivity'].min():.2e} – {df['normalized_conductivity'].max():.2e} S/cm")

    # Part 2: Plots
    print("\n" + "=" * 60)
    print("📊 Part 2: Generating exploratory plots")
    print("=" * 60)
    make_plots(df, output_dir)

    # Part 3: LLM Design Rules
    if not args.skip_llm:
        print("\n" + "=" * 60)
        print("🤖 Part 3: Generating design rules with Gemini")
        print("=" * 60)
        generate_design_rules(df, output_dir)
    else:
        print("\n⏭️  Skipping LLM design rule generation (--skip-llm)")

    print("\n✅ Done! All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
