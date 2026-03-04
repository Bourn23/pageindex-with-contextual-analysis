"""
T0 Provenance Tracer
====================
Second-pass script that traces the complete ordered processing chain for each
measurement group in a v7 extraction output.  Measurements are grouped by
**data provenance** (which figure / table / text paragraph they came from),
then Gemini + the cached full paper text reconstructs the fabrication chain
backwards from the reported data to the raw materials.

Usage:
    mamba activate pokeagent
    python t0_provenance_tracer.py --model gemini-2.5-flash
    python t0_provenance_tracer.py --sample "2021-PEO based polymer-ceramic..."
    python t0_provenance_tracer.py --force --delete-cache
"""

import os
import re
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

# We will override PAPERS_DIR and OUTPUT_DIR dynamically.
# Setting some default fallback values.
PAPERS_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output/downselectedpapers_jiyoung")
PARSED_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/output_parsed/downselectedpapers_jiyoung")
v8_dir = Path("/Users/bourn23/Downloads/general/PageIndex/output/downselectedpapers_jiyoung")
OUTPUT_DIR = Path("/Users/bourn23/Downloads/general/PageIndex/actionable_analysis/provenance_output")

DEFAULT_MODEL = "gemini-2.5-flash"
CONCURRENCY_LIMIT = 4
BATCH_SIZE = 3  # smaller than KG extractor — provenance groups have richer context

CACHE_PREFIX = "prov-"


# ============================================================================
# Provenance Models
# ============================================================================

class Parameter(BaseModel):
    name: str = Field(..., description="Parameter name, e.g. 'temperature_c'")
    value: str = Field(..., description="Parameter value, e.g. '60'")


class ProcessStep(BaseModel):
    step_order: int = Field(..., description="1-indexed, chronological order")
    step_type: Literal[
        "precursor_prep", "milling", "mixing", "dissolution", "casting", "drying",
        "annealing", "pressing", "sintering", "assembly", "characterization", "equilibration", "other"
    ]
    step_name: Optional[str] = Field(default=None, description="Short 1-4 word specific readable name for the step (e.g. 'Ball Milling', 'Vacuum Drying', 'Thermal Equilibration')")
    description: str = Field(..., description="Concise description, e.g. 'PEO dissolved in ACN at 60C for 12h'")
    materials_involved: List[str] = Field(default_factory=list, description="Materials used in this step")
    parameters: List[Parameter] = Field(default_factory=list, description="Key parameters, e.g. [{'name': 'temperature_c', 'value': '60'}]")
    evidence_section: Optional[str] = Field(default=None, description="Paper section where evidence found")
    branch_id: Optional[str] = Field(default=None, description="Branch identifier for parallel synthesis paths, e.g. 'LLZO_synthesis', 'PEO_solution'. None for linear chains.")


class ProvenanceChain(BaseModel):
    group_key: str
    source_type: str = Field(..., description="'figure', 'table', or 'text'")
    source_id: str = Field(..., description="e.g. 'Fig. 3' or 'Table 1'")
    measurement_indices: List[int] = Field(default_factory=list)
    compositions_in_group: List[str] = Field(default_factory=list)
    process_chain: List[ProcessStep] = Field(default_factory=list)
    cell_configuration: Optional[str] = None
    measurement_technique: Optional[str] = None
    measurement_conditions: Optional[str] = None
    chain_completeness: Literal["full", "partial", "minimal"] = "minimal"
    missing_info: List[str] = Field(default_factory=list)
    cited_method_references: List["CiteRef"] = Field(default_factory=list, description="References cited for missing processing steps, e.g. 'prepared according to [14]'")


class CiteRef(BaseModel):
    ref_number: str = Field(..., description="Reference number cited, e.g. '14'")
    ref_text: Optional[str] = Field(default=None, description="Full bibliographic text from the References section, e.g. 'J. Zheng et al., Angew. Chem., 2016, 55, 12538'")
    what_is_missing: str = Field(..., description="What information is missing, e.g. 'Full LLZO synthesis procedure'")


class GroupPreAnalysis(BaseModel):
    grouping_verdict: Literal["shared", "split"] = Field(..., description="Whether compositions in this group share the same fabrication route or should be split")
    split_clusters: Optional[List[List[str]]] = Field(default=None, description="Composition clusters if verdict is 'split'")
    synthesis_structure: Literal["linear", "branching"] = Field(..., description="Whether the synthesis is a linear chain or has parallel branches")
    branch_hints: Optional[List[str]] = Field(default=None, description="Names of parallel synthesis branches, e.g. ['LLZO ceramic synthesis', 'PEO polymer solution']")


# Resolve forward reference for CiteRef in ProvenanceChain
ProvenanceChain.model_rebuild()


# ============================================================================
# Utility
# ============================================================================

def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "unknown"


async def cache_heartbeat(client, cache_name: str, stop_event: asyncio.Event, interval=120, ttl=600):
    while not stop_event.is_set():
        try:
            client.caches.update(
                name=cache_name,
                config=types.UpdateCachedContentConfig(ttl=f"{ttl}s"),
            )
        except Exception as exc:
            print(f"  ⚠️  Heartbeat failed for {cache_name}: {exc}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


def load_v7_data(paper_name: str, v8_dir: Path) -> Optional[dict]:
    """Load the v7 extraction JSON for a given paper."""
    v8_dir = v8_dir / paper_name / "robust_results_v8.json"
    if not v8_dir.exists():
        return None
    with open(v8_dir, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_papers(papers_dir: Path, v8_dir: Path) -> List[str]:
    """Find all papers that have both a markdown file and v7 extraction output."""
    paper_names = []
    if not papers_dir.exists():
        return paper_names
    for directory in sorted(papers_dir.iterdir()):
        if not directory.is_dir():
            continue
        v7_path = v8_dir / directory.name / "robust_results_v8.json"
        md_files = list(directory.glob("*.md"))
        if v7_path.exists() and md_files:
            paper_names.append(directory.name)
    return paper_names


def group_by_provenance(measurements: list) -> List[dict]:
    """
    Group measurements by where the data appears (provenance), not by composition.

    Returns list of dicts with keys:
      - group_key: str  (e.g. "fig__Fig. 3__image_005.png")
      - source_type: str  ("figure", "table", "text")
      - source_id: str  (e.g. "Fig. 3")
      - indices: List[int]
      - compositions: List[str]
      - captions: List[str]
    """
    groups: Dict[str, dict] = {}

    for i, m in enumerate(measurements):
        source = m.get("source", "text")

        # Skip cited measurements — no process chain in this paper
        if source and source.startswith("cited"):
            continue

        figure_id = m.get("source_figure_id") or ""
        image_filename = m.get("source_image_filename") or ""
        section = m.get("source_section") or "Unknown"
        paragraph_indices = m.get("source_paragraph_indices") or []
        caption = m.get("source_caption") or ""

        if source == "figure":
            key = f"fig__{figure_id}__{image_filename}"
            source_type = "figure"
            source_id = figure_id or image_filename
        elif source == "table":
            key = f"tab__{figure_id}"
            source_type = "table"
            source_id = figure_id or "Unknown Table"
        else:
            # text source — group by section + paragraph indices
            para_str = "_".join(str(p) for p in sorted(paragraph_indices)) if paragraph_indices else "all"
            key = f"txt__{section}__{para_str}"
            source_type = "text"
            source_id = f"{section} P{para_str}"

        if key not in groups:
            groups[key] = {
                "group_key": key,
                "source_type": source_type,
                "source_id": source_id,
                "indices": [],
                "compositions": [],
                "captions": [],
            }

        groups[key]["indices"].append(i)
        comp = m.get("raw_composition", "Unknown")
        if comp and comp not in groups[key]["compositions"]:
            groups[key]["compositions"].append(comp)
        if caption and caption not in groups[key]["captions"]:
            groups[key]["captions"].append(caption)

    return list(groups.values())


async def pre_analyze_group(
    client,
    sem: asyncio.Semaphore,
    model_name: str,
    group: dict,
    paper_context: dict,
    full_paper_text: Optional[str] = None,
    cache_name: Optional[str] = None,
) -> Optional[GroupPreAnalysis]:
    """Cheap pre-analysis call for multi-composition groups.

    Determines whether compositions share a fabrication route (shared vs split)
    and whether the synthesis is linear or branching.
    Only called for groups with 2+ distinct compositions.
    """
    compositions = ", ".join(group["compositions"])
    captions = " | ".join(group["captions"][:3]) or "N/A"

    ctx_lines = []
    for field_name in ["experimental_procedure_summary", "material_systems_overview"]:
        val = paper_context.get(field_name, "")
        if val:
            ctx_lines.append(f"[{field_name}]: {val}")
    context_block = "\n".join(ctx_lines)

    paper_block = f"\n=== Full Paper Text ===\n{full_paper_text}\n=== End Full Paper Text ===\n" if full_paper_text else ""

    prompt = f"""Analyze this provenance group from a materials science paper.

Group: {group['group_key']}
Source: {group['source_type']} — {group['source_id']}
Compositions: {compositions}
Caption(s): {captions}
{paper_block}
=== Paper context ===
{context_block}
=== End context ===

Answer these two questions:
1. Do all these compositions share the SAME fabrication route (just different ratios/loadings), or do some have fundamentally DIFFERENT processing? Answer "shared" or "split". If "split", cluster the compositions into groups that share fabrication.
2. Is the synthesis a simple LINEAR chain (one step after another), or does it have BRANCHING parallel paths (e.g., ceramic prepared separately from polymer, then merged)? If branching, list the parallel branch names.
"""

    if cache_name:
        config = types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=GroupPreAnalysis,
            temperature=0.0,
        )
    else:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroupPreAnalysis,
            temperature=0.0,
        )

    try:
        async with sem:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        raw = json.loads(response.text)
        return GroupPreAnalysis.model_validate(raw)
    except Exception as exc:
        print(f"  ⚠️  Pre-analysis failed for group {group['group_key']}: {exc}")
        return None


def split_group_by_clusters(group: dict, split_clusters: List[List[str]], measurements: Optional[list] = None) -> List[dict]:
    """Split a provenance group into sub-groups based on composition clusters.

    Appends _c0, _c1, etc. to group_key to differentiate sub-groups.
    If measurements are provided, indices are properly partitioned by matching
    each measurement's raw_composition to the cluster. Otherwise, all indices
    are conservatively kept in every sub-group.
    """
    sub_groups = []
    for ci, cluster in enumerate(split_clusters):
        cluster_set = set(cluster)
        sub_compositions = [c for c in group["compositions"] if c in cluster_set]
        if not sub_compositions:
            continue

        if measurements is not None:
            sub_indices = [
                idx for idx in group["indices"]
                if measurements[idx].get("raw_composition", "Unknown") in cluster_set
            ]
        else:
            sub_indices = list(group["indices"])

        sub_groups.append({
            "group_key": f"{group['group_key']}_c{ci}",
            "source_type": group["source_type"],
            "source_id": group["source_id"],
            "indices": sub_indices,
            "compositions": sub_compositions,
            "captions": group["captions"],
        })

    return sub_groups if sub_groups else [group]


def build_prompt(paper_key: str, paper_context: dict, batch_groups: list, full_paper_text: Optional[str] = None, branch_hints: Optional[Dict[str, List[str]]] = None) -> str:
    """Build the LLM prompt for a batch of provenance groups.

    Args:
        branch_hints: Optional mapping of group_key → list of branch names for branching syntheses.
    """

    # Preamble from pre-extracted paper context
    ctx_lines = []
    for field_name in [
        "experimental_procedure_summary",
        "nomenclature_key",
        "material_systems_overview",
        "measurement_and_testing_setup",
        "baseline_and_champion_samples",
    ]:
        val = paper_context.get(field_name, "")
        if val:
            ctx_lines.append(f"[{field_name}]: {val}")
    context_block = "\n".join(ctx_lines)
    
    # If not using caching, insert the full paper directly into the prompt
    full_paper_block = f"\n=== Full Paper Text ===\n{full_paper_text}\n=== End Full Paper Text ===\n" if full_paper_text else ""

    # Per-group specifications
    group_lines = []
    for group in batch_groups:
        compositions = ", ".join(group["compositions"][:10]) or "Unknown"
        captions = " | ".join(group["captions"][:3]) or "N/A"
        entry = (
            f"- group_key={group['group_key']}\n"
            f"  source_type={group['source_type']}, source_id={group['source_id']}\n"
            f"  compositions: {compositions}\n"
            f"  caption(s): {captions}"
        )
        # Inject branch hints if available for this group
        gk = group["group_key"]
        if branch_hints and gk in branch_hints:
            hints_str = ", ".join(branch_hints[gk])
            entry += f"\n  SYNTHESIS STRUCTURE: BRANCHING — parallel paths: {hints_str}"
        group_lines.append(entry)
    groups_block = "\n".join(group_lines)

    prompt = f"""You are an expert materials-science process extraction system.
For each provenance group below, trace the COMPLETE ordered processing chain
from raw materials to final measurement, using ONLY the provided paper text context.

Paper key: {paper_key}
{full_paper_block}
=== Pre-extracted paper context (from a prior extraction pass) ===
{context_block}
=== End context ===

Provenenance groups to trace:
{groups_block}

For each group, return a ProvenanceChain with:
1. process_chain: ordered list of ProcessStep objects describing every fabrication
   step mentioned in the paper for the sample(s) in this group. Start from raw
   materials/precursors and end at the characterization/measurement step.
2. cell_configuration: the electrochemical cell setup if applicable (e.g., "Li|SPE|Li")
3. measurement_technique: how conductivity was measured (e.g., "EIS", "DC polarization")
4. measurement_conditions: conditions during measurement (e.g., "25°C, Ar atmosphere")
5. chain_completeness: "full" if all steps from precursor to measurement are captured,
   "partial" if some steps are described but gaps exist,
   "minimal" if very little process info is available.
6. missing_info: list of what's missing (e.g., ["drying temperature", "pressing pressure"])

IMPORTANT:
- Do NOT invent steps not stated in the paper.
- Use the pre-extracted context as hints but verify against the full paper text.
- If multiple compositions share one group, they share the same processing chain
  (this is correct — they come from the same figure/table).
- step_type must be one of: precursor_prep, milling, mixing, dissolution, casting, drying,
  annealing, pressing, sintering, assembly, characterization, equilibration, other
- Provide a concise `step_name` (1-4 words) for every step (e.g. "Ball Milling", "Thermal Equilibration", "Solvent Casting").
- When a processing step refers to another publication (e.g., "prepared according to [14]"),
  do NOT invent those steps. Instead, populate `cited_method_references` with the reference
  number, the full bibliographic text from the References section, and what information is missing.
- Use `branch_id` on ProcessStep when the synthesis involves parallel preparation paths
  (e.g., ceramic and polymer prepared separately before combining). Set to null for simple linear chains.
  Use descriptive branch_ids like "LLZO_synthesis" or "PEO_solution".
  A merge step (e.g., mixing ceramic into polymer) starts a new branch like "composite_assembly".
- Return a JSON array with one ProvenanceChain per group.
"""
    return prompt


# ============================================================================
# LLM Extraction
# ============================================================================

async def extract_provenance_batch(
    client,
    sem: asyncio.Semaphore,
    model_name: str,
    cache_name: Optional[str],
    paper_key: str,
    paper_context: dict,
    batch_groups: list,
    batch_index: int,
    full_paper_text: Optional[str] = None,
    group_hints: Optional[Dict[str, "GroupPreAnalysis"]] = None,
) -> Optional[List[ProvenanceChain]]:
    """Extract provenance chains for a batch of groups, either via cached paper context or direct injection."""
    # Build branch_hints dict from group pre-analysis results
    batch_branch_hints: Optional[Dict[str, List[str]]] = None
    if group_hints:
        batch_branch_hints = {}
        for g in batch_groups:
            gk = g["group_key"]
            hint = group_hints.get(gk)
            if hint and hint.synthesis_structure == "branching" and hint.branch_hints:
                batch_branch_hints[gk] = hint.branch_hints
        if not batch_branch_hints:
            batch_branch_hints = None

    prompt = build_prompt(paper_key, paper_context, batch_groups, full_paper_text, branch_hints=batch_branch_hints)

    # Dynamically setup GenerateContentConfig based on cache existence
    if cache_name:
        config = types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=list[ProvenanceChain],
            temperature=0.1,
        )
    else:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[ProvenanceChain],
            temperature=0.1,
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with sem:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

            raw = json.loads(response.text)
            validated = [ProvenanceChain.model_validate(item) for item in raw]

            # Overwrite measurement_indices from ground truth
            for chain, group in zip(validated, batch_groups):
                chain.group_key = group["group_key"]
                chain.source_type = group["source_type"]
                chain.source_id = group["source_id"]
                chain.measurement_indices = group["indices"]
                chain.compositions_in_group = group["compositions"]

            return validated

        except Exception as exc:
            err_str = str(exc)
            transient = any(code in err_str for code in ["403", "429", "500", "503"])
            if transient and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  ⚠️  Batch {batch_index} failed (attempt {attempt+1}), retrying in {wait_time}s: {exc}")
                await asyncio.sleep(wait_time)
                continue

            print(f"  ❌ Provenance extraction failed for batch {batch_index}: {exc}")
            return None


# ============================================================================
# Per-paper Processing
# ============================================================================

async def process_paper(
    client,
    paper_name: str,
    model_name: str,
    v8_dir: Path,
    papers_dir: Path,
    force: bool,
    delete_cache: bool,
    concurrency_limit: int,
    batch_size: int,
) -> dict:
    paper_slug = slugify(paper_name)
    paper_dir = papers_dir / paper_name
    
    # OUTPUT_DIR is overridden to be the paper's own directory.
    output_path = paper_dir / f"{paper_slug}_provenance.json"

    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {paper_name}: already processed")
        return {"paper": paper_name, "status": "skipped"}

    # Load v7 data
    v7_data = load_v7_data(paper_name, v8_dir)
    if not v7_data:
        print(f"  ⚠️  No v7 extraction found for {paper_name}")
        return {"paper": paper_name, "status": "no_v7"}

    measurements = v7_data.get("measurements", [])
    if not measurements:
        print(f"  ⚠️  No measurements in {paper_name}")
        return {"paper": paper_name, "status": "no_measurements"}

    paper_context = v7_data.get("paper_context", {})

    # Find paper markdown
    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        print(f"  ⚠️  No markdown file found for {paper_name}")
        return {"paper": paper_name, "status": "no_md"}

    with open(md_files[0], "r", encoding="utf-8") as f:
        md_content = f.read()

    # Group by provenance
    groups = group_by_provenance(measurements)
    cited_count = sum(
        1 for m in measurements
        if (m.get("source") or "").startswith("cited")
    )

    print(f"\n{'='*70}")
    print(f"📄 {paper_name}")
    print(f"   Measurements: {len(measurements)} | Provenance groups: {len(groups)} | "
          f"Cited (skipped): {cited_count} | Model: {model_name}")
    print(f"{'='*70}")

    if not groups:
        print("  ⚠️  No non-cited provenance groups found")
        return {
            "paper": paper_name,
            "status": "no_groups",
            "cited_measurements_skipped": cited_count,
        }

    # ---- Pre-analysis: check multi-composition groups for split/branch ----
    sem = asyncio.Semaphore(concurrency_limit)
    multi_comp_groups = [g for g in groups if len(g["compositions"]) >= 2]
    group_hints: Dict[str, GroupPreAnalysis] = {}

    if multi_comp_groups:
        print(f"  🔍 Pre-analyzing {len(multi_comp_groups)} multi-composition group(s)...")
        pre_tasks = [
            pre_analyze_group(
                client=client,
                sem=sem,
                model_name=model_name,
                group=g,
                paper_context=paper_context,
                full_paper_text=md_content,
            )
            for g in multi_comp_groups
        ]
        pre_results = await asyncio.gather(*pre_tasks)

        for g, result in zip(multi_comp_groups, pre_results):
            if result is None:
                continue
            group_hints[g["group_key"]] = result
            print(f"    {g['group_key']}: grouping={result.grouping_verdict}, "
                  f"structure={result.synthesis_structure}")

        # Apply splits where needed
        new_groups = []
        for g in groups:
            hint = group_hints.get(g["group_key"])
            if hint and hint.grouping_verdict == "split" and hint.split_clusters:
                sub_groups = split_group_by_clusters(g, hint.split_clusters, measurements)
                print(f"    Split {g['group_key']} into {len(sub_groups)} sub-groups")
                # Propagate branch hints to sub-groups
                for sg in sub_groups:
                    if hint.synthesis_structure == "branching" and hint.branch_hints:
                        group_hints[sg["group_key"]] = hint
                new_groups.extend(sub_groups)
            else:
                new_groups.append(g)
        groups = new_groups

    # Figure out if we have enough batches to warrant cache creation
    batches = [groups[i:i + batch_size] for i in range(0, len(groups), batch_size)]
    use_cache = len(batches) > 1

    cache_display_name = f"{CACHE_PREFIX}{paper_name[:47]}"
    cache = None
    heartbeat_task = None
    stop_heartbeat = asyncio.Event()

    if use_cache:
        print(f"  🔄 Step 1: Checking cache '{cache_display_name}' for {len(batches)} batches...")
        try:
            for cached in client.caches.list():
                if cached.display_name == cache_display_name:
                    cache = cached
                    print(f"  ✅ Found existing cache: {cache.name}")
                    break
        except Exception as exc:
            print(f"  ⚠️  Could not list caches: {exc}")

        if not cache:
            print(f"  🔄 Step 2: Creating cache from markdown ({len(md_content)} chars)...")
            try:
                cache = client.caches.create(
                    model=model_name,
                    config=types.CreateCachedContentConfig(
                        display_name=cache_display_name,
                        system_instruction=(
                            "You are an expert in solid-state battery material synthesis. "
                            "Your task is to trace the complete fabrication process chain "
                            "for measurements reported in this paper."
                        ),
                        contents=[md_content],
                        ttl="600s",
                    ),
                )
                print(f"  ✅ Cache created: {cache.name}")
            except Exception as exc:
                print(f"  ❌ Cache creation failed: {exc}")
                return {"paper": paper_name, "status": "cache_failed"}

        heartbeat_task = asyncio.create_task(cache_heartbeat(client, cache.name, stop_heartbeat))
    else:
        print(f"  ⚡️ Skipping cache creation (only {len(batches)} batch to process)")

    # Extract provenance chains in batches
    print(f"  🔄 Step 3: Extracting provenance chains ({len(groups)} groups in batches of {batch_size})...")

    try:
        tasks = []
        for bi, batch in enumerate(batches):
            tasks.append(
                extract_provenance_batch(
                    client=client,
                    sem=sem,
                    model_name=model_name,
                    cache_name=cache.name if cache else None,
                    paper_key=slugify(paper_name),
                    paper_context=paper_context,
                    batch_groups=batch,
                    batch_index=bi,
                    full_paper_text=md_content if not cache else None,
                    group_hints=group_hints if group_hints else None,
                )
            )
        raw_results = await asyncio.gather(*tasks)
    finally:
        if heartbeat_task:
            stop_heartbeat.set()
            await heartbeat_task

    # Clean up cache
    if cache:
        if delete_cache:
            print(f"  🧹 Deleting cache: {cache.name}")
            try:
                client.caches.delete(name=cache.name)
            except Exception as exc:
                print(f"  ⚠️  Cache deletion failed: {exc}")
        else:
            print(f"  ⏳ Cache left intact: {cache.name} (auto-expire)")

    # Collect results
    chains: List[ProvenanceChain] = []
    for bi, batch_result in enumerate(raw_results):
        if batch_result:
            chains.extend(batch_result)
        else:
            print(f"  ⚠️  Batch {bi} returned no provenance chains")

    if not chains:
        print("  ❌ No provenance chains extracted")
        return {"paper": paper_name, "status": "extraction_failed"}

    # Build output
    completeness_counts = {"full": 0, "partial": 0, "minimal": 0}
    total_steps = 0
    for chain in chains:
        completeness_counts[chain.chain_completeness] += 1
        total_steps += len(chain.process_chain)

    output = {
        "paper": paper_name,
        "doc_name": v7_data.get("doc_name", ""),
        "created_at": datetime.now().isoformat(),
        "model": model_name,
        "stats": {
            "total_measurements": len(measurements),
            "cited_measurements_skipped": cited_count,
            "provenance_groups": len(groups),
            "chains_extracted": len(chains),
            "total_process_steps": total_steps,
            "completeness": completeness_counts,
        },
        "paper_context": paper_context,
        "provenance_groups": [chain.model_dump() for chain in chains],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved: {output_path.name}")
    print(f"     Groups: {len(chains)} | Steps: {total_steps} | "
          f"Full: {completeness_counts['full']} | Partial: {completeness_counts['partial']} | "
          f"Minimal: {completeness_counts['minimal']}")

    return {
        "paper": paper_name,
        "status": "success",
        "provenance_groups": len(chains),
        "total_steps": total_steps,
        "completeness": completeness_counts,
        "cited_measurements_skipped": cited_count,
    }


# ============================================================================
# Consolidation
# ============================================================================

def consolidate_provenance(output_dir: Path):
    """Merge all per-paper provenance JSONs into a single consolidated file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prov_files = sorted(output_dir.glob("*_provenance.json"))
    if not prov_files:
        print("⚠️  No per-paper provenance files found for consolidation")
        return

    all_groups = []
    total_stats = {
        "papers": 0,
        "total_measurements": 0,
        "cited_measurements_skipped": 0,
        "provenance_groups": 0,
        "total_process_steps": 0,
        "completeness": {"full": 0, "partial": 0, "minimal": 0},
    }

    for pf in prov_files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_name = data.get("paper", pf.stem)
        stats = data.get("stats", {})

        total_stats["papers"] += 1
        total_stats["total_measurements"] += stats.get("total_measurements", 0)
        total_stats["cited_measurements_skipped"] += stats.get("cited_measurements_skipped", 0)
        total_stats["provenance_groups"] += stats.get("provenance_groups", 0)
        total_stats["total_process_steps"] += stats.get("total_process_steps", 0)
        for level in ["full", "partial", "minimal"]:
            total_stats["completeness"][level] += stats.get("completeness", {}).get(level, 0)

        for group in data.get("provenance_groups", []):
            group["paper"] = paper_name
            all_groups.append(group)

    consolidated = {
        "created_at": datetime.now().isoformat(),
        "stats": total_stats,
        "provenance_groups": all_groups,
    }

    out_path = output_dir / "consolidated_provenance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    print(f"💾 Consolidated provenance: {out_path}")
    print(f"📊 Papers: {total_stats['papers']} | Groups: {total_stats['provenance_groups']} | "
          f"Steps: {total_stats['total_process_steps']}")
    print(f"   Completeness — Full: {total_stats['completeness']['full']}, "
          f"Partial: {total_stats['completeness']['partial']}, "
          f"Minimal: {total_stats['completeness']['minimal']}")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="T0 Provenance Tracer — trace fabrication chains from v8 extraction")
    parser.add_argument("--sample", help="Process only one paper by exact folder name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--v8-dir", type=Path, default=v8_dir, help="Directory containing v8 output folders")
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR, help="Directory containing the paper folders")
    parser.add_argument("--force", action="store_true", help="Re-process even if provenance output exists")
    parser.add_argument("--delete-cache", action="store_true", help="Delete Gemini cache after each paper")
    parser.add_argument("--skip-consolidate", action="store_true", help="Skip consolidated output generation")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT, help="Max concurrent LLM batch calls")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Provenance groups per LLM batch")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: set GEMINI_API_KEY or GOOGLE_API_KEY")
        return

    client = genai.Client(api_key=api_key)

    if args.sample:
        paper_names = [args.sample]
    else:
        paper_names = discover_papers(args.papers_dir, args.v8_dir)

    if not paper_names:
        print("❌ No papers found to process")
        return

    print("🚀 T0 Provenance Tracer")
    print(f"   Model: {args.model}")
    print(f"   V7 dir: {args.v8_dir}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Papers to process: {len(paper_names)}")

    results = []
    for paper_name in paper_names:
        result = await process_paper(
            client=client,
            paper_name=paper_name,
            model_name=args.model,
            v8_dir=args.v8_dir,
            papers_dir=args.papers_dir,
            force=args.force,
            delete_cache=args.delete_cache,
            concurrency_limit=max(1, args.concurrency),
            batch_size=max(1, args.batch_size),
        )
        results.append(result)

    # Summary
    success = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") not in ("success", "skipped")]

    print(f"\n{'='*70}")
    print("📊 Run Summary")
    print(f"{'='*70}")
    print(f"   ✅ Success: {len(success)}")
    print(f"   ⏭️  Skipped: {len(skipped)}")
    print(f"   ❌ Failed:  {len(failed)}")

    if success:
        total_groups = sum(r.get("provenance_groups", 0) for r in success)
        total_steps = sum(r.get("total_steps", 0) for r in success)
        total_cited = sum(r.get("cited_measurements_skipped", 0) for r in success)
        print(f"   🔗 Total groups: {total_groups} | Steps: {total_steps} | Cited skipped: {total_cited}")

    if failed:
        print("\n   Failed papers:")
        for r in failed:
            print(f"      - {r.get('paper')}: {r.get('status')}")

    if not args.skip_consolidate:
        print(f"\n{'='*70}")
        print("📦 Generating Consolidated Provenance Output")
        print(f"{'='*70}")
        consolidate_provenance(args.v8_dir)

    # Save run metadata
    args.v8_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "papers": paper_names,
        "results": results,
    }
    run_meta_path = args.v8_dir / f"provenance_run_{args.model.replace('-', '_')}.json"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Run metadata: {run_meta_path}")


if __name__ == "__main__":
    asyncio.run(main())
