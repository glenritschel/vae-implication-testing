"""
06_pathway_scores.py
---------------------
Computes proper pathway activity scores using curated gene signatures
via scanpy's score_genes() (Seurat-style aggregate z-score).

This replaces the broken pathway columns from 01_download_and_prep.py,
which produced uniform noise because the target genes were filtered out
during HVG selection.

score_genes() works on ALL genes (pre-HVG filtering), so it bypasses
the HVG bottleneck entirely.

Run BEFORE 05_reanalysis.py. Overwrites data/processed/latent.csv
with proper pathway score columns.

Requirements: scanpy, anndata (already in requirements.txt)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import stats

DATA_PATH   = "./data/replogle2022.h5ad"
LATENT_PATH = "./data/processed/latent.csv"
OUT_PATH    = "./data/processed/latent_with_pathways.csv"

# ── Curated gene signatures ────────────────────────────────────────────────────
# These are compact, well-validated signatures. scanpy score_genes handles
# genes not present in the dataset gracefully (skips them).
SIGNATURES = {
    # MYC KO => E2F targets de-repressed (MYC and E2F co-regulate)
    "E2F_targets": [
        "E2F1","E2F2","E2F3","CCNE1","CCNE2","CDC6","MCM2","MCM3",
        "MCM4","MCM5","MCM7","PCNA","RRM1","RRM2","TYMS","DHFR",
    ],
    # TP53 KO => p53 target genes silenced
    "p53_targets": [
        "CDKN1A","MDM2","GADD45A","BBC3","PUMA","BAX","NOXA",
        "TIGAR","GDF15","SESN1","SESN2","TP53I3","FAS","TNFRSF10B",
    ],
    # BRCA1 KO => DNA damage response activated
    "DNA_damage": [
        "BRCA1","BRCA2","RAD51","FANCD2","FANCI","RPA1","RPA2",
        "H2AFX","CHEK1","CHEK2","ATM","ATR","ATRIP","MRE11","NBN",
    ],
    # KRAS KO => MAPK/ERK pathway suppressed
    "MAPK_ERK": [
        "DUSP6","DUSP4","SPRY2","SPRY4","ETV4","ETV5","ETV1",
        "FOSL1","FOSL2","JUN","FOS","EGR1","MYC","CCND1",
    ],
    # PTEN KO => PI3K/AKT pathway activated
    "PI3K_AKT": [
        "AKT1","AKT2","AKT3","PHLPP1","PHLPP2","FOXO1","FOXO3",
        "GSK3B","TSC2","RPTOR","RPS6KB1","EIF4EBP1","SGK1","PDPK1",
    ],
}

print("Loading AnnData...")
adata = ad.read_h5ad(DATA_PATH)
print(f"Shape: {adata.shape}")

# score_genes needs log-normalised counts in adata.X
# If X still has raw counts, normalise first
import scipy.sparse as sp
X = adata.X
vals = X.data if sp.issparse(X) else X.flatten()
if vals.max() > 20:
    print("Normalising counts for scoring...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# ── Score each signature ───────────────────────────────────────────────────────
for sig_name, gene_list in SIGNATURES.items():
    # Filter to genes actually present in the dataset
    present = [g for g in gene_list if g in adata.var_names]
    missing = [g for g in gene_list if g not in adata.var_names]
    print(f"\n{sig_name}: {len(present)}/{len(gene_list)} genes found")
    if missing:
        print(f"  Missing: {missing}")
    if len(present) < 3:
        print(f"  WARNING: fewer than 3 genes found — score will be unreliable")

    sc.tl.score_genes(
        adata,
        gene_list=present,
        score_name=f"score_{sig_name}",
        use_raw=False,
    )

# ── Load existing latent CSV and merge pathway scores ─────────────────────────
latent = pd.read_csv(LATENT_PATH, index_col=0)

for sig_name in SIGNATURES:
    col = f"score_{sig_name}"
    if col in adata.obs.columns:
        latent[col] = adata.obs[col].values
        # Report distribution
        ctrl_mask = adata.obs["perturbation"] == "non-targeting"
        ctrl_scores = adata.obs.loc[ctrl_mask, col]
        print(f"\n{col}:")
        print(f"  Global mean={latent[col].mean():.4f}  std={latent[col].std():.4f}")
        print(f"  Ctrl  mean={ctrl_scores.mean():.4f}  std={ctrl_scores.std():.4f}")
        for gene in ["MYC","TP53","BRCA1","KRAS","PTEN"]:
            ko_mask = adata.obs["perturbation"] == gene
            if ko_mask.sum() > 0:
                ko_scores = adata.obs.loc[ko_mask, col]
                delta = ko_scores.mean() - ctrl_scores.mean()
                mw = stats.mannwhitneyu(ko_scores, ctrl_scores, alternative="two-sided")
                print(f"  {gene:6s} KO: mean={ko_scores.mean():.4f}  delta={delta:+.4f}  p={mw.pvalue:.4f}")

latent.to_csv(OUT_PATH, index=True)
print(f"\nSaved enriched latent CSV to {OUT_PATH}")
print("\nNext step: re-run 05_reanalysis.py after updating LATENT_PATH to point to")
print(f"  {OUT_PATH}")
print("and updating the pathway column names to use 'score_' prefix.")
