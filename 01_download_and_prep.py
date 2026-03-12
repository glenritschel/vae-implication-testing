"""
01_download_and_prep.py
-----------------------
Downloads the Replogle 2022 Perturb-seq dataset (K562 essential screen)
and prepares it for VAE training.

Requirements:
    pip install scvi-tools anndata scanpy gdown

Dataset: ~4GB download. If bandwidth is limited, the script will use a
cached version if it exists at DATA_DIR/replogle2022.h5ad.
"""

import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"
OUT_DIR    = "./data/processed"
CACHE_FILE = os.path.join(DATA_DIR, "replogle2022.h5ad")

# Figshare direct link for Replogle et al. 2022 K562 essential screen
# (Norman et al. subset — smaller, ~800MB, good for development)
# Full dataset: https://figshare.com/articles/dataset/21224008
FIGSHARE_URL = (
    "https://figshare.com/ndownloader/files/35773219"
)

# Implication pairs to test: (perturbation_gene, readout_gene_or_pathway)
# These are biologically motivated A => B claims.
IMPLICATION_PAIRS = [
    ("MYC",    "E2F_target_genes"),   # MYC KO => E2F targets down
    ("TP53",   "CDKN1A"),             # TP53 KO => p21 (CDKN1A) down
    ("BRCA1",  "DNA_damage_response"),
    ("KRAS",   "MAPK_pathway"),
    ("PTEN",   "PI3K_AKT_pathway"),
]

# Minimum cells per perturbation to keep
MIN_CELLS = 50

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────────────
if not os.path.exists(CACHE_FILE):
    print("Downloading Perturb-seq dataset...")
    import urllib.request
    urllib.request.urlretrieve(FIGSHARE_URL, CACHE_FILE)
    print(f"Saved to {CACHE_FILE}")
else:
    print(f"Using cached file: {CACHE_FILE}")

# ── Load & basic QC ───────────────────────────────────────────────────────────
print("Loading AnnData...")
adata = sc.read_h5ad(CACHE_FILE)
print(f"Raw shape: {adata.shape}")   # (cells, genes)
print(adata.obs.columns.tolist())    # inspect metadata columns

# Standardise perturbation column name (varies by dataset version)
PERT_COL = None
for candidate in ["perturbation", "gene_name", "target_gene", "guide_ids"]:
    if candidate in adata.obs.columns:
        PERT_COL = candidate
        break
assert PERT_COL, "Could not find perturbation column — inspect adata.obs"
print(f"Using perturbation column: '{PERT_COL}'")

# ── Filter low-count perturbations ───────────────────────────────────────────
pert_counts = adata.obs[PERT_COL].value_counts()
keep_perts  = pert_counts[pert_counts >= MIN_CELLS].index
adata       = adata[adata.obs[PERT_COL].isin(keep_perts)].copy()
print(f"After filtering (<{MIN_CELLS} cells): {adata.shape}")

# ── Standard scRNA-seq preprocessing ─────────────────────────────────────────
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)

# Store raw counts for scVI (scVI expects raw integer counts)
adata.layers["counts"] = adata.X.copy()

# Normalise & log for downstream visualization only
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)

# ── Build binary implication labels ──────────────────────────────────────────
# For each (A, B) pair:
#   A_col = 1 if gene A was knocked out, else 0
#   B_col = 1 if the readout B is "active" (above median in control cells)

control_mask = adata.obs[PERT_COL] == "non-targeting"

for (gene_a, readout_b) in IMPLICATION_PAIRS:
    # A: perturbation indicator
    a_col = f"KO_{gene_a}"
    adata.obs[a_col] = (adata.obs[PERT_COL] == gene_a).astype(int)

    # B: readout indicator — if readout_b is a single gene
    if readout_b in adata.var_names:
        expr     = np.array(adata[:, readout_b].X.todense()).flatten()
        ctrl_med = np.median(expr[control_mask])
        b_col    = f"HIGH_{readout_b}"
        adata.obs[b_col] = (expr > ctrl_med).astype(int)
    else:
        # readout_b is a pathway label — expect it in adata.obs already
        # or mark as missing for manual addition
        print(f"  Pathway '{readout_b}' not found as gene; "
              f"add pathway scores separately (see 01b_pathway_scores.py)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "perturb_prepped.h5ad")
adata.write_h5ad(out_path)
print(f"Saved preprocessed data to {out_path}")
print(f"Final shape: {adata.shape}")
