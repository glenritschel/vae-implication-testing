"""
02_train_vae.py
---------------
Trains an scVI VAE on the preprocessed Perturb-seq data.
scVI is a VAE designed for single-cell RNA-seq — it uses a negative
binomial likelihood (appropriate for count data) and learns a latent
space Z that captures cell state independent of technical noise.

Requirements:
    pip install scvi-tools

GPU strongly recommended. On CPU, reduce N_LATENT and MAX_EPOCHS.
"""

import os
import torch
import scvi
import anndata as ad
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./data/processed/perturb_prepped.h5ad"
MODEL_DIR   = "./models/scvi_model"
LATENT_PATH = "./data/processed/latent.csv"

N_LATENT    = 32       # latent dimensions; increase to 64 for full dataset
MAX_EPOCHS  = 400      # reduce to 100 for quick dev run
BATCH_SIZE  = 256
TRAIN_FRAC  = 0.9

os.makedirs("./models", exist_ok=True)

print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
scvi.settings.seed = 42

# ── Load data ─────────────────────────────────────────────────────────────────
adata = ad.read_h5ad(DATA_PATH)
print(f"Loaded: {adata.shape}")

# ── Setup scVI ────────────────────────────────────────────────────────────────
# scVI needs to know which layer holds raw counts
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",               # raw integer counts
    # batch_key="batch",          # uncomment if you have batch metadata
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = scvi.model.SCVI(
    adata,
    n_latent=N_LATENT,
    n_layers=2,
    n_hidden=256,
    dropout_rate=0.1,
    dispersion="gene",            # per-gene dispersion (negative binomial)
    gene_likelihood="nb",         # negative binomial for count data
)

print(model)

model.train(
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    train_size=TRAIN_FRAC,
    early_stopping=True,
    early_stopping_patience=20,
    plan_kwargs={"lr": 1e-3},
)

# ── Extract latent representation ─────────────────────────────────────────────
Z = model.get_latent_representation()   # (n_cells, N_LATENT)

# Save latent coords alongside perturbation labels
latent_df = pd.DataFrame(
    Z,
    columns=[f"z_{i}" for i in range(N_LATENT)],
    index=adata.obs_names,
)
# Attach all obs columns (perturbation labels, KO_* and HIGH_* flags)
latent_df = pd.concat([adata.obs, latent_df], axis=1)
latent_df.to_csv(LATENT_PATH, index=True)
print(f"Latent representation saved to {LATENT_PATH}")

# ── Save model ────────────────────────────────────────────────────────────────
model.save(MODEL_DIR, overwrite=True)
print(f"Model saved to {MODEL_DIR}")

# ── Quick sanity: UMAP of latent space coloured by perturbation ───────────────
import scanpy as sc
adata.obsm["X_scVI"] = Z
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color=["perturbation"],
    save="_perturbations.png",
    show=False,
)
print("UMAP saved to figures/umap_perturbations.png")
