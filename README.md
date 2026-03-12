# VAE Implication Testing Pipeline
## Perturb-seq × Causal Inference × PAC Learning

Tests whether biological implications of the form **KO(gene A) ⇒ gene B active**
hold statistically using a VAE latent space, causal discovery, generative
stress-testing, and invariance analysis.

---

## Setup

```bash
# Python 3.10+ recommended
pip install scvi-tools anndata scanpy causal-learn lingam \
            scipy scikit-learn matplotlib seaborn torch
```

GPU strongly recommended for step 02. On CPU, set:
- `N_LATENT = 16`
- `MAX_EPOCHS = 100`

---

## Run order

```bash
python 01_download_and_prep.py   # ~10 min (download + preprocess)
python 02_train_vae.py           # ~30 min GPU / ~3 hr CPU
python 03_implication_testing.py # ~15 min
python 04_visualize.py           # ~1 min
```

---

## Output structure

```
data/
  raw/replogle2022.h5ad          # downloaded dataset
  processed/perturb_prepped.h5ad # QC-filtered AnnData
  processed/latent.csv           # Z coordinates + labels

models/
  scvi_model/                    # saved scVI model

results/
  implication_results.json       # full results per pair

figures/
  pac_bounds.png                 # PAC violation bound bar chart
  invariance_heatmap.png         # violation rates × environments
  summary_table.png              # final verdict table
  umap_perturbations.png         # UMAP coloured by perturbation
```

---

## What each script does

| Script | Role |
|--------|------|
| `01` | Downloads Replogle 2022 K562 Perturb-seq, runs QC, creates binary A/B labels |
| `02` | Trains scVI VAE (negative binomial, 32-dim latent), extracts Z, plots UMAP |
| `03` | For each (A,B) pair: PAC bound, PC-algorithm causal discovery, generative stress-test via decoder, ICP-style invariance across latent partitions, Rosenbaum sensitivity |
| `04` | Summary figures |

---

## Verdict criteria

| Verdict | Criteria |
|---------|----------|
| **STRONG** | Zero violations observed + invariant across environments + causal edge found + stress-test violation rate < 5% |
| **MODERATE** | Zero violations + (invariant OR causal edge) |
| **WEAK** | Zero violations only |
| **REJECTED** | Violations observed |

---

## Implication pairs tested

These are biologically motivated claims — all have prior literature support
but are not exhaustively proven:

- `MYC KO ⇒ CDKN1A high` — MYC suppresses p21; knockout releases it
- `TP53 KO ⇒ CDKN1A low` — TP53 activates p21; knockout silences it
- `BRCA1 KO ⇒ FANCD2 active` — BRCA1/FANCD2 co-operate in DNA repair
- `KRAS KO ⇒ DUSP6 low` — KRAS drives DUSP6 via MAPK; knockout drops it
- `PTEN KO ⇒ AKT1 high` — PTEN suppresses AKT; knockout activates it

To add your own pair, extend the `IMPLICATION_PAIRS` list in both
`01_download_and_prep.py` and `03_implication_testing.py`.

---

## Key design decisions

**Why scVI?**  
Designed for single-cell count data. Uses negative binomial likelihood,
which correctly handles the zero-inflation and overdispersion in RNA-seq.

**Why latent stress-testing instead of raw sampling?**  
Random noise in 20,000-gene space generates biologically impossible cells.
The VAE decoder constrains samples to the learned data manifold — violations
found this way are meaningful.

**Why KMeans partitions for ICP?**  
Real heterogeneous environments aren't available for a single dataset.
Latent space partitions approximate different "contexts" — cell cycle states,
metabolic states, etc. A relationship that survives across all partitions
is evidence of invariance.

**Why Rosenbaum Gamma?**  
Quantifies how much unmeasured confounding would be needed to explain
away zero violations. A Gamma of 1.0 = no robustness; Gamma of 10 = very robust.
