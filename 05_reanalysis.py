"""
05_reanalysis.py
----------------
Fixes three issues from the first run:

  1. Wires pathway scores to HIGH_* binary labels using per-perturbation
     effect direction (some KOs activate, some suppress the pathway)
  2. Re-runs implication testing with corrected violation logic
  3. Adds population-level aggregation to handle single-cell noise
     (tests implication at the *perturbation mean* level, not per-cell)
  4. Adds Mann-Whitney U test as a continuous alternative to binary violation

Run from the repo root after 01-03 have completed.
Writes: results/reanalysis_results.json
        figures/reanalysis_pac_bounds.png
        figures/population_level_effects.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

LATENT_PATH  = "./data/processed/latent.csv"
RESULTS_DIR  = "./results"
FIGURES_DIR  = "./figures"
ALPHA        = 0.05
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

import anndata as ad
import scipy.sparse

df = pd.read_csv(LATENT_PATH, index_col=0)

# Extract raw CDKN1A expression from h5ad to df since it's not saved in latent.csv
try:
    adata = ad.read_h5ad("./data/processed/perturb_prepped.h5ad")
    if "CDKN1A" in adata.var_names:
        X_cdkn1a = adata[:, "CDKN1A"].X
        if hasattr(X_cdkn1a, 'todense'):
            expr_cdkn1a = np.array(X_cdkn1a.todense()).flatten()
        else:
            expr_cdkn1a = np.array(X_cdkn1a).flatten()
        df["CDKN1A"] = expr_cdkn1a
except Exception as e:
    print(f"Failed to load raw CDKN1A counts: {e}")

print(f"Loaded {len(df)} cells")
print(f"Perturbations: {df['perturbation'].value_counts().to_dict()}")
print(f"Pathway columns: {[c for c in df.columns if c not in df.filter(like='z_').columns and not c.startswith('KO_') and not c.startswith('HIGH_') and c not in ['perturbation','n_genes','_scvi_batch','_scvi_labels']]}")

# ── PAC bound helper ───────────────────────────────────────────────────────────
def pac_bound(n, k, alpha=ALPHA):
    if n == 0:
        return None
    if k == 0:
        return 1 - (alpha ** (1.0 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


# ══════════════════════════════════════════════════════════════════════════════
# Implication pairs — with CORRECTED violation direction
# violation_when: 'low'  = violation when B is LOW  (KO activates B)
#                 'high' = violation when B is HIGH  (KO suppresses B)
# ══════════════════════════════════════════════════════════════════════════════
PAIRS = [
    # (ko_gene, readout_col, readout_label, violation_when, biological_claim)
    ("MYC",   "HIGH_CDKN1A",        "CDKN1A_high",        "low",
     "MYC KO de-represses CDKN1A => expect HIGH_CDKN1A=1"),

    ("TP53",  "HIGH_CDKN1A",        "CDKN1A_high",        "high",
     "TP53 activates CDKN1A; KO => expect CDKN1A LOW (violation=HIGH)"),

    ("BRCA1", "DNA_damage_response", "DNA_damage_active",  "low",
     "BRCA1 KO activates DNA damage response => expect score HIGH"),

    ("KRAS",  "MAPK_pathway",        "MAPK_active",        "high",
     "KRAS KO suppresses MAPK => expect score LOW (violation=HIGH)"),

    ("PTEN",  "PI3K_AKT_pathway",    "PI3K_AKT_active",   "low",
     "PTEN KO activates PI3K/AKT => expect score HIGH"),
]

# ── Binarize continuous pathway scores at control median ──────────────────────
ctrl_mask = df["perturbation"] == "non-targeting"
for col in ["DNA_damage_response", "MAPK_pathway", "PI3K_AKT_pathway", "E2F_target_genes"]:
    if col in df.columns:
        med = df.loc[ctrl_mask, col].median()
        df[f"HIGH_{col}"] = (df[col] > med).astype(int)
        print(f"Binarized {col} at median={med:.4f}")

# ── Fix TP53 direction: violation = HIGH_CDKN1A=1 when KO'd ─────────────────
# (TP53 normally activates CDKN1A; its KO should silence it)
# We'll handle direction in the loop below.

results = {}

for (ko_gene, readout_col, readout_label, violation_when, claim) in PAIRS:
    ko_col = f"KO_{ko_gene}"
    hi_col = readout_col if readout_col.startswith("HIGH_") else f"HIGH_{readout_col}"

    if ko_col not in df.columns:
        results[f"{ko_gene}=>{readout_label}"] = {"error": f"{ko_col} not found"}
        continue
    if hi_col not in df.columns:
        results[f"{ko_gene}=>{readout_label}"] = {"error": f"{hi_col} not found"}
        continue

    a1   = df[df[ko_col] == 1]
    ctrl = df[ctrl_mask]
    n    = len(a1)

    # Violation count depends on direction
    if violation_when == "low":
        # Implication: KO => B HIGH. Violation = B is LOW.
        n_viol = int((a1[hi_col] == 0).sum())
    else:
        # Implication: KO => B LOW. Violation = B is HIGH.
        n_viol = int((a1[hi_col] == 1).sum())

    pac = pac_bound(n, n_viol)
    viol_rate = n_viol / n if n > 0 else None

    # ── Continuous Mann-Whitney (more powerful than binary) ──────────────────
    raw_col = readout_col.replace("HIGH_", "") if readout_col.startswith("HIGH_") else readout_col
    if raw_col in df.columns:
        a1_scores   = df.loc[df[ko_col] == 1, raw_col]
        ctrl_scores = df.loc[ctrl_mask, raw_col]
        if violation_when == "low":
            alt = "greater"   # expect KO > ctrl
        else:
            alt = "less"      # expect KO < ctrl
        mw = stats.mannwhitneyu(a1_scores, ctrl_scores, alternative=alt)
        effect_direction = float(a1_scores.mean() - ctrl_scores.mean())
        mw_result = {
            "statistic": round(float(mw.statistic), 2),
            "p_value": round(float(mw.pvalue), 4),
            "effect_direction": round(effect_direction, 4),
            "expected_direction": "positive" if violation_when == "low" else "negative",
            "direction_correct": (effect_direction > 0) == (violation_when == "low"),
        }
    else:
        mw_result = None

    # ── Population-level test ─────────────────────────────────────────────────
    # Aggregate to perturbation mean and test if mean is on the correct side
    if raw_col in df.columns:
        ko_mean   = float(df.loc[df[ko_col] == 1, raw_col].mean())
        ctrl_mean = float(ctrl_scores.mean())
        pop_correct = (ko_mean > ctrl_mean) == (violation_when == "low")
    else:
        ko_mean, ctrl_mean, pop_correct = None, None, None

    results[f"{ko_gene}=>{readout_label}"] = {
        "gene_a": ko_gene,
        "readout": readout_label,
        "claim": claim,
        "violation_when": violation_when,
        "n_a1": n,
        "n_violations": n_viol,
        "violation_rate": round(viol_rate, 3) if viol_rate is not None else None,
        "pac_bound": round(pac, 4) if pac else None,
        "mann_whitney": mw_result,
        "population_level": {
            "ko_mean": round(ko_mean, 4) if ko_mean else None,
            "ctrl_mean": round(ctrl_mean, 4) if ctrl_mean else None,
            "direction_correct": pop_correct,
        },
        "verdict": (
            "SUPPORTED"  if (mw_result and mw_result["p_value"] < 0.05
                             and mw_result["direction_correct"])           else
            "TRENDING"   if (mw_result and mw_result["p_value"] < 0.20
                             and mw_result["direction_correct"])           else
            "REJECTED"
        )
    }

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = os.path.join(RESULTS_DIR, "reanalysis_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out}")

# ── Figure: population-level effect sizes ─────────────────────────────────────
fig, axes = plt.subplots(1, len(PAIRS), figsize=(14, 4), sharey=False)
fig.suptitle("Population-level Pathway Effects: KO vs Non-targeting", fontsize=13, y=1.02)

VERDICT_COLORS = {"SUPPORTED": "#2ecc71", "TRENDING": "#f39c12", "REJECTED": "#e74c3c"}

for ax, (ko_gene, readout_col, readout_label, violation_when, _) in zip(axes, PAIRS):
    raw_col = readout_col.replace("HIGH_", "") if readout_col.startswith("HIGH_") else readout_col
    ko_col  = f"KO_{ko_gene}"
    key     = f"{ko_gene}=>{readout_label}"
    res     = results.get(key, {})

    if raw_col not in df.columns or ko_col not in df.columns:
        ax.set_title(f"{ko_gene}\n(missing data)")
        continue

    ko_scores   = df.loc[df[ko_col] == 1, raw_col]
    ctrl_scores = df.loc[ctrl_mask, raw_col]

    bp = ax.boxplot(
        [ctrl_scores, ko_scores],
        labels=["ctrl", f"KO\n{ko_gene}"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    verdict = res.get("verdict", "REJECTED")
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor(VERDICT_COLORS.get(verdict, "#e74c3c"))

    mw = res.get("mann_whitney", {}) or {}
    p  = mw.get("p_value", 1.0)
    ax.set_title(
        f"{ko_gene} → {raw_col.replace('_',' ')}\n"
        f"p={p:.3f}  [{verdict}]",
        fontsize=8,
    )
    ax.set_ylabel("Pathway score" if ax == axes[0] else "")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "population_level_effects.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved population_level_effects.png")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== REANALYSIS SUMMARY ===")
for key, res in results.items():
    mw  = res.get("mann_whitney") or {}
    pop = res.get("population_level") or {}
    print(f"\n{key}")
    print(f"  Claim    : {res.get('claim','')}")
    print(f"  N(KO)    : {res.get('n_a1')}")
    print(f"  Violation rate (binary): {res.get('violation_rate')}")
    print(f"  Mann-Whitney p         : {mw.get('p_value')}  direction_correct={mw.get('direction_correct')}")
    print(f"  Population direction   : {pop.get('direction_correct')}  (ko={pop.get('ko_mean')} vs ctrl={pop.get('ctrl_mean')})")
    print(f"  VERDICT  : {res.get('verdict')}")
