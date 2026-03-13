"""
07_reanalysis_v2.py
--------------------
Re-runs implication testing using proper scanpy gene signature scores
from 06_pathway_scores.py instead of the uniform-noise pathway columns.

Run after 06_pathway_scores.py.
Writes: results/reanalysis_v2_results.json
        figures/population_effects_v2.png
        figures/pathway_score_distributions.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

LATENT_PATH = "./data/processed/latent_with_pathways.csv"
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
ALPHA       = 0.05
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(LATENT_PATH, index_col=0)
print(f"Loaded {len(df)} cells")

# ── PAC bound ─────────────────────────────────────────────────────────────────
def pac_bound(n, k, alpha=ALPHA):
    if n == 0: return None
    if k == 0: return 1 - (alpha ** (1.0 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))

# ── Implication pairs using proper signature scores ───────────────────────────
# (ko_gene, score_col, direction, claim)
# direction='up'   => KO activates pathway (violation = score below ctrl median)
# direction='down' => KO suppresses pathway (violation = score above ctrl median)
PAIRS = [
    ("MYC",   "score_E2F_targets", "up",
     "MYC KO de-represses E2F target genes"),

    ("TP53",  "score_p53_targets", "down",
     "TP53 KO silences p53 target genes including CDKN1A"),

    ("BRCA1", "score_DNA_damage",  "up",
     "BRCA1 KO activates DNA damage response"),

    ("KRAS",  "score_MAPK_ERK",   "down",
     "KRAS KO suppresses MAPK/ERK signalling"),

    ("PTEN",  "score_PI3K_AKT",   "up",
     "PTEN KO activates PI3K/AKT pathway"),
]

ctrl_mask = df["perturbation"] == "non-targeting"
results   = {}

for (ko_gene, score_col, direction, claim) in PAIRS:
    ko_col = f"KO_{ko_gene}"
    key    = f"{ko_gene}=>{score_col}"

    if score_col not in df.columns:
        results[key] = {"error": f"{score_col} not in CSV — run 06 first"}
        continue

    ctrl_scores = df.loc[ctrl_mask, score_col]
    ko_scores   = df.loc[df[ko_col] == 1, score_col]
    n           = len(ko_scores)
    ctrl_median = ctrl_scores.median()

    # Binarize at control median
    if direction == "up":
        # Expect KO > ctrl_median. Violation = KO score below median.
        n_viol = int((ko_scores < ctrl_median).sum())
        mw_alt = "greater"
    else:
        # Expect KO < ctrl_median. Violation = KO score above median.
        n_viol = int((ko_scores > ctrl_median).sum())
        mw_alt = "less"

    pac  = pac_bound(n, n_viol)
    mw   = stats.mannwhitneyu(ko_scores, ctrl_scores, alternative=mw_alt)
    delta = float(ko_scores.mean() - ctrl_scores.mean())
    direction_correct = (delta > 0) == (direction == "up")

    # Cohen's d effect size
    pooled_std = np.sqrt((ko_scores.std()**2 + ctrl_scores.std()**2) / 2)
    cohens_d   = delta / pooled_std if pooled_std > 0 else 0.0

    verdict = (
        "SUPPORTED"  if mw.pvalue < 0.05  and direction_correct else
        "TRENDING"   if mw.pvalue < 0.20  and direction_correct else
        "REJECTED"
    )

    results[key] = {
        "gene_a": ko_gene, "score": score_col,
        "direction": direction, "claim": claim,
        "n_ko": n, "n_ctrl": int(ctrl_mask.sum()),
        "n_violations_binary": n_viol,
        "violation_rate": round(n_viol / n, 3) if n > 0 else None,
        "pac_bound": round(pac, 4) if pac else None,
        "ko_mean": round(float(ko_scores.mean()), 4),
        "ctrl_mean": round(float(ctrl_scores.mean()), 4),
        "delta": round(delta, 4),
        "cohens_d": round(cohens_d, 4),
        "mannwhitney_p": round(float(mw.pvalue), 4),
        "direction_correct": direction_correct,
        "verdict": verdict,
    }

    print(f"\n{key}")
    print(f"  {claim}")
    print(f"  N(KO)={n}  violations={n_viol}  rate={n_viol/n:.3f}")
    print(f"  delta={delta:+.4f}  Cohen's d={cohens_d:+.4f}  p={mw.pvalue:.4f}")
    print(f"  VERDICT: {verdict}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = os.path.join(RESULTS_DIR, "reanalysis_v2_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out}")

# ── Figure 1: pathway score distributions (violin) ────────────────────────────
VERDICT_COLORS = {"SUPPORTED":"#2ecc71","TRENDING":"#f39c12","REJECTED":"#e74c3c"}

fig, axes = plt.subplots(1, len(PAIRS), figsize=(15, 5), sharey=False)
fig.suptitle("Pathway Scores: KO vs Non-targeting Controls\n(Scanpy gene signature scores)", fontsize=12)

for ax, (ko_gene, score_col, direction, claim) in zip(axes, PAIRS):
    ko_col = f"KO_{ko_gene}"
    key    = f"{ko_gene}=>{score_col}"
    res    = results.get(key, {})
    verdict = res.get("verdict", "REJECTED")

    if score_col not in df.columns:
        ax.set_title(f"{ko_gene}\n(missing)")
        continue

    plot_df = pd.DataFrame({
        "score": pd.concat([
            df.loc[ctrl_mask, score_col],
            df.loc[df[ko_col]==1, score_col]
        ]),
        "group": (
            ["ctrl"] * ctrl_mask.sum() +
            [f"KO\n{ko_gene}"] * int((df[ko_col]==1).sum())
        )
    })

    sns.violinplot(
        data=plot_df, x="group", y="score", ax=ax,
        palette=["#3498db", VERDICT_COLORS[verdict]],
        inner="box", cut=0,
    )
    p   = res.get("mannwhitney_p", 1.0)
    d   = res.get("cohens_d", 0.0)
    ax.set_title(
        f"{ko_gene} → {score_col.replace('score_','')}\n"
        f"p={p:.3f}  d={d:+.3f}  [{verdict}]",
        fontsize=8
    )
    ax.set_xlabel("")
    ax.set_ylabel("Signature score" if ax == axes[0] else "")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "population_effects_v2.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved population_effects_v2.png")

# ── Figure 2: Effect size summary ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
labels, deltas, colors, ps = [], [], [], []
for key, res in results.items():
    if "error" in res: continue
    labels.append(f"{res['gene_a']} KO\n→ {res['score'].replace('score_','')}")
    deltas.append(res.get("cohens_d", 0))
    colors.append(VERDICT_COLORS.get(res.get("verdict","REJECTED"), "#e74c3c"))
    ps.append(res.get("mannwhitney_p", 1.0))

x = np.arange(len(labels))
bars = ax.bar(x, deltas, color=colors, edgecolor="white", width=0.6)
ax.axhline(0, color="white", linewidth=0.8)
ax.axhline(0.2,  color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="small effect (d=0.2)")
ax.axhline(-0.2, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cohen's d  (KO − ctrl)")
ax.set_title("Effect Sizes: Gene Knockout on Pathway Signatures", fontsize=12)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, p in zip(bars, ps):
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.005 if bar.get_height() >= 0 else -0.015),
            sig, ha="center", fontsize=11, color="white")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=v) for v,c in VERDICT_COLORS.items()]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
          facecolor="#2a2a3e", labelcolor="white")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "effect_sizes_v2.png"), dpi=150,
            bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("Saved effect_sizes_v2.png")
