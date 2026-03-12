"""
04_visualize.py
---------------
Produces summary figures from the implication testing results.

Figures generated:
  - pac_bounds.png        : PAC violation bounds per implication pair
  - latent_umap.png       : UMAP with A=1/B=0 cells highlighted
  - invariance_heatmap.png: violation rates across pseudo-environments
  - summary_table.png     : final verdict table
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

RESULTS_DIR  = "./results"
FIGURES_DIR  = "./figures"
LATENT_PATH  = "./data/processed/latent.csv"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "implication_results.json")) as f:
    results = json.load(f)

latent = pd.read_csv(LATENT_PATH, index_col=0)

VERDICT_COLORS = {
    "STRONG":   "#2ecc71",
    "MODERATE": "#f39c12",
    "WEAK":     "#e67e22",
    "REJECTED": "#e74c3c",
    "UNKNOWN":  "#95a5a6",
}

# ── Figure 1: PAC bounds bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
pairs, bounds, colors = [], [], []

for key, res in results.items():
    obs = res.get("observational", {})
    bound = obs.get("pac_bound_alpha05")
    if bound is None:
        continue
    pairs.append(key)
    bounds.append(bound)
    colors.append(VERDICT_COLORS.get(res.get("verdict", "UNKNOWN"), "#95a5a6"))

y = np.arange(len(pairs))
bars = ax.barh(y, bounds, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(pairs, fontsize=11)
ax.set_xlabel("Upper bound on P(violation)  [95% confidence]", fontsize=11)
ax.set_title("PAC Violation Bounds per Implication Pair", fontsize=13, pad=12)
ax.axvline(0.01, color="gray", linestyle="--", linewidth=1, label="p=0.01")
ax.axvline(0.05, color="gray", linestyle=":",  linewidth=1, label="p=0.05")
ax.legend(fontsize=9)
ax.set_xlim(0, max(bounds) * 1.2 if bounds else 0.1)

for bar, val in zip(bars, bounds):
    ax.text(val + max(bounds)*0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pac_bounds.png"), dpi=150)
plt.close()
print("Saved pac_bounds.png")

# ── Figure 2: Invariance heatmap ─────────────────────────────────────────────
icp_data = {}
for key, res in results.items():
    inv = res.get("invariance", {})
    envs = inv.get("environments", {})
    for env_name, env_res in envs.items():
        if not env_res.get("skipped"):
            viol = env_res.get("n_violations", 0)
            n    = env_res.get("n_a1", 1)
            icp_data.setdefault(key, {})[env_name] = viol / max(n, 1)

if icp_data:
    df_icp = pd.DataFrame(icp_data).T.fillna(0)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df_icp)*0.8)))
    sns.heatmap(
        df_icp, annot=True, fmt=".3f", cmap="RdYlGn_r",
        vmin=0, vmax=0.1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Violation rate"},
    )
    ax.set_title("Violation Rate per Implication × Pseudo-Environment", fontsize=12)
    ax.set_xlabel("Pseudo-environment (latent partition)")
    ax.set_ylabel("Implication pair")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "invariance_heatmap.png"), dpi=150)
    plt.close()
    print("Saved invariance_heatmap.png")

# ── Figure 3: Summary verdict table ──────────────────────────────────────────
rows = []
for key, res in results.items():
    obs  = res.get("observational", {})
    sens = res.get("sensitivity", {})
    rows.append({
        "Pair": key,
        "N(A=1)": obs.get("n_a1", "—"),
        "Violations": obs.get("n_violations", "—"),
        "PAC bound": f"{obs.get('pac_bound_alpha05', float('nan')):.4f}",
        "Causal edge": str(res.get("causal_discovery", {}).get("has_edge", "—")),
        "Invariant": str(res.get("invariance", {}).get("invariant", "—")),
        "Gamma": f"{sens.get('gamma_bound', '—')}",
        "Verdict": res.get("verdict", "—"),
    })

df_summary = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(14, max(2, len(rows) * 0.7 + 1.5)))
ax.axis("off")
tbl = ax.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# Color verdict cells
verdict_col_idx = df_summary.columns.get_loc("Verdict")
for i, row in enumerate(rows):
    v = row["Verdict"]
    tbl[(i+1, verdict_col_idx)].set_facecolor(
        VERDICT_COLORS.get(v, "#ffffff")
    )
    tbl[(i+1, verdict_col_idx)].set_text_props(color="white", fontweight="bold")

# Header style
for j in range(len(df_summary.columns)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

ax.set_title("Implication Testing Summary", fontsize=13, pad=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved summary_table.png")

print(f"\nAll figures saved to {FIGURES_DIR}/")
print("\nFinal verdicts:")
for key, res in results.items():
    print(f"  {key:30s}  {res.get('verdict', '—')}")
