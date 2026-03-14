"""
heart_implication_test.py
--------------------------
Statistical implication testing framework applied to the UCI Heart Disease
(Cleveland) dataset.

Tests four clinically motivated implications of the form A => B where
B = "heart disease present" (target > 0).

Methods per implication:
  1. PAC violation bound     — upper bounds P(violation) with confidence 1-alpha
  2. Chi-squared test        — tests association between A and B
  3. Causal discovery (PC)   — checks for directed A->B edge in causal graph
  4. Invariance test (ICP)   — checks implication holds across subgroups
  5. Rosenbaum sensitivity   — how much confounding would explain away result

Requirements:
    pip3 install pandas numpy scipy scikit-learn causal-learn matplotlib seaborn
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]

df = pd.read_csv("processed.cleveland.data", header=None, names=COLS,
                 na_values="?")
df = df.dropna().reset_index(drop=True)

# Binarize target: 0 = no disease, 1 = disease present
df["disease"] = (df["target"] > 0).astype(int)

print(f"Loaded {len(df)} rows after dropping missing values")
print(f"Disease prevalence: {df['disease'].mean():.1%}")
print()

# ── Define implication pairs ──────────────────────────────────────────────────
# Each entry: (name, A_column, A_condition_fn, description, clinical_source)
def make_a(df, col, fn):
    return fn(df[col]).astype(int)

IMPLICATIONS = [
    {
        "name":    "AsymptomaticCP => Disease",
        "a_col":   "cp",
        "a_fn":    lambda x: x == 4,          # asymptomatic chest pain
        "b_col":   "disease",
        "claim":   "Asymptomatic chest pain implies heart disease present",
        "source":  "Diamond & Forrester 1979; ACC/AHA Guidelines",
    },
    {
        "name":    "ExerciseAngina => Disease",
        "a_col":   "exang",
        "a_fn":    lambda x: x == 1,           # exercise-induced angina
        "b_col":   "disease",
        "claim":   "Exercise-induced angina implies heart disease present",
        "source":  "Gibbons et al. 2002 ACC/AHA Exercise Testing Guidelines",
    },
    {
        "name":    "STDepression>2 => Disease",
        "a_col":   "oldpeak",
        "a_fn":    lambda x: x > 2.0,          # ST depression > 2mm
        "b_col":   "disease",
        "claim":   "ST depression > 2mm implies heart disease present",
        "source":  "Kligfield et al. 2007 AHA/ACC ECG Guidelines",
    },
    {
        "name":    "BlockedVessels => Disease",
        "a_col":   "ca",
        "a_fn":    lambda x: x > 0,            # any blocked vessel
        "b_col":   "disease",
        "claim":   "Any fluoroscopy-visible blocked vessel implies disease",
        "source":  "Detrano et al. 1989 (original Cleveland dataset paper)",
    },
]

ALPHA = 0.05   # confidence level

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def pac_bound(n, k, alpha=ALPHA):
    """Clopper-Pearson upper bound on violation probability."""
    if n == 0: return None
    if k == 0: return 1 - (alpha ** (1.0 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def chi_squared_test(a, b):
    """Pearson chi-squared test of association between binary A and B."""
    ct = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # Phi coefficient as effect size for 2x2
    n = len(a)
    phi = np.sqrt(chi2 / n)
    return {"chi2": round(float(chi2), 3),
            "p_value": round(float(p), 4),
            "dof": int(dof),
            "phi": round(float(phi), 3)}


def causal_discovery(df, a_col, extra_cols, alpha=0.05):
    """
    Run PC algorithm on a subset of variables.
    Returns whether a directed or undirected edge exists from a_col to 'disease'.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cols = extra_cols + [a_col, "disease"]
        sub  = df[cols].dropna().values.astype(float)
        cg   = pc(sub, alpha=alpha, indep_test=fisherz, show_progress=False)
        g    = cg.G
        a_idx = cols.index(a_col)
        b_idx = cols.index("disease")
        has_edge    = bool(g.graph[a_idx, b_idx] != 0 or
                          g.graph[b_idx, a_idx] != 0)
        directed_ab = bool(g.graph[a_idx, b_idx] == -1 and
                          g.graph[b_idx, a_idx] ==  1)
        return {"has_edge": has_edge,
                "directed_A_to_B": directed_ab,
                "note": "PC algorithm (Fisher-Z independence test)"}
    except ImportError:
        return {"has_edge": None,
                "note": "causal-learn not installed — pip3 install causal-learn"}
    except Exception as e:
        return {"has_edge": None, "error": str(e)}


def invariance_test(df, a_vec, b_col, groupby_col, min_size=10):
    """
    Tests whether A=>B holds across subgroups defined by groupby_col.
    A genuine implication should be invariant across contexts.
    """
    results = {}
    for grp_val in sorted(df[groupby_col].unique()):
        mask = df[groupby_col] == grp_val
        sub_a = a_vec[mask]
        sub_b = df.loc[mask, b_col]
        a1    = sub_b[sub_a == 1]
        n     = len(a1)
        if n < min_size:
            results[f"{groupby_col}={grp_val}"] = {"n": n, "skipped": True}
            continue
        k   = int((a1 == 0).sum())   # violations: A=1 but B=0
        pac = pac_bound(n, k)
        results[f"{groupby_col}={grp_val}"] = {
            "n_a1": n,
            "n_violations": k,
            "violation_rate": round(k/n, 3),
            "pac_bound": round(pac, 4) if pac else None,
        }
    any_viol = any(v.get("n_violations", 0) > 0
                   for v in results.values() if not v.get("skipped"))
    return {"invariant": not any_viol, "groups": results}


def rosenbaum_gamma(n, k, alpha=ALPHA):
    """
    Simplified Rosenbaum sensitivity bound.
    Returns Gamma: unmeasured confounding odds ratio needed to
    explain away the observed violation rate.
    """
    pac = pac_bound(n, k, alpha)
    if pac is None or pac >= 1.0:
        return None
    gamma = pac / (1.0 - pac)
    return round(gamma, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}

# Confounding variables to include in causal discovery
CONFOUNDERS = ["age", "sex", "chol", "trestbps"]

for imp in IMPLICATIONS:
    name   = imp["name"]
    a_vec  = make_a(df, imp["a_col"], imp["a_fn"])
    b_vec  = df[imp["b_col"]]
    claim  = imp["claim"]

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Claim:   {claim}")
    print(f"{'='*60}")

    # Contingency table
    ct = pd.crosstab(a_vec, b_vec,
                     rownames=["A (antecedent)"],
                     colnames=["B (disease)"])
    print(f"\nContingency table:\n{ct}\n")

    a1      = b_vec[a_vec == 1]
    n_a1    = len(a1)
    n_a0    = int((a_vec == 0).sum())
    n_viol  = int((a1 == 0).sum())   # A=1 but B=0
    viol_rate = n_viol / n_a1 if n_a1 > 0 else None

    # 1. PAC bound
    pac = pac_bound(n_a1, n_viol)
    print(f"N(A=1): {n_a1}  Violations: {n_viol}  Rate: {viol_rate:.3f}")
    print(f"PAC bound on P(violation) < {pac:.4f}  [95% confidence]")

    # 2. Chi-squared
    chi = chi_squared_test(a_vec, b_vec)
    print(f"Chi-squared: chi2={chi['chi2']}  p={chi['p_value']}  phi={chi['phi']}")

    # 3. Causal discovery
    print("Running causal discovery...")
    causal = causal_discovery(df, imp["a_col"], CONFOUNDERS)
    print(f"Causal edge: {causal}")

    # 4. Invariance test across sex
    inv_sex = invariance_test(df, a_vec, imp["b_col"], "sex")
    print(f"Invariant across sex: {inv_sex['invariant']}")
    print(f"  Groups: {inv_sex['groups']}")

    # 5. Rosenbaum sensitivity
    gamma = rosenbaum_gamma(n_a1, n_viol)
    print(f"Rosenbaum Gamma: {gamma}")

    # Verdict
    chi_sig    = chi["p_value"] < ALPHA
    low_viol   = viol_rate < 0.3 if viol_rate is not None else False
    causal_ok  = causal.get("has_edge", False)
    invariant  = inv_sex.get("invariant", False)

    verdict = (
        "STRONG"   if (low_viol and chi_sig and causal_ok and invariant) else
        "MODERATE" if (low_viol and chi_sig and (causal_ok or invariant))  else
        "WEAK"     if (low_viol and chi_sig)                               else
        "TRENDING" if chi_sig                                               else
        "REJECTED"
    )
    print(f"\n*** VERDICT: {verdict} ***")

    all_results[name] = {
        "claim":          claim,
        "source":         imp["source"],
        "n_a1":           n_a1,
        "n_a0":           n_a0,
        "n_violations":   n_viol,
        "violation_rate": round(viol_rate, 3) if viol_rate else None,
        "pac_bound":      round(pac, 4) if pac else None,
        "chi_squared":    chi,
        "causal":         causal,
        "invariance_sex": inv_sex,
        "rosenbaum_gamma": gamma,
        "verdict":        verdict,
    }

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open("results/heart_implication_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nSaved results/heart_implication_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════

VERDICT_COLORS = {
    "STRONG":   "#2ecc71",
    "MODERATE": "#3498db",
    "WEAK":     "#f39c12",
    "TRENDING": "#e67e22",
    "REJECTED": "#e74c3c",
}

# Figure 1: Violation rates + PAC bounds
fig, ax = plt.subplots(figsize=(11, 5))

# Short readable labels — no truncation
short_labels = [
    "Asymptomatic CP\n=> Disease",
    "Exercise Angina\n=> Disease",
    "ST Depression >2mm\n=> Disease",
    "Blocked Vessels\n=> Disease",
]
names, rates, bounds, colors = [], [], [], []
for name, res in all_results.items():
    names.append(name)
    rates.append(res["violation_rate"] or 0)
    bounds.append(res["pac_bound"] or 0)
    colors.append(VERDICT_COLORS.get(res["verdict"], "#95a5a6"))

x = np.arange(len(names))
bars = ax.bar(x - 0.2, rates,  width=0.35, label="Observed violation rate",
              color=colors, alpha=0.9, edgecolor="white")
ax.bar(x + 0.2, bounds, width=0.35, label="PAC upper bound (95% CI)",
       color=colors, alpha=0.35, edgecolor=colors, linewidth=1.5, hatch="//")
ax.axhline(0.3, color="gray", linestyle="--", linewidth=1.2,
           label="30% violation threshold")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Violation probability", fontsize=10)
ax.set_title("Implication Violation Rates and PAC Bounds\nUCI Heart Disease Dataset",
             fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.75)

# Annotate chi-sq p-values above each group
for i, (v, b, name) in enumerate(zip(rates, bounds, names)):
    res = all_results[name]
    p   = res["chi_squared"]["p_value"]
    ax.text(i, b + 0.03, f"χ² p={p:.4f}", ha="center", fontsize=7.5,
            color="dimgray")
    ax.text(x[i] - 0.2, v + 0.015, f"{v:.2f}", ha="center",
            fontsize=9, fontweight="bold", color="white" if v > 0.1 else "black")

plt.tight_layout()
plt.savefig("figures/pac_bounds_heart.png", dpi=150)
plt.close()
print("Saved figures/pac_bounds_heart.png")

# Figure 2: Stacked bar — contingency breakdown per implication
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 4))
for ax, imp in zip(axes, IMPLICATIONS):
    a_vec = make_a(df, imp["a_col"], imp["a_fn"])
    ct = pd.crosstab(a_vec, df["disease"])
    ct.index = ["A=0", "A=1"]
    ct.columns = ["No disease", "Disease"]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax, color=["#3498db","#e74c3c"],
                edgecolor="white", width=0.6)
    res = all_results[imp["name"]]
    ax.set_title(f"{imp['name']}\n[{res['verdict']}]", fontsize=8)
    ax.set_ylabel("% of cells" if ax == axes[0] else "")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)
fig.suptitle("Disease Prevalence by Antecedent Status", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figures/contingency_heart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/contingency_heart.png")

# Figure 3: Summary table
rows = []
for name, res in all_results.items():
    rows.append({
        "Implication": name,
        "N(A=1)": res["n_a1"],
        "Violations": res["n_violations"],
        "Viol. rate": f"{res['violation_rate']:.2f}",
        "PAC bound": f"{res['pac_bound']:.3f}",
        "Chi-sq p": f"{res['chi_squared']['p_value']:.4f}",
        "Phi": f"{res['chi_squared']['phi']:.3f}",
        "Causal edge": str(res["causal"].get("has_edge", "?")),
        "Invariant": str(res["invariance_sex"].get("invariant", "?")),
        "Gamma": str(res["rosenbaum_gamma"]),
        "Verdict": res["verdict"],
    })

df_summary = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(18, max(2, len(rows)*0.9 + 2)))
ax.axis("off")
tbl = ax.table(cellText=df_summary.values,
               colLabels=df_summary.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)

# Auto-size first column to avoid truncation
tbl.auto_set_column_width([0])

verdict_col = df_summary.columns.get_loc("Verdict")
for i, row in enumerate(rows):
    tbl[(i+1, verdict_col)].set_facecolor(
        VERDICT_COLORS.get(row["Verdict"], "#ffffff"))
    tbl[(i+1, verdict_col)].set_text_props(color="white", fontweight="bold")
for j in range(len(df_summary.columns)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")
ax.set_title("Heart Disease Implication Testing Summary",
             fontsize=13, pad=16, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/summary_table_heart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/summary_table_heart.png")

# Figure 4: Sex-stratified violation rates
fig, axes = plt.subplots(1, len(IMPLICATIONS), figsize=(14, 5), sharey=True)
fig.suptitle("Violation Rates by Sex: Where Implications Break Down\n"
             "(Female = sex=0, Male = sex=1)", fontsize=12)

SEX_COLORS = {
    "Female\n(sex=0)": "#e91e8c",
    "Male\n(sex=1)":   "#1e88e5",
    "Overall":         "#607d8b",
}

for ax, imp in zip(axes, IMPLICATIONS):
    a_vec  = make_a(df, imp["a_col"], imp["a_fn"])
    name   = imp["name"]
    res    = all_results[name]

    group_labels, viol_rates, pac_ups, bar_colors = [], [], [], []

    for sex_val, sex_label in [(0.0, "Female\n(sex=0)"), (1.0, "Male\n(sex=1)")]:
        mask  = (df["sex"] == sex_val) & (a_vec == 1)
        sub_b = df.loc[mask, "disease"]
        n     = len(sub_b)
        if n < 10:
            continue
        k   = int((sub_b == 0).sum())
        vr  = k / n
        pac = pac_bound(n, k)
        group_labels.append(sex_label)
        viol_rates.append(vr)
        pac_ups.append(pac)
        bar_colors.append(SEX_COLORS[sex_label])

    # Add overall
    group_labels.append("Overall")
    viol_rates.append(res["violation_rate"])
    pac_ups.append(res["pac_bound"])
    bar_colors.append(SEX_COLORS["Overall"])

    x    = np.arange(len(group_labels))
    bars = ax.bar(x, viol_rates, color=bar_colors, edgecolor="white",
                  width=0.5, alpha=0.88)

    # Error bars showing gap between observed rate and PAC bound
    yerr_upper = [p - v for p, v in zip(pac_ups, viol_rates)]
    ax.errorbar(x, viol_rates, yerr=[np.zeros(len(x)), yerr_upper],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    ax.axhline(0.3, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_title(name.split(" =>")[0].replace("=>","").strip() +
                 "\n=> Disease", fontsize=8)
    ax.set_ylim(0, 0.85)
    if ax == axes[0]:
        ax.set_ylabel("Violation rate  (↑ error bar = PAC bound)", fontsize=9)
    for bar, v in zip(bars, viol_rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l)
                   for l, c in SEX_COLORS.items()]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("figures/sex_stratified_heart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/sex_stratified_heart.png")

print("\nAll done. Results in results/ and figures/")
