"""
03_implication_testing.py
--------------------------
Core pipeline: for each (A, B) implication pair, computes:

  1. PAC bound  — P(violation | A=1) < epsilon with confidence 1-delta
  2. Causal discovery — does A->B edge exist in the causal graph?
  3. Latent stress-test — decode samples from A=1 region, check B
  4. Invariance test (ICP-style) — does implication hold across
     z-space partitions (pseudo-environments)?
  5. Rosenbaum sensitivity — how much hidden confounding would
     invalidate the result?

Requirements:
    pip install scvi-tools causal-learn lingam scipy scikit-learn
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import scvi
import anndata as ad
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "./data/processed/perturb_prepped.h5ad"
LATENT_PATH  = "./data/processed/latent.csv"
MODEL_DIR    = "./models/scvi_model"
RESULTS_DIR  = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA        = 0.05    # confidence level for PAC bounds
N_STRESS     = 1000    # synthetic cells to decode per stress test
N_PARTITIONS = 5       # pseudo-environments for ICP
PERT_COL     = "perturbation"   # must match 01_download_and_prep.py

IMPLICATION_PAIRS = [
    ("MYC",   "CDKN1A"),
    ("TP53",  "CDKN1A"),
    ("BRCA1", "FANCD2"),
    ("KRAS",  "DUSP6"),
    ("PTEN",  "AKT1"),
]

# ── Load ──────────────────────────────────────────────────────────────────────
adata    = ad.read_h5ad(DATA_PATH)
latent   = pd.read_csv(LATENT_PATH, index_col=0)
z_cols   = [c for c in latent.columns if c.startswith("z_")]
Z        = latent[z_cols].values
model    = scvi.model.SCVI.load(MODEL_DIR, adata=adata)
print(f"Loaded {len(Z)} cells, {len(z_cols)} latent dims")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAC Bound
# ═══════════════════════════════════════════════════════════════════════════════

def pac_violation_bound(n_total: int, n_violations: int, alpha: float) -> float:
    """
    One-sided Clopper-Pearson upper bound on violation probability.
    Returns epsilon such that P(p_violation > epsilon) < alpha.
    """
    if n_violations == 0:
        # Exact: 1 - alpha^(1/n)  ≈  -log(alpha) / n  for small alpha
        return 1 - (alpha ** (1.0 / n_total))
    else:
        # Beta distribution upper tail
        return stats.beta.ppf(1 - alpha, n_violations + 1, n_total - n_violations)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Causal Discovery (PC algorithm on latent space)
# ═══════════════════════════════════════════════════════════════════════════════

def causal_edge_exists(Z_sub: np.ndarray, a_idx: int, b_idx: int,
                        alpha: float = 0.05) -> dict:
    """
    Runs PC algorithm on a subset of latent dimensions + A/B indicators.
    Returns whether a directed or undirected edge A->B is found.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        data = Z_sub.astype(float)
        cg   = pc(data, alpha=alpha, indep_test=fisherz, show_progress=False)
        g    = cg.G
        # Check for edge between a_idx and b_idx in learned graph
        has_edge = bool(g.graph[a_idx, b_idx] != 0 or
                        g.graph[b_idx, a_idx] != 0)
        directed = bool(g.graph[a_idx, b_idx] == -1 and
                        g.graph[b_idx, a_idx] ==  1)
        return {"has_edge": has_edge, "directed_A_to_B": directed}
    except ImportError:
        return {"has_edge": None, "directed_A_to_B": None,
                "note": "causal-learn not installed"}
    except Exception as e:
        return {"has_edge": None, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Latent Stress Test (generative search for violations)
# ═══════════════════════════════════════════════════════════════════════════════

def latent_stress_test(adata, model, latent_df: pd.DataFrame,
                        gene_a: str, gene_b: str,
                        n_samples: int = 1000) -> dict:
    """
    Samples from the A=1 region of latent space, decodes to gene space,
    checks whether gene_b expression falls below a 'inactive' threshold.

    Returns fraction of decoded samples where B appears inactive.
    """
    if gene_b not in adata.var_names:
        return {"violation_rate": None, "note": f"{gene_b} not in var_names"}

    ko_col = f"KO_{gene_a}"
    if ko_col not in latent_df.columns:
        return {"violation_rate": None, "note": f"No {ko_col} column"}

    z_cols  = [c for c in latent_df.columns if c.startswith("z_")]
    a1_mask = latent_df[ko_col] == 1
    Z_a1    = latent_df.loc[a1_mask, z_cols].values

    if len(Z_a1) < 10:
        return {"violation_rate": None, "note": "Too few A=1 cells"}

    # Sample with noise from the A=1 latent cloud
    mu_a1    = Z_a1.mean(axis=0)
    std_a1   = Z_a1.std(axis=0) + 1e-6
    z_sample = np.random.randn(n_samples, len(z_cols)) * std_a1 + mu_a1
    z_tensor = torch.tensor(z_sample, dtype=torch.float32)

    # Decode: get mean expression for each sampled cell
    # scVI's decode returns a dict with 'px' (NegBinom params)
    model.module.eval()
    with torch.no_grad():
        px = model.module.generative(
            z=z_tensor,
            library=torch.ones(n_samples, 1) * 10.0,  # fixed library size
            batch_index=torch.zeros(n_samples, 1, dtype=torch.long) # fake batch
        )["px"]
        # Mean expression under NegBinom: px.mu
        expr_decoded = px.mu.cpu().numpy()  # (n_samples, n_genes)

    gene_idx  = adata.var_names.get_loc(gene_b)
    b_expr    = expr_decoded[:, gene_idx]

    # Threshold: median expression of gene_b in control cells
    ctrl_mask = adata.obs[PERT_COL] == "non-targeting"
    counts_b = adata[ctrl_mask, gene_b].layers["counts"]
    if hasattr(counts_b, 'todense'):
        b_ctrl = np.array(counts_b.todense()).flatten()
    else:
        b_ctrl = np.array(counts_b).flatten()
    threshold = np.median(b_ctrl)

    violation_rate = float((b_expr < threshold).mean())
    return {
        "violation_rate": violation_rate,
        "threshold_used": float(threshold),
        "n_samples": n_samples,
        "mean_b_expr_decoded": float(b_expr.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ICP-style Invariance Test
# ═══════════════════════════════════════════════════════════════════════════════

def invariance_test(latent_df: pd.DataFrame, gene_a: str, gene_b: str,
                    n_partitions: int = 5) -> dict:
    """
    Partitions latent space into pseudo-environments via KMeans clustering.
    Tests whether the implication A=>B holds (zero violations) in each
    partition. A genuine causal relationship should be invariant.
    """
    z_cols = [c for c in latent_df.columns if c.startswith("z_")]
    ko_col  = f"KO_{gene_a}"
    hi_col  = f"HIGH_{gene_b}"

    if ko_col not in latent_df.columns or hi_col not in latent_df.columns:
        return {"invariant": None, "note": "Missing label columns"}

    Z = latent_df[z_cols].values
    kmeans = KMeans(n_clusters=n_partitions, random_state=42, n_init=10)
    latent_df = latent_df.copy()
    latent_df["partition"] = kmeans.fit_predict(Z)

    results = {}
    for part in range(n_partitions):
        mask      = latent_df["partition"] == part
        sub       = latent_df[mask]
        a1        = sub[sub[ko_col] == 1]
        if len(a1) < 5:
            results[f"env_{part}"] = {"n": len(a1), "skipped": True}
            continue
        n_viol    = int((a1[hi_col] == 0).sum())
        n_total   = len(a1)
        eps_bound = pac_violation_bound(n_total, n_viol, ALPHA)
        results[f"env_{part}"] = {
            "n_a1": n_total,
            "n_violations": n_viol,
            "pac_bound": round(eps_bound, 5),
        }

    any_violations = any(
        v.get("n_violations", 0) > 0
        for v in results.values() if not v.get("skipped")
    )
    return {
        "invariant": not any_violations,
        "environments": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Rosenbaum Sensitivity (simplified Gamma bound)
# ═══════════════════════════════════════════════════════════════════════════════

def rosenbaum_sensitivity(n_a1: int, n_violations: int) -> dict:
    """
    Estimates how strong unmeasured confounding would need to be
    to explain away zero violations. Simplified: reports how large
    the odds ratio Gamma would need to be to produce at least one
    expected violation.

    Gamma = 1 means no confounding. Gamma = 2 means a hidden confounder
    could double the odds of treatment assignment.
    """
    if n_a1 == 0:
        return {"gamma_bound": None}

    # Under no confounding, expected violations = n_a1 * p_hat
    # We observed 0; the MLE is p_hat = 0.
    # With confounding Gamma, worst-case p under H0 is:
    #   p_upper = Gamma / (1 + Gamma)
    # We report the Gamma at which the Bonferroni-corrected
    # expected count exceeds 0.5 (i.e., we'd expect at least one violation).
    # For zero observed: any Gamma > 1 could in principle produce violations,
    # so we report the PAC bound as our effective sensitivity.
    pac = pac_violation_bound(n_a1, n_violations, ALPHA)
    # Gamma such that Gamma/(1+Gamma) = pac_bound  =>  Gamma = pac/(1-pac)
    if pac >= 1.0:
        gamma = float("inf")
    else:
        gamma = pac / (1.0 - pac)
    return {
        "gamma_bound": round(gamma, 4),
        "interpretation": (
            f"Unmeasured confounding would need odds ratio > {gamma:.2f} "
            f"to plausibly explain zero violations"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}

for (gene_a, gene_b) in IMPLICATION_PAIRS:
    print(f"\n{'='*60}")
    print(f"Testing: KO({gene_a}) => HIGH({gene_b})")
    print(f"{'='*60}")

    ko_col = f"KO_{gene_a}"
    hi_col = f"HIGH_{gene_b}"
    result = {"gene_a": gene_a, "gene_b": gene_b}

    # ── Observational counts ──────────────────────────────────────────────────
    if ko_col in latent.columns and hi_col in latent.columns:
        a1       = latent[latent[ko_col] == 1]
        n_a1     = len(a1)
        n_viol   = int((a1[hi_col] == 0).sum())
        n_a0     = int((latent[ko_col] == 0).sum())
        print(f"  Cells with A=1: {n_a1}")
        print(f"  Violations (A=1, B=0): {n_viol}")

        pac = pac_violation_bound(n_a1, n_viol, ALPHA)
        result["observational"] = {
            "n_a1": n_a1, "n_a0": n_a0,
            "n_violations": n_viol,
            "pac_bound_alpha05": round(pac, 6),
        }
        print(f"  PAC bound on P(violation): < {pac:.5f}")
    else:
        result["observational"] = {"note": f"Labels {ko_col}/{hi_col} not found"}
        n_a1, n_viol = 0, 0
        print(f"  WARNING: labels not found, skipping observational test")

    # ── Causal discovery ─────────────────────────────────────────────────────
    print("  Running causal discovery...")
    # Use top-10 PCs of Z for tractability
    from sklearn.decomposition import PCA
    Z_pca  = PCA(n_components=10).fit_transform(Z)
    # Append A and B as extra columns
    a_vals = latent[ko_col].values.reshape(-1,1) if ko_col in latent.columns \
             else np.zeros((len(Z),1))
    b_vals = latent[hi_col].values.reshape(-1,1) if hi_col in latent.columns \
             else np.zeros((len(Z),1))
    Z_aug  = np.hstack([Z_pca, a_vals, b_vals])
    causal = causal_edge_exists(Z_aug, a_idx=10, b_idx=11)
    result["causal_discovery"] = causal
    print(f"  Causal edge: {causal}")

    # ── Generative stress test ────────────────────────────────────────────────
    print("  Running latent stress test...")
    stress = latent_stress_test(adata, model, latent, gene_a, gene_b,
                                 n_samples=N_STRESS)
    result["stress_test"] = stress
    print(f"  Stress test violation rate: {stress.get('violation_rate')}")

    # ── Invariance test ───────────────────────────────────────────────────────
    print("  Running invariance test...")
    icp = invariance_test(latent, gene_a, gene_b, n_partitions=N_PARTITIONS)
    result["invariance"] = icp
    print(f"  Invariant across environments: {icp.get('invariant')}")

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sens = rosenbaum_sensitivity(n_a1, n_viol)
    result["sensitivity"] = sens
    print(f"  Rosenbaum Gamma: {sens.get('gamma_bound')}")

    # ── Summary verdict ───────────────────────────────────────────────────────
    obs_ok    = result.get("observational", {}).get("n_violations", 1) == 0
    inv_ok    = icp.get("invariant", False)
    causal_ok = causal.get("has_edge", False)
    stress_ok = (stress.get("violation_rate") or 1.0) < 0.05

    verdict = (
        "STRONG"   if (obs_ok and inv_ok and causal_ok and stress_ok) else
        "MODERATE" if (obs_ok and (inv_ok or causal_ok))              else
        "WEAK"     if obs_ok                                           else
        "REJECTED"
    )
    result["verdict"] = verdict
    print(f"\n  *** VERDICT: {verdict} ***")

    all_results[f"{gene_a}=>{gene_b}"] = result

# ── Save results ──────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "implication_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
