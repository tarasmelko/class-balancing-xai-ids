"""
Computes every number reported in the article from results_<tag>.json.

Usage: python3 analysis.py results/results_clean10.json [--json out.json]

Statistical tests (paired by split, n = number of splits):
  * attribution shift vs the noise control: one-sided Wilcoxon signed-rank test
    (balanced configuration < seed control), Bonferroni correction for 5 comparisons;
  * random oversampling vs SMOTE: two-sided Wilcoxon signed-rank test;
  * detection quality vs no balancing: one-sided Wilcoxon signed-rank test (balanced > baseline);
  * interventional vs path-dependent TreeSHAP: one-sided Wilcoxon (interventional > path-dependent);
  * LIME stability vs no balancing: one-sided Wilcoxon signed-rank test (balanced < baseline).
"""
import json
import sys
from collections import Counter

import numpy as np
from scipy import stats

BALANCED = ["class_weight", "random_over", "smote", "borderline_smote", "adasyn"]
RESAMPLERS = ["random_over", "smote", "borderline_smote", "adasyn"]
NAMES = {"baseline": "No balancing", "seed_control": "Seed control",
         "class_weight": "class_weight", "random_over": "Random Oversampling",
         "smote": "SMOTE", "borderline_smote": "Borderline-SMOTE", "adasyn": "ADASYN"}
TCP_WIN = {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}

PACKET_COUNTS = ["Total Fwd Packets", "Total Backward Packets", "act_data_pkt_fwd"]
BYTE_TOTALS = ["Total Length of Fwd Packets", "Total Length of Bwd Packets",
               "Fwd Header Length", "Bwd Header Length"]
INTACT = ["Destination Port", "Init_Win_bytes_forward", "FIN Flag Count", "SYN Flag Count",
          "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
          "ECE Flag Count", "Fwd PSH Flags"]


def per_split(res, scen, group, key):
    """{split seed: value} for one metric."""
    out = {}
    for r in res["runs"]:
        s = r["scenarios"].get(scen, {})
        if group in s and key in s[group]:
            out[r["seed"]] = s[group][key]
    return out


def vec(res, scen, group, key):
    d = per_split(res, scen, group, key)
    return np.array([d[k] for k in sorted(d)])


def paired(res, a, b):
    """Two dicts -> aligned arrays over common splits."""
    ks = sorted(set(a) & set(b))
    return np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


def wilcoxon(x, y, alternative):
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, alternative=alternative).pvalue)


def ms(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std())


def main(path, json_out=None):
    res = json.load(open(path))
    n = len(res["runs"])
    R = {}
    P = print

    P(f"Data: {res['shape'][0]} flows x {res['shape'][1]} features, {res['n_attacks']} attacks "
      f"(1:{(res['shape'][0] - res['n_attacks']) / res['n_attacks']:.0f}); "
      f"{res['n_integer_features']} whole-number features; {n} splits\n")

    # ---------------------------------------------------------------- attribution shift
    P("1. Global SHAP attribution vs the unbalanced reference (path-dependent TreeSHAP)")
    ctrl = {k: per_split(res, "seed_control", "attr", k) for k in ["spearman", "jaccard5", "kendall"]}
    for scen in ["seed_control"] + BALANCED:
        rho, j5, tau, cos = (ms(vec(res, scen, "attr", k)) for k in
                             ["spearman", "jaccard5", "kendall", "shap_cosine_distance"])
        line = (f"  {NAMES[scen]:<20} rho {rho[0]:.3f} ± {rho[1]:.3f}   Jaccard@5 {j5[0]:.3f} ± {j5[1]:.3f}"
                f"   tau {tau[0]:.3f}   cosine dist {cos[0]:.4f}")
        if scen != "seed_control":
            a, b = paired(res, per_split(res, scen, "attr", "spearman"), ctrl["spearman"])
            p_rho = wilcoxon(a, b, "less")
            a, b = paired(res, per_split(res, scen, "attr", "jaccard5"), ctrl["jaccard5"])
            p_j5 = wilcoxon(a, b, "less")
            line += f"   p(rho) {p_rho:.4f}  p(J@5) {p_j5:.4f}  Bonferroni x5: {min(1, 5*p_rho):.4f} / {min(1, 5*p_j5):.4f}"
            R[f"p_rho_{scen}"], R[f"p_j5_{scen}"] = p_rho, p_j5
        P(line)
        R[f"rho_{scen}"], R[f"j5_{scen}"], R[f"tau_{scen}"] = rho, j5, tau

    # ROS vs SMOTE
    P("\n2. Synthesis vs duplication (Random Oversampling vs SMOTE, two-sided Wilcoxon)")
    for key in ["jaccard5", "spearman", "shap_cosine_distance"]:
        a, b = paired(res, per_split(res, "random_over", "attr", key), per_split(res, "smote", "attr", key))
        p = wilcoxon(a, b, "two-sided")
        P(f"  {key:<22} ROS {a.mean():.4f}  SMOTE {b.mean():.4f}  diff {abs(a.mean()-b.mean()):.4f}  p = {p:.3f}")
        R[f"ros_vs_smote_{key}"] = (float(a.mean()), float(b.mean()), p)
    shift = R["rho_seed_control"][0] - R["rho_random_over"][0]
    P(f"  shift of ROS from the noise control in rho: {shift:.3f}")

    # TCP window in top five
    P("\n3. Initial TCP window sizes among the five highest-ranked features")
    bal_any = bal_both = 0
    unbal_any = 0
    for r in res["runs"]:
        for scen in BALANCED:
            top = set(r["scenarios"][scen]["attr"]["top5_new"])
            bal_any += bool(top & TCP_WIN)
            bal_both += TCP_WIN <= top
        for top in [r["scenarios"]["baseline"]["attr"]["top5_new"],
                    r["scenarios"]["seed_control"]["attr"]["top5_new"]]:
            unbal_any += bool(set(top) & TCP_WIN)
    P(f"  balanced runs with at least one: {bal_any} of {n*len(BALANCED)}; with both: {bal_both}")
    P(f"  unbalanced runs (reference + seed control) with any: {unbal_any} of {2*n}")
    fd = sum("Flow Duration" in r["scenarios"][s]["attr"]["top5_new"] for r in res["runs"] for s in BALANCED)
    P(f"  Flow Duration in the top five of balanced models: {fd} of {n*len(BALANCED)} runs")
    top_unbal = Counter(f for r in res["runs"] for f in r["scenarios"]["baseline"]["attr"]["top5_new"])
    P(f"  most frequent top-five features of the unbalanced model: {top_unbal.most_common(5)}")

    # ---------------------------------------------------------------- quality
    P("\n4. Detection quality and calibration")
    base = {k: per_split(res, "baseline", "quality", k) for k in
            ["pr_auc", "f1_best", "mcc_at_05", "brier", "calib_slope", "obs_exp_ratio"]}
    for scen in ["baseline", "seed_control"] + BALANCED:
        q = {k: ms(vec(res, scen, "quality", k)) for k in base if len(vec(res, scen, "quality", k))}
        line = (f"  {NAMES[scen]:<20} PR-AUC {q['pr_auc'][0]:.3f} ± {q['pr_auc'][1]:.3f}  F1best {q['f1_best'][0]:.3f}"
                f"  MCC {q['mcc_at_05'][0]:.3f}  Brier {q['brier'][0]:.5f}  slope {q['calib_slope'][0]:.2f}")
        if "obs_exp_ratio" in q:
            line += f"  obs/pred {q['obs_exp_ratio'][0]:.2f}"
        if scen in BALANCED:
            a, b = paired(res, per_split(res, scen, "quality", "pr_auc"), base["pr_auc"])
            p = wilcoxon(a, b, "greater")
            line += f"  p(PR-AUC > baseline) {p:.4f}"
            R[f"p_prauc_{scen}"] = p
        P(line)
        R[f"quality_{scen}"] = q

    # ---------------------------------------------------------------- integrity
    P("\n5. Generated records with values no network flow can take")
    for scen in RESAMPLERS:
        rows = vec(res, scen, "integrity", "pct_rows_noninteger")
        vals = vec(res, scen, "integrity", "pct_values_noninteger")
        nsyn = vec(res, scen, "integrity", "n_synth")
        P(f"  {NAMES[scen]:<20} generated {nsyn.mean():.0f}   records with a fractional value {rows.mean():.1f} %"
          f"   fractional values {vals.mean():.1f} %")
        R[f"integrity_{scen}"] = (float(rows.mean()), float(vals.mean()))
    ntr = np.mean([r["n_train_attacks"] for r in res["runs"]])
    nsm = vec(res, "smote", "integrity", "n_synth").mean()
    P(f"  share of the minority class after SMOTE that was never observed: {100*nsm/(nsm+ntr):.2f} %")
    pf = [r["scenarios"]["smote"]["integrity"].get("per_feature") for r in res["runs"]]
    if all(pf):
        feat = {c: np.mean([d[c] for d in pf]) for c in pf[0]}
        P("  SMOTE, share of generated records with a fractional value, by feature (%):")
        for c in sorted(feat, key=lambda c: -feat[c]):
            P(f"    {c:<32} {feat[c]:6.1f}")
        grp = lambda cols: [round(feat[c], 1) for c in cols if c in feat]
        P(f"  packet counts {grp(PACKET_COUNTS)}; byte totals {grp(BYTE_TOTALS)}; "
          f"Init_Win_bytes_backward {grp(['Init_Win_bytes_backward'])}; intact {grp(INTACT)}")
        R["integrity_per_feature_smote"] = feat
    else:
        P("  (per-feature breakdown not stored in this results file)")

    # ---------------------------------------------------------------- remedy: SHAP
    P("\n6. Interventional TreeSHAP with a background of real records")
    ivf = per_split(res, "seed_control", "attr_interventional", "spearman")
    ivf_j = per_split(res, "seed_control", "attr_interventional", "jaccard5")
    pdf = ctrl["spearman"]
    if ivf:
        P(f"  noise level (seed control), interventional: rho {np.mean(list(ivf.values())):.3f} ± "
          f"{np.std(list(ivf.values())):.3f}   Jaccard@5 {np.mean(list(ivf_j.values())):.3f}")
    gaps_pd, gaps_iv = [], []
    for scen in RESAMPLERS:
        pd_k = per_split(res, scen, "attr", "spearman")
        iv_k = per_split(res, scen, "attr_interventional", "spearman")
        iv_j = per_split(res, scen, "attr_interventional", "jaccard5")
        a, b = paired(res, iv_k, pd_k)
        p_imp = wilcoxon(a, b, "greater")
        line = f"  {NAMES[scen]:<20} rho path-dep {b.mean():.3f} -> interventional {a.mean():.3f}  p = {p_imp:.4f}"
        line += f"   Jaccard@5 interventional {np.mean(list(iv_j.values())):.3f}"
        if ivf:
            x, y = paired(res, iv_j, ivf_j)
            line += f"  p(J@5 < iv floor) {wilcoxon(x, y, 'less'):.4f}"
            x, y = paired(res, iv_k, ivf)
            line += f"  p(rho < iv floor) {wilcoxon(x, y, 'less'):.4f}"
            gaps_iv.append(np.mean(y - x))
        x, y = paired(res, pd_k, pdf)
        gaps_pd.append(np.mean(y - x))
        P(line)
    if gaps_iv:
        P(f"  mean shortfall below the own noise level: path-dependent {np.mean(gaps_pd):.3f}, "
          f"interventional {np.mean(gaps_iv):.3f}  (removed {100*(1-np.mean(gaps_iv)/np.mean(gaps_pd)):.0f} %)")

    # ---------------------------------------------------------------- remedy: LIME
    P("\n7. Local stability of LIME (VSI), one-sided Wilcoxon vs no balancing")
    b_l = per_split(res, "baseline", "lime", "vsi_jaccard5")
    P(f"  No balancing        VSI {np.mean(list(b_l.values())):.3f}   splits at 1.0: "
      f"{sum(v >= 0.9999 for v in b_l.values())} of {len(b_l)}")
    for scen in BALANCED:
        for grp, lab in [("lime", "neighbourhood from the data at hand"),
                         ("lime_realref", "neighbourhood from real records")]:
            d = per_split(res, scen, grp, "vsi_jaccard5")
            if not d:
                continue
            a, b = paired(res, d, b_l)
            P(f"  {NAMES[scen]:<20} {lab:<38} VSI {a.mean():.3f}  p = {wilcoxon(a, b, 'less'):.3f}")

    if json_out:
        json.dump(R, open(json_out, "w"), indent=1, default=float)


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    main(sys.argv[1], out)
