"""
Main experiment: how the class-balancing strategy affects feature attribution
(SHAP, LIME) in machine-learning intrusion detection.

Data: CIC-IDS2017, Thursday-WorkingHours-Morning-WebAttacks, cleaned with
      clean_cicids.py (CSV-level part of the corrections of Engelen et al., 2021).

Three controls make the comparison interpretable:
  seed_control : same data and settings, another model seed -> noise level of the measurement
  class_weight : rebalancing without changing the data
  random_over  : rebalancing by duplicating real records, without synthesis
Comparing random_over with smote isolates the effect of synthetic generation.

Usage: python3 experiment.py <clean.csv> <tag> [n_splits]
Writes results_<tag>.json in the current directory.
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, brier_score_loss
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
import shap

BG_N = 1500          # test records explained by TreeSHAP (global attribution)
REAL_BG_N = 500      # real training records used as background for interventional TreeSHAP
LIME_RUNS = 15       # repeated LIME explanations of the same record
N_TREES = 150


def load(path):
    d = pd.read_csv(path, low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    d["Label"] = (d["Label"].astype(str).str.replace("�", "-", regex=False)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    y = (d["Label"] != "BENIGN").astype(int).values
    X = d.drop(columns=["Label"])
    X = X.select_dtypes(include=[np.number]).astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = ~X.isna().any(axis=1).values
    return X[keep].reset_index(drop=True), y[keep]


def integer_columns(X):
    """Features that take whole-number values only; interpolation breaks them."""
    return [c for c in X.columns
            if np.all(np.isclose(X[c].values % 1, 0))]


SAMPLERS = {
    "random_over":      lambda s: RandomOverSampler(random_state=s),
    "smote":            lambda s: SMOTE(random_state=s, k_neighbors=5),
    "borderline_smote": lambda s: BorderlineSMOTE(random_state=s, k_neighbors=5),
    "adasyn":           lambda s: ADASYN(random_state=s, n_neighbors=5),
}


def rf(X, y, seed, cw=None):
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                  class_weight=cw, n_jobs=-1,
                                  random_state=seed).fit(X, y)


def quality(model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    ths = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(yte, (p >= t).astype(int), zero_division=0) for t in ths]
    best = int(np.argmax(f1s))
    # calibration intercept and slope: logistic recalibration on the logits
    eps = 1e-6
    lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    try:
        cal = LogisticRegression(solver="lbfgs").fit(lo, yte)
        cal_int, cal_slope = float(cal.intercept_[0]), float(cal.coef_[0][0])
    except Exception:
        cal_int = cal_slope = float("nan")
    return {
        "pr_auc": float(average_precision_score(yte, p)),
        "f1_at_05": float(f1_score(yte, (p >= .5).astype(int), zero_division=0)),
        "f1_best": float(f1s[best]),
        "thr_best": float(ths[best]),
        "mcc_at_05": float(matthews_corrcoef(yte, (p >= .5).astype(int))),
        "brier": float(brier_score_loss(yte, p)),
        "calib_intercept": cal_int,
        "calib_slope": cal_slope,
        "obs_exp_ratio": float(yte.sum() / max(p.sum(), 1e-12)),  # observed / predicted attacks
    }


def shap_ranks(model, bg, mode="tree_path_dependent", data=None):
    ex = shap.TreeExplainer(model, data=data, feature_perturbation=mode)
    sv = ex.shap_values(bg, check_additivity=False)
    sv = sv[..., 1] if np.ndim(sv) == 3 else sv
    g = np.abs(sv).mean(0)
    return pd.Series(g, index=bg.columns)


def compare(g0, g1):
    r0, r1 = g0.rank(ascending=False), g1.rank(ascending=False)
    j = lambda k: len(set(r0.nsmallest(k).index) & set(r1.nsmallest(k).index)) / \
                  len(set(r0.nsmallest(k).index) | set(r1.nsmallest(k).index))
    n0, n1 = g0 / g0.sum(), g1 / g1.sum()
    cos = float(np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-12))
    return {
        "jaccard5": j(5), "jaccard10": j(10),
        "spearman": float(stats.spearmanr(r0, r1)[0]),
        "kendall": float(stats.kendalltau(r0, r1)[0]),
        "rank_agree_top5": float(np.mean([a == b for a, b in
                                          zip(r0.nsmallest(5).index, r1.nsmallest(5).index)])),
        "shap_cosine_distance": 1.0 - cos,
        "max_attr_change": float(np.abs(n0 - n1).max()),
        "top5_ref": list(r0.nsmallest(5).index),
        "top5_new": list(r1.nsmallest(5).index),
    }


def lime_tabular(model, x, Xref, mu, sd, n_samples=1500, kw=None, seed=0):
    """Tabular LIME following the specification of Ribeiro et al. (2016)."""
    r = np.random.default_rng(seed)
    kw = kw or np.sqrt(len(mu)) * 0.75
    Z = r.normal(0, 1, (n_samples, len(mu)))
    Xp = Z * sd + mu
    Xp[0] = x
    d = np.sqrt((((Xp - x) / sd) ** 2).sum(1))
    w = np.exp(-(d ** 2) / kw ** 2)
    yp = model.predict_proba(pd.DataFrame(Xp, columns=Xref.columns))[:, 1]
    g = Ridge(alpha=1.0).fit((Xp - mu) / sd, yp, sample_weight=w)
    return pd.Series(np.abs(g.coef_), index=Xref.columns)


def lime_stability(model, target, Xref):
    """VSI (Visani et al., 2022) computed as the mean pairwise Jaccard overlap of the
    five highest-weighted features over LIME_RUNS explanations of one record.
    Xref is the reference set from which the LIME neighbourhood is sampled."""
    mu, sd = Xref.mean(0).values, Xref.std(0).values + 1e-9
    tops, coefs = [], []
    for s in range(LIME_RUNS):
        c = lime_tabular(model, target, Xref, mu, sd, seed=s)
        tops.append(set(c.nlargest(5).index)); coefs.append(c.values)
    pair = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    C = np.array(coefs)
    return {"vsi_jaccard5": float(np.mean(pair)),
            "csi_mean_cv": float(np.mean(C.std(0) / (np.abs(C).mean(0) + 1e-12)))}


def integrity(synth, int_cols):
    """Semantic validity of generated records: fractional values in whole-number columns."""
    if len(synth) == 0:
        return {}
    v = synth[int_cols].values
    frac = ~np.isclose(v % 1, 0)
    per_feature = {c: round(float(f) * 100, 1) for c, f in zip(int_cols, frac.mean(0))}
    return {
        "n_synth": int(len(synth)),
        "pct_rows_noninteger": round(float(frac.any(1).mean()) * 100, 2),
        "pct_values_noninteger": round(float(frac.mean()) * 100, 2),
        "n_integer_features": len(int_cols),
        "worst_features": {c: per_feature[c] for c in
                           sorted(per_feature, key=lambda c: -per_feature[c])[:6]},
        "per_feature": per_feature,
    }


def run(path, tag, n_seeds=10):
    X, y = load(path)
    int_cols = integer_columns(X)
    print(f"[{tag}] {X.shape}, attacks {y.sum()} (1:{int((y == 0).sum() / max(1, y.sum()))}), "
          f"whole-number features {len(int_cols)}/{X.shape[1]}", flush=True)

    out = {"tag": tag, "path": path, "shape": list(X.shape), "n_attacks": int(y.sum()),
           "n_integer_features": len(int_cols), "integer_features": int_cols, "runs": []}

    for seed in range(n_seeds):
        t0 = time.time()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y,
                                              random_state=seed)
        bg = Xte.sample(BG_N, random_state=seed)
        real_bg = Xtr.sample(min(REAL_BG_N, len(Xtr)), random_state=seed)  # real records only
        target = Xte[yte == 1].iloc[0].values                              # attack record for LIME

        rec = {"seed": seed, "n_train_attacks": int(ytr.sum()),
               "n_test_attacks": int(yte.sum()), "scenarios": {}}

        # -- reference: no balancing ------------------------------------------
        m_ref = rf(Xtr, ytr, seed)
        g_ref = shap_ranks(m_ref, bg)
        g_ref_int = shap_ranks(m_ref, bg, "interventional", real_bg)
        rec["scenarios"]["baseline"] = {
            "quality": quality(m_ref, Xte, yte),
            "attr": compare(g_ref, g_ref),
            "lime": lime_stability(m_ref, target, Xtr),
        }

        # -- noise control: another model seed, same data -----------------------
        m_c = rf(Xtr, ytr, seed + 9999)
        rec["scenarios"]["seed_control"] = {
            "quality": quality(m_c, Xte, yte),
            "attr": compare(g_ref, shap_ranks(m_c, bg)),
            # noise level of the interventional mode (same real background)
            "attr_interventional": compare(
                g_ref_int, shap_ranks(m_c, bg, "interventional", real_bg)),
        }

        # -- rebalancing without changing the data ------------------------------
        m_w = rf(Xtr, ytr, seed, cw="balanced")
        rec["scenarios"]["class_weight"] = {
            "quality": quality(m_w, Xte, yte),
            "attr": compare(g_ref, shap_ranks(m_w, bg)),
            "lime": lime_stability(m_w, target, Xtr),
        }

        # -- resampling --------------------------------------------------------
        for name, mk in SAMPLERS.items():
            try:
                Xs, ys = mk(seed).fit_resample(Xtr, ytr)
                Xs = pd.DataFrame(np.asarray(Xs), columns=X.columns)
                synth = Xs.iloc[len(Xtr):] if len(Xs) > len(Xtr) else Xs.iloc[:0]
                m = rf(Xs, ys, seed)
                g = shap_ranks(m, bg)
                sc = {
                    "quality": quality(m, Xte, yte),
                    "attr": compare(g_ref, g),
                    "integrity": integrity(synth, int_cols),
                    # LIME as a practitioner would run it: neighbourhood from the resampled data
                    "lime": lime_stability(m, target, Xs),
                    # remedy: LIME neighbourhood from real training records
                    "lime_realref": lime_stability(m, target, Xtr),
                }
                # remedy: interventional TreeSHAP with a background of real records
                sc["attr_interventional"] = compare(
                    g_ref_int, shap_ranks(m, bg, "interventional", real_bg))
                rec["scenarios"][name] = sc
            except Exception as e:
                rec["scenarios"][name] = {"error": str(e)[:200]}

        out["runs"].append(rec)
        print(f"  split {seed} done in {time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(f"results_{tag}.json", "w"), ensure_ascii=False, indent=1)

    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
