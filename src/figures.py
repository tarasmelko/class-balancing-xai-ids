"""
Figures for the article: 300 dpi, black-and-white safe (identity is carried by
hatching and direct labels, not by colour). Captions are set in the article.

Usage: python3 figures.py [results/results_clean10.json] [output_dir]

Output files:
  figure1_smote_geometry.png     -> Fig. 1
  figure2_pipeline_coupling.png  -> Fig. 2
  figure3_attribution_shift.png  -> Fig. 3
  figure4_integrity.png          -> Fig. 4
  figure5_interventional.png     -> Fig. 5 (shortfall below each mode's own noise level)
  figure6_lime_stability.png     -> Fig. 6
"""
import sys
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results_clean10.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})

INK, MUTED, LIGHT = "#1A1A1A", "#666666", "#B0B0B0"
FILL_MAIN, FILL_CTRL, FILL_ALT = "#8C8C8C", "#FFFFFF", "#C9C9C9"

R = json.load(open(RESULTS))
ORDER = ["seed_control", "class_weight", "random_over",
         "smote", "borderline_smote", "adasyn"]
FLAT = {"random_over": "Random Oversampling", "smote": "SMOTE",
        "borderline_smote": "Borderline-SMOTE", "adasyn": "ADASYN",
        "class_weight": "class_weight", "baseline": "No balancing"}
LAB = {
    "baseline": "No\nbalancing",
    "seed_control": "Control:\nother model seed",
    "class_weight": "class_weight",
    "random_over": "Random\nOversampling",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-\nSMOTE",
    "adasyn": "ADASYN",
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def grab(scen, group, key):
    return [r["scenarios"][scen][group][key] for r in R["runs"]
            if scen in r["scenarios"] and group in r["scenarios"][scen]]


def by_seed(scen, group, key):
    return {r["seed"]: r["scenarios"][scen][group][key] for r in R["runs"]
            if scen in r["scenarios"] and group in r["scenarios"][scen]}


def clean_axes(ax):
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bars(ax, labels, mean, err, hatches, fills, edge=INK):
    x = np.arange(len(labels))
    b = ax.bar(x, mean, yerr=err, width=.62, color=fills, edgecolor=edge,
               linewidth=.9, hatch=hatches, capsize=3,
               error_kw=dict(ecolor=MUTED, elinewidth=.9))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    clean_axes(ax)
    return b


# ── Fig. 3 (file fig1). Attribution shift against the noise floor ─────────
fig, ax = plt.subplots(figsize=(6.3, 3.4))
m = [np.mean(grab(k, "attr", "spearman")) for k in ORDER]
e = [np.std(grab(k, "attr", "spearman")) for k in ORDER]
bars(ax, [LAB[k] for k in ORDER], m, e, ["//"] + [""] * 5,
     [FILL_CTRL] + [FILL_MAIN] * 5)
ax.axhline(m[0], color=INK, lw=1.0, ls=(0, (5, 3)), zorder=1)
ax.text(5.45, m[0] + .012, "noise floor of the measurement", ha="right",
        va="bottom", fontsize=8.5, color=INK, style="italic")
ax.set_ylabel("Spearman ρ of SHAP ranks\nagainst the unbalanced model")
ax.set_ylim(0, 1.0)
for i, v in enumerate(m):
    ax.text(i, v + e[i] + .03, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
save(fig, "figure3_attribution_shift.png")


# ── Fig. 4 (file fig2). Semantic integrity of generated records ───────────
K = ["random_over", "smote", "borderline_smote", "adasyn"]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
rows = [np.mean(grab(k, "integrity", "pct_rows_noninteger")) for k in K]
vals = [np.mean(grab(k, "integrity", "pct_values_noninteger")) for k in K]
x = np.arange(len(K)); w = .36
ax.bar(x - w/2, rows, w, color=FILL_MAIN, edgecolor=INK, lw=.9,
       label="generated records with at least one fractional value")
ax.bar(x + w/2, vals, w, color=FILL_CTRL, edgecolor=INK, lw=.9, hatch="///",
       label="values in whole-number columns that became fractional")
ax.set_xticks(x); ax.set_xticklabels([FLAT[k] for k in K], fontsize=8.5)
ax.set_ylabel("Share, %")
ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(fontsize=8.5, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=1)
clean_axes(ax)
for i in range(len(K)):
    ax.text(i - w/2, rows[i] + 2, f"{rows[i]:.0f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, vals[i] + 2, f"{vals[i]:.1f}", ha="center", fontsize=8.5)
ax.text(0, 14, "duplication violates\nno constraint", ha="center",
        fontsize=8, style="italic", color=MUTED)
save(fig, "figure4_integrity.png")


# ── Fig. 5 (file fig3). Remedy: shortfall below each mode's own floor ─────
# path-dependent: floor = seed control in the default mode (results_clean10.json)
# interventional: floor = seed control computed with the same real background
#                 (seed_control / attr_interventional in the same results file)
iv_floor = by_seed("seed_control", "attr_interventional", "spearman")
pd_floor = by_seed("seed_control", "attr", "spearman")

gap_pd, gap_iv = [], []
for k in K:
    pd_k = by_seed(k, "attr", "spearman")
    iv_k = by_seed(k, "attr_interventional", "spearman")
    seeds = sorted(set(pd_k) & set(iv_k) & set(pd_floor) & set(iv_floor))
    gap_pd.append([pd_floor[s] - pd_k[s] for s in seeds])
    gap_iv.append([iv_floor[s] - iv_k[s] for s in seeds])

mp = [np.mean(g) for g in gap_pd]; sp = [np.std(g) for g in gap_pd]
mi = [np.mean(g) for g in gap_iv]; si = [np.std(g) for g in gap_iv]

fig, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(K)); w = .36
ax.bar(x - w/2, mp, w, yerr=sp, color=FILL_MAIN, edgecolor=INK, lw=.9,
       capsize=3, error_kw=dict(ecolor=MUTED, elinewidth=.9),
       label="path-dependent TreeSHAP (reference: branch shares of resampled data)")
ax.bar(x + w/2, mi, w, yerr=si, color=FILL_CTRL, edgecolor=INK, lw=.9,
       hatch="\\\\\\", capsize=3, error_kw=dict(ecolor=MUTED, elinewidth=.9),
       label="interventional TreeSHAP (reference: 500 real training records)")
ax.axhline(0, color=INK, lw=.9)
ax.set_xticks(x); ax.set_xticklabels([FLAT[k] for k in K], fontsize=8.5)
ax.set_ylabel("Shortfall below the noise floor,\nΔρ (lower is better)")
ax.set_ylim(0, 0.46)
ax.legend(fontsize=8.5, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=1)
clean_axes(ax)
for i in range(len(K)):
    ax.text(i - w/2, mp[i] + sp[i] + .012, f"{mp[i]:.2f}", ha="center", fontsize=8.5)
    ax.text(i + w/2, mi[i] + si[i] + .012, f"{mi[i]:.2f}", ha="center", fontsize=8.5)
share = 1 - np.mean(mi) / np.mean(mp)
ax.text(-0.45, 0.455, f"mean shortfall {np.mean(mp):.2f} → {np.mean(mi):.2f}\n"
                    f"(about {share*100:.0f} % removed)",
        ha="left", va="top", fontsize=8.5, style="italic", color=INK)
save(fig, "figure5_interventional.png")
print(f"fig3: pd gap {np.mean(mp):.3f}, iv gap {np.mean(mi):.3f}, removed {share:.2f}, "
      f"iv floor {np.mean(list(iv_floor.values())):.3f}")


# ── Fig. 6 (file fig4). Stability of local LIME explanations ──────────────
KL = ["baseline", "class_weight", "random_over", "smote", "borderline_smote", "adasyn"]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
m = [np.mean(grab(k, "lime", "vsi_jaccard5")) for k in KL]
e = [np.std(grab(k, "lime", "vsi_jaccard5")) for k in KL]
# VSI cannot exceed 1.0, so the upper error bar is capped at the maximum
e_up = [min(sd, 1.0 - v) for v, sd in zip(m, e)]
bars(ax, [LAB[k] for k in KL], m, [e, e_up], ["//"] + [""] * 5,
     [FILL_CTRL, FILL_ALT] + [FILL_MAIN] * 4)
ax.set_ylabel("VSI: mean pairwise Jaccard@5\nacross 15 LIME runs")
ax.set_ylim(0, 1.12)
for i, v in enumerate(m):
    ax.text(i, v + e_up[i] + .03, f"{v:.3f}", ha="center", fontsize=8.5)
# bracket under the four resampling methods, below the tick labels
import matplotlib.transforms as mtransforms
tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax.plot([1.7, 5.3], [-0.215, -0.215], transform=tr, color=MUTED, lw=.8, clip_on=False)
for xx in (1.7, 5.3):
    ax.plot([xx, xx], [-0.215, -0.19], transform=tr, color=MUTED, lw=.8, clip_on=False)
ax.text(3.5, -0.235, "data resampling", transform=tr, ha="center", va="top",
        fontsize=8, style="italic", color=MUTED)
save(fig, "figure6_lime_stability.png")


# ── helpers for the two diagrams ──────────────────────────────────────────
def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=INK, lw=0.9, fs=9, style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, style=style, zorder=4, linespacing=1.45)


def arrow(ax, p0, p1, ls="-", lw=1.0, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 linestyle=ls, linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2,
                                 shrinkA=0, shrinkB=0))


# ── Fig. 1 (file fig5). Coupling of the two pipelines ─────────────────────
fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.axis("off")
W, H, Y = 0.232, 0.17, 0.70
xs = [0.0, 0.256, 0.512, 0.768]
labels = ["Imbalanced\ndata", "Resampling\n(SMOTE)",
          "Model\ntraining", "Post-hoc explanation\n(SHAP, LIME)"]
for xx, t in zip(xs, labels):
    box(ax, xx, Y, W, H, t, fs=8.8)
for i in range(3):
    arrow(ax, (xs[i] + W, Y + H / 2), (xs[i + 1], Y + H / 2))
box(ax, 0.235, 0.17, 0.53, 0.155, "Empirical distribution of the training data",
    fc="#EDEDED", ec=INK, fs=9.5)
arrow(ax, (xs[1] + W / 2, Y), (0.38, 0.325), ls=(0, (4, 2.5)))
ax.text(0.27, 0.50, "rewrites", fontsize=8.5, color=INK, style="italic", ha="center")
arrow(ax, (0.66, 0.325), (xs[3] + W / 2, Y), ls=(0, (4, 2.5)))
ax.text(0.885, 0.40, "draws its\nsubstitutions from here", fontsize=8.5, color=INK,
        style="italic", ha="center", linespacing=1.3)
arrow(ax, (xs[3] + W / 2, Y + H), (xs[3] + W / 2, 0.96))
ax.text(xs[3] + W / 2, 0.975, "SOC analyst's decision", fontsize=8.5,
        color=MUTED, ha="center", va="bottom")
ax.text(0.5, 0.045, "The two steps are designed independently, "
                    "yet they are coupled through one and the same distribution",
        fontsize=8.5, color=MUTED, style="italic", ha="center")
save(fig, "figure2_pipeline_coupling.png")


# ── Fig. 2 (file fig6). Geometry of SMOTE interpolation ───────────────────
fig, ax = plt.subplots(figsize=(6.3, 3.6))
benign = np.array([
    [0.09, 0.42], [0.20, 0.72], [0.36, 0.86], [0.55, 0.81], [0.70, 0.45],
    [0.42, 0.33], [0.50, 0.38], [0.45, 0.45], [0.57, 0.36], [0.52, 0.55],
    [0.33, 0.58], [0.26, 0.47], [0.66, 0.22], [0.78, 0.30], [0.88, 0.23],
    [0.95, 0.44], [0.15, 0.46], [0.92, 0.85], [0.68, 0.90], [0.36, 0.09],
    [0.55, 0.10], [0.78, 0.13], [0.95, 0.09], [0.99, 0.58], [0.60, 0.28],
    [0.64, 0.52], [0.30, 0.25], [0.84, 0.62], [0.24, 0.90], [0.47, 0.66],
])
A = np.array([[0.12, 0.16], [0.21, 0.27], [0.15, 0.34]])
B = np.array([[0.80, 0.74], [0.90, 0.64]])
segs = [(A[0], A[1]), (A[1], A[2]), (A[0], A[2]), (B[0], B[1]), (A[1], B[0])]
for p, q in segs:
    ax.plot([p[0], q[0]], [p[1], q[1]], ls=(0, (4, 2.5)), lw=1.0, color=MUTED, zorder=2)
synth = []
for p, q in segs:
    n = 5 if np.hypot(*(q - p)) > 0.4 else 2
    for t in np.linspace(0.18, 0.82, n):
        synth.append(p + t * (q - p))
synth = np.array(synth)
ax.scatter(benign[:, 0], benign[:, 1], s=26, facecolor=LIGHT, edgecolor=LIGHT,
           zorder=3, label="benign traffic")
ax.scatter(synth[:, 0], synth[:, 1], s=34, facecolor="white", edgecolor=INK,
           linewidth=1.0, zorder=4, label="synthetic SMOTE points")
ax.scatter(np.r_[A[:, 0], B[:, 0]], np.r_[A[:, 1], B[:, 1]], s=62,
           facecolor=INK, edgecolor=INK, zorder=5, label="real attacks")
mid = (A[1] + B[0]) / 2
ax.annotate("a segment between distant clusters\ncrosses a zone with no real attacks",
            xy=(mid[0], mid[1]), xytext=(0.30, 0.94), fontsize=8.5, color=INK,
            ha="center", va="top", linespacing=1.45, zorder=6,
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none"),
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xlim(-0.02, 1.06); ax.set_ylim(-0.02, 1.06)
ax.set_xlabel("feature $x_1$"); ax.set_ylabel("feature $x_2$")
ax.set_xticks([]); ax.set_yticks([])
ax.grid(False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#444444"); ax.spines[s].set_linewidth(0.7)
lg = ax.legend(fontsize=8.5, frameon=True, loc="lower right", handletextpad=0.4,
               borderpad=0.5, facecolor="white", edgecolor="none", framealpha=0.95)
lg.get_frame().set_linewidth(0)
save(fig, "figure1_smote_geometry.png")

print("done:", OUT)
