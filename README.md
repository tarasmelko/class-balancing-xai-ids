# Class balancing shifts SHAP and LIME explanations in intrusion detection

Code, results and figures for the article

> Melko, T., & Kotsun, V. *Class balancing shifts SHAP and LIME explanations in intrusion
> detection: causes and a practical remedy.* Submitted to *Cybersecurity: Education, Science,
> Technique*, 2026.

The experiment trains a random forest on the web-attack subset of CIC-IDS2017 (cleaned
following Engelen et al., 2021) under seven configurations and measures how much the global
SHAP attribution and the local LIME explanation change relative to the unbalanced model.

| Configuration | What it does | Role |
|---|---|---|
| No balancing | trains on the data as they are | reference |
| Different model seed | same data, another random seed | noise level of the measurement |
| `class_weight="balanced"` | reweights the loss | rebalancing without changing the data |
| Random oversampling | duplicates real minority records | rebalancing without synthesis |
| SMOTE, Borderline-SMOTE, ADASYN | interpolate new minority records | synthetic oversampling |

## Main results (10 splits)

| Configuration | PR-AUC | SHAP rank agreement, ρ | Generated records with impossible values, % | LIME stability, VSI |
|---|---|---|---|---|
| No balancing | 0.985 ± 0.009 | reference | – | 0.952 |
| Different model seed | 0.987 ± 0.006 | 0.829 ± 0.030 | – | – |
| class_weight | 0.994 ± 0.005 | 0.579 ± 0.033 | – | 0.948 |
| Random oversampling | 0.994 ± 0.004 | 0.565 ± 0.042 | 0 | 0.770 |
| SMOTE | 0.996 ± 0.002 | 0.562 ± 0.041 | 100 | 0.744 |
| Borderline-SMOTE | 0.993 ± 0.004 | 0.562 ± 0.059 | 100 | 0.882 |
| ADASYN | 0.996 ± 0.002 | 0.534 ± 0.030 | 100 | 0.881 |

Computing interventional TreeSHAP against 500 real training records raises ρ for the
resampled models to 0.69–0.73, and building the LIME neighbourhood from real records restores
VSI to 0.95–0.99. The full set of numbers, with the statistical tests, is in
[`results/analysis_clean10.txt`](results/analysis_clean10.txt).

## Repository layout

```
src/clean_cicids.py   CSV-level cleaning of CIC-IDS2017 (constant/duplicate features, Inf/NaN,
                      impossible values, exact duplicates, zero-payload flows)
src/experiment.py     the experiment: 7 configurations x 10 splits -> results/results_clean10.json
src/analysis.py       every number reported in the article, with the statistical tests
src/figures.py        Figures 1-6
results/              cleaning report, raw results, run log, analysis output
figures/              the figures as published in the article
data/README.md        where to obtain the input CSV and its checksum
run_all.sh            runs everything end to end
```

## Reproducing the results

```
python3 -m pip install -r requirements.txt
curl -L -o data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv \
  https://raw.githubusercontent.com/HY21797/CICIDS2017-models/main/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
./run_all.sh
```

The run takes about 40 minutes on a 2-core machine. All random states are fixed (split
`seed`, model `seed`, reference model `seed + 9999`, LIME runs `0..14`), so with the pinned
versions of scikit-learn, imbalanced-learn and shap the results file is reproduced exactly.

Notes:

* The reference `lime` package does not build with current tooling, so tabular LIME is
  implemented directly from the specification of Ribeiro et al. (2016) in `src/experiment.py`
  (Gaussian perturbations, exponential kernel, weighted ridge surrogate).
* LIME stability is the variables stability index of Visani et al. (2022), computed as the mean
  pairwise Jaccard overlap of the five highest-weighted features over 15 explanations of the
  same attack record.
* Resampling is fitted on the training part of each split only.

## Licence

MIT, see [`LICENSE`](LICENSE). The CIC-IDS2017 data are subject to the terms of the
Canadian Institute for Cybersecurity and are not included.
