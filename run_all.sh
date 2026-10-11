#!/usr/bin/env bash
# Reproduces every number and figure in the article.
# Requires data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (see data/README.md).
# Runtime: about 40 minutes on a 2-core machine.
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
python3 src/clean_cicids.py "$RAW" data/thursday_clean.csv results/clean_report_thursday.json > /dev/null
( cd results && python3 ../src/experiment.py ../data/thursday_clean.csv clean10 10 | tee run_clean10.log )
python3 src/analysis.py results/results_clean10.json | tee results/analysis_clean10.txt
python3 src/figures.py results/results_clean10.json figures
