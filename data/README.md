# Data

The raw data are not redistributed here. The experiments use the CIC-IDS2017 file
`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` from the `MachineLearningCVE`
set published by the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

The copy used in the article was taken from a public GitHub mirror and checked against
the published label counts (168 186 BENIGN, 1 507 Brute Force, 652 XSS, 21 SQL Injection):

```
curl -L -o data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv \
  https://raw.githubusercontent.com/HY21797/CICIDS2017-models/main/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

SHA-256 of the file used:
`d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d`

After cleaning with `src/clean_cicids.py`, the file has 146 314 flows and 61 features
(SHA-256 of `thursday_clean.csv`:
`46bd1b1db76f307e09f1ed67e4b5f403679a1716fd0c43269f4a6c9153ffee1a`).
