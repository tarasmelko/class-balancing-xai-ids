"""
Очищення CIC-IDS2017 (набір MachineLearningCVE) на рівні CSV.

Реалізує ті виправлення з Engelen та ін. (2021) і Rosay та ін. (2022), які
принципово можливі без перегенерації з PCAP, і ЯВНО документує ті, що неможливі.

Запуск:  python3 clean_cicids.py <вхідний.csv> <вихідний.csv>
"""
import sys, json
import numpy as np
import pandas as pd

# Колонки, які CICFlowMeter заповнює нулями у всіх файлах CIC-IDS2017
# (підтверджено емпірично, див. звіт audit)
KNOWN_CONSTANT = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# Пари ознак з тотожним вмістом. Лишаємо першу, другу відкидаємо.
# 'Fwd Header Length.1' — прямий дублікат 'Fwd Header Length' (дефект CICFlowMeter).
REDUNDANT = [
    "Fwd Header Length.1",        # == Fwd Header Length
    "Avg Fwd Segment Size",       # == Fwd Packet Length Mean
    "Avg Bwd Segment Size",       # == Bwd Packet Length Mean
    "Subflow Fwd Packets",        # == Total Fwd Packets
    "Subflow Bwd Packets",        # == Total Backward Packets
    "Subflow Fwd Bytes",          # == Total Length of Fwd Packets
    "Subflow Bwd Bytes",          # == Total Length of Bwd Packets
]

# Опубліковані Engelen та ін. (2021) кількості після виправлення — для звірки
ENGELEN_REFERENCE = {
    "Web Attack - Brute Force": (1507, 151),
    "Web Attack - XSS": (652, 27),
    "Bot": (1956, 738),
}


def clean(path_in, path_out, report_path=None):
    rep = {"input_file": path_in}

    # ── 0. Завантаження, нормалізація назв колонок і міток ────────────────
    df = pd.read_csv(path_in, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rep["rows_raw"], rep["cols_raw"] = df.shape

    # Мітки у файлах CIC-IDS2017 записані в cp1252: en-dash приходить як 0x96
    # і після читання як UTF-8 перетворюється на U+FFFD. Нормалізуємо на дефіс.
    df["Label"] = (df["Label"].astype(str)
                   .str.replace("�", "-", regex=False)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip())
    rep["labels_raw"] = df["Label"].value_counts().to_dict()

    # ── 1. Константні та надлишкові ознаки ────────────────────────────────
    const = [c for c in df.columns
             if c != "Label" and df[c].nunique(dropna=False) <= 1]
    redundant = [c for c in REDUNDANT if c in df.columns]
    drop_cols = sorted(set(const) | set(redundant))
    df = df.drop(columns=drop_cols)
    rep["dropped_constant"] = const
    rep["dropped_redundant"] = redundant
    rep["cols_after_feature_cleanup"] = df.shape[1]

    num = df.select_dtypes(include=[np.number]).columns

    # ── 2. Inf та NaN ─────────────────────────────────────────────────────
    # 'Flow Bytes/s' і 'Flow Packets/s' діляться на тривалість; за нульової
    # тривалості CICFlowMeter пише Infinity або порожнє значення.
    inf_mask = np.isinf(df[num]).any(axis=1)
    nan_mask = df[num].isna().any(axis=1)
    rep["rows_with_inf"] = int(inf_mask.sum())
    rep["rows_with_nan"] = int(nan_mask.sum())
    df = df[~(inf_mask | nan_mask)].copy()

    # ── 3. Фізично неможливі значення ─────────────────────────────────────
    # Від'ємна тривалість потоку і від'ємні міжпакетні інтервали — прямий
    # наслідок некоректного завершення TCP-потоків (Rosay та ін., 2022).
    # Init_Win_bytes_* = -1 НЕ є помилкою: це легітимний маркер «вікно невідоме».
    impossible = pd.Series(False, index=df.index)
    for c in ["Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
              "Flow Bytes/s", "Flow Packets/s"]:
        if c in df.columns:
            impossible |= (df[c] < 0)
    rep["rows_impossible_negative"] = int(impossible.sum())
    df = df[~impossible].copy()

    # ── 4. Точні дублікати рядків ─────────────────────────────────────────
    dup = df.duplicated()
    rep["rows_exact_duplicates"] = int(dup.sum())
    rep["duplicates_by_label"] = df[dup]["Label"].value_counts().to_dict()
    df = df[~dup].copy()

    # ── 5. Потоки без корисного навантаження («TCP appendices») ───────────
    # Engelen та ін. (2021) показали, що некоректна обробка завершення TCP
    # породжує потоки-фрагменти лише з керуючих пакетів. Такий потік не
    # переносить прикладних даних, тому не може бути ані веб-атакою, ані
    # осмисленим сеансом.
    #
    # ВАЖЛИВО: правило застосовується ОДНАКОВО до всіх класів. Якби ми
    # прибрали такі потоки лише з атак, ознака «нульове навантаження»
    # ідеально передбачала б клас BENIGN, і модель вивчила б артефакт
    # очищення замість властивостей трафіку.
    payload = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    empty = payload == 0
    rep["rows_zero_payload"] = int(empty.sum())
    rep["zero_payload_by_label"] = df[empty]["Label"].value_counts().to_dict()
    df = df[~empty].copy()

    # ── 6. Звірка з опублікованими числами Engelen та ін. ─────────────────
    after = df["Label"].value_counts().to_dict()
    check = {}
    for lab, (pub_before, pub_after) in ENGELEN_REFERENCE.items():
        if lab in rep["labels_raw"]:
            check[lab] = {
                "our_before": int(rep["labels_raw"][lab]),
                "our_after": int(after.get(lab, 0)),
                "engelen_before": pub_before,
                "engelen_after": pub_after,
                "match": int(after.get(lab, 0)) == pub_after,
            }
    rep["engelen_crosscheck"] = check

    rep["rows_clean"], rep["cols_clean"] = df.shape
    rep["labels_clean"] = after
    n_att = sum(v for k, v in after.items() if k != "BENIGN")
    rep["imbalance_raw"] = round(
        rep["labels_raw"].get("BENIGN", 0)
        / max(1, sum(v for k, v in rep["labels_raw"].items() if k != "BENIGN")), 1)
    rep["imbalance_clean"] = round(after.get("BENIGN", 0) / max(1, n_att), 1)

    df.to_csv(path_out, index=False)
    if report_path:
        json.dump(rep, open(report_path, "w"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    r = clean(sys.argv[1], sys.argv[2],
              sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
