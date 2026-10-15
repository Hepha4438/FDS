"""Hợp nhất số liệu cho báo cáo FROST.

Nguồn 1: notebook 36 cell 10 (BẢNG 2) — 18 chỉ số × 12 cặp × 11 phương pháp.
Nguồn 2: note 22 `F/<tổ hợp>/f200/{s0,s1,s2}` — vá SD 3 seed cho dòng
         `USHER-ST (sàn 200)` ở 4 cặp COAD (notebook 36 chỉ có seed 0).

Ra: data.json  {metric: {pair: {method: {"m": mean, "sd": sd|None}}}}
"""
import json, re, statistics as st, os

NB = "/root/.claude/uploads/215cbb2f-e5f0-53ef-81b7-aa79733d8893/a87b6d47-36-usher.ipynb"
N22 = "/mnt/user-data/uploads/research/results/22_summary.json"
OUT = os.path.dirname(os.path.abspath(__file__))

METRICS = ["ilisi", "kbet", "bras", "graph_conn", "batch_v2", "batch_asw", "nmi", "ari",
           "asw", "clisi", "iso_labels", "bio_v2", "total_v2", "xfer_main", "xfer_all",
           "rare_purity", "knn_xen", "imputation"]
METHODS = ["raw", "center", "CORAL", "MNN-ref", "ridge",
           "FROST_base", "FROST_cc", "FROST", "Harmony", "Scanorama", "Symphony"]

# --- nguồn 1: notebook 36 ---------------------------------------------------
nb = json.load(open(NB))
blocks = [''.join(o.get('data', {}).get('text/plain', '')) for o in nb['cells'][10]['outputs']]
blocks = [b for b in blocks if 'cặp' in b and 'raw' in b]
assert len(blocks) == len(METRICS), f"mong {len(METRICS)} bảng, thấy {len(blocks)}"

DATA = {}
for name, blk in zip(METRICS, blocks):
    rows = {}
    for line in blk.split("\n"):
        mm = re.match(r'^((?:coad|ov|hcc)/[a-z0-9_]+)', line.strip())
        if not mm:
            continue
        toks = re.findall(r'[-+]?\d+\.\d+(?:±\d+\.\d+)?|—', line)
        if len(toks) < 11:
            continue
        cell = {}
        for meth, t in zip(METHODS, toks[:11]):
            if t == '—':
                cell[meth] = {"m": None, "sd": None}
            elif '±' in t:
                a, b = t.split('±')
                cell[meth] = {"m": float(a), "sd": float(b)}
            else:
                cell[meth] = {"m": float(t), "sd": None}
        rows[mm.group(1)] = cell
    assert len(rows) == 12, f"{name}: {len(rows)} cặp"
    DATA[name] = rows

# --- nguồn 2: vá SD 3 seed cho COAD / FROST_base ----------------------------
# ánh xạ tên tổ hợp của note 22 -> tên cặp của notebook 36
COMBO2PAIR = {"cosmx6k/pca": "coad/cosmx6k", "visiumhd_ff/scgpt": "coad/visiumhd_ff",
              "visiumhd_ffpe/pca": "coad/visiumhd_ffpe", "stereoseq/scgpt": "coad/stereoseq"}
f22 = json.load(open(N22))["F"]
patched = []
for combo, pair in COMBO2PAIR.items():
    for met in METRICS:
        if met == "imputation":
            continue          # note 22 chỉ có `sp` = giao thức nội suy CŨ (có rò rỉ) -> không dùng
        vals = [f22[combo]["f200"][f"s{i}"].get(met) for i in range(3)]
        if any(v is None for v in vals):
            continue
        old = DATA[met][pair]["FROST_base"]["m"]
        DATA[met][pair]["FROST_base"] = {"m": round(st.mean(vals), 4),
                                         "sd": round(st.stdev(vals), 4),
                                         "seed0_nb36": old}
        patched.append((pair, met))

json.dump(DATA, open(f"{OUT}/data.json", "w"), indent=1)
print(f"data.json: {len(METRICS)} chỉ số × 12 cặp × {len(METHODS)} phương pháp")
print(f"đã vá SD 3 seed cho FROST_base: {len(patched)} ô (4 cặp COAD × 17 chỉ số, trừ nội suy)")

# --- kiểm tra tính toàn vẹn -------------------------------------------------
miss = [(m, p, k) for m in METRICS for p in DATA[m] for k in METHODS
        if DATA[m][p][k]["m"] is None]
print(f"ô trống: {len(miss)}" + (f"  (vd {miss[:3]})" if miss else ""))
n_sd = sum(1 for m in METRICS for p in DATA[m] for k in METHODS if DATA[m][p][k].get("sd"))
print(f"ô có SD: {n_sd}")
