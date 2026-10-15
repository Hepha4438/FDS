"""Rà soát: (A) data.json có khớp notebook 36 không, (B) hình có khớp số không."""
import json, re, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
NB = "/root/.claude/uploads/215cbb2f-e5f0-53ef-81b7-aa79733d8893/a87b6d47-36-usher.ipynb"
D = json.load(open(f"{HERE}/data.json"))
METRICS = list(D)
METHODS = ["raw", "center", "CORAL", "MNN-ref", "ridge", "FROST_base", "FROST_cc",
           "FROST", "Harmony", "Scanorama", "Symphony"]
PAIRS = list(D["xfer_main"])

# ---------- A. đọc lại notebook ĐỘC LẬP rồi đối chiếu ----------
nb = json.load(open(NB))
blocks = [''.join(o.get('data', {}).get('text/plain', ''))
          for o in nb['cells'][10]['outputs']]
blocks = [b for b in blocks if 'cặp' in b and 'raw' in b]
bad = patched = 0
for name, blk in zip(METRICS, blocks):
    for line in blk.split("\n"):
        mm = re.match(r'^((?:coad|ov|hcc)/[a-z0-9_]+)', line.strip())
        if not mm:
            continue
        toks = re.findall(r'[-+]?\d+\.\d+(?:±\d+\.\d+)?|—', line)
        if len(toks) < 11:
            continue
        for meth, t in zip(METHODS, toks[:11]):
            cell = D[name][mm.group(1)][meth]
            if t == '—':
                ok = cell["m"] is None
            else:
                v = float(t.split('±')[0])
                ok = (cell["m"] is not None and abs(cell["m"] - v) < 5e-5)
                if not ok and "seed0_nb36" in cell:      # ô đã vá bằng 3 seed
                    ok = abs(cell["seed0_nb36"] - v) < 5e-5
                    patched += ok
            if not ok:
                bad += 1
                print(f"  LỆCH {name} {mm.group(1)} {meth}: data={cell['m']} nb={t}")
print(f"A. đối chiếu notebook: {bad} ô lệch  ({patched} ô đã vá bằng 3 seed, "
      f"khớp với seed 0 gốc)")

# ---------- B. trung bình và trung vị có cùng dấu không ----------
print("\nB. các ô mà HIỆU CỦA TRUNG BÌNH và TRUNG VỊ HIỆU SỐ lệch dấu:")
BASE = ["raw", "center", "CORAL", "MNN-ref", "ridge", "Harmony", "Scanorama", "Symphony"]
COLS = ["batch_v2", "bio_v2", "total_v2", "xfer_main", "xfer_all",
        "rare_purity", "knn_xen", "imputation"]
n_flip = 0
for m in COLS:
    for b in BASE:
        d = [D[m][p]["FROST"]["m"] - D[m][p][b]["m"] for p in PAIRS
             if D[m][p]["FROST"]["m"] is not None and D[m][p][b]["m"] is not None]
        if not d:
            continue
        mu_f = np.mean([D[m][p]["FROST"]["m"] for p in PAIRS
                        if D[m][p]["FROST"]["m"] is not None])
        mu_b = np.mean([D[m][p][b]["m"] for p in PAIRS if D[m][p][b]["m"] is not None])
        dmean, dmed = mu_f - mu_b, float(np.median(d))
        if np.sign(dmean) != np.sign(dmed):
            n_flip += 1
            print(f"  {m:<12} vs {b:<10} hiệu TB {dmean:+.4f} | trung vị {dmed:+.4f} "
                  f"| p={wilcoxon(d).pvalue:.2f}")
print(f"  -> {n_flip} ô lệch dấu. Thanh vẽ theo TB mà số in theo trung vị => "
      f"{n_flip} ô hình MÂU THUẪN với số.")

# ---------- C. số liệu sẽ in ra hình sau khi sửa ----------
print("\nC. knn_xen — kiểm chứng chi tiết (sau khi sửa, thanh vẽ theo trung vị):")
for b in ["center", "Harmony", "CORAL"]:
    d = [D["knn_xen"][p]["FROST"]["m"] - D["knn_xen"][p][b]["m"] for p in PAIRS]
    med = float(np.median(d))
    print(f"  {b:<9} trung vị hiệu = {med:+.4f}  p = {wilcoxon(d).pvalue:.2f}  "
          f"-> vạch đen {'LẤN PHẢI đường xanh' if med < 0 else 'nằm trái đường xanh'}"
          f"  ({sum(1 for x in d if x > 0)}/12 cặp FROST hơn)")
