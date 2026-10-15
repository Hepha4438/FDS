# Chạy: cd fig_code && python3 fig_pareto.py   (cần matplotlib, numpy; KHÔNG dùng python3 -I)
# Dữ liệu: data.json md5 33c26b15e01a157762f4cb111b208c0b
"""Mặt phẳng trộn lô ↔ tác vụ: mỗi phương pháp là một điểm = trung bình 12 cặp (batch_v2, y).
Đường gạch = biên Pareto của các điểm trung bình (không bị điểm nào khác trội ở cả hai trục).
Ba hình: y = xfer_main (bài chính), xfer_all, rare_purity (phụ lục). Tiếng Anh, không tiêu đề.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import style as S

S.rc()
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/data.json"))
NAME = {"raw": "raw", "center": "center", "CORAL": "CORAL", "MNN-ref": "MNN-ref", "ridge": "ridge",
        "Harmony": "Harmony*", "Scanorama": "Scanorama*", "Symphony": "Symphony*",
        "FROST_base": "USHER-ST", "FROST_cc": "USHER-CC", "FROST": "FROST"}
OURS = {"FROST_base", "FROST_cc", "FROST"}
# lệch nhãn để hai cặp điểm gần trùng (FROST/USHER-CC, USHER-ST/ridge) không đè chữ
OFF = {"FROST": (0, -10, "center"), "FROST_cc": (0, 5, "center"), "FROST_base": (-4, 5, "right"),
       "ridge": (4, -8, "left"), "center": (0, -10, "center"), "MNN-ref": (4, -9, "left"),
       ("rare_purity", "ridge"): (-4, 3, "right"), ("rare_purity", "CORAL"): (0, -10, "center"),
       ("xfer_main", "center"): (-4, 3, "right"), ("xfer_all", "ridge"): (0, 5, "center"),
       ("xfer_all", "MNN-ref"): (-4, 3, "right"), ("rare_purity", "MNN-ref"): (4, 3, "left"),
       ("rare_purity", "Symphony"): (4, -9, "left"), ("xfer_main", "raw"): (0, -10, "center"),
       ("rare_purity", "Scanorama"): (6, -5, "left")}
YS = [("xfer_main", "xfer_main"), ("xfer_all", "xfer_all"), ("rare_purity", "rare_purity")]

def mean(k, m): return float(np.mean([D[k][p][m]["m"] for p in S.PAIRS]))

def frontier(pts):
    keep = [i for i, (x, y) in enumerate(pts)
            if not any((x2 >= x and y2 >= y) and (x2 > x or y2 > y) for x2, y2 in pts)]
    return sorted(keep, key=lambda i: pts[i][0])

def draw(ax, yk, ylab):
    ms = list(NAME)
    pts = [(mean("batch_v2", m), mean(yk, m)) for m in ms]
    f = frontier(pts)
    ax.plot([pts[i][0] for i in f], [pts[i][1] for i in f], ls="--", lw=0.9, color=S.BASELINE, zorder=1)
    for (x, y), m in zip(pts, ms):
        hi = m in OURS
        ax.scatter(x, y, s=34 if m == "FROST" else 22, zorder=3,
                   c=S.INK if m == "FROST" else (S.BAR_HI if hi else S.BAR), edgecolors="none",
                   marker="D" if m in ("Harmony", "Scanorama", "Symphony") else "o")
        dx, dy, ha = OFF.get((yk, m), OFF.get(m, (4, 3, "left")))
        ax.annotate(NAME[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.4, ha=ha,
                    color=S.INK if hi else S.INK_2, fontweight="bold" if m == "FROST" else "normal")
    ax.set_xlabel("batch mixing (batch_v2)", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)

os.makedirs(f"{HERE}/figs", exist_ok=True)
fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
draw(ax, "xfer_main", "label transfer (xfer_main)")
fig.tight_layout(); fig.savefig(f"{HERE}/figs/fig_pareto_main.png", dpi=300, bbox_inches="tight", facecolor=S.SURFACE)
fig, axs = plt.subplots(1, 3, figsize=(6.69, 2.3), dpi=300)
for ax, (k, lab) in zip(axs, YS): draw(ax, k, lab)
fig.tight_layout(); fig.savefig(f"{HERE}/figs/fig_pareto_supp.png", dpi=300, bbox_inches="tight", facecolor=S.SURFACE)
for k, _ in YS:
    pts = [(mean("batch_v2", m), mean(k, m)) for m in NAME]
    print(k, "biên Pareto:", [list(NAME)[i] for i in frontier(pts)])
