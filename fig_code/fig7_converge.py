# Chạy: cd fig_code && python3 fig7_converge.py   (cần matplotlib, numpy)
# Dữ liệu: converge.json md5 64ed6a67f221a09b7d27bc0e5f54da9a
"""HÌNH 7 — hội tụ bước E: ba cách dựng Cˢ, đặt cạnh bản cài đặt USHER công bố.

Đại lượng: tỉ lệ ánh xạ `T.argmax(1)` đổi giữa hai vòng EM liên tiếp, gộp theo nhóm —
đúng định nghĩa Fig. S6 của bài USHER. Vẽ TRUNG BÌNH 4 cặp COAD.

BỎ giá trị đầu tiên (v0 -> v1): ở vòng 0 code đặt gamma = 1 nên ghép hoàn toàn theo
M_aux, đổi gần 100% là tất yếu chứ không phải dấu hiệu chưa hội tụ.

USHER gốc chỉ có 5 mốc đo (v5/10/15/20/25) nên vẽ bằng điểm, không nội suy.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S

S.rc()
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/converge.json", encoding="utf-8"))

FW, FH = 6.69, 3.55            # 17 cm — bề rộng thật để Word không co hình
ARMS = [("it25",   r"(a) landmark $C^s$, recomputed every iteration", "#eb6834", ":",  1.8),
        ("both25", r"(b) exact $C^s$, cached",                          "#2a78d6", "-.", 1.8),
        ("cmix25", r"(c) FROST: exact $C^s$ + microenvironment, cached", S.INK,     "-",  2.4)]

def mean_curve(byp):
    L = min(len(a) for a in byp.values())
    return np.array([np.mean([a[i] for a in byp.values()]) for i in range(L)])

fig, ax = plt.subplots(figsize=(FW, FH), dpi=220)
for key, lab, col, ls, lw in ARMS:
    m = mean_curve(D["nhanh"][key])
    ax.plot(range(1, len(m)), m[1:], ls=ls, lw=lw, color=col, label=lab,
            marker="o", ms=3.0, markevery=2, zorder=3)

ug = D["usher_goc_moc"]
xs = sorted(int(k) for k in next(iter(ug.values())))
ys = [np.mean([ug[p][str(x)] if str(x) in ug[p] else ug[p][x] for p in ug]) for x in xs]
ax.plot(xs, ys, ls="--", lw=1.6, color=S.MUTED, marker="s", ms=4.5, zorder=2,
        label="USHER reference implementation (5 checkpoints)")

ax.axhline(0.05, color=S.BASELINE, lw=0.9, ls=":", zorder=1)
ax.text(1.0, 0.0525, "5%", color=S.MUTED, fontsize=7, ha="left", va="bottom")
ax.axvline(10, color=S.BASELINE, lw=0.9, ls=":", zorder=1)
ax.text(10.3, 0.47, "iteration 10\n(benchmark setting)", color=S.MUTED, fontsize=7, va="top")

ax.set_yscale("log")
ax.set_xlabel("EM iteration", fontsize=9)
ax.set_ylabel("fraction of assignments changed\n(mean over four COAD pairs)", fontsize=9)
ax.set_xlim(0.5, 25.2); ax.set_xticks([1, 5, 10, 15, 20, 25])
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.4, frameon=False, loc="lower left",
          bbox_to_anchor=(0.015, 0.02), handlelength=2.6)
fig.tight_layout()
out = f"{HERE}/figs/fig_converge.png"
os.makedirs(f"{HERE}/figs", exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=S.SURFACE)
print("->", out)

# số dùng trong bài, in ra để đối chiếu với §6.4
def med(key, sl): return float(np.median([x for a in D["nhanh"][key].values() for x in a[sl]]))
for k, lab, *_ in ARMS:
    cross = [next((i + 1 for i in range(1, len(a)) if a[i] < 0.05), None)
             for a in D["nhanh"][k].values()]
    print("%-8s @v10 %.4f | đuôi %.4f | qua 5%% %d/4 cặp %s"
          % (k, med(k, slice(9, 10)), med(k, slice(-3, None)),
             sum(x is not None for x in cross), [x or "—" for x in cross]))
print("USHER gốc @v25 TB %.4f" % np.mean([ug[p]["25"] for p in ug]))
