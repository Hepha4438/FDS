# Chạy: cd fig_code && python3 fig_umap.py [combo]   (mặc định stereoseq_scgpt = COAD; cần umap_panel.npz từ notebook 44)
# Dữ liệu: umap_panel.npz md5 63ceaee58e21b2f3cd10257b0d3625d5
"""Hình UMAP cho bài báo: 5 panel (raw · Harmony · Scanorama · USHER-ST · FROST) × 2 hàng.

Hàng trên tô theo CÔNG NGHỆ (Xenium / nguồn) -> trộn lô; hàng dưới tô theo LOẠI TẾ BÀO -> cấu trúc.
Toạ độ: out23u/umap_panel.npz (run44.py), khoá "<combo_>__<panel>"; mỗi mảng có n_x hàng Xenium rồi
n_s hàng nguồn (thứ tự của run23.umap2). Không tiêu đề trong hình; chú giải vào caption LaTeX.
"""
import hashlib, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.rc()
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = f"{HERE}/umap_panel.npz"
COMBO = sys.argv[1] if len(sys.argv) > 1 else "stereoseq_scgpt"   # khoá npz = "<assay>_<rep>" (COAD)
PANELS = ["raw", "Harmony", "Scanorama", "USHER-ST", "FROST"]
TITLES = {"raw": "No alignment", "Harmony": "Harmony *", "Scanorama": "Scanorama *",
          "USHER-ST": "USHER-ST", "FROST": "FROST"}
SRC = COMBO.rsplit("_", 1)[0]
C_X, C_S = "#9b9a93", S.TECH_COLOR.get(SRC, "#eda100")

print("md5", NPZ, hashlib.md5(open(NPZ, "rb").read()).hexdigest())
D = np.load(NPZ, allow_pickle=True)
lab_x, lab_s = D[f"{COMBO}__lab_x"].astype(str), D[f"{COMBO}__lab_s"].astype(str)
nx = len(lab_x)
lab = np.concatenate([lab_x, lab_s])
tech = np.r_[np.zeros(nx, int), np.ones(len(lab_s), int)]
types = sorted(set(lab))
cmap = plt.get_cmap("tab20")
TYPE_C = {t: cmap(i % 20) for i, t in enumerate(types)}
order = np.random.RandomState(0).permutation(len(lab))   # trộn thứ tự vẽ, tránh lớp sau đè lớp trước

FW = 6.69
fig, axes = plt.subplots(2, len(PANELS), figsize=(FW, 3.05), dpi=300)
for j, p in enumerate(PANELS):
    k = f"{COMBO}__{p}"
    for i in range(2):
        ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if k not in D:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", color=S.MUTED, transform=ax.transAxes); continue
        Z = D[k][order]
        col = (np.where(tech[order] == 0, C_X, C_S) if i == 0
               else [TYPE_C[t] for t in lab[order]])
        ax.scatter(Z[:, 0], Z[:, 1], s=0.6, c=col, linewidths=0, rasterized=True)
    axes[0, j].set_title(TITLES[p], fontsize=7.5, color=S.INK,
                         fontweight="bold" if p == "FROST" else "normal")
axes[0, 0].set_ylabel("platform", fontsize=7, color=S.INK_2)
axes[1, 0].set_ylabel("cell type", fontsize=7, color=S.INK_2)

h1 = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=C_X, mec="none", label="Xenium 5K (reference)"),
      Line2D([], [], marker="o", ls="", ms=3.5, mfc=C_S, mec="none", label=S.TECH_LABEL.get(SRC, SRC))]
h2 = [Line2D([], [], marker="o", ls="", ms=3.5, mfc=TYPE_C[t], mec="none", label=t) for t in types]
fig.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False, fontsize=6.2)
fig.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=min(8, len(h2)), frameon=False,
           fontsize=5.8, handletextpad=0.2, columnspacing=0.8)
fig.subplots_adjust(left=0.04, right=0.995, top=0.92, bottom=0.02, wspace=0.04, hspace=0.04)
os.makedirs(f"{HERE}/figs", exist_ok=True)
out = f"{HERE}/figs/fig_umap_{COMBO}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=S.SURFACE)
print("đã lưu", out)
