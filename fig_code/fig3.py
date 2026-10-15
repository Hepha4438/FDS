"""HÌNH 3 — khoảng cách từ mỗi baseline tới FROST, kiểm định Wilcoxon signed-rank.

  đường xanh = mức của FROST (mốc chung, đứng yên ở mọi ô)
  vạch đen   = mức của baseline, đặt tại −(trung vị hiệu số); baseline nào HƠN
               FROST thì vạch lấn sang phải đường xanh
  thanh      = khoảng giữa hai mức
  dòng số    = trung vị hiệu số và p; IN ĐẬM khi p < 0,05

Vạch, thanh và số đều lấy từ CÙNG một đại lượng — trung vị của 12 hiệu số theo
cặp (FROST − baseline), chính đại lượng Wilcoxon kiểm. (Bản trước vẽ thanh theo
hiệu của hai TRUNG BÌNH còn số in theo TRUNG VỊ nên 3 ô lệch dấu giữa hình và
số: knn_xen vs center, knn_xen vs Harmony, xfer_all vs ridge.)

Xếp hai dải như Hình 1–2 để vừa bề ngang trang dọc mà chữ giữ cỡ thật.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
import style as S

S.rc()
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/data.json"))

BASE = [("raw", "raw"), ("center", "center"), ("CORAL", "CORAL"),
        ("MNN-ref", "MNN-ref"), ("ridge", "ridge"),
        ("Harmony *", "Harmony"), ("Scanorama *", "Scanorama"),
        ("Symphony *", "Symphony")]
COLS = [("batch_v2", "batch_v2"), ("bio_v2", "bio_v2"), ("total_v2", "total_v2"),
        ("xfer_main", "xfer_main"), ("xfer_all", "xfer_all"),
        ("rare_purity", "rare_purity"), ("knn_xen", "knn_xen")]   # bỏ nội suy (giao thức xuôi, rò rỉ) — §6
NR = len(BASE)
BARC, REF = "#9b9a93", "#2a78d6"

# Hình 3 để NGANG (một dải, 8 cột): mỗi ô cần chỗ cho dòng "trung vị + p" nên
# không ép xuống khổ trang dọc được như Hình 1–2.
FW = 11.9
X_LAB, LW = 1.26, 1.38
H_ROW, H_HEAD, H_TITLE, H_FOOT = 0.50, 0.40, 0.15, 0.70
FS_LAB, FS_COL, FS_NUM, FS_TICK = 9.0, 9.0, 7.4, 6.6


def stat(mkey, bkey):
    d = [D[mkey][p]["FROST"]["m"] - D[mkey][p][bkey]["m"] for p in S.PAIRS
         if D[mkey][p]["FROST"]["m"] is not None and D[mkey][p][bkey]["m"] is not None]
    return (float(np.median(d)), wilcoxon(d).pvalue) if d else (None, None)


FH = H_TITLE + H_HEAD + NR * H_ROW + H_FOOT
fig = plt.figure(figsize=(FW, FH), dpi=220)
fx, fy = (lambda v: v / FW), (lambda v: v / FH)
bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
bg.set_xlim(0, 1); bg.set_ylim(0, 1)


def panel(cols, top_in):
    nc = len(cols)
    cw_in = (FW - LW - 0.20) / nc
    cw, x0 = fx(cw_in), fx(LW)
    y1 = 1 - fy(top_in + H_HEAD)
    rh = fy(H_ROW)
    y0 = y1 - NR * rh
    row_y = lambda i: y1 - (i + 1) * rh

    for i, (lab, _) in enumerate(BASE):
        yy = row_y(i)
        if i % 2 == 0:
            bg.add_patch(plt.Rectangle((x0 - fx(0.04), yy), fx(nc * cw_in + 0.04),
                                       rh, fc="#f6f5f2", ec="none", zorder=0))
        bg.text(fx(X_LAB), yy + rh / 2, lab, ha="right", va="center",
                fontsize=FS_LAB, color=S.INK_2)

    for j, (mkey, mlab) in enumerate(cols):
        xs = x0 + j * cw
        gaps = [-stat(mkey, bk)[0] for _, bk in BASE if stat(mkey, bk)[0] is not None]
        lo = min(min(gaps) * 1.16, -max(gaps) * 0.34)
        hi = max(max(gaps) * 1.30, -min(gaps) * 0.17)

        for i, (_, bkey) in enumerate(BASE):
            ax = fig.add_axes([xs + fx(0.03), row_y(i), cw - fx(0.06), rh])
            ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
            ax.set_facecolor("none"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.axvline(0, color=REF, lw=1.2, zorder=2)

            if i == NR - 1:
                tr = ax.get_xaxis_transform()
                ax.plot([lo, hi], [-0.08, -0.08], color=S.BASELINE, lw=0.6,
                        clip_on=False, transform=tr)
                for xv, al in ((min(gaps), "left"), (0.0, "center")):
                    ax.plot([xv, xv], [-0.08, -0.18], color=S.BASELINE, lw=0.6,
                            clip_on=False, transform=tr)
                    ax.text(xv, -0.24, f"{xv:+.2f}" if xv else "0", ha=al,
                            va="top", fontsize=FS_TICK, color=S.MUTED,
                            clip_on=False, transform=tr)

            med, pv = stat(mkey, bkey)
            if med is None:
                ax.text((lo + hi) / 2, 0.58, "n/a", ha="center", va="center",
                        fontsize=5.2, color=S.BASELINE)
                continue
            g = -med
            ax.barh(0.66, -g, height=0.36, left=g, color=BARC, ec="none", zorder=1)
            ax.plot([g, g], [0.44, 0.88], color=S.INK, lw=1.7, zorder=4,
                    solid_capstyle="butt")
            ps = (f"p={pv:.0e}".replace("e-0", "e−") if pv < 0.01
                  else f"p={pv:.2f}")
            ax.text(0.5, 0.16, f"{med:+.3f}  {ps}", ha="center", va="center",
                    fontsize=FS_NUM, color=S.INK if pv < 0.05 else S.MUTED,
                    fontweight="bold" if pv < 0.05 else "normal",
                    transform=ax.transAxes)

        bg.text(xs + cw / 2, y1 + fy(0.05), mlab, ha="center", va="bottom",
                fontsize=FS_COL, color=S.INK, fontweight="bold")


panel(COLS, H_TITLE)

h = [Line2D([], [], color=S.INK, lw=1.7, label="baseline level"),
     Line2D([], [], color=REF, lw=1.5, label="FROST level"),
     Patch(fc=BARC, ec="none", label="gap between levels"),
     Line2D([], [], ls="", label="bold = p < 0.05")]
L = fig.legend(handles=h, loc="lower left",
               bbox_to_anchor=(fx(0.22), fy(0.06)), ncol=4,
               frameon=False, fontsize=8.0, handletextpad=0.4,
               columnspacing=1.0)
L._legend_box.align = "left"

# tiêu đề + chú thích chân hình chuyển vào caption LaTeX
out = f"{HERE}/figs/fig_gaps.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=S.SURFACE)
print(f"đã lưu {os.path.basename(out)}  {FW:.1f}×{FH:.1f} in  "
      f"{os.path.getsize(out)/1e6:.2f} MB")
