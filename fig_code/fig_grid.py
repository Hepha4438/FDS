# Chạy: cd fig_code && python3 fig_grid.py   (cần matplotlib, numpy, scipy; KHÔNG dùng python3 -I)
# Dữ liệu: data.json md5 33c26b15e01a157762f4cb111b208c0b
"""Lưới toàn cảnh: 11 phương pháp × 16 chỉ số, xếp thành HAI DẢI chồng nhau.

Vì sao hai dải: 16 cột trên một hàng ngang cần ~15 inch; đặt vừa bề ngang trang
dọc (17 cm ≈ 6,7 in) thì chữ co còn ~3,5 pt, không đọc được. Tách làm hai dải —
dải trên Trộn lô + Bảo toàn sinh học (10 cột), dải dưới Tác vụ hạ nguồn + Nội
suy + Tổng hợp (6 cột) — thì hình chỉ rộng 8,3 in và chữ vẫn trên 5 pt khi đặt
vừa trang dọc.

Hai hình cùng bộ chỉ số, khác CÁCH GOM 12 cặp:
  HÌNH 1  gom theo CÔNG NGHỆ -> 4 điểm/phương pháp (mỗi điểm = TB 3 mô)
  HÌNH 2  gom theo MÔ        -> 3 điểm/phương pháp (mỗi điểm = TB 4 công nghệ)
Lưới 3 mô × 4 công nghệ cân bằng nên TB của 4 điểm = TB của 3 điểm = TB 12 cặp.

Hai bảng màu TÁCH BIỆT, mỗi kênh đúng một nghĩa trong toàn báo cáo:
  Hình 1 màu = công nghệ (violet/vàng/magenta/xanh lá)
  Hình 2 màu = mô        (xanh/cam/aqua)
Cả hai đã chạy validate_palette.js --pairs all và PASS mọi cổng cứng.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.rc()
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/data.json"))

# Bản tiếng Anh cho bài báo (ghi đè nhãn của style.py tại chỗ; style.py không đổi)
S.BLOCKS = [
    ("Reference-\npreserving",
     [("raw", "raw"), ("center", "center"), ("CORAL", "CORAL"), ("MNN-ref", "MNN-ref")]),
    ("Reference-\npreserving,\nlabel-informed",
     [("ridge", "ridge"), ("USHER-ST", "FROST_base"), ("USHER-CC", "FROST_cc"), ("FROST", "FROST")]),
    ("Reference-\nmodifying *",
     [("Harmony *", "Harmony"), ("Scanorama *", "Scanorama"), ("Symphony *", "Symphony")]),
]
ROWS = [(lab, key) for _, items in S.BLOCKS for lab, key in items]
NR = len(ROWS)

BAND_A = [("Batch mixing", "#2a78d6",
           [("ilisi", "iLISI"), ("kbet", "kBET"), ("bras", "BRAS"),
            ("graph_conn", "graph conn"), ("batch_asw", "batch ASW"),
            ("batch_v2", "batch_v2")]),
          ("Total", "#52514e", [("total_v2", "total_v2")])]   # bỏ cột nội suy (giao thức xuôi, rò rỉ) — §6
BAND_B = [("Biological conservation", "#1baf7a",
           [("nmi", "NMI"), ("ari", "ARI"), ("asw", "ASW"), ("bio_v2", "bio_v2")]),
          ("Downstream tasks", "#eb6834",
           [("xfer_main", "xfer_main"), ("xfer_all", "xfer_all"),
            ("rare_purity", "rare_purity"), ("knn_xen", "knn_xen")])]
COMPOSITE = {"batch_v2", "bio_v2", "total_v2"}

TECHS, TISSUES = list(S.TECH), list(S.TISSUE)
FW = 6.69                       # = 17 cm: dựng ĐÚNG khổ cuối nên khi
                                # chèn vào Word không phải co, chữ giữ cỡ thật
X_GRP, X_BRK, X_LAB, LW = 0.19, 0.42, 1.46, 1.58
H_ROW, H_HEAD, H_GAP, H_TITLE, H_FOOT = 0.225, 0.42, 0.46, 0.12, 0.75
FS_LAB, FS_COL, FS_GRP, FS_TICK = 7.0, 6.6, 6.0, 5.6


def seed_sd(mkey, mcol):
    """SD qua 3 seed, gộp theo kiểu căn quân phương trên 12 cặp.

    Đây là NHIỄU SEED CỦA MỘT CẶP — mốc để đọc "chênh lệch nhỏ hơn mức này là
    nhiễu", đúng đại lượng mà quy tắc quyết định của dự án dùng (ngưỡng
    max(0,02; 2·SD)). KHÔNG phải sai số chuẩn của trung bình 12 cặp; đại lượng
    đó nhỏ hơn khoảng √12 lần và hầu như luôn vô hình ở thang này.
    """
    v = [D[mkey][p][mcol]["sd"] for p in S.PAIRS
         if D[mkey][p][mcol].get("sd") is not None]
    return float(np.sqrt(np.mean(np.square(v)))) if v else None


def agg(mkey, mcol, mode):
    keys = TECHS if mode == "tech" else TISSUES
    out = []
    for g in keys:
        vs = [D[mkey][p][mcol]["m"] for p in S.PAIRS
              if (p.split("/")[1] if mode == "tech" else p.split("/")[0]) == g
              and D[mkey][p][mcol]["m"] is not None]
        out.append((g, float(np.mean(vs)) if vs else None))
    return out


def draw(mode, title, subtitle, outfile):
    keys = TECHS if mode == "tech" else TISSUES
    lanes = np.linspace(0.80, 0.24, len(keys))
    band_h = NR * H_ROW
    FH = H_TITLE + 2 * (H_HEAD + band_h) + H_GAP + H_FOOT
    fig = plt.figure(figsize=(FW, FH), dpi=220)
    fx, fy = (lambda v: v / FW), (lambda v: v / FH)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    def panel(groups, top_in):
        cols = [(g, k, lb) for g, _, items in groups for k, lb in items]
        nc = len(cols)
        cw_in = (FW - LW - 0.20) / nc
        cw, x0 = fx(cw_in), fx(LW)
        y1 = 1 - fy(top_in + H_HEAD)
        rh = fy(H_ROW)
        y0 = y1 - NR * rh
        row_y = lambda i: y1 - (i + 1) * rh

        for i, (lab, key) in enumerate(ROWS):
            yy = row_y(i)
            if key == "FROST":
                bg.add_patch(plt.Rectangle((x0 - fx(0.04), yy),
                                           fx(nc * cw_in + 0.04), rh,
                                           fc=S.BAND, ec="none", zorder=0))
            bg.text(fx(X_LAB), yy + rh / 2, lab, ha="right", va="center",
                    fontsize=FS_LAB, color=S.INK if key == "FROST" else S.INK_2,
                    fontweight="bold" if key == "FROST" else "normal")
        i = 0
        for blk, items in S.BLOCKS:
            top, bot = row_y(i) + rh, row_y(i + len(items) - 1)
            bx = fx(X_BRK)
            bg.plot([bx, bx], [bot + fy(0.02), top - fy(0.02)],
                    color=S.BASELINE, lw=0.9)
            for yy in (bot + fy(0.02), top - fy(0.02)):
                bg.plot([bx, bx + fx(0.05)], [yy, yy], color=S.BASELINE, lw=0.9)
            bg.text(fx(X_GRP), (top + bot) / 2, blk, ha="center", va="center",
                    fontsize=FS_GRP, color=S.MUTED, rotation=90, linespacing=1.3)
            i += len(items)

        gb, acc = set(), 0
        for _, _, items in groups:
            acc += len(items); gb.add(acc)
        for k in range(nc + 1):
            xv = x0 + k * cw
            bg.plot([xv, xv], [y0, y1 + fy(0.03)],
                    color=S.BASELINE if k in gb or k == 0 else S.GRID,
                    lw=0.8 if k in gb or k == 0 else 0.55, zorder=0)

        span = {}
        for j, (grp, mkey, mlab) in enumerate(cols):
            xs = x0 + j * cw
            span.setdefault(grp, [xs, 0])[1] = xs + cw
            if mkey in COMPOSITE:
                bg.add_patch(plt.Rectangle((xs, y0), cw, y1 - y0 + fy(0.03),
                                           fc="#f4f3ef", ec="none", zorder=0))
            allv = [v for _, k in ROWS for _, v in agg(mkey, k, mode)
                    if v is not None]
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.11 or 0.01
            lo, hi = lo - pad, hi + pad

            for i, (_, key) in enumerate(ROWS):
                ax = fig.add_axes([xs + fx(0.015), row_y(i),
                                   cw - fx(0.030), rh])
                ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
                ax.set_facecolor("none"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if i == NR - 1:
                    tr = ax.get_xaxis_transform()
                    ax.plot([lo, hi], [-0.10, -0.10], color=S.BASELINE, lw=0.6,
                            clip_on=False, transform=tr)
                    for xv, al in ((lo + pad, "left"), (hi - pad, "right")):
                        ax.plot([xv, xv], [-0.10, -0.26], color=S.BASELINE,
                                lw=0.6, clip_on=False, transform=tr)
                        ax.text(xv, -0.36, f"{xv:.2f}", ha=al, va="top",
                                fontsize=FS_TICK, color=S.MUTED,
                                clip_on=False, transform=tr)
                pts = agg(mkey, key, mode)
                good = [v for _, v in pts if v is not None]
                if not good:
                    ax.text((lo + hi) / 2, 0.52, "n/a", ha="center",
                            va="center", fontsize=5.0, color=S.BASELINE)
                    continue
                for (g, v), ly in zip(pts, lanes):
                    if v is None:
                        continue
                    ax.scatter(v, ly, s=9,
                               c=S.TECH_COLOR[g] if mode == "tech" else S.TISSUE[g],
                               edgecolors="#1a1a19", linewidths=0.25, zorder=3)
                ax.plot([np.mean(good)] * 2, [0.10, 0.94], color=S.INK_2,
                        lw=1.0, zorder=4, solid_capstyle="round")

            bg.text(xs + cw / 2, y1 + fy(0.045), mlab, ha="center", va="bottom",
                    fontsize=FS_COL, color=S.INK)

        for grp, color, _ in groups:
            a, b = span[grp]
            yy = 1 - fy(top_in + 0.20)
            bg.add_patch(plt.Rectangle((a + fx(0.012), yy),
                                       b - a - fx(0.024), fy(0.17),
                                       fc=color, ec="none", alpha=0.92))
            bg.text((a + b) / 2, yy + fy(0.085), grp, ha="center", va="center",
                    fontsize=FS_GRP, color="white", fontweight="bold")

    panel(BAND_A, H_TITLE)
    panel(BAND_B, H_TITLE + H_HEAD + band_h + H_GAP)

    if mode == "tech":
        h = [Line2D([], [], marker="o", ls="", ms=4.6, mfc=S.TECH_COLOR[k],
                    mec="#1a1a19", mew=0.25, label=S.TECH_LABEL[k]) for k in TECHS]
        ti = "Each dot = mean over 3 tissues for one source technology"
    else:
        h = [Line2D([], [], marker="o", ls="", ms=4.6, mfc=S.TISSUE[t],
                    mec="#1a1a19", mew=0.25, label=S.TISSUE_LABEL[t])
             for t in TISSUES]
        ti = "Each dot = mean over 4 source technologies for one tissue"
    h += [Line2D([], [], color=S.INK_2, lw=1.0, label="mean of 12 pairs")]
    L = fig.legend(handles=h, loc="lower left",
                   bbox_to_anchor=(fx(X_BRK), fy(0.06)),
                   ncol=len(h), frameon=False, fontsize=FS_COL, title=ti,
                   title_fontsize=FS_COL, handletextpad=0.35, columnspacing=0.9)
    L._legend_box.align = "left"
    L.get_title().set_color(S.INK_2)

    fig.text(fx(X_BRK), 1 - fy(0.20), title, fontsize=9.6, color=S.INK,
             va="top", fontweight="bold")
    fig.text(fx(X_BRK), 1 - fy(0.42), subtitle, fontsize=6.8, color=S.INK_2,
             va="top")
    # chú thích chân hình chuyển vào caption LaTeX

    fig.savefig(outfile, dpi=220, bbox_inches="tight", facecolor=S.SURFACE)
    plt.close(fig)
    print(f"  {os.path.basename(outfile)}  {FW:.1f}×{FH:.1f} in  "
          f"{os.path.getsize(outfile)/1e6:.2f} MB  | ở khổ 17 cm: nhãn cột "
          f"{FS_COL * 6.69 / FW:.1f} pt, nhãn method {FS_LAB * 6.69 / FW:.1f} pt")


print("đang dựng:")
draw("tech", "", "", f"{HERE}/figs/fig_grid_tech.png")
draw("tissue", "", "", f"{HERE}/figs/fig_grid_tissue.png")
