"""Hệ màu và thông số mark dùng chung cho mọi hình của báo cáo FROST.

Bảng màu phân loại lấy 3 slot đầu của bảng chuẩn (đã chạy validate_palette.js,
chế độ light, --pairs all): PASS cả dải sáng, sàn chroma, tách CVD (ΔE 9.2) và
sàn thị giác thường (ΔE 24.0). Cảnh báo duy nhất: aqua #1baf7a tương phản 2.74
(<3:1) trên nền sáng -> áp quy tắc bù: chú giải có chữ + viền tối quanh chấm +
bảng phụ lục đầy đủ.
"""
import matplotlib as mpl

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
BAND      = "#f1f0ec"     # dải nền cho hàng FROST
BAR       = "#b9b8b1"     # thân bar: ink trung tính, KHÔNG mang nghĩa phân loại
BAR_HI    = "#6f6e68"     # bar của 3 dòng FROST

# tổ chức mô -> màu (3 slot đã kiểm định)
TISSUE = {"coad": "#2a78d6", "ov": "#eb6834", "hcc": "#1baf7a"}
TISSUE_LABEL = {"coad": "COAD", "ov": "OV", "hcc": "HCC"}

# công nghệ nguồn -> ký hiệu (kênh mã hoá thứ hai)
TECH = {"cosmx6k": "o", "stereoseq": "s", "visiumhd_ff": "^", "visiumhd_ffpe": "D"}
TECH_LABEL = {"cosmx6k": "CosMx 6K", "stereoseq": "Stereo-seq",
              "visiumhd_ff": "Visium HD FF", "visiumhd_ffpe": "Visium HD FFPE"}
TECH_COLOR = {"cosmx6k": "#4a3aa7", "stereoseq": "#eda100",
              "visiumhd_ff": "#e87ba4", "visiumhd_ffpe": "#008300"}

PAIRS = ["coad/cosmx6k", "coad/stereoseq", "coad/visiumhd_ff", "coad/visiumhd_ffpe",
         "ov/cosmx6k", "ov/stereoseq", "ov/visiumhd_ff", "ov/visiumhd_ffpe",
         "hcc/cosmx6k", "hcc/stereoseq", "hcc/visiumhd_ff", "hcc/visiumhd_ffpe"]

# cặp nhãn nguồn không được biểu hiện ủng hộ (methods.md §4): vẽ viền rỗng
FLAGGED = "hcc/stereoseq"

# thứ tự hàng + khối
BLOCKS = [
    ("Giữ th.chiếu\nkhông tiên nghiệm",
     [("raw", "raw"), ("center", "center"), ("CORAL", "CORAL"),
      ("MNN-ref", "MNN-ref")]),
    ("Giữ th.chiếu\nxác suất loại TB",
     [("ridge", "ridge"), ("USHER-ST", "FROST_base"),
      ("USHER-CC", "FROST_cc"), ("FROST", "FROST")]),
    ("Đổi th.chiếu *",
     [("Harmony *", "Harmony"), ("Scanorama *", "Scanorama"),
      ("Symphony_h *", "Symphony")]),
]
FROST_KEYS = {"FROST_base", "FROST_cc", "FROST"}

# cột meta: (nhãn, {method: ký hiệu})
META = [
    ("Giữ\ntham chiếu", {"raw": 1, "center": 1, "CORAL": 1, "MNN-ref": 1, "ridge": 1,
                         "FROST_base": 1, "FROST_cc": 1, "FROST": 1,
                         "Harmony": 0, "Scanorama": 0, "Symphony": 0}),
    ("Dùng\nnhãn",      {"raw": 0, "center": 0, "CORAL": 0, "MNN-ref": 0, "ridge": 1,
                         "FROST_base": 1, "FROST_cc": 1, "FROST": 1,
                         "Harmony": 0, "Scanorama": 0, "Symphony": 0}),
    ("Nghiệm\nđóng",    {"raw": 1, "center": 1, "CORAL": 1, "MNN-ref": 0, "ridge": 1,
                         "FROST_base": 0, "FROST_cc": 0, "FROST": 0,
                         "Harmony": 0, "Scanorama": 0, "Symphony": 0}),
]

# nhóm chỉ số: (tên nhóm, màu tiêu đề, [(khoá, nhãn hiển thị)])
GROUPS = [
    ("Trộn lô", "#2a78d6", [("ilisi", "iLISI"), ("kbet", "kBET"), ("bras", "BRAS"),
                            ("graph_conn", "graph conn"), ("batch_asw", "batch ASW"),
                            ("batch_v2", "Tổng nhóm")]),
    ("Bảo toàn sinh học", "#1baf7a", [("nmi", "NMI"), ("ari", "ARI"), ("asw", "ASW"),
                                      ("bio_v2", "Tổng nhóm")]),
    ("Tác vụ hạ nguồn  (FROST bổ sung)", "#eb6834",
     [("xfer_main", "xfer_main"), ("xfer_all", "xfer_all"),
      ("rare_purity", "rare_purity"), ("knn_xen", "knn_xen")]),
    ("Nội suy", "#4a3aa7", [("imputation", "Spearman")]),
    ("Tổng hợp", "#52514e", [("total_v2", "total_v2")]),
]


def rc():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "text.color": INK,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    })
