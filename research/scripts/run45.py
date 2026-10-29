"""45 — Chỉ số ST không cần chú giải (Moran's I trên nhãn, PAS) + chẩn đoán Moran's I của PHẦN DƯ.

KHÔNG FGW, KHÔNG huấn luyện: chỉ nạp map đã lưu rồi tính. CPU là đủ.

Phần D — chẩn đoán, quyết định FROST 1.0.2 sống hay chết:
    Bản LIÊN TỤC của run20 Phần D. Vòng 20 đo η²(niche) của phần dư (cấu trúc theo LỚP) và được
    0.001-0.005. Nhưng phần dư có thể có tự tương quan KHÔNG GIAN mà không có cấu trúc theo lớp —
    hai thứ khác nhau. Ở đây: fit Θ_glob trên T1 (đúng như run20:237), lấy phần dư trên T2, rồi đo
    Moran's I của từng chiều phần dư theo toạ độ vật lý.
      Moran's I ≈ 0  -> số hạng Laplacian không có gì để cắn; đóng hướng 1.0.2 bằng số liệu.
      Moran's I > 0  -> còn tín hiệu không gian chưa được sửa; số hạng mục tiêu có cơ sở.
    Mốc so: hoán vị toạ độ (giữ nguyên phần dư) -> Moran's I của nhiễu thuần.

Phần M — hai chỉ số ST mới, cho MỌI phương pháp trong bảng chính:
    Nhãn chuyển sang = bỏ phiếu 15-NN nhãn tham chiếu trong không gian đã căn chỉnh (đúng
    run22._knn_bacc), rồi so TỔ CHỨC KHÔNG GIAN của nhãn chuyển với nhãn THẬT:
      - Moran's I trên nhãn: từng loại tế bào một, chỉ báo nhị phân, trọng số từ toạ độ.
      - PAS: tỉ lệ tế bào có nhãn khác đa số láng giềng KHÔNG GIAN của nó.
    CẢ HAI báo cáo dưới dạng |Δ so với nhãn thật|, KHÔNG phải giá trị thô: gán tất cả cùng một nhãn
    cho PAS = 0 và Moran's I suy biến — thấp-hơn-là-tốt-hơn sẽ thưởng cho nhãn vô dụng.

HAI BẪY ĐÃ XỬ LÝ:
  1. Q nằm trên tile giữ lại của bàn cờ 500µm -> kNN không gian nối được hai tile cách nhau qua khe.
     Cắt mọi cạnh dài hơn R_MAX_UM (mặc định 100µm, bằng bán kính niche lớn nhất vòng 19). Log tỉ lệ
     cạnh bị cắt; nếu > 30% thì cảnh báo vì đồ thị đã rời rạc quá.
  2. Moran's I chỉ có nghĩa với nhãn CÓ tổ chức không gian. Lọc theo I của nhãn THẬT >= MORAN_MIN
     (mặc định 0.10), đúng cách bài sketching (NAR 2026) làm. Số nhãn còn lại được ghi vào summary.

ENV: TISSUE, RUN_PREFIX, COMBOS_ALL, OUT_DIR_45, ONLY_PART (D,M,A,B,T,R,K,S,V,W,X,Y), R_MAX_UM, MORAN_MIN, K_SPATIAL,
     ALPHAS, ALPHA_TAG, PART_B_ARMS,
     SPC_TAG, SPC_TS, SPC_FULL, PATCH_N, PATCH_MIN, N_PATCH, PK, GW_N,
     PATCH_SEEDS, EXTRA_MAPS, IMP_METHODS (dùng lại danh sách của run30).
"""
import os, sys, json, glob, time, gc, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score

import run20 as R20
import run22 as R22
import run30 as R30
R17, R18, R19 = R20.R17, R20.R18, R20.R19
R23, R23C, R26G = R30.R23, R30.R23C, R30.R26G
log, lite_obs, _unit, mode_rows = R20.log, R20.lite_obs, R20._unit, R20.mode_rows
ridge_fit = R20.ridge_fit

TISSUE = os.environ.get("TISSUE", "coad").lower()
P = os.environ.get("RUN_PREFIX", {"coad": "23", "ov": "26", "hcc": "27"}[TISSUE])
OUT = os.environ.get("OUT_DIR_45", f"/kaggle/working/out{P}s")
COMBOS = [c for c in os.environ.get("COMBOS_ALL", "").split(",") if c.strip()]
PARTS = {p.strip().upper() for p in os.environ.get("ONLY_PART", "D,M,A").split(",") if p.strip()}
SEEDS = R30.SEEDS            # dùng CHUNG với run30 để phần D và danh sách map của phần M không lệch seed
R_MAX_UM = float(os.environ.get("R_MAX_UM", 100.0))
MORAN_MIN = float(os.environ.get("MORAN_MIN", 0.10))
K_SPATIAL = int(os.environ.get("K_SPATIAL", 15))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.05,0.1,0.2,0.3,0.4,0.5").split(",") if a.strip()]
VER = "45.24-tieuchi"   # TEM PHIÊN BẢN: in ra dòng log ĐẦU TIÊN. Hai lần notebook cũ trên Kaggle đã
                    # ngốn 4 giờ CPU mà không ai biết. Thấy tem khác -> notebook trên Kaggle là bản cũ.
ALPHA_TAG = os.environ.get("ALPHA_TAG", "cmix")    # map nền của phần A: cmix = FROST CHƯA CORAL
WN_TOP = [int(x) for x in os.environ.get("WN_TOP", "10,25,50").split(",") if x.strip()]
AGGS = [a.strip() for a in os.environ.get("AGGS", "tb,trungvi").split(",") if a.strip()]
SK_KS = [int(k) for k in os.environ.get("SK_KS", "5,10,15,30").split(",") if k.strip()]
KC = [float(c) for c in os.environ.get("KC", "0.25,0.5,1,2,4").split(",") if c.strip()]  # sigma^2 = c * trung vị s^2
ROT_TAG = os.environ.get("ROT_TAG", "cmix_levns")       # map NỀN của phần R (chưa CORAL)
ROT_TS = [float(t) for t in os.environ.get("ROT_TS", "0,0.25,0.5,0.75,1").split(",") if t.strip()]
ROT_FULL = [float(t) for t in os.environ.get("ROT_FULL", "0,0.5,1").split(",") if t.strip()]
SPC_TAG = os.environ.get("SPC_TAG", "cmix")             # map NỀN của phần X (CHƯA CORAL) -> t=0 = FROST
SPC_TS = [float(t) for t in os.environ.get("SPC_TS", "0,0.25,0.5,0.75,1").split(",") if t.strip()]
SPC_FULL = [float(t) for t in os.environ.get("SPC_FULL", "0,0.5,1").split(",") if t.strip()]
PATCH_N = int(os.environ.get("PATCH_N", 1500))       # số tế bào mục tiêu mỗi bản vá (~cỡ batch USHER)
PATCH_MIN = int(os.environ.get("PATCH_MIN", 300))    # ô dưới ngưỡng này bị bỏ
N_PATCH = int(os.environ.get("N_PATCH", 8))          # số bản vá lấy mỗi bên
PK = int(os.environ.get("PK", 15))                   # k cho tính khiết lân cận
GW_N = int(os.environ.get("GW_N", 800))              # cỡ bản vá khi giải GW thật
N_T_MAX = int(os.environ.get("N_T_MAX", 20000))   # cỡ tập giữ lại của phần T
PART_B_ARMS = [a.strip() for a in os.environ.get(
    "PART_B_ARMS", "pre@0,pre@0.2,pre@0.3,pre@0.4,pre@0.5").split(",") if a.strip()]
FLOOR = R30.FLOOR
T0 = time.time()
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", 11 * 3600))


class OutOfTime(Exception): pass


def step(msg):
    log(f"    [{time.time() - T0:6.0f}s] {msg}")
    if time.time() - T0 > TIME_BUDGET_S: raise OutOfTime(msg)


# =============================================================================== đồ thị không gian
def spatial_graph(xy, k=None, r_max=None):
    """kNN không gian ĐÃ CẮT cạnh dài. Trả (W hàng-chuẩn-hoá, nn_idx có mặt nạ, tỉ lệ cạnh bị cắt).

    Cắt cạnh là BẮT BUỘC: Q lấy từ tile giữ lại của bàn cờ nên hai tế bào ở hai tile khác nhau có thể
    thành "láng giềng" dù cách nhau qua một khe 500µm. Không cắt thì mọi chỉ số không gian đều sai.
    """
    k = k or K_SPATIAL
    r_max = R_MAX_UM if r_max is None else r_max
    n = len(xy)
    k_eff = min(k, n - 1)
    d, idx = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=4).fit(xy).kneighbors(xy)
    d, idx = d[:, 1:], idx[:, 1:]           # bỏ chính nó
    ok = d <= r_max
    cut = float(1.0 - ok.mean())
    rows = np.repeat(np.arange(n), k_eff)[ok.ravel()]
    cols = idx.ravel()[ok.ravel()]
    W = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)                       # đối xứng hoá
    deg = np.asarray(W.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    Wn = sp.diags(inv.astype(np.float32)) @ W   # hàng chuẩn hoá; hàng cô lập = toàn 0
    return Wn.tocsr(), idx, ok, cut, deg


def moran_i(Wn, y):
    """Moran's I với W ĐÃ chuẩn hoá hàng. y: (n,) hoặc (n, p). Trả vô hướng hoặc (p,)."""
    y = np.asarray(y, np.float64)
    one_d = y.ndim == 1
    Y = y[:, None] if one_d else y
    Z = Y - Y.mean(0, keepdims=True)
    num = np.einsum("ij,ij->j", Z, Wn @ Z)
    den = np.einsum("ij,ij->j", Z, Z)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(out[0]) if one_d else out


def pas(nn_idx, ok, lab):
    """PAS — tỉ lệ tế bào có nhãn KHÁC đa số láng giềng không gian của nó.

    Chỉ tính trên tế bào còn ít nhất một láng giềng sau khi cắt cạnh; tế bào cô lập bị loại và số
    lượng được trả về để ghi log.
    """
    lab = np.asarray(lab)
    codes, _ = pd.factorize(lab)             # nhãn THIẾU -> -1
    nb = np.where(ok, codes[nn_idx], -1)
    # HAI SỬA (phát hiện khi dựng phần S, chạy trên TOÀN BỘ tế bào thay vì chỉ tập high_quality):
    #  1. Tế bào KHÔNG có nhãn phải bị loại khỏi mẫu số, trước đây codes = -1 luôn khác đa số nên
    #     bị tính là "bất thường" -> pas bị thổi lên.
    #  2. Tế bào mà MỌI láng giềng đều không có nhãn làm np.bincount([]) .argmax() ném ValueError.
    #     Phần M không dính vì Q lấy từ ds["H"] (high_quality); phần S chạy trên tất cả thì dính.
    keep = ok.any(1) & (codes >= 0)
    maj = np.full(len(lab), -1)
    n_trong = 0
    for i in np.where(keep)[0]:
        v = nb[i][nb[i] >= 0]
        if len(v) == 0: keep[i] = False; n_trong += 1; continue
        maj[i] = np.bincount(v).argmax()
    n_bo = int(len(lab) - keep.sum())
    if keep.sum() == 0: return None, n_bo
    return float((maj[keep] != codes[keep]).mean()), n_bo


# =============================================================================== Phần M
def transferred_labels(cx, Fmap, q):
    """Nhãn chuyển sang: bỏ phiếu 15-NN nhãn tham chiếu — ĐÚNG run22._knn_bacc, chỉ trả nhãn thay vì bacc."""
    ref, Fx = cx["pctx"]["ref"], cx["Fx"]
    mu = Fx.mean(0)
    nn = NearestNeighbors(n_neighbors=15, n_jobs=4).fit(_unit(Fx[ref] - mu)).kneighbors(
        _unit(np.asarray(Fmap, np.float32)[q] - mu), return_distance=False)
    return mode_rows(np.asarray(cx["dx"]["lab"])[ref][nn])


def st_metrics(Wn, nn_idx, ok, lab_pred, lab_true, labels_keep):
    """Moran's I trên nhãn + PAS, báo cáo dưới dạng ĐỘ LỆCH so với nhãn thật."""
    oh_p = np.stack([(lab_pred == c).astype(np.float64) for c in labels_keep], 1)
    oh_t = np.stack([(lab_true == c).astype(np.float64) for c in labels_keep], 1)
    I_p, I_t = moran_i(Wn, oh_p), moran_i(Wn, oh_t)
    pas_p, n_iso = pas(nn_idx, ok, lab_pred)
    pas_t, _ = pas(nn_idx, ok, lab_true)
    return {"moran_abs_dev": round(float(np.abs(I_p - I_t).mean()), 4),
            "moran_pred_mean": round(float(I_p.mean()), 4),
            "moran_true_mean": round(float(I_t.mean()), 4),
            "pas_pred": None if pas_p is None else round(pas_p, 4),
            "pas_true": None if pas_t is None else round(pas_t, 4),
            "pas_abs_dev": None if (pas_p is None or pas_t is None) else round(abs(pas_p - pas_t), 4),
            "n_label": int(len(labels_keep)), "n_isolated": n_iso}


def part_m(combo, cx, S, save, W):
    Sc = S.setdefault("M", {}).setdefault(combo, {})
    pair = cx["pair"]
    q = cx["Q"]
    xy = lite_obs(pair)[["x", "y"]].to_numpy()[q]
    Wn, nn_idx, ok, cut, deg = spatial_graph(xy)
    lab_true = np.asarray(cx["ds"]["lab"])[q]

    # lọc nhãn theo tự tương quan NỀN (trên nhãn thật) — bài sketching NAR 2026 làm đúng bước này
    cnt = pd.Series(lab_true).value_counts()
    cand = [c for c in cnt.index if cnt[c] >= 20]
    I_true_all = moran_i(Wn, np.stack([(lab_true == c).astype(np.float64) for c in cand], 1)) if cand else np.array([])
    labels_keep = [c for c, i_ in zip(cand, I_true_all) if i_ >= MORAN_MIN]
    S.setdefault("M_meta", {})[combo] = {"canh_bi_cat": round(cut, 4), "bac_tb": round(float(deg.mean()), 2),
                                         "n_nhan_xet": len(cand), "n_nhan_giu": len(labels_keep),
                                         "nhan_giu": list(map(str, labels_keep)),
                                         "r_max_um": R_MAX_UM, "moran_min": MORAN_MIN}
    save()
    log(f"[{combo}] đồ thị: cắt {cut:.1%} cạnh, bậc TB {deg.mean():.1f} | nhãn giữ {len(labels_keep)}/{len(cand)}")
    if cut > 0.30:
        W.append(f"{combo}: cắt {cut:.0%} cạnh không gian (> 30%) — đồ thị rời rạc, đọc chỉ số ST thận trọng")
    if not labels_keep:
        W.append(f"{combo}: KHÔNG nhãn nào đạt Moran's I >= {MORAN_MIN} trên nhãn thật -> bỏ phần M")
        return

    G = R30.build_maps(combo, cx, W)
    thieu = []
    for name in R30.wanted(combo):
        if name in Sc: continue
        fn = G.get(name)
        if fn is None: thieu.append(name + "(không có trong build_maps)"); continue
        step(f"[{combo}] M: {name}")
        try:
            Fmap = fn()
        except Exception as e:
            W.append(f"{combo}/{name}: dựng map lỗi {e!r}"); continue
        if Fmap is None:
            thieu.append(name); continue        # KHÔNG im lặng: map không có trên đĩa -> ghi ra
        pred = transferred_labels(cx, Fmap, q)
        Sc[name] = st_metrics(Wn, nn_idx, ok, pred, lab_true, labels_keep)
        save()
        del Fmap; gc.collect()
    if thieu:
        S.setdefault("M_thieu", {})[combo] = thieu
        W.append(f"{combo}: KHÔNG tìm thấy map cho {len(thieu)} mục -> {', '.join(thieu)}")
        log(f"[{combo}] thiếu map: {', '.join(thieu)}")
    save()


# =============================================================================== Phần D
def part_d(combo, cx, S, save, W):
    """Moran's I của phần dư — bản liên tục của run20 Phần D."""
    Sc = S.setdefault("D", {}).setdefault(combo, {})
    pair, Fs = cx["pair"], cx["Fs"]
    ds = cx["ds"]
    T1, T2 = cx["T1"], cx["T2"]
    xy_all = lite_obs(pair)[["x", "y"]].to_numpy()
    for sd in SEEDS:
        key = f"s{sd}"
        if key in Sc: continue
        _, ybar, has = R22.get_map(combo, f"f{FLOOR}", sd, None, R30.ALIAS.get(FLOOR))
        if ybar is None:
            W.append(f"{combo} s{sd}: không có ȳ (ybar npz) -> bỏ phần D"); continue
        step(f"[{combo}] D: seed {sd}")
        Xhas = np.asarray(ds["T"])[has]
        m1h, m2h = np.isin(Xhas, T1), np.isin(Xhas, T2)
        if m1h.sum() < 50 or m2h.sum() < 50:
            W.append(f"{combo} s{sd}: T1/T2 quá nhỏ ({m1h.sum()}/{m2h.sum()}) -> bỏ"); continue
        Xt = np.hstack([Fs[Xhas], np.ones((len(Xhas), 1), np.float32)]).astype(np.float64)
        Yt = np.asarray(ybar)[has].astype(np.float64)
        Theta = ridge_fit(Xt[m1h], Yt[m1h], 1e-3)
        e2 = Yt[m2h] - Xt[m2h] @ Theta                      # phần dư trên nửa GIỮ LẠI
        xy2 = xy_all[Xhas[m2h]]
        Wn, _, _, cut, deg = spatial_graph(xy2)
        I = moran_i(Wn, e2)
        rng = np.random.RandomState(0)
        I_perm = np.array([moran_i(Wn, e2[rng.permutation(len(e2))]) for _ in range(5)])
        Sc[key] = {"moran_tb": round(float(I.mean()), 5), "moran_p90": round(float(np.percentile(I, 90)), 5),
                   "moran_max": round(float(I.max()), 5),
                   "moran_hoanvi_tb": round(float(I_perm.mean()), 5),
                   "moran_hoanvi_sd": round(float(I_perm.std()), 5),
                   "n_T2": int(m2h.sum()), "d": int(e2.shape[1]),
                   "canh_bi_cat": round(cut, 4), "bac_tb": round(float(deg.mean()), 2)}
        save()
        log(f"[{combo}] s{sd} (D): {Sc[key]}")


# =============================================================================== Phần A
def part_a(combo, cx, S, save, W):
    """Sàng α — làm trơn không gian HẬU KỲ trên map FROST. CHẶN TRÊN của 1.0.2, không phải 1.0.2.

    Nghiệm của  min_Z ‖Z−Z₀‖² + α·tr(ZᵀLZ)  là  (I+αL)⁻¹Z₀ ≈ (1−α)Z₀ + α·W_sp·Z₀ — đúng toán tử
    gần kề của số hạng phạt Laplacian định thêm vào M-step. KHÁC: ở đây làm trơn theo TỪNG tế bào,
    còn M-step phải thể hiện qua MỘT ma trận W tuyến tính duy nhất, tự do ít hơn hẳn. Nên:
      đường α hỏng -> 1.0.2 CHẮC CHẮN hỏng, dừng.  đường α đẹp -> 1.0.2 CÓ THỂ được, mới đáng viết.

    Ba nhánh (α=0 của `pre` chính là FROST 1.0; α=0 của `nocoral` chính là u_cmix):
      nocoral smooth(F) · pre coral(smooth(F)) KHỚP Laplacian ở M-step · post smooth(coral(F))

    d_cov = ‖Σ(Z[T])−Σ_x‖_F/‖Σ_x‖_F. CẢNH BÁO ĐỌC: ở nhánh `pre` nó ≈ 0 với MỌI α một cách tầm
    thường, vì CORAL chạy sau cùng nên ép lại khớp mô-men. Đừng đọc đó là "an toàn" — nó chỉ có
    nghĩa cho `nocoral` và `post`. Với `pre`, rủi ro trộn lô phải chấm bằng R23.evaluate (chưa làm
    ở đây): Σ khớp KHÔNG bảo đảm iLISI khớp, vì iLISI phụ thuộc trộn lô CỤC BỘ.
    """
    Sc = S.setdefault("A", {}).setdefault(combo, {})
    Fx, T, q = cx["Fx"], cx["ds"]["T"], cx["Q"]
    sd = SEEDS[0]
    F0 = R22.get_map(combo, ALPHA_TAG, sd, None)[0]
    if F0 is None:                                   # KHÔNG im lặng
        W.append(f"{combo}: không có map {ALPHA_TAG} s{sd} -> bỏ phần A"); return
    F0 = np.asarray(F0, np.float32)

    xy = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    Wn_all, _, _, cut_all, _ = spatial_graph(xy)          # đồ thị TOÀN BỘ tế bào -> để làm trơn
    Wn_q, nn_q, ok_q, cut_q, _ = spatial_graph(xy[q])     # đồ thị trên Q -> để chấm (y hệt part_m)
    iso = np.asarray(Wn_all.sum(1)).ravel() <= 0          # hàng không láng giềng: Wn@Z = 0 -> phải giữ nguyên

    lab_true = np.asarray(cx["ds"]["lab"])[q]
    cnt = pd.Series(lab_true).value_counts()
    cand = [c for c in cnt.index if cnt[c] >= 20]
    I_t = moran_i(Wn_q, np.stack([(lab_true == c).astype(np.float64) for c in cand], 1)) if cand else np.array([])
    keep = [c for c, i_ in zip(cand, I_t) if i_ >= MORAN_MIN]
    if not keep:
        W.append(f"{combo}: không nhãn nào đạt Moran >= {MORAN_MIN} -> bỏ phần A"); return
    S.setdefault("A_meta", {})[combo] = {"n_nhan_giu": len(keep), "nhan_giu": list(map(str, keep)),
                                         "cat_toan_bo": round(cut_all, 4), "cat_Q": round(cut_q, 4),
                                         "n_co_lap": int(iso.sum()), "map_nen": f"{ALPHA_TAG}@s{sd}"}
    save()
    Cx = np.cov(Fx.astype(np.float64), rowvar=False); nCx = float(np.linalg.norm(Cx))

    def smooth(Z, a):
        Zs = (1.0 - a) * Z + a * np.asarray(Wn_all @ Z)
        Zs[iso] = Z[iso]                                  # giữ nguyên tế bào cô lập
        return Zs.astype(np.float32)

    def score(Z):
        pred = transferred_labels(cx, Z, q)
        r = st_metrics(Wn_q, nn_q, ok_q, pred, lab_true, keep)
        r["bacc_xfer"] = round(float(balanced_accuracy_score(lab_true, pred)), 4)
        r["d_cov"] = round(float(np.linalg.norm(np.cov(Z[T].astype(np.float64), rowvar=False) - Cx) / nCx), 4)
        return r

    jobs = [("nocoral", 0.0, lambda: F0), ("pre", 0.0, lambda: R23C.coral(F0, Fx, T))]
    for a in ALPHAS:
        jobs += [("nocoral", a, (lambda a=a: smooth(F0, a))),
                 ("pre",     a, (lambda a=a: R23C.coral(smooth(F0, a), Fx, T))),
                 ("post",    a, (lambda a=a: smooth(R23C.coral(F0, Fx, T), a)))]
    for arm, a, mk in jobs:
        k = f"{arm}@{a:g}"
        if k in Sc: continue
        step(f"[{combo}] A: {k}")
        try:
            Sc[k] = score(np.asarray(mk(), np.float32))
        except Exception as e:
            W.append(f"{combo}/A/{k}: lỗi {e!r}"); log(f"[{combo}] A {k} LỖI {e!r}"); continue
        v = Sc[k]
        log(f"[{combo}] A {k}: pas_dev={v['pas_abs_dev']} moran_dev={v['moran_abs_dev']} "
            f"bacc={v['bacc_xfer']} d_cov={v['d_cov']}")
        save(); gc.collect()
    del Wn_all, Wn_q; gc.collect()


# =============================================================================== Phần B
def part_b(combo, cx, S, save, W):
    """Chấm ĐẦY ĐỦ bằng R23.evaluate ở vài điểm α — cổng CUỐI trước khi viết M-step của 1.0.2.

    Phần A chỉ có `bacc_xfer` (proxy rẻ) và `d_cov`. Thứ CHƯA biết là `batch_v2`/iLISI: `d_cov ≈ 0`
    là điều kiện CẦN chứ không đủ, vì iLISI phụ thuộc trộn lô CỤC BỘ, không chỉ mô-men toàn cục.

    `pre@0` ở đây CHÍNH LÀ FROST 1.0 và là mốc so DUY NHẤT đúng — đừng so với `u_cmix_cc` ở bảng
    của vòng 23e: cùng một map nhưng khác đường tính (seed, Fs/Fx nạp lại), lệch vài phần nghìn.
    """
    if R20.KM16 is None:   # nổ ở giây 0 thay vì sau ~6 phút eval_all16d rồi mới chết trong etrack
        W.append("phần B: R20.KM16 chưa đặt -> etrack sẽ nổ; BỎ phần B")
        log("phần B: THIẾU R20.KM16 -> bỏ"); return
    Sc = S.setdefault("B", {}).setdefault(combo, {})
    Fs, Fx, T, q = cx["Fs"], cx["Fx"], cx["ds"]["T"], cx["Q"]
    xy = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    Wn_all, _, _, _, _ = spatial_graph(xy)
    iso = np.asarray(Wn_all.sum(1)).ravel() <= 0

    # Chỉ số ST tính NGAY TẠI ĐÂY, trên CÙNG map mà R23.evaluate chấm. Trước đây chúng chỉ có ở
    # phần A, đi một đường tính khác, nên không ghép được với total_v2 để kiểm tiêu chí
    # "cải thiện ST mà giữ total". Đó chính là lý do 1.0.2 bị kết luận nhầm là chết.
    Wn_q, nn_q, ok_q, _, _ = spatial_graph(xy[q])
    lab_true = np.asarray(cx["ds"]["lab"])[q]
    cnt = pd.Series(lab_true).value_counts()
    cand = [c for c in cnt.index if cnt[c] >= 20]
    I_t = moran_i(Wn_q, np.stack([(lab_true == c).astype(np.float64) for c in cand], 1)) if cand else np.array([])
    keep = [c for c, i_ in zip(cand, I_t) if i_ >= MORAN_MIN]
    if not keep:
        W.append(f"{combo}: không nhãn nào đạt Moran >= {MORAN_MIN} -> phần B không có chỉ số ST")

    def smooth(Z, a):
        if not a: return Z
        Zs = (1.0 - a) * Z + a * np.asarray(Wn_all @ Z)
        Zs[iso] = Z[iso]                             # giữ nguyên tế bào cô lập
        return Zs.astype(np.float32)

    # Di cư khoá PHẲNG cũ (chỉ 1 seed, trước khi phần B biết tới seed) sang khoá có seed, để
    # không phải chạy lại ~6 phút/nhánh cho những gì đã có trong out<P>s.
    for k0 in [k for k in Sc if "@" in k and "/" not in k]:
        Sc[f"s{SEEDS[0]}/{k0}"] = Sc.pop(k0)

    for sd in SEEDS:
        F0 = R22.get_map(combo, ALPHA_TAG, sd, None)[0]
        if F0 is None:                               # thiếu MỘT seed thì bỏ seed đó, không bỏ cả cặp
            W.append(f"{combo}: không có map {ALPHA_TAG} s{sd} -> bỏ seed này ở phần B"); continue
        F0 = np.asarray(F0, np.float32)
        for k in PART_B_ARMS:
            ten = f"s{sd}/{k}"
            cu = Sc.get(ten, {})
            if "pas_abs_dev" in cu: continue
            if "@" not in k:
                W.append(f"phần B: tên nhánh sai {k!r} (cần dạng pre@0.2)"); continue
            arm, a_ = k.split("@", 1)
            try: a = float(a_)
            except ValueError: W.append(f"phần B: α sai ở {k!r}"); continue
            if arm not in ("nocoral", "pre", "post"):
                W.append(f"phần B: nhánh lạ {arm!r} ở {k!r}"); continue
            step(f"[{combo}] B: {ten}")
            Fm = np.asarray(smooth(F0, a) if arm == "nocoral" else
                            R23C.coral(smooth(F0, a), Fx, T) if arm == "pre" else
                            smooth(R23C.coral(F0, Fx, T), a), np.float32)
            v = dict(cu)
            if "total_v2" not in v:                  # phần đắt; bản di cư đã có thì bỏ qua
                try:
                    v.update(R23.evaluate(cx, f"{combo}/B/{ten}", Fs, Fm, Fx, W))
                except Exception as e:
                    W.append(f"{combo}/B/{ten}: lỗi {e!r}"); log(f"[{combo}] B {ten} LỖI {e!r}")
                    del Fm; gc.collect(); continue
            if keep:
                try:
                    v.update(st_metrics(Wn_q, nn_q, ok_q,
                                        transferred_labels(cx, Fm, q), lab_true, keep))
                except Exception as e:
                    W.append(f"{combo}/B/{ten}: chỉ số ST lỗi {e!r}")
            Sc[ten] = v
            log(f"[{combo}] B {ten}: xfer_main={v.get('xfer_main')} total_v2={v.get('total_v2')} "
                f"batch_v2={v.get('batch_v2')} pas_dev={v.get('pas_abs_dev')} "
                f"moran_dev={v.get('moran_abs_dev')}")
            save(); del Fm; gc.collect()
    del Wn_all, Wn_q; gc.collect()


def import_cache(W):
    """Chép cache niche/labcomp từ BẤT KỲ out<P>* nào trong Input vào OUT.

    `R17.get_niche_label_comp` và `get_pair_niche` chỉ đọc cache từ OUT, không tự quét Input. OUT
    của 45 (`out<P>s`) khởi đầu RỖNG mỗi lần chạy -> dựng lại từ h5ad đầy đủ, ~25 phút trước khi
    tới cặp đầu tiên. Thấy rõ ở mọi log trước (2350s, 1176s...). Chép sang là xong.
    """
    import shutil
    n = 0
    for d in sorted(glob.glob(f"/kaggle/input/**/out{P}*", recursive=True)):
        # labcomp*  bắt CẢ labcomp_<pair>.npy (niche 8 nhóm) LẪN labcomp16_<pair>.npy (niche mịn 16,
        # run20.py:532) — cái sau là thứ KM16 của phần B/T cần, glob "labcomp_*" KHÔNG khớp nó.
        for pat in ("niche_*.npy", "labcomp*.npy"):
            for f in glob.glob(os.path.join(d, pat)):
                dst = os.path.join(OUT, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy(f, dst); n += 1
    log(f"import_cache: chép {n} file cache vào {OUT}")
    if n == 0:
        W.append("import_cache: KHÔNG thấy cache niche/labcomp nào trong Input -> sẽ dựng lại từ đầu (~25 phút)")
    return n


# =============================================================================== Phần R
def coral_xoay(F, Fx, T, L, t):
    """CORAL với bậc tự do QUAY được dùng để làm trơn không gian.

    CORAL chuẩn: Z = (F-mu_s)·Sigma_s^-1/2·Sigma_x^1/2 + mu_x. Nghiệm đó KHÔNG duy nhất: với MỌI
    R trực giao, A_R = Sigma_s^-1/2·R·Sigma_x^1/2 cho Cov(Z) = Sigma_x^1/2·RᵀR·Sigma_x^1/2 = Sigma_x,
    y HỆT nhau. Chọn R = I là tuỳ tiện — đó là một bậc tự do chuẩn đang bị bỏ phí. Nhưng
    Sigma_s^-1/2·R·Sigma_x^1/2 KHÔNG trực giao nên R có đổi hình học kNN, tức đổi nhãn chuyển.

    Dùng nó để tối thiểu độ gồ ghề không gian. Với B = (F-mu_s)Sigma_s^-1/2 và M = BᵀLB:
        tr(ZᵀLZ) = tr(Rᵀ M R Sigma_x)
    Bất đẳng thức vết Ruhe: cực tiểu trên nhóm trực giao đạt khi ghép trị riêng NHỎ nhất của M với
    trị riêng LỚN nhất của Sigma_x, tức R* = U_M[:, ::-1] @ U_Xᵀ (cả hai từ eigh, tăng dần).
    Đổi dấu cột KHÔNG đổi mục tiêu (S·Lambda·S = Lambda với S chéo ±1) nên ép det=+1 được miễn phí.

    t nội suy: R(t) = cực-phân-rã((1-t)I + t R*). t=0 cho ĐÚNG CORAL hiện tại, t=1 cho trơn nhất.
    MỌI t khớp Sigma_x NHƯ NHAU -> phần mô-men bậc hai của batch_v2 bất biến theo cấu tạo. (Sai số
    còn lại ~RIDGE=0.01 là do Ws dựng từ Sigma_s đã ridge; nó GIỐNG nhau ở mọi t nên không ảnh hưởng.)

    Dùng lại RIDGE và _mat_pow của run23c để t=0 tái lập CORAL hiện hành đến từng chữ số.
    """
    F64, Fx64 = np.asarray(F, np.float64), np.asarray(Fx, np.float64)
    mu_s, mu_x = F64[T].mean(0), Fx64.mean(0)
    Cs, Cx = np.cov(F64[T], rowvar=False), np.cov(Fx64, rowvar=False)
    d = Cs.shape[0]
    Cs = Cs + R23C.RIDGE * np.trace(Cs) / d * np.eye(d)
    Cx = Cx + R23C.RIDGE * np.trace(Cx) / d * np.eye(d)
    Ws, Wx = R23C._mat_pow(Cs, -0.5), R23C._mat_pow(Cx, 0.5)
    if t == 0:
        return ((F64 - mu_s) @ (Ws @ Wx) + mu_x).astype(np.float32), None
    B = (F64 - mu_s) @ Ws
    M = B.T @ (L @ B)
    M = 0.5 * (M + M.T)                       # đối xứng hoá chống trôi số
    _, UM = np.linalg.eigh(M); _, UX = np.linalg.eigh(Cx)
    Rs = UM[:, ::-1] @ UX.T
    if np.linalg.det(Rs) < 0:                 # ép về SO(d) cho nội suy mượt; mục tiêu không đổi
        UM2 = UM[:, ::-1].copy(); UM2[:, -1] *= -1; Rs = UM2 @ UX.T
    U_, _, Vt_ = np.linalg.svd((1.0 - t) * np.eye(d) + t * Rs)
    Rt = U_ @ Vt_                             # cực phân rã -> trực giao gần nhất
    return ((F64 - mu_s) @ (Ws @ Rt @ Wx) + mu_x).astype(np.float32), Rs


def part_r(combo, cx, S, save, W):
    """FROST 1.0.3 — CORAL quay theo không gian. Hậu kỳ trên map đã có, CPU, KHÔNG dựng lại map.

    Khuyết tật ST (nhãn chuyển vụn trong không gian, PAS vượt sự thật 4/4) do CHÍNH CORAL gây ra:
    nó thổi các hướng phương sai thấp, lân cận kNN nhiễu hơn, nhãn lật giữa hai tế bào kề nhau.
    Phần A đã thử làm trơn đầu ra -> được PAS nhưng BÓP phương sai. Ở đây Sigma bị ghim cứng, chỉ
    còn quay -> ràng buộc chặt hơn hẳn.
    """
    Sc = S.setdefault("R", {}).setdefault(combo, {})
    Fx, T, q = cx["Fx"], cx["ds"]["T"], cx["Q"]
    sd = SEEDS[0]
    F0 = R22.get_map(combo, ROT_TAG, sd, None)[0]
    if F0 is None:
        W.append(f"{combo}: không có map {ROT_TAG} s{sd} -> bỏ phần R"); return
    F0 = np.asarray(F0, np.float32)

    xy = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    Wn_all, _, _, cut_all, deg_all = spatial_graph(xy)
    # L = D - A = D(I - Wn). Đối xứng vì A đã được đối xứng hoá ở spatial_graph.
    L = sp.diags(deg_all.astype(np.float64)) @ (sp.eye(len(xy), format="csr") - Wn_all.astype(np.float64))
    asym = abs((L - L.T)).max()
    if asym > 1e-6: W.append(f"{combo}: Laplacian lệch đối xứng {asym:.2e} -> nghiệm Ruhe không còn đúng")

    Wn_q, nn_q, ok_q, cut_q, _ = spatial_graph(xy[q])
    lab_true = np.asarray(cx["ds"]["lab"])[q]
    cnt = pd.Series(lab_true).value_counts()
    cand = [c for c in cnt.index if cnt[c] >= 20]
    I_t = moran_i(Wn_q, np.stack([(lab_true == c).astype(np.float64) for c in cand], 1)) if cand else np.array([])
    keep = [c for c, i_ in zip(cand, I_t) if i_ >= MORAN_MIN]
    if not keep:
        W.append(f"{combo}: không nhãn nào đạt Moran >= {MORAN_MIN} -> bỏ phần R"); return
    S.setdefault("R_meta", {})[combo] = {"map_nen": f"{ROT_TAG}@s{sd}", "n_nhan_giu": len(keep),
                                         "cat_toan_bo": round(cut_all, 4), "cat_Q": round(cut_q, 4)}
    Cx0 = np.cov(np.asarray(Fx, np.float64), rowvar=False); nCx = float(np.linalg.norm(Cx0))

    for t in ROT_TS:
        k = f"t{t:g}"
        if k in Sc and ("xfer_main" in Sc[k] or t not in ROT_FULL): continue
        step(f"[{combo}] R: {k}")
        try:
            Z, _ = coral_xoay(F0, Fx, T, L, t)
        except Exception as e:
            W.append(f"{combo}/R/{k}: dựng map lỗi {e!r}"); log(f"[{combo}] R {k} LỖI {e!r}"); continue
        pred = transferred_labels(cx, Z, q)
        r = st_metrics(Wn_q, nn_q, ok_q, pred, lab_true, keep)
        # d_cov ~0.01 (mức RIDGE) và PHẢI không đổi theo t; tăng theo t nghĩa là code sai.
        r["d_cov"] = round(float(np.linalg.norm(
            np.cov(Z[T].astype(np.float64), rowvar=False) - Cx0) / nCx), 4)
        r["rough"] = round(float(np.einsum("ij,ij->", Z.astype(np.float64), L @ Z.astype(np.float64))
                                 / (len(Z) * Z.shape[1])), 6)
        if t in ROT_FULL:
            if R20.KM16 is None:
                W.append(f"{combo}/R/{k}: R20.KM16 chưa đặt -> bỏ chấm đầy đủ")
            else:
                ev = R23.evaluate(cx, f"{combo}/R/{k}", cx["Fs"], Z, Fx, W)
                r.update({kk: ev.get(kk) for kk in
                          ("xfer_main", "xfer_all", "total_v2", "batch_v2", "bio_v2")})
        Sc[k] = r; save()
        log(f"[{combo}] R {k}: pas={r['pas_abs_dev']} moran={r['moran_abs_dev']} d_cov={r['d_cov']} "
            f"rough={r['rough']} total={r.get('total_v2')} xfer={r.get('xfer_main')}")
        del Z; gc.collect()
    del Wn_all, Wn_q, L; gc.collect()


# =============================================================================== Phần W
def part_w(combo, cx, S, save, W):
    """Tự tương quan không gian làm bộ lọc CHIỀU — không phải bộ lọc tế bào.

    CHỖ MỌI THỨ TRƯỚC ĐÂY SAI: tín hiệu không gian đúng 62-80% (pas_true 0.20-0.38). Dùng nó ở cấp
    TỪNG TẾ BÀO thì vô dụng — mỗi lân cận là mẫu cỡ 15 với 1/3 nhiễm bẩn. Nhưng cùng tín hiệu đó ở
    cấp QUẦN THỂ trên 45.000 tế bào có sai số chuẩn ~1/sqrt(n) ~ 0.005. Yếu ở cá thể, rất chắc ở
    tổng thể. Sáu ý tưởng trước (cmix, beta, Laplacian, CORAL quay, co rút, trung vị) đều là MỘT ý
    tưởng duy nhất — gộp láng giềng vào tế bào — dùng ở cấp sai.

    Ở đây không gian KHÔNG nói "tế bào này giống hàng xóm". Nó nói "CHIỀU này mới là sinh học":
    nhiễu kỹ thuật (độ sâu, hiệu ứng lô, tràn tín hiệu giữa bin) KHÔNG có tổ chức không gian; cấu
    trúc mô thì có. Moran's I của từng chiều nhúng, tính trên toàn bộ tế bào, tách được hai thứ.

    BA PHÉP ĐO, và phép thứ hai là cổng:
      1. Phân bố Moran theo chiều, trên nguồn và trên Xenium, kèm sàn hoán vị. Nếu mọi chiều đều
         ~0 thì không có gì để lọc, ý tưởng chết tại chỗ.
      2. TƯƠNG QUAN SPEARMAN giữa Moran_nguồn(d) và Moran_đích(d). Hai bên dùng CÙNG cơ sở (joint
         PCA hoặc cùng không gian scGPT) nên chiều so được với nhau. Tương quan CAO -> "chiều có
         cấu trúc không gian" là tính chất CHUNG, độc lập công nghệ -> đánh trọng số theo nó có
         nghĩa. Tương quan ~0 -> cấu trúc không gian là đặc thù từng công nghệ, trọng số sẽ tuỳ tiện.
      3. Chọn chiều theo Moran có hơn chọn theo PHƯƠNG SAI không: dùng top-N chiều theo mỗi tiêu
         chí rồi chấm bằng chính probe của Phần K/V. Đây là phép thử trực tiếp "không gian có chọn
         ra sinh học không", chứ không phải suy luận gián tiếp.
    """
    Sc = S.setdefault("W", {}).setdefault(combo, {})
    if Sc: return
    pctx, ds = cx["pctx"], cx["ds"]
    Fs, Fx = np.asarray(cx["Fs"], np.float64), np.asarray(cx["Fx"], np.float64)
    g = pctx["groups_ev"]; lab_s = np.asarray(pctx["lab_src_ev"])
    T_, Q_ = np.asarray(ds["T"]), np.asarray(cx["Q"])
    rng = np.random.RandomState(0)

    def moran_cot(Z, xy):
        Wn, _, _, cut, _ = spatial_graph(xy)
        I = moran_i(Wn, Z)
        Ip = np.stack([moran_i(Wn, Z[rng.permutation(len(Z))]) for _ in range(3)])
        return I, float(Ip.std()), cut

    xy_s = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    xy_x = lite_obs(R18.TARGET)[["x", "y"]].to_numpy()
    # Lệch hàng giữa ma trận đặc trưng và bảng toạ độ sẽ cho Moran SAI MÀ KHÔNG NỔ -> chặn ở đây.
    if len(xy_s) != len(Fs) or len(xy_x) != len(Fx):
        W.append(f"{combo}: lệch hàng toạ độ/đặc trưng (nguồn {len(xy_s)} vs {len(Fs)}, "
                 f"đích {len(xy_x)} vs {len(Fx)}) -> bỏ phần W")
        log(f"[{combo}] W: LỆCH HÀNG, bỏ"); return
    if Fs.shape[1] != Fx.shape[1]:
        W.append(f"{combo}: nguồn và đích khác số chiều ({Fs.shape[1]} vs {Fx.shape[1]}) -> "
                 "không so được Moran theo chiều, bỏ phần W")
        return
    step(f"[{combo}] W: Moran theo chiều (nguồn)")
    I_s, sd_s, cut_s = moran_cot(Fs, xy_s)
    step(f"[{combo}] W: Moran theo chiều (Xenium)")
    I_x, sd_x, cut_x = moran_cot(Fx, xy_x)

    from scipy.stats import spearmanr
    var_s = Fs.var(0)
    qs = lambda v: {f"p{p}": round(float(np.percentile(v, p)), 4) for p in (0, 10, 50, 90, 100)}
    Sc["moran_nguon"] = qs(I_s); Sc["moran_dich"] = qs(I_x)
    Sc["sd_hoanvi"] = {"nguon": round(sd_s, 5), "dich": round(sd_x, 5)}
    Sc["cat_canh"] = {"nguon": round(cut_s, 4), "dich": round(cut_x, 4)}
    Sc["d"] = int(Fs.shape[1])
    Sc["tl_chieu_co_tin_hieu"] = {"nguon": round(float((I_s > 3 * sd_s).mean()), 4),
                                  "dich": round(float((I_x > 3 * sd_x).mean()), 4)}
    Sc["spearman_nguon_dich"] = round(float(spearmanr(I_s, I_x).statistic), 4)   # <- CỔNG
    Sc["spearman_moran_var"] = round(float(spearmanr(I_s, var_s).statistic), 4)
    save()

    # 3) chọn chiều theo Moran so với theo phương sai
    def probe(cols):
        Z = np.asarray(Fs[:, cols], np.float32)
        mu = Z[T_].mean(0)
        nn = NearestNeighbors(n_neighbors=15, n_jobs=4).fit(_unit(Z[T_] - mu)).kneighbors(
            _unit(Z[Q_] - mu), return_distance=False)
        return bacc_on(lab_s[Q_], mode_rows(lab_s[T_][nn]), g)

    d = Fs.shape[1]
    Sc["bacc_toanbo"] = probe(np.arange(d))
    for N in WN_TOP:
        if N >= d: continue
        step(f"[{combo}] W: top-{N} chiều")
        Sc[f"bacc_moran_top{N}"] = probe(np.argsort(-I_s)[:N])
        Sc[f"bacc_var_top{N}"] = probe(np.argsort(-var_s)[:N])
        save()
    log(f"[{combo}] W: spearman(nguồn,đích)={Sc['spearman_nguon_dich']} "
        f"spearman(moran,var)={Sc['spearman_moran_var']} "
        f"moran_nguồn p50={Sc['moran_nguon']['p50']} p90={Sc['moran_nguon']['p90']} "
        f"sàn={Sc['sd_hoanvi']['nguon']} | bacc toàn bộ={Sc['bacc_toanbo']}")
    save(); gc.collect()


# =============================================================================== Phần X
def part_x(combo, cx, S, save, W):
    """FROST 1.0.4 — Spatial-CORAL: khớp thêm mô-men bậc hai CÓ TRỌNG SỐ KHÔNG GIAN.

    LÝ DO CÓ TÍNH CHỨNG MINH, không phải quan sát. CORAL tính Cs, Cx từ ma trận tế_bào×chiều. Hoán
    vị thứ tự hàng KHÔNG đổi hiệp phương sai. Vậy CORAL BẤT BIẾN VỚI HOÁN VỊ TẾ BÀO, nên nó mù cấu
    trúc không gian về mặt CẤU TRÚC — không phải vì chưa tinh chỉnh đủ. Phần W đo được chỗ mù đó
    tốn gì: Moran trung vị theo chiều lệch 40-50% giữa nguồn và Xenium (vhd_ff 0.410 vs 0.285;
    stereoseq 0.438 vs 0.285; cosmx6k 0.033 vs 0.131 — ngược chiều). CORAL không chạm tới được.

    Mô-men bậc hai CÓ TRỌNG SỐ KHÔNG GIAN, trong hệ ĐÃ TẨY TRẮNG:
        A = Bᵀ W̃_s B / n_s    với B  = (F - mu_s) Cs^-1/2    (nguồn)
        G = Bxᵀ W̃_x Bx / n_x   với Bx = (X - mu_x) Cx^-1/2    (đích)
    W̃ = (Wn + Wnᵀ)/2: Wn chuẩn hoá theo hàng nên KHÔNG đối xứng, mà nghiệm Ruhe đòi đối xứng.

    Bậc tự do còn lại sau CORAL đúng là nhóm trực giao (lập luận ở coral_xoay):
    Z = (F-mu_s)Cs^-1/2 R Cx^1/2 giữ nguyên Cov(Z) = Cx với MỌI R. Trong hệ tẩy trắng mô-men không
    gian của Z là RᵀAR, nên
        min_R ‖RᵀAR - G‖_F   <=>   max_R tr(RᵀAR G)
    Ruhe: cực đại = Σ λ↑(A)λ↑(G), đạt tại R* = U_A U_Gᵀ (cả hai eigh, TĂNG dần). Khác phần R ở chỗ
    phần R ghép NGƯỢC (cực tiểu vết); ở đây ghép THUẬN.

    HAI CỔNG RẺ, chạy trước mọi phép chấm đắt:
      1. `d_sp_san` — chặn DƯỚI của ‖RᵀAR-G‖/‖G‖ trên TOÀN nhóm trực giao, tính thẳng từ hai phổ.
         Nếu nó ≈ `d_sp_t0` thì phép quay về nguyên tắc không sửa được gì -> chết, khỏi chấm.
      2. nhánh `hoanvi@t1` — R* dựng từ đồ thị nguồn ĐÃ HOÁN VỊ. Bài học phần K: đồ thị thật từng
         THUA đồ thị hoán vị 4/4. Nếu hoanvi ngang t=1 thì thứ mua được không phải thông tin không gian.

    `d_cov` phải ĐỨNG YÊN ở mức RIDGE với mọi t (bất biến theo cấu tạo); tăng theo t nghĩa là code sai.
    Với SPC_TAG="cmix", t=0 tái lập ĐÚNG FROST (u_cmix_cc) nên là mốc so duy nhất đúng của phần này.
    """
    Sc = S.setdefault("X", {}).setdefault(combo, {})
    Fx, T, q = cx["Fx"], cx["ds"]["T"], cx["Q"]
    sd = SEEDS[0]
    F0 = R22.get_map(combo, SPC_TAG, sd, None)[0]
    if F0 is None:
        W.append(f"{combo}: không có map {SPC_TAG} s{sd} -> bỏ phần X"); return
    F0 = np.asarray(F0, np.float64); Fx64 = np.asarray(Fx, np.float64)
    d = F0.shape[1]
    if d != Fx64.shape[1]:
        W.append(f"{combo}: nguồn {d} chiều, đích {Fx64.shape[1]} chiều -> bỏ phần X"); return
    xy_s = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    xy_x = lite_obs(R18.TARGET)[["x", "y"]].to_numpy()
    # Lệch hàng giữa toạ độ và đặc trưng cho mô-men không gian SAI MÀ KHÔNG NỔ -> chặn ở đây (bài học phần W).
    if len(xy_s) != len(F0) or len(xy_x) != len(Fx64):
        W.append(f"{combo}: lệch hàng toạ độ/đặc trưng (nguồn {len(xy_s)}/{len(F0)}, "
                 f"đích {len(xy_x)}/{len(Fx64)}) -> bỏ phần X")
        log(f"[{combo}] X: LỆCH HÀNG, bỏ"); return

    step(f"[{combo}] X: đồ thị + mô-men không gian")
    Wn_s, _, _, cut_s, _ = spatial_graph(xy_s)
    Wn_x, _, _, cut_x, _ = spatial_graph(xy_x)
    Ss = (0.5 * (Wn_s + Wn_s.T)).tocsr()
    Sx = (0.5 * (Wn_x + Wn_x.T)).tocsr()
    del Wn_x; gc.collect()

    mu_s, mu_x = F0[T].mean(0), Fx64.mean(0)
    Cs, Cx0 = np.cov(F0[T], rowvar=False), np.cov(Fx64, rowvar=False)
    nCx = float(np.linalg.norm(Cx0))
    Cs = Cs + R23C.RIDGE * np.trace(Cs) / d * np.eye(d)
    Cx = Cx0 + R23C.RIDGE * np.trace(Cx0) / d * np.eye(d)
    Wsq, Wxq = R23C._mat_pow(Cs, -0.5), R23C._mat_pow(Cx, 0.5)
    sym = lambda Mx: 0.5 * (Mx + Mx.T)
    Bx = (Fx64 - mu_x) @ R23C._mat_pow(Cx, -0.5)
    G = sym(Bx.T @ (Sx @ Bx) / len(Bx))
    del Bx, Sx, Fx64; gc.collect()
    B = (F0 - mu_s) @ Wsq            # giữ B, BỎ F0: Z = B·Rt·Wxq + mu_x, khỏi trừ mu_s lại mỗi vòng
    del F0; gc.collect()
    A = sym(B.T @ (Ss @ B) / len(B))
    perm = np.random.RandomState(0).permutation(len(B))
    Bp = B[perm].astype(np.float32)                       # float32: bản sao này chỉ để dựng nhánh null
    A_h = sym((Bp.T @ (Ss.astype(np.float32) @ Bp)).astype(np.float64) / len(Bp))
    del Bp, Ss; gc.collect()

    nG = float(np.linalg.norm(G))
    lA, lG = np.linalg.eigvalsh(A), np.linalg.eigvalsh(G)
    _, UG = np.linalg.eigh(G)
    san = float(np.sqrt(max(0.0, float(lA @ lA) + float(lG @ lG) - 2.0 * float(lA @ lG)))) / nG
    S.setdefault("X_meta", {})[combo] = {
        "map_nen": f"{SPC_TAG}@s{sd}", "d": int(d),
        "cat_canh": {"nguon": round(cut_s, 4), "dich": round(cut_x, 4)},
        "d_sp_t0": round(float(np.linalg.norm(A - G)) / nG, 4),   # lệch mô-men không gian CỦA FROST
        "d_sp_san": round(san, 4),                                # CỔNG 1: chặn dưới trên MỌI phép quay
        "moran_tb_nguon_trang": round(float(np.trace(A) / d), 4),
        "moran_tb_dich_trang": round(float(np.trace(G) / d), 4)}
    mt = S["X_meta"][combo]
    log(f"[{combo}] X: d_sp t0={mt['d_sp_t0']} sàn={mt['d_sp_san']} | moran(tẩy trắng) "
        f"nguồn={mt['moran_tb_nguon_trang']} đích={mt['moran_tb_dich_trang']}")
    save()

    def R_sao(Am):
        """R* = U_A U_Gᵀ, ghép THUẬN λ↑ với λ↑ -> cực đại tr(RᵀAR G). det<0 thì lật một cột:
        đổi dấu cột không đổi mục tiêu (S·Λ·S = Λ) nên ép về SO(d) cho nội suy cực mượt là miễn phí."""
        _, U = np.linalg.eigh(Am)
        Rs = U @ UG.T
        if np.linalg.det(Rs) < 0:
            U2 = U.copy(); U2[:, -1] *= -1; Rs = U2 @ UG.T
        return Rs

    Rs_that, Rs_hoanvi = R_sao(A), R_sao(A_h)
    del A_h; gc.collect()

    Wn_q, nn_q, ok_q, cut_q, _ = spatial_graph(xy_s[q])
    lab_true = np.asarray(cx["ds"]["lab"])[q]
    cnt = pd.Series(lab_true).value_counts()
    cand = [c for c in cnt.index if cnt[c] >= 20]
    I_t = moran_i(Wn_q, np.stack([(lab_true == c).astype(np.float64) for c in cand], 1)) if cand else np.array([])
    keep = [c for c, i_ in zip(cand, I_t) if i_ >= MORAN_MIN]
    if not keep:
        W.append(f"{combo}: không nhãn nào đạt Moran >= {MORAN_MIN} -> bỏ chấm ST phần X"); return
    mt["n_nhan_giu"] = len(keep); mt["cat_Q"] = round(cut_q, 4)

    for k, t, Rs in [(f"t{t:g}", t, Rs_that) for t in SPC_TS] + [("hoanvi@t1", 1.0, Rs_hoanvi)]:
        if k in Sc and ("xfer_main" in Sc[k] or t not in SPC_FULL): continue
        step(f"[{combo}] X: {k}")
        U_, _, Vt_ = np.linalg.svd((1.0 - t) * np.eye(d) + t * Rs)
        Rt = U_ @ Vt_                                 # cực phân rã -> trực giao gần nhất
        Z = (B @ (Rt @ Wxq) + mu_x).astype(np.float32)
        pred = transferred_labels(cx, Z, q)
        r = st_metrics(Wn_q, nn_q, ok_q, pred, lab_true, keep)
        r["d_sp"] = round(float(np.linalg.norm(Rt.T @ A @ Rt - G)) / nG, 4)
        r["d_cov"] = round(float(np.linalg.norm(
            np.cov(Z[T].astype(np.float64), rowvar=False) - Cx0) / nCx), 4)
        if t in SPC_FULL:
            if R20.KM16 is None:
                W.append(f"{combo}/X/{k}: R20.KM16 chưa đặt -> bỏ chấm đầy đủ")
            else:
                ev = R23.evaluate(cx, f"{combo}/X/{k}", cx["Fs"], Z, Fx, W)
                r.update({kk: ev.get(kk) for kk in
                          ("xfer_main", "xfer_all", "total_v2", "batch_v2", "bio_v2")})
        Sc[k] = r; save()
        log(f"[{combo}] X {k}: d_sp={r['d_sp']} d_cov={r['d_cov']} pas={r['pas_abs_dev']} "
            f"moran={r['moran_abs_dev']} total={r.get('total_v2')} xfer={r.get('xfer_main')} "
            f"batch={r.get('batch_v2')}")
        del Z; gc.collect()
    del B, A, G, Wn_s, Wn_q; gc.collect()


# =============================================================================== Phần Y
def _ban_va(xy, n_muc, n_toi_thieu, n_lay):
    """Chia lát thành ô vuông cạnh L (µm) sao cho ô TRUNG VỊ có ~n_muc tế bào; trả các ô gần n_muc nhất.

    Không dùng diện tích hộp bao: mô có hình dạng bất kỳ nên hộp bao thổi diện tích lên và L bị chọn
    quá nhỏ. Dò L bằng 3 vòng theo trung vị số tế bào thực tế trong ô — bất biến với hình dạng lát.
    """
    rng = xy.max(0) - xy.min(0)
    L = float(np.sqrt(max(rng[0] * rng[1], 1.0) / max(len(xy) / n_muc, 1.0)))
    o = None
    for _ in range(3):
        g = np.floor((xy - xy.min(0)) / L).astype(np.int64)
        key = g[:, 0] * (g[:, 1].max() + 2) + g[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        ok = cnt >= n_toi_thieu
        if not ok.any(): break
        med = float(np.median(cnt[ok]))
        o = (inv, u, cnt, ok, L)
        if abs(med - n_muc) / n_muc < 0.15: break
        L *= float(np.sqrt(n_muc / max(med, 1.0)))
    if o is None: return [], float("nan")
    inv, u, cnt, ok, L = o
    thu_tu = np.where(ok)[0][np.argsort(np.abs(cnt[ok] - n_muc))][:n_lay]
    return [np.where(inv == j)[0] for j in thu_tu], L


def _tinh_khiet(D, lab, k):
    """Tỉ lệ cặp (tế bào, 1 trong k láng giềng gần nhất theo D) CÙNG nhãn. D là ma trận khoảng cách."""
    k = min(k, len(D) - 1)
    if k < 1: return None
    nb = np.argsort(D, axis=1)[:, 1:k + 1]
    codes, _ = pd.factorize(np.asarray(lab))
    m = codes >= 0
    if not m.any(): return None
    return round(float((codes[nb[m]] == codes[m][:, None]).mean()), 4)


def _hist_cx(D, nbin=20):
    """Lược đồ khoảng cách đôi một SAU chuẩn hoá D/D.max() — đúng phép chuẩn hoá USHER dùng cho C."""
    v = D[np.triu_indices_from(D, 1)]
    mx = v.max()
    if not np.isfinite(mx) or mx <= 0: return np.zeros(nbin)
    h, _ = np.histogram(v / mx, bins=nbin, range=(0, 1))
    return h / max(h.sum(), 1)


def _dong_y_ghep(Pi, lab_a, lab_b):
    """Tỉ lệ KHỐI LƯỢNG mà kế hoạch vận chuyển đặt vào cặp cùng nhãn, kèm mức ngẫu nhiên Σ p_a(c)p_b(c)."""
    la, lb = np.asarray(lab_a).astype(str), np.asarray(lab_b).astype(str)
    uniq = np.unique(np.concatenate([la, lb]))
    tot = float(Pi.sum())
    if tot <= 0: return None, None
    s = 0.0; ch = 0.0
    for c in uniq:
        ia, ib = la == c, lb == c
        if ia.any() and ib.any(): s += float(Pi[np.ix_(ia, ib)].sum())
        ch += (ia.mean()) * (ib.mean())
    return round(s / tot, 4), round(float(ch), 4)


def part_y(combo, cx, S, save, W):
    """CỔNG CPU cho C_sp (toạ độ làm ma trận cấu trúc GW kiểu PASTE) — chạy TRƯỚC khi sửa src_cuda.

    Vì sao phải là cổng chứ không dựng thẳng: đổi cách lấy batch + C_sp là sửa src_cuda rồi chạy lại
    USHER trên GPU, 4 cặp × 3 seed. Tám ý tưởng trước đã chết. Mọi thứ cần biết để đoán ý tưởng này
    sống hay chết đều đo được trên CPU trong vài phút, nên đo trước.

    Vì sao C_sp là khe DUY NHẤT còn hợp lệ cho toạ độ ở bộ dữ liệu này: vòng 17 Phần A đo đăng ký
    affine THẤT BẠI 0/4 (same_label chỉ nhỉnh ngẫu nhiên 2-6 điểm, top1/top2 ≈ 1, stereoseq lệch
    174°/2556µm). Nên mọi cách dùng toạ độ ở chi phí LIÊN-MIỀN (vòng 18 `spatial_b`) là nạp nhiễu.
    Số hạng cấu trúc GW chỉ cần khoảng cách TRONG một lát nên bất biến với xoay/tịnh tiến/lật —
    chính lý do PASTE chọn thiết kế đó, và chính lý do nó miễn nhiễm với việc đăng ký thất bại.

    USHER có sẵn `spatial_utils` cửa sổ không gian nhưng nó dùng CÙNG biên cửa sổ cho hai bên
    ("For affine-aligned data"), tức đòi đúng cái đăng ký bất khả -> không dùng được, phải tự lấy
    bản vá ĐỘC LẬP mỗi bên, cùng cỡ vật lý, vị trí tuỳ ý.

    BA CỔNG:
      1. `tk_kgian` so với `tk_bieuhien` TRÊN CẢ HAI BÊN. Việc duy nhất của ma trận cấu trúc là nói
         "hai tế bào này gần nhau về cấu trúc". Nếu biểu hiện nói chính xác hơn toạ độ thì trộn
         toạ độ vào C chỉ làm bẩn. Đây đúng là logic đã giết Phần W (phương sai thắng Moran).
      2. `chong_lap_sp`: lược đồ khoảng cách đôi một sau chuẩn hoá D/D.max(), nguồn so với đích.
         Thấp -> hai bên ở hai thang mật độ khác nhau, GW sẽ khớp bằng cách CO GIÃN mật độ chứ không
         phải khớp sinh học. Phần S đã đo d1 = 14.9-16.0µm khác nhau giữa các công nghệ.
      3. `gw_sp` so với `gw_kgian_ngau` (toạ độ đều cùng hộp bao) và `gw_ex`: giải GW thật trên một cặp bản vá
         rồi đo tỉ lệ khối lượng ghép vào cặp CÙNG nhãn. Đây là ước lượng TRỰC TIẾP thứ C_sp sẽ mua
         được bên trong USHER, không cần chạy USHER. (Cần POT; thiếu thì bỏ, hai cổng trên vẫn đủ
         để kết luận âm.)
    """
    Sc = S.setdefault("Y", {}).setdefault(combo, {})
    if Sc: return
    pctx = cx["pctx"]
    Fs, Fx = np.asarray(cx["Fs"], np.float32), np.asarray(cx["Fx"], np.float32)
    lab_s, lab_x = np.asarray(pctx["lab_src_ev"]), np.asarray(pctx["lab_tgt_ev"])
    xy_s = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    xy_x = lite_obs(R18.TARGET)[["x", "y"]].to_numpy()
    for ten, a, b, c in (("nguồn", xy_s, Fs, lab_s), ("đích", xy_x, Fx, lab_x)):
        if not (len(a) == len(b) == len(c)):
            W.append(f"{combo}: lệch hàng bên {ten} ({len(a)}/{len(b)}/{len(c)}) -> bỏ phần Y")
            log(f"[{combo}] Y: LỆCH HÀNG {ten}, bỏ"); return

    rng = np.random.RandomState(0)
    ket = {}
    for ten, xy, F, lab in (("nguon", xy_s, Fs, lab_s), ("dich", xy_x, Fx, lab_x)):
        step(f"[{combo}] Y: bản vá bên {ten}")
        bv, L = _ban_va(xy, PATCH_N, PATCH_MIN, N_PATCH)
        if not bv:
            W.append(f"{combo}: bên {ten} không có ô nào đủ {PATCH_MIN} tế bào -> bỏ phần Y"); return
        tk_sp, tk_ex, d1s, hs, hx, ns = [], [], [], [], [], []
        for idx in bv:
            if len(idx) > PATCH_N: idx = idx[rng.choice(len(idx), PATCH_N, replace=False)]
            ns.append(len(idx))
            Dsp = cdist(xy[idx], xy[idx])
            Zn = _unit(F[idx].astype(np.float32))
            Dex = 1.0 - Zn @ Zn.T
            np.fill_diagonal(Dex, 0.0)
            tk_sp.append(_tinh_khiet(Dsp, lab[idx], PK)); tk_ex.append(_tinh_khiet(Dex, lab[idx], PK))
            d = np.partition(Dsp + np.eye(len(idx)) * 1e9, 1, axis=1)[:, 1]
            d1s.append(float(np.median(d)))
            hs.append(_hist_cx(Dsp)); hx.append(_hist_cx(Dex))
        f = lambda v: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
        ket[ten] = {"L_um": round(L, 1), "n_ban_va": len(bv), "n_te_bao_tb": int(np.mean(ns)),
                    "d1_um": round(float(np.mean(d1s)), 2),
                    "tk_kgian": f(tk_sp), "tk_bieuhien": f(tk_ex),
                    "_h_sp": np.mean(hs, 0), "_h_ex": np.mean(hx, 0), "_bv": bv}
    for ten in ("nguon", "dich"):
        v = ket[ten]
        Sc[ten] = {k: v[k] for k in v if not k.startswith("_")}
        Sc[ten]["chenh_kgian_tru_bieuhien"] = round(v["tk_kgian"][0] - v["tk_bieuhien"][0], 4)

    tv = lambda p, q: round(float(1.0 - 0.5 * np.abs(p - q).sum()), 4)
    Sc["chong_lap_sp"] = tv(ket["nguon"]["_h_sp"], ket["dich"]["_h_sp"])
    Sc["chong_lap_ex"] = tv(ket["nguon"]["_h_ex"], ket["dich"]["_h_ex"])
    Sc["d1_ti_le_nguon_tren_dich"] = round(ket["nguon"]["d1_um"] / max(ket["dich"]["d1_um"], 1e-9), 3)
    save()
    log(f"[{combo}] Y: tk_kgian nguồn={Sc['nguon']['tk_kgian']} bieuhien={Sc['nguon']['tk_bieuhien']} | "
        f"đích {Sc['dich']['tk_kgian']} vs {Sc['dich']['tk_bieuhien']} | "
        f"chồng lấp sp={Sc['chong_lap_sp']} ex={Sc['chong_lap_ex']} d1 tỉ lệ={Sc['d1_ti_le_nguon_tren_dich']}")

    # ---- cổng 3: giải GW thật trên MỘT cặp bản vá
    try:
        import ot as POT
    except Exception as e:
        W.append(f"{combo}/Y: không có POT -> bỏ cổng GW ({e!r})"); save(); return
    ia, ib = ket["nguon"]["_bv"][0], ket["dich"]["_bv"][0]
    if len(ia) > GW_N: ia = ia[rng.choice(len(ia), GW_N, replace=False)]
    if len(ib) > GW_N: ib = ib[rng.choice(len(ib), GW_N, replace=False)]
    p, q = np.ones(len(ia)) / len(ia), np.ones(len(ib)) / len(ib)
    nrm = lambda D: (D / max(D.max(), 1e-9)).astype(np.float64)
    def Dex_(F, idx):
        Z = _unit(F[idx].astype(np.float32)); D = 1.0 - Z @ Z.T; np.fill_diagonal(D, 0.0); return D
    # NULL phải là HÌNH HỌC NGẪU NHIÊN, không phải hoán vị. GW BẤT BIẾN với phép hoán vị đồng thời
    # hàng-cột (Ca[perm][:,perm] cho đúng cùng một nghiệm), nên "hoán vị toạ độ" là null RỖNG — nó
    # trả đúng con số của nhánh thật. Thay bằng toạ độ đều trong CÙNG hộp bao: phá cấu trúc không
    # gian mà giữ nguyên thang khoảng cách. Mức ngẫu nhiên của `dong_y` đã có sẵn ở `ngau_nhien`.
    bb0, bb1 = xy_s[ia].min(0), xy_s[ia].max(0)
    xy_rnd = bb0 + rng.rand(len(ia), 2) * (bb1 - bb0)
    nhanh = {"sp": (nrm(cdist(xy_s[ia], xy_s[ia])), nrm(cdist(xy_x[ib], xy_x[ib]))),
             "kgian_ngau": (nrm(cdist(xy_rnd, xy_rnd)), nrm(cdist(xy_x[ib], xy_x[ib]))),
             "ex": (nrm(Dex_(Fs, ia)), nrm(Dex_(Fx, ib)))}
    for k, (Ca, Cb) in nhanh.items():
        step(f"[{combo}] Y: GW {k} ({len(ia)}x{len(ib)})")
        try:                         # chữ ký POT đổi giữa các bản -> thử vị trí rồi thử từ khoá
            try:
                Pi, inf = POT.gromov.gromov_wasserstein(Ca, Cb, p, q, "square_loss", log=True, max_iter=200)
            except TypeError:
                Pi, inf = POT.gromov.gromov_wasserstein(C1=Ca, C2=Cb, p=p, q=q, log=True)
        except Exception as e:
            W.append(f"{combo}/Y/GW {k}: lỗi {e!r}"); log(f"[{combo}] Y GW {k} LỖI {e!r}"); continue
        gwd = next((inf[kk] for kk in ("gw_dist", "gw_dist_est", "loss") if kk in inf), np.nan)
        try: gwd = round(float(np.asarray(gwd).ravel()[-1]), 6)
        except Exception: gwd = None
        dy, ch = _dong_y_ghep(np.asarray(Pi), lab_s[ia], lab_x[ib])
        Sc[f"gw_{k}"] = {"gw_dist": gwd,
                         "dong_y": dy, "ngau_nhien": ch,
                         "loi_tren_ngau_nhien": None if dy is None else round(dy - ch, 4)}
        save()
        log(f"[{combo}] Y: GW {k}: gw={gwd} đồng ý={dy} "
            f"(ngẫu nhiên {ch}, lợi {Sc[f'gw_{k}']['loi_tren_ngau_nhien']})")
    gc.collect()


# =============================================================================== Phần V
def _gop(Z, nn_idx, ok, how, chunk=4000):
    """Ước lượng VỊ TRÍ của lân cận: trung bình hoặc TRUNG VỊ theo từng chiều.

    VÌ SAO CÓ HÀM NÀY: pas_true = 0.20-0.38 nghĩa là lân cận là một mẫu bị NHIỄM BẨN 20-38%.
    Điểm sụp đổ của trung bình là 0 — một tế bào khác loại đủ kéo lệch. Của trung vị là 0.5.
    Mức nhiễm bẩn đo được nằm GỌN trong vùng trung vị còn sống mà trung bình đã chết. Mọi thứ dự
    án đã thử (cmix, beta, làm trơn alpha, phạt Laplacian, CORAL quay, Phần K) đều là TRUNG BÌNH.

    Chia khối vì Z[nn_idx] cỡ (n, k, d); với n=45k, k=15, d=512 là 1.4 GB nếu làm một lần.
    """
    n, d = Z.shape
    out = np.empty((n, d), np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        v = np.asarray(Z[nn_idx[a:b]], np.float32)
        v = np.where(ok[a:b][:, :, None], v, np.nan)
        trong = ~np.isfinite(v).any(1)                 # hàng không còn láng giềng nào
        with warnings.catch_warnings():                # nanmean/nanmedian của lát toàn NaN cảnh báo
            warnings.simplefilter("ignore", RuntimeWarning)   # -> đã xử lý ngay dưới, đừng rác log
            r = np.nanmean(v, 1) if how == "tb" else np.nanmedian(v, 1)
        r[trong] = np.asarray(Z[a:b], np.float32)[trong]   # giữ nguyên tế bào cô lập
        out[a:b] = r
    return out


def part_v(combo, cx, S, save, W):
    """Gộp BỀN VỮNG thay cho trung bình — hệ quả trực tiếp của pas_true = 0.20-0.38.

    Giữ NGUYÊN cơ chế co rút của Phần K (w_i = sigma^2/(sigma^2+s2_i), s2 tính quanh TRUNG BÌNH cho
    MỌI nhánh) để `w` giống hệt nhau giữa các nhánh. Chỉ đổi đúng một thứ: ước lượng vị trí `m_i`.
    Như vậy chênh lệch quy được về ước lượng, không lẫn với việc đổi mức co rút.

    DỰ ĐOÁN KIỂM CHỨNG ĐƯỢC (đặt TRƯỚC khi chạy): nếu nguyên nhân là nhiễm bẩn thì mức hơn của
    trung vị so với trung bình phải TĂNG THEO pas_true. Thứ tự pas_true (đồ thị toàn bộ, k=15):
    cosmx6k .378 > visiumhd_ff .288 > visiumhd_ffpe .258 > stereoseq .197. Mức hơn KHÔNG theo thứ
    tự này thì giả thuyết sai — không được giải thích lại sau khi nhìn số.
    """
    Sc = S.setdefault("V", {}).setdefault(combo, {})
    pctx, ds, Fs = cx["pctx"], cx["ds"], cx["Fs"]
    g = pctx["groups_ev"]; lab_s = np.asarray(pctx["lab_src_ev"])
    T_, Q_ = np.asarray(ds["T"]), np.asarray(cx["Q"])
    xy = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    Wn, nn_idx, ok, cut, _ = spatial_graph(xy)

    def probe(Z):
        mu = np.asarray(Z[T_], np.float32).mean(0)
        nn = NearestNeighbors(n_neighbors=15, n_jobs=4).fit(_unit(np.asarray(Z[T_], np.float32) - mu)
             ).kneighbors(_unit(np.asarray(Z[Q_], np.float32) - mu), return_distance=False)
        return bacc_on(lab_s[Q_], mode_rows(lab_s[T_][nn]), g)

    if "raw" not in Sc:
        step(f"[{combo}] V: raw"); Sc["raw"] = {"bacc": probe(Fs)}; save()
    F64 = np.asarray(Fs, np.float64)
    # s2 và w: tính MỘT LẦN, dùng chung cho mọi nhánh -> cô lập được biến duy nhất là `how`.
    m_tb = np.asarray(Wn @ F64)
    s2 = np.maximum(np.asarray(Wn @ (F64 ** 2).sum(1)) - (m_tb ** 2).sum(1), 0.0) / F64.shape[1]
    iso = np.asarray(Wn.sum(1)).ravel() <= 0
    med_s2 = float(np.median(s2[~iso])) if (~iso).any() else 1.0
    for c in KC:
        for how in AGGS:
            k = f"{how}_c{c:g}"
            if k in Sc: continue
            step(f"[{combo}] V: {k}")
            w = (c * med_s2) / (c * med_s2 + np.maximum(s2, 1e-12)); w[iso] = 0.0
            m = _gop(Fs, nn_idx, ok, how)
            Z = (F64 + w[:, None] * (np.asarray(m, np.float64) - F64)).astype(np.float32)
            Sc[k] = {"bacc": probe(Z), "w_tb": round(float(w.mean()), 4)}
            save(); log(f"[{combo}] V {k}: bacc={Sc[k]['bacc']} (raw={Sc['raw']['bacc']}) w_tb={Sc[k]['w_tb']}")
            del Z, m; gc.collect()
    S.setdefault("V_meta", {})[combo] = {"cat_canh": round(cut, 4),
                                         "nhan": "thô" if pctx.get("use_coarse") else "mịn"}
    save(); del Wn; gc.collect()


# =============================================================================== Phần S
def part_s(combo, cx, S, save, W):
    """THANG ĐO: pas_true thật sự bằng bao nhiêu, ở thang nào, đo bằng đơn vị đo được.

    LÝ DO: con số `pas_true` 0.21-0.40 từng được dùng để kết luận "mô quá dị thể ngay ở thang tế
    bào" là SAI THANG. Nó tính trên `xy[q]` (N_Q = 10000 trên 40-45k tế bào, run45.py:170), tức
    mật độ chỉ ~22% -> khoảng cách 15-NN giãn ra ~sqrt(n/10000) lần. Trong khi Phần K lại dùng đồ
    thị TOÀN BỘ tế bào (run45.py:348). Hai thang khác nhau, không được dùng số này giải thích số kia.

    Ở đây đo TRỰC TIẾP, không ước lượng:
      d1_um      — trung vị khoảng cách tới láng giềng GẦN NHẤT = khoảng cách giữa các tế bào.
                   Đây là thang quy chiếu ĐO ĐƯỢC, thay cho việc giả định đường kính tế bào ~10um.
      dk_um      — trung vị khoảng cách tới láng giềng thứ k.
      dk_tren_d1 — bán kính lân cận tính theo ĐƠN VỊ khoảng cách tế bào. Không cần giả định gì.
      pas_true, moran_true — trên nhãn THẬT, cho từng k, trên CẢ HAI đồ thị.

    PHÁN QUYẾT: pas_true ở thang chặt (đồ thị toàn bộ, k nhỏ) vẫn ~0.2-0.4 -> mô thật sự dị thể ở
    thang tế bào, và kết quả âm của Phần K có lời giải. Nếu nó tụt về ~0.05-0.1 -> lân cận chặt CÓ
    đồng nhất, Phần K hỏng vì lý do khác, và việc đóng hướng ST là vội.
    """
    Sc = S.setdefault("S", {}).setdefault(combo, {})
    xy_full = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    lab_full = np.asarray(cx["ds"]["lab"])
    q = np.asarray(cx["Q"])
    for ten, xy, lab in (("toanbo", xy_full, lab_full), ("Q", xy_full[q], lab_full[q])):
        for kk in SK_KS:
            key = f"{ten}_k{kk}"
            if key in Sc: continue
            step(f"[{combo}] S: {key}")
            nn = NearestNeighbors(n_neighbors=min(kk + 1, len(xy)), n_jobs=4).fit(xy)
            dist, _ = nn.kneighbors(xy)
            Wn, nn_idx, ok, cut, deg = spatial_graph(xy, k=kk, r_max=R_MAX_UM)
            pt, n_iso = pas(nn_idx, ok, lab)
            cnt = pd.Series(lab).value_counts()
            cand = [c for c in cnt.index if cnt[c] >= 20]
            I = moran_i(Wn, np.stack([(lab == c).astype(np.float64) for c in cand], 1)) if cand else np.array([0.0])
            d1, dk = float(np.median(dist[:, 1])), float(np.median(dist[:, -1]))
            Sc[key] = {"n": int(len(xy)), "d1_um": round(d1, 2), "dk_um": round(dk, 2),
                       "dk_tren_d1": round(dk / max(d1, 1e-9), 2), "cat_canh": round(cut, 4),
                       "bac_tb": round(float(deg.mean()), 2), "n_co_lap": int(n_iso),
                       "pas_true": None if pt is None else round(pt, 4),
                       "moran_true_tb": round(float(I.mean()), 4), "n_nhan": len(cand)}
            save(); log(f"[{combo}] S {key}: {Sc[key]}")
            del Wn, dist; gc.collect()


# =============================================================================== Phần K
def _lam_truot(Z, Wn, c):
    """Co rút thích ứng theo ĐỘ DỊ THỂ cục bộ (James-Stein trên đồ thị không gian).

        m_i   = trung bình nhúng của láng giềng không gian
        s2_i  = phương sai TRONG lân cận  -> cao ở ranh giới mô, thấp trong niche đồng nhất
        w_i   = sigma^2 / (sigma^2 + s2_i),   sigma^2 = c * trung vị(s2)
        z'_i  = z_i + w_i * (m_i - z_i)

    Ở ranh giới s2 lớn -> w nhỏ -> KHÔNG làm trơn (giữ tín hiệu thật). Trong vùng đồng nhất -> làm
    trơn mạnh (khử nhiễu). Đây là trung bình hậu nghiệm dưới mô hình Gauss cục bộ; hợp lệ vì Phần D
    đo được phần dư TRẮNG về không gian (Moran 0.007-0.017), tức nhiễu đẳng hướng theo toạ độ.

    sigma^2 cho dưới dạng BỘI của trung vị s2 nên c không thứ nguyên, khỏi phải ước lượng tuyệt đối.
    """
    Z = np.asarray(Z, np.float64)
    m = np.asarray(Wn @ Z)
    s2 = np.maximum(np.asarray(Wn @ (Z ** 2).sum(1)) - (m ** 2).sum(1), 0.0) / Z.shape[1]
    iso = np.asarray(Wn.sum(1)).ravel() <= 0
    med = float(np.median(s2[~iso])) if (~iso).any() else 1.0
    w = (c * med) / (c * med + np.maximum(s2, 1e-12))
    w[iso] = 0.0                                  # tế bào cô lập: giữ nguyên
    return (Z + w[:, None] * (m - Z)).astype(np.float32), w, s2


def part_k(combo, cx, S, save, W):
    """CỔNG: khử nhiễu không gian có nâng được LƯỢNG THÔNG TIN NHÃN trong nhúng nguồn không?

    Phần T đo trần: oracle tuyến tính chỉ hơn FROST 0.006-0.010 ở 3/4 cặp, trong khi Xenium tự
    chuyển cho mình đạt 0.91-0.94. Tức MỌI can thiệp ở tầng ánh xạ đều đã hết dư địa, và đòn bẩy
    duy nhất còn lại là chính BIỂU DIỄN. FM (scGPT-spatial / PCA) nhúng từng tế bào từ vector biểu
    hiện của nó và KHÔNG BAO GIỜ thấy toạ độ -> toạ độ là kênh thông tin nằm ngoài nhúng.

    BA NHÁNH, cùng một mức làm trơn trung bình, nếu không thì so sánh vô nghĩa:
      raw       — mốc, chính là tran_nguon của Phần T
      hoanvi    — làm trơn trên đồ thị dựng từ toạ độ ĐÃ HOÁN VỊ: sàn, phá thông tin không gian
      deu       — làm trơn ĐỀU với alpha = trung bình(w) của nhánh thích ứng
      thichung  — co rút thích ứng

    BẪY ĐÃ BIẾT: làm trơn khiến tế bào gần nhau giống nhau hơn, mà tế bào gần nhau VỐN thường cùng
    nhãn -> tran_nguon tăng một phần do rò rỉ tự tương quan nhãn, không phải do khử nhiễu. Nhánh
    `deu` hứng trọn phần rò rỉ đó. Chỉ phần `thichung` VƯỢT `deu` mới là đóng góp của tính thích ứng.
    """
    Sc = S.setdefault("K", {}).setdefault(combo, {})
    pctx, ds, Fs = cx["pctx"], cx["ds"], cx["Fs"]
    g = pctx["groups_ev"]; lab_s = np.asarray(pctx["lab_src_ev"])
    T_, Q_ = np.asarray(ds["T"]), np.asarray(cx["Q"])
    xy = lite_obs(cx["pair"])[["x", "y"]].to_numpy()
    Wn, _, _, cut, _ = spatial_graph(xy)
    rng = np.random.RandomState(0)
    Wn_p, _, _, _, _ = spatial_graph(xy[rng.permutation(len(xy))])

    def probe(Z):
        mu = np.asarray(Z[T_], np.float32).mean(0)
        nn = NearestNeighbors(n_neighbors=15, n_jobs=4).fit(_unit(np.asarray(Z[T_], np.float32) - mu)
             ).kneighbors(_unit(np.asarray(Z[Q_], np.float32) - mu), return_distance=False)
        return bacc_on(lab_s[Q_], mode_rows(lab_s[T_][nn]), g)

    if "raw" not in Sc:
        step(f"[{combo}] K: raw")
        Sc["raw"] = {"bacc": probe(Fs)}; save()
    for c in KC:
        k = f"c{c:g}"
        if k in Sc: continue
        step(f"[{combo}] K: {k}")
        Za, w, s2 = _lam_truot(Fs, Wn, c)
        a = float(w.mean())
        Zu = ((1 - a) * np.asarray(Fs, np.float64) + a * np.asarray(Wn @ np.asarray(Fs, np.float64))
              ).astype(np.float32)
        Zp, _, _ = _lam_truot(Fs, Wn_p, c)
        Sc[k] = {"thichung": probe(Za), "deu": probe(Zu), "hoanvi": probe(Zp),
                 "w_tb": round(a, 4), "w_p01": round(float(np.percentile(w, 1)), 4),
                 "w_min": round(float(w[w > 0].min()) if (w > 0).any() else 0.0, 4),
                 # ĐUÔI DƯỚI mới là chỗ tính thích ứng hoạt động: đa số tế bào nằm xa ranh giới nên
                 # p10-p90 hẹp là BÌNH THƯỜNG, không có nghĩa là co rút ~ đều (kiểm bằng số: biên có
                 # s2 gấp 3.4 lần, w thấp hơn ~30%, nhưng p10-p90 chỉ rộng 0.025).
                 "tl_bao_ve": round(float((w < 0.5 * a).mean()), 4),
                 "s2_cv": round(float(np.std(s2) / max(np.mean(s2), 1e-12)), 3)}
        save()
        v = Sc[k]
        log(f"[{combo}] K {k}: thichung={v['thichung']} deu={v['deu']} hoanvi={v['hoanvi']} "
            f"raw={Sc['raw']['bacc']} | w_tb={v['w_tb']} w_p01={v['w_p01']} "
            f"bảo_vệ={v['tl_bao_ve']:.1%} s2_cv={v['s2_cv']}")
        del Za, Zu, Zp; gc.collect()
    S.setdefault("K_meta", {})[combo] = {"cat_canh": round(cut, 4), "n_nhom": int(len(g)),
                                         "nhan": "thô" if pctx.get("use_coarse") else "mịn"}
    save(); del Wn, Wn_p; gc.collect()


# =============================================================================== Phần T
def bacc_on(y, p, keep):
    """Chép đúng run05.py:65-67 để số so sánh thẳng được với `xfer_main`."""
    m = np.isin(y, list(keep))
    return round(float(balanced_accuracy_score(y[m], p[m])), 4) if m.sum() else None


def part_t(combo, cx, S, save, W):
    """TRẦN của `xfer_main`: chuyển nhãn 15-NN khi KHÔNG có khoảng cách công nghệ.

    VÌ SAO: ở n=12, `xfer_main` KHÔNG phân biệt được bốn can thiệp khác hẳn nhau về cấu trúc —
    bỏ CORAL (p=0.79), usher_cc (p=0.85), usher_final (p=0.68), class_ridge_pl_cc (p=0.29) — trong
    khi cùng chỉ số đó tách FROST khỏi harmony/scanorama rất rõ. Đó là dấu hiệu bão hoà trong nội
    bộ họ USHER. Phần T đo xem trần nằm ở đâu, để biết `xfer_main` còn dùng làm chỉ số phân biệt
    được không, hay chỉ còn là ràng buộc "không được tụt".

    Oracle theo NHÃN là vô dụng ở đây: đặt mỗi tế bào nguồn cạnh một tế bào Xenium cùng nhãn thật
    thì kNN đương nhiên đúng -> ~1.0 một cách tầm thường. Nên đo hai đại lượng KHÔNG suy biến:
      tran_dich  — Xenium -> Xenium (ref -> phần giữ lại). Cùng chú giải, cùng nhúng, khoảng cách
                   công nghệ BẰNG 0. Đây là trần của chính thước đo.
      tran_nguon — nguồn -> nguồn (T -> Q) trong không gian nguồn. Đo lượng thông tin nhãn mà nhúng
                   nguồn tự mang. M-step là ánh xạ TUYẾN TÍNH nên nó không tạo ra được khả năng
                   phân tách vốn không có; đây là chặn mềm phía nguồn.
      *_ngau     — mốc ngẫu nhiên (hoán vị nhãn), để biết sàn.

    Dùng `lab_tgt_ev`/`lab_src_ev` + `groups_ev` của pctx, tức ĐÚNG nhãn và ĐÚNG nhóm mà
    `xfer_main` chấm (thô hay mịn tuỳ cặp), nếu không thì hai con số không so được với nhau.
    """
    Sc = S.setdefault("T", {}).setdefault(combo, {})
    # KHÔNG dùng `if Sc: return` ở đây: lần đầu viết hàm này chỉ có một khối việc, rồi khối oracle
    # được thêm vào CUỐI -> `if Sc: return` thấy tran_dich đã có là thoát ngay, oracle không bao giờ
    # chạy và KHÔNG ghi cảnh báo nào. Mỗi khối tự gác lấy khoá của nó.
    pctx, ds, Fx, Fs = cx["pctx"], cx["ds"], cx["Fx"], cx["Fs"]
    g = pctx["groups_ev"]
    lab_t, lab_s = np.asarray(pctx["lab_tgt_ev"]), np.asarray(pctx["lab_src_ev"])
    ref = np.asarray(pctx["ref"]); rng = np.random.RandomState(0)

    def knn_vote(Xfit, yfit, Xq, k=15):
        mu = np.asarray(Xfit, np.float32).mean(0)     # cùng giao thức với transferred_labels
        nn = NearestNeighbors(n_neighbors=min(k, len(Xfit)), n_jobs=4).fit(
            _unit(np.asarray(Xfit, np.float32) - mu)).kneighbors(
            _unit(np.asarray(Xq, np.float32) - mu), return_distance=False)
        return mode_rows(np.asarray(yfit)[nn])

    if "tran_dich" not in Sc:
        step(f"[{combo}] T: trần đích (Xenium->Xenium)")
        held = np.setdiff1d(np.arange(len(Fx)), ref)
        if len(held) == 0:
            W.append(f"{combo}: ref phủ toàn bộ Xenium -> không có phần giữ lại, bỏ trần đích")
        else:
            if len(held) > N_T_MAX: held = np.sort(rng.choice(held, N_T_MAX, replace=False))
            pr = knn_vote(Fx[ref], lab_t[ref], Fx[held])
            Sc["tran_dich"] = bacc_on(lab_t[held], pr, g)
            Sc["tran_dich_ngau"] = bacc_on(lab_t[held], rng.permutation(pr), g)
            Sc["n_ref"], Sc["n_giu_lai"] = int(len(ref)), int(len(held))
            save()

    if "tran_nguon" not in Sc:
        step(f"[{combo}] T: trần nguồn (nguồn->nguồn)")
        T_, Q_ = np.asarray(ds["T"]), np.asarray(cx["Q"])
        ps = knn_vote(Fs[T_], lab_s[T_], Fs[Q_])
        Sc["tran_nguon"] = bacc_on(lab_s[Q_], ps, g)
        Sc["tran_nguon_ngau"] = bacc_on(lab_s[Q_], rng.permutation(ps), g)
        Sc["n_nhom"] = int(len(g)); Sc["nhan"] = "thô" if pctx.get("use_coarse") else "mịn"
        save()

    # TRẦN THỨ BA — "ánh xạ tuyến tính tốt nhất nếu biết tương ứng LỚP hoàn hảo".
    # `class_ridge_pl_cc` của bảng chính dùng NHÃN GIẢ (cx["lab_def"], từ M_aux). Đổi sang nhãn THẬT
    # của nguồn là được oracle. KHÔNG suy biến về 1.0: base_class_ridge chỉ ghép theo lớp trên T rồi
    # hồi quy ridge, từng tế bào vẫn phải rơi đúng chỗ; và nó chỉ thấy nhãn của T, đánh giá trên Q.
    # Hiệu số oracle − nhãn giả = CÁI GIÁ CỦA VIỆC ĐOÁN SAI LỚP, tức đúng thứ M_niche của 1.0.3 nhắm vào.
    if "oracle_ridge_cc" not in Sc:
        if R20.KM16 is None:
            W.append("phần T: R20.KM16 chưa đặt -> bỏ oracle_ridge_cc"); log("phần T: THIẾU KM16, bỏ oracle")
        else:
            step(f"[{combo}] T: oracle ridge theo lớp (nhãn THẬT)")
            try:
                Fo = R30.M.base_class_ridge(Fs, cx["Fx"], ds["T"], np.asarray(ds["lab"]),
                                            np.asarray(cx["dx"]["lab"]))[0]
                Fo = R23C.coral(np.asarray(Fo, np.float32), cx["Fx"], ds["T"])   # khớp class_ridge_pl_cc (có CORAL)
                ev = R23.evaluate(cx, f"{combo}/T/oracle_ridge_cc", Fs, np.asarray(Fo, np.float32), cx["Fx"], W)
                Sc["oracle_ridge_cc"] = {k: ev.get(k) for k in
                                         ("xfer_main", "xfer_all", "total_v2", "batch_v2", "bio_v2")}
                save()
            except Exception as e:
                W.append(f"{combo}/T/oracle_ridge_cc: lỗi {e!r}"); log(f"[{combo}] T oracle LỖI {e!r}")
    log(f"[{combo}] T: {Sc}")


# =============================================================================== báo cáo
def report(S):
    L = [f"# 45 — chỉ số ST không chú giải + chẩn đoán phần dư (mô={TISSUE})",
         f"CẤU HÌNH: combos={COMBOS} seeds={SEEDS} k_spatial={K_SPATIAL} r_max={R_MAX_UM}µm "
         f"moran_min={MORAN_MIN} floor={FLOOR} | nhãn chuyển = bỏ phiếu 15-NN (đúng run22._knn_bacc)", ""]
    if S.get("D"):
        L += ["## Phần D — Moran's I của PHẦN DƯ (fit Θ trên T1, dư trên T2)", "```"]
        rows = []
        for c, sc in S["D"].items():
            for k, v in sc.items():
                rows.append({"tổ_hợp": c, "seed": k, **{kk: v[kk] for kk in
                            ("moran_tb", "moran_p90", "moran_hoanvi_tb", "n_T2", "canh_bi_cat")}})
        L += [pd.DataFrame(rows).to_string(index=False), "```"]
        for c, sc in S["D"].items():
            v = [x["moran_tb"] for x in sc.values()]
            h = [x["moran_hoanvi_tb"] for x in sc.values()]
            if not v: continue
            d = float(np.mean(v) - np.mean(h))
            L.append(f"- D1 {c}: Moran's I phần dư = {np.mean(v):.5f} so với hoán vị {np.mean(h):.5f} "
                     f"-> chênh {d:+.5f} -> {'CÓ tín hiệu không gian' if d >= 0.02 else 'không đáng kể'}")
        L.append("- Đọc: chênh < 0.02 nghĩa là phần dư gần như không mang cấu trúc không gian liên tục; "
                 "số hạng Laplacian trong 1.0.2 sẽ không có gì để tác động.")
        L.append("")
    if S.get("M"):
        L += ["## Phần M — Moran's I trên nhãn + PAS (|Δ| so với nhãn thật; THẤP hơn = trung thực hơn)", "```"]
        rows = []
        for c, sc in S["M"].items():
            for m, v in sc.items():
                rows.append({"tổ_hợp": c, "map": m, **{kk: v[kk] for kk in
                            ("moran_abs_dev", "moran_pred_mean", "moran_true_mean", "pas_pred", "pas_true",
                             "pas_abs_dev")}})
        L += [pd.DataFrame(rows).to_string(index=False), "```",
              "- CẢ HAI cột `*_abs_dev` là ĐỘ LỆCH so với nhãn thật, không phải giá trị thô: nhãn suy biến "
              "(gán tất cả cùng một loại) cho PAS = 0 và Moran's I = 0, nên 'thấp hơn là tốt hơn' trên giá trị "
              "thô sẽ thưởng cho nhãn vô dụng.", ""]
    if S.get("A"):
        L += ["## Phần A — sàng α: làm trơn không gian hậu kỳ (CHẶN TRÊN của 1.0.2)", "```"]
        rows = []
        for c, sc in S["A"].items():
            for k, v in sc.items():
                arm, a = k.split("@")
                rows.append({"tổ_hợp": c, "nhánh": arm, "α": float(a),
                             **{kk: v[kk] for kk in ("pas_abs_dev", "moran_abs_dev", "bacc_xfer", "d_cov")}})
        df = pd.DataFrame(rows).sort_values(["tổ_hợp", "nhánh", "α"])
        L += [df.to_string(index=False), "```",
              "- `pre` = coral(smooth(F)) là nhánh KHỚP với Laplacian ở M-step. α=0 của `pre` chính là "
              "FROST 1.0; α=0 của `nocoral` chính là u_cmix.",
              "- `d_cov` ở nhánh `pre` ≈ 0 với MỌI α một cách TẦM THƯỜNG (CORAL chạy sau cùng nên ép lại "
              "khớp mô-men). Đừng đọc đó là an toàn: nó chỉ có nghĩa cho `nocoral` và `post`.", ""]
        for c, sc in S["A"].items():
            b = sc.get("pre@0")
            if not b: continue
            # Tối ưu CÓ RÀNG BUỘC: lọc trước theo bacc rồi mới chọn pas_dev thấp nhất. Lấy min pas_dev
            # trước rồi mới kiểm bacc là SAI — một α hơi kém pas_dev nhưng giữ được bacc sẽ bị che mất.
            cands = [(v["pas_abs_dev"], k) for k, v in sc.items()
                     if k.startswith("pre@") and k != "pre@0" and v.get("pas_abs_dev") is not None]
            if not cands: continue
            ok_b = [t for t in cands if sc[t[1]]["bacc_xfer"] - b["bacc_xfer"] >= -0.02]
            if not ok_b:
                worst = min(cands)
                v_ = (f"MỌI α đều làm bacc tụt > 0.02 (tốt nhất {worst[1]}: "
                      f"{sc[worst[1]]['bacc_xfer'] - b['bacc_xfer']:+.4f}) -> bào vào tín hiệu sinh học: HỎNG")
            else:
                dp, kk = min(ok_b)
                dp -= b["pas_abs_dev"]; db = sc[kk]["bacc_xfer"] - b["bacc_xfer"]
                if dp > -0.01:
                    v_ = ("trong vùng giữ được bacc, pas_dev KHÔNG giảm (<0.01) -> Laplacian không cắn "
                          "qua được CORAL: 1.0.2 CHẾT")
                else:
                    v_ = (f"pas_dev giảm {-dp:.4f} tại {kk}, bacc đổi {db:+.4f} -> ĐƯỜNG SỐNG, "
                          f"chấm đầy đủ R23.evaluate ở α đó")
            L.append(f"- A1 {c}: {v_}")
        L += ["- Đối chiếu chéo: `nocoral` cải thiện mạnh mà `pre` thì không => CORAL xoá công làm trơn "
              "-> phải chuyển sang bản phạt có trọng số A (‖(Wxᵢ−Wxⱼ)A‖²), KHÔNG phải bỏ Laplacian.", ""]
    if S.get("B"):
        MB = ["xfer_main", "xfer_all", "total_v2", "batch_v2", "bio_v2",
              "pas_abs_dev", "moran_abs_dev"]
        gom = {}                                   # (tổ_hợp, nhánh) -> {chỉ số: [giá trị theo seed]}
        for c, sc in S["B"].items():
            for k, v in sc.items():
                nh = k.split("/", 1)[1] if "/" in k else k
                d = gom.setdefault((c, nh), {m: [] for m in MB})
                for m in MB:
                    if isinstance(v.get(m), (int, float)): d[m].append(v[m])
        rows = [{"tổ_hợp": c, "nhánh": nh, "n_seed": max(len(x) for x in d.values()),
                 **{m: (round(float(np.mean(d[m])), 4) if d[m] else None) for m in MB}}
                for (c, nh), d in sorted(gom.items())]
        L += ["## Phần B — chấm ĐẦY ĐỦ ở vài điểm α (TB qua seed)", "```",
              pd.DataFrame(rows).to_string(index=False), "```",
              "- `pre@0` CHÍNH LÀ FROST 1.0 — mốc so duy nhất đúng ở đây, đừng lấy `u_cmix_cc` của bảng 23e.", ""]
        TB = {(r["tổ_hợp"], r["nhánh"]): r for r in rows}
        for c in sorted({r["tổ_hợp"] for r in rows}):
            b = TB.get((c, "pre@0"))
            if not b: L.append(f"- B1 {c}: THIẾU mốc `pre@0` -> không kết luận được"); continue
            for nh in sorted(r["nhánh"] for r in rows if r["tổ_hợp"] == c):
                if nh == "pre@0": continue
                v = TB[(c, nh)]
                g = lambda m: (v.get(m) or 0) - (b.get(m) or 0)
                # TIÊU CHÍ: total_v2 không tụt quá 0.02, VÀ phần ST bù lại TƯƠNG XỨNG.
                # `mat` = phần total_v2 mất (0 nếu tăng); `bu` = trung bình mức giảm của hai chỉ số
                # ST (chúng là ĐỘ LỆCH nên giảm = tốt). Trung bình chứ không lấy max: cả hai chỉ số
                # đều được tính, nên một chỉ số xấu đi sẽ ăn vào phần bù thay vì bị bỏ qua.
                mat = max(0.0, -g("total_v2"))
                bu = (-g("pas_abs_dev") - g("moran_abs_dev")) / 2
                dat = g("total_v2") >= -0.02 and bu > 0 and bu >= mat
                L.append(f"- B1 {c} {nh} (n={v['n_seed']}): total_v2 {g('total_v2'):+.4f}  "
                         f"xfer_main {g('xfer_main'):+.4f}  batch_v2 {g('batch_v2'):+.4f}  | "
                         f"pas_dev {g('pas_abs_dev'):+.4f}  moran_dev {g('moran_abs_dev'):+.4f}  | "
                         f"bù {bu:+.4f} so với mất {mat:.4f}"
                         + ("  -> ĐẠT" if dat else
                            "  -> total tụt > 0.02" if g("total_v2") < -0.02 else
                            "  -> ST không cải thiện" if bu <= 0 else "  -> bù KHÔNG tương xứng"))
        # Wilcoxon ghép cặp trên (tổ_hợp, seed) — 4 cặp × 3 seed = n=12, đúng chuẩn của dự án.
        try:
            from scipy.stats import wilcoxon
            cap = {}
            for c, sc in S["B"].items():
                for k, v in sc.items():
                    if "/" not in k: continue
                    sd, nh = k.split("/", 1)
                    cap.setdefault(nh, {})[(c, sd)] = v
            goc = cap.get("pre@0", {})
            if goc:
                L += ["### Wilcoxon ghép cặp so với `pre@0` (n = số tổ_hợp × seed)", "```"]
                for nh in sorted(k for k in cap if k != "pre@0"):
                    for m in ("total_v2", "xfer_main", "pas_abs_dev", "moran_abs_dev"):
                        a = [(goc[t].get(m), cap[nh][t].get(m)) for t in cap[nh]
                             if t in goc and isinstance(goc[t].get(m), (int, float))
                             and isinstance(cap[nh][t].get(m), (int, float))]
                        if len(a) < 6: continue
                        x = np.array([u for u, _ in a]); y = np.array([v for _, v in a])
                        pv = float("nan") if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
                        huong = "THẤP hơn tốt" if m.endswith("_dev") else "CAO hơn tốt"
                        L.append(f"{nh:<10} {m:<14} n={len(a):<3} Δ={np.mean(y - x):+.4f}  "
                                 f"p={pv:.4f}  ({huong})")
                L += ["```", "- p >= 0.05 trên `total_v2` = GIỮ VỮNG (hoà). p < 0.05 với Δ ÂM trên "
                      "`pas_abs_dev`/`moran_abs_dev` = chỉ số ST cải thiện có ý nghĩa.", ""]
        except Exception as e:
            L.append(f"- (Wilcoxon bỏ qua: {e!r})")
        L += ["- Tiêu chí: `total_v2` không tụt quá **0.02** so với FROST 1.0, VÀ phần ST bù lại "
              "TƯƠNG XỨNG: `bù = trung bình(−Δpas_dev, −Δmoran_dev)` phải dương VÀ không nhỏ hơn "
              "`mất = max(0, −Δtotal_v2)`. `xfer_main` in ra để theo dõi, KHÔNG phải điều kiện — đọc "
              "nó thay cho `total_v2` chính là sai lầm đã chôn nhầm 1.0.2 một lần.",
              "- TB qua seed chỉ để xem nhanh; kết luận phải chạy Wilcoxon n=12 trên `stat33`.", ""]
    if S.get("T"):
        L += ["## Phần T — TRẦN của `xfer_main` (chuyển nhãn 15-NN, không khoảng cách công nghệ)", "```",
              pd.DataFrame(S["T"]).T.to_string(), "```",
              "- `tran_dich` = Xenium -> Xenium: trần của chính thước đo. So nó với `xfer_main` của FROST "
              "ở `out<P>e`. Sát nhau -> `xfer_main` đã bão hoà, chỉ còn dùng làm ràng buộc KHÔNG ĐƯỢC TỤT, "
              "không dùng để xếp hạng biến thể.",
              "- `tran_nguon` = nguồn -> nguồn: lượng thông tin nhãn nhúng nguồn tự mang. M-step là ánh xạ "
              "TUYẾN TÍNH nên không tạo ra khả năng phân tách vốn không có.",
              "- `*_ngau` là sàn (hoán vị nhãn dự đoán). Nhãn thô hay mịn theo cột `nhan` — đúng cái "
              "`xfer_main` dùng cho cặp đó.",
              "- `tran_nguon` CÓ giám sát (huấn luyện trên nhãn nguồn ở T); FROST không bao giờ thấy nhãn "
              "nguồn. Nên FROST thấp hơn nó không phải là điều đáng trách.", ""]
        orc = {c: (v.get("oracle_ridge_cc") or {}).get("xfer_main") for c, v in S["T"].items()}
        if any(x is not None for x in orc.values()):
            L += ["### Oracle: ridge theo lớp với nhãn THẬT (trần của ÁNH XẠ TUYẾN TÍNH)", "```",
                  pd.DataFrame({c: (v.get("oracle_ridge_cc") or {}) for c, v in S["T"].items()}).T.to_string(),
                  "```",
                  "- So với `class_ridge_pl_cc` (nhãn GIẢ) ở `out<P>e`: hiệu số = cái giá của việc đoán sai lớp.",
                  "- Oracle ≈ `xfer_main` của FROST → căn chỉnh tuyến tính đã sát trần của chính nó; "
                  "muốn hơn phải BỎ ràng buộc tuyến tính, không phải tinh chỉnh chi phí ghép cặp.",
                  "- Oracle cao hơn nhiều → phần lớn khoảng cách đến từ TƯƠNG ỨNG SAI, đúng chỗ ý tưởng A nhắm.", ""]
    if S.get("R"):
        L += ["## Phần R — FROST 1.0.3: CORAL QUAY theo không gian", "```"]
        rows = []
        for c, sc in S["R"].items():
            for k, v in sc.items():
                rows.append({"tổ_hợp": c.split("/")[0], "t": k, **{m: v.get(m) for m in
                            ("pas_abs_dev", "moran_abs_dev", "rough", "d_cov",
                             "xfer_main", "xfer_all", "total_v2", "batch_v2")}})
        L += [pd.DataFrame(rows).to_string(index=False), "```",
              "- `t=0` CHÍNH LÀ CORAL hiện hành (R = I) — mốc so duy nhất đúng. `t=1` là R* tối ưu Ruhe.",
              "- `d_cov` ≈ 0.01 ở MỌI t, KHÔNG phải 0: Ws dựng từ Sigma_s đã cộng RIDGE=0.01 nên "
              "Ws·Sigma_s_thô·Ws ≠ I đúng bằng lượng ridge. Thứ phải kiểm là nó **không đổi theo t** "
              "(bất biến quay). d_cov tăng theo t mới là lỗi code.",
              "- `rough` = tr(ZᵀLZ)/(n·d), hàm mục tiêu được tối thiểu. Chỉ đảm bảo `rough(t=1) < "
              "rough(t=0)`; ở t trung gian nó CÓ THỂ nhô lên vì nội suy cực phân rã không phải đường "
              "trắc địa trên SO(d) (đã thấy +0.1% tại t=0.25 khi thử bằng số). `rough(1) >= rough(0)` "
              "mới là dấu hiệu R* tính sai.", ""]
        for c, sc in S["R"].items():
            b = sc.get("t0")
            if not b: continue
            for k, v in sc.items():
                if k == "t0" or v.get("total_v2") is None or b.get("total_v2") is None: continue
                dp = (v.get("pas_abs_dev") or 0) - (b.get("pas_abs_dev") or 0)
                dt = v["total_v2"] - b["total_v2"]; dx = (v.get("xfer_main") or 0) - (b.get("xfer_main") or 0)
                ok = dp < -0.005 and dt >= -0.01
                L.append(f"- R1 {c} {k}: pas {dp:+.4f}  total_v2 {dt:+.4f}  xfer_main {dx:+.4f}"
                         + ("  -> ĐẠT (ST tốt lên, total giữ)" if ok else "  -> không đạt"))
        L += ["- Tiêu chí 1.0.3: `pas_abs_dev` giảm > 0.005 và `total_v2` không tụt quá 0.01 so với t=0. "
              "`xfer_main` in ra để biết giá phải trả, không dùng làm điều kiện.", ""]
    if S.get("K"):
        L += ["## Phần K — CỔNG: khử nhiễu không gian có nâng thông tin nhãn trong nhúng nguồn không?", "```"]
        rows = []
        for c, sc in S["K"].items():
            r0 = sc.get("raw", {}).get("bacc")
            for k, v in sc.items():
                if k == "raw": continue
                rows.append({"tổ_hợp": c.split("/")[0], "c": k, "raw": r0, "hoanvi": v["hoanvi"],
                             "deu": v["deu"], "thichung": v["thichung"],
                             "thichung-deu": round((v["thichung"] or 0) - (v["deu"] or 0), 4),
                             "w_tb": v["w_tb"], "w_p01": v["w_p01"],
                             "tl_bao_ve": v["tl_bao_ve"], "s2_cv": v["s2_cv"]})
        L += [pd.DataFrame(rows).to_string(index=False), "```",
              "- `raw` = chính `tran_nguon` của Phần T. `hoanvi` = làm trơn trên toạ độ đã hoán vị (sàn).",
              "- **Cột quyết định là `thichung-deu`.** `deu` có CÙNG mức làm trơn trung bình nên nó hứng "
              "trọn phần tăng do rò rỉ tự tương quan nhãn theo không gian. Chỉ phần vượt mới là đóng "
              "góp của TÍNH THÍCH ỨNG — tức của ý tưởng, chứ không phải của việc làm trơn nói chung.",
              "- Tính thích ứng hoạt động ở ĐUÔI DƯỚI của `w`, không ở phần giữa: đa số tế bào nằm xa "
              "ranh giới mô nên p10-p90 hẹp là BÌNH THƯỜNG. Hai cột cần nhìn là `w_p01` (mức bảo vệ "
              "ở tế bào dị thể nhất) và `tl_bao_ve` = tỉ lệ tế bào có w < 0.5·w_tb. `tl_bao_ve` ~ 0 "
              "mới nghĩa là co rút thực sự ĐỀU và ý tưởng rỗng.",
              "- Kiểm bằng số trên dữ liệu tổng hợp có ranh giới rõ: s2 ở biên gấp 3.4 lần, w ở biên "
              "thấp hơn ~30%, MSE so với trạng thái thật: thô 1.004 / đều 0.144 / thích ứng 0.110.",
              "- CẢNH BÁO ĐỌC: cổng này chỉ nói về thông tin trong ĐẶC TRƯNG NGUỒN. Việc khử nhiễu có "
              "giúp CĂN CHỈNH hay không thì phải dựng lại map (GPU) mới biết. Nó dùng để GIẾT ý tưởng "
              "rẻ, không phải để xác nhận.", ""]
    if S.get("S"):
        L += ["## Phần S — THANG ĐO: pas_true theo bán kính lân cận, đo bằng đơn vị đo được", "```"]
        rows = []
        for c, sc in S["S"].items():
            for k, v in sc.items():
                g_, kk_ = k.rsplit("_k", 1)
                rows.append({"tổ_hợp": c.split("/")[0], "đồ_thị": g_, "k": int(kk_), **v})
        df = pd.DataFrame(rows).sort_values(["tổ_hợp", "đồ_thị", "k"])
        L += [df.to_string(index=False), "```",
              "- `d1_um` = trung vị khoảng cách tới láng giềng GẦN NHẤT = khoảng cách giữa các tế bào, "
              "ĐO ĐƯỢC. `dk_tren_d1` = bán kính lân cận tính theo đơn vị đó — không cần giả định đường "
              "kính tế bào.",
              "- Hàng `đồ_thị=toanbo` là thang mà **Phần K** dùng; hàng `Q` là thang mà **Phần M** "
              "(pas_true 0.21-0.40 đã công bố) dùng. Trước đây tôi lấy số của hàng này giải thích hàng "
              "kia — SAI THANG, Phần S có để sửa đúng chỗ đó.",
              "- PHÁN QUYẾT: `pas_true` ở `toanbo` với k nhỏ vẫn ~0.2-0.4 -> mô dị thể thật ở thang tế "
              "bào, kết quả âm của Phần K có lời giải, hướng ST đóng. Tụt về ~0.05-0.1 -> lân cận chặt "
              "CÓ đồng nhất, Phần K hỏng vì lý do khác, và việc đóng hướng ST là vội.", ""]
    if S.get("V"):
        L += ["## Phần V — gộp BỀN VỮNG thay cho trung bình", "```"]
        PAS = {"cosmx6k": .378, "visiumhd_ff": .288, "visiumhd_ffpe": .258, "stereoseq": .197}
        rows, hieu = [], {}
        for c, sc in S["V"].items():
            nm = c.split("/")[0]; r0 = sc.get("raw", {}).get("bacc")
            for cc in sorted({k.split("_c")[1] for k in sc if k != "raw"}, key=float):
                d = {"tổ_hợp": nm, "c": cc, "raw": r0, "pas_true": PAS.get(nm)}
                for how in ("tb", "trungvi"):
                    d[how] = sc.get(f"{how}_c{cc}", {}).get("bacc")
                if d.get("tb") is not None and d.get("trungvi") is not None:
                    d["trungvi-tb"] = round(d["trungvi"] - d["tb"], 4)
                    hieu.setdefault(nm, []).append(d["trungvi-tb"])
                rows.append(d)
        L += [pd.DataFrame(rows).to_string(index=False), "```",
              "- `w` giống HỆT nhau giữa các nhánh (s2 tính quanh trung bình, một lần). Biến duy nhất "
              "là ước lượng vị trí -> chênh lệch quy thẳng về nó.", ""]
        if hieu:
            xh = sorted(hieu.items(), key=lambda t: -PAS.get(t[0], 0))
            L.append("- DỰ ĐOÁN đặt TRƯỚC: mức hơn của trung vị phải TĂNG THEO `pas_true`. Thứ tự "
                     "pas_true giảm dần: " + " > ".join(k for k, _ in xh))
            L.append("  Mức hơn tốt nhất theo thứ tự đó: "
                     + " | ".join(f"{k} {max(v):+.4f}" for k, v in xh))
            best = [max(v) for _, v in xh]
            L.append("  -> " + ("ĐƠN ĐIỆU GIẢM, khớp dự đoán: nhiễm bẩn là nguyên nhân"
                                if all(best[i] >= best[i + 1] for i in range(len(best) - 1))
                                else "KHÔNG đơn điệu -> giả thuyết nhiễm bẩn KHÔNG được số liệu đỡ; "
                                     "không được diễn giải lại sau khi nhìn kết quả"))
        L += ["- Mốc phải vượt là `raw`, không phải `tb`: trung vị hơn trung bình mà cả hai vẫn dưới "
              "`raw` thì vẫn là làm trơn có hại, chỉ là hại ít hơn.", ""]
    if S.get("W"):
        L += ["## Phần W — tự tương quan không gian làm bộ lọc CHIỀU (không phải bộ lọc tế bào)", "```"]
        rows = []
        for c, v in S["W"].items():
            r = {"tổ_hợp": c.split("/")[0], "d": v.get("d"),
                 "moran_ng_p50": (v.get("moran_nguon") or {}).get("p50"),
                 "moran_ng_p90": (v.get("moran_nguon") or {}).get("p90"),
                 "moran_ng_max": (v.get("moran_nguon") or {}).get("p100"),
                 "sàn_hoánvị": (v.get("sd_hoanvi") or {}).get("nguon"),
                 "tl_có_tín_hiệu": (v.get("tl_chieu_co_tin_hieu") or {}).get("nguon"),
                 "SPEARMAN_ng_đích": v.get("spearman_nguon_dich"),
                 "spearman_moran_var": v.get("spearman_moran_var"),
                 "bacc_toànbộ": v.get("bacc_toanbo")}
            for k in sorted(v):
                if k.startswith("bacc_moran_top") or k.startswith("bacc_var_top"): r[k[5:]] = v[k]
            rows.append(r)
        L += [pd.DataFrame(rows).to_string(index=False), "```",
              "- **Cổng là `SPEARMAN_ng_đích`**: tương quan giữa Moran theo chiều của nguồn và của "
              "Xenium. Hai bên cùng cơ sở nên chiều so được. CAO -> 'chiều có cấu trúc không gian' là "
              "tính chất chung, độc lập công nghệ, đánh trọng số theo nó có nghĩa. ~0 -> cấu trúc "
              "không gian là đặc thù công nghệ, trọng số sẽ tuỳ tiện và ý tưởng chết.",
              "- `tl_có_tín_hiệu` = tỉ lệ chiều có Moran > 3·sàn hoán vị. ~0 -> không có gì để lọc.",
              "- `spearman_moran_var` cao -> chiều Moran cao cũng là chiều phương sai cao, nên đánh "
              "trọng số gần như không đổi gì. Thấp hoặc âm -> không gian surface được chiều mà phương "
              "sai bỏ sót; ĐÓ mới là chỗ có lợi.",
              "- `moran_topN` so với `var_topN`: phép thử TRỰC TIẾP 'chọn chiều theo không gian có hơn "
              "chọn theo phương sai không'. Đây là bằng chứng, không phải suy luận gián tiếp.", ""]
    if S.get("X"):
        L += ["## Phần X — FROST 1.0.4 Spatial-CORAL (khớp mô-men bậc hai CÓ TRỌNG SỐ KHÔNG GIAN)", "```"]
        rows = []
        for c, sc in S["X"].items():
            mt = S.get("X_meta", {}).get(c, {})
            for k, v in sc.items():
                rows.append({"tổ_hợp": c, "nhánh": k, "sàn_d_sp": mt.get("d_sp_san"),
                             **{kk: v.get(kk) for kk in ("d_sp", "d_cov", "pas_abs_dev", "moran_abs_dev",
                                "xfer_main", "total_v2", "batch_v2", "bio_v2")}})
        L += [pd.DataFrame(rows).to_string(index=False), "```"]
        if S.get("X_meta"):
            L += ["", "```", pd.DataFrame(S["X_meta"]).T.to_string(), "```"]
        L += ["", "CÁCH ĐỌC — ba cổng, theo thứ tự; trượt cổng nào thì dừng tại đó:",
              "- CỔNG 1 `d_sp_san` so với `d_sp_t0`: sàn là chặn dưới trên TOÀN nhóm trực giao. "
              "Sàn ≈ t0 -> phép quay về nguyên tắc không sửa được lệch mô-men không gian, 1.0.4 chết "
              "bất kể các số còn lại. Sàn << t0 -> có chỗ để sửa.",
              "- CỔNG 2 `d_cov` phải đứng yên ở mức RIDGE≈0.01 với MỌI t. Tăng theo t = code sai, "
              "KHÔNG phải phát hiện khoa học (bài học phần R: RIDGE phá bất biến chuẩn ở d=512).",
              "- CỔNG 3 `hoanvi@t1` so với `t1`: R* dựng từ đồ thị ĐÃ HOÁN VỊ. Ngang nhau -> thứ mua "
              "được là bậc tự do quay nói chung, KHÔNG phải thông tin không gian; hết tính mới.",
              "- Chỉ khi qua cả ba mới đọc tới `total_v2`/`xfer_main`. Mốc là `t0` (= FROST khi "
              "SPC_TAG=cmix), KHÔNG phải bảng u_cmix_cc của vòng 23e.", ""]
    if S.get("Y"):
        L += ["## Phần Y — CỔNG CPU cho C_sp (toạ độ làm ma trận cấu trúc GW kiểu PASTE)", "```"]
        rows = []
        for c, sc in S["Y"].items():
            for ben in ("nguon", "dich"):
                v = sc.get(ben, {})
                rows.append({"tổ_hợp": c, "bên": ben, "L_µm": v.get("L_um"), "n_vá": v.get("n_ban_va"),
                             "n_tb": v.get("n_te_bao_tb"), "d1_µm": v.get("d1_um"),
                             "tk_kgian": v.get("tk_kgian"), "tk_biểu_hiện": v.get("tk_bieuhien"),
                             "chênh": v.get("chenh_kgian_tru_bieuhien")})
        L += [pd.DataFrame(rows).to_string(index=False), "```", "", "```"]
        rows = []
        for c, sc in S["Y"].items():
            r = {"tổ_hợp": c, "chồng_lấp_sp": sc.get("chong_lap_sp"), "chồng_lấp_ex": sc.get("chong_lap_ex"),
                 "d1_tỉ_lệ": sc.get("d1_ti_le_nguon_tren_dich")}
            for k in ("sp", "kgian_ngau", "ex"):
                g = sc.get(f"gw_{k}") or {}
                r[f"gw_{k}"] = g.get("gw_dist"); r[f"đồng_ý_{k}"] = g.get("dong_y")
                r[f"lợi_{k}"] = g.get("loi_tren_ngau_nhien")
            rows.append(r)
        L += [pd.DataFrame(rows).to_string(index=False), "```"]
        L += ["", "CÁCH ĐỌC — ba cổng; đây là cổng CPU đặt TRƯỚC việc sửa src_cuda và chạy lại USHER trên GPU:",
              "- CỔNG 1 `chênh = tk_kgian − tk_biểu_hiện`, TRÊN CẢ HAI BÊN. Việc duy nhất của ma trận "
              "cấu trúc là nói 'hai tế bào này gần nhau về cấu trúc'. Chênh âm rõ ở cả hai bên -> "
              "biểu hiện đã nói chính xác hơn toạ độ, trộn toạ độ vào C chỉ làm bẩn. Đúng logic đã "
              "giết Phần W (phương sai thắng Moran).",
              "- CỔNG 2 `chồng_lấp_sp` (lược đồ khoảng cách sau chuẩn hoá D/D.max(), nguồn vs đích). "
              "Thấp, hoặc `d1_tỉ_lệ` lệch xa 1 -> hai bên ở hai thang mật độ khác nhau, GW sẽ khớp "
              "bằng cách CO GIÃN mật độ chứ không khớp sinh học.",
              "- CỔNG 3 `lợi_sp` so với `lợi_kgian_ngau` và `lợi_ex`: đồng ý nhãn của kế hoạch GW "
              "trừ mức ngẫu nhiên. `lợi_sp` ≈ `lợi_kgian_ngau` -> cấu trúc không gian không mang gì. "
              "`lợi_sp` << `lợi_ex` -> C_sp thua C biểu hiện, chỉ đáng trộn với λ nhỏ (PASTE thấy "
              "α=0.1 tốt nhất, α=1 khớp đúng 0 cặp).",
              "- Null là HÌNH HỌC NGẪU NHIÊN, không phải hoán vị: GW bất biến với hoán vị đồng thời "
              "hàng-cột nên 'hoán vị toạ độ' là null rỗng.", ""]
    if S.get("M_meta"):
        L += ["## Đồ thị không gian", "```", pd.DataFrame(S["M_meta"]).T.to_string(), "```", ""]
    if S.get("A_meta"):
        L += ["## Đồ thị không gian — phần A", "```", pd.DataFrame(S["A_meta"]).T.to_string(), "```", ""]
    L += ["## Cảnh báo"] + ([f"- {w}" for w in dict.fromkeys(S.get("warnings", []))] or ["- không có"])
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    log("\n".join(L))


def main():
    assert COMBOS, "COMBOS_ALL rỗng"
    os.makedirs(OUT, exist_ok=True)
    # PHẢI là việc ĐẦU TIÊN: run30.py:47 gán OUT của mọi module về out<P>i ngay lúc import, nên mọi
    # hàm có cache (get_niche_label_comp, setup_pairs, get_pair_niche...) sẽ ghi vào thư mục đó nếu
    # gọi trước khi khôi phục -> FileNotFoundError: out<P>i/labcomp_*.npy.
    for m in (R17, R18, R19, R20, R22, R22.R21, R23, R23C, R26G):
        if hasattr(m, "OUT"): m.OUT = OUT
    log(f"45 [{VER}] | mô={TISSUE} | combos={COMBOS} | phần={sorted(PARTS)} | OUT={OUT}")
    W0 = []
    import_cache(W0)     # PHẢI trước get_niche_label_comp ở dưới, nếu không nó dựng lại từ đầu
    S = json.load(open(f"{OUT}/summary.json")) if os.path.exists(f"{OUT}/summary.json") else {}
    if not S:
        for fj in sorted(glob.glob(f"/kaggle/input/**/{os.path.basename(OUT)}/summary.json", recursive=True)):
            S = json.load(open(fj)); log(f"tiếp nối <- {fj}"); break

    def save(): json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=str)

    needed = sorted({c.split("/")[0] for c in COMBOS})
    D, F, PCTX = R18.setup_pairs(needed)
    from sklearn.cluster import KMeans
    km8 = KMeans(n_clusters=R20.K_NICHE_TYPES, random_state=0, n_init=10).fit(
        R17.get_niche_label_comp(needed[0], S, save)[1])
    R22.G.update(D=D, F=F, PCTX=PCTX, S=S, save=save, kmeans8=km8, pca_cache={})
    if PARTS & {"B", "T", "R", "X"}:
        # BẮT BUỘC cho phần B VÀ T: R23.evaluate -> R20.etrack -> eval_e_track(..., KM16, ...) -> KM16.predict.
        # run20.py:101 khai KM16 = None và CHỈ run20.main() đặt; sáu module khác đều phải tự đặt lại
        # (run21:308, run22:346, run23:168, run23c:40, run26d:147, run26g:150 — run21:306 ghi rõ đây là
        # lỗi từng mắc). Chỉ đặt khi chạy B vì fine16_label_comp có thể phải dựng cache, phần D/M/A không cần.
        step("dựng KMeans niche mịn 16 (cho phần B/T/R/X)")
        R20.KM16 = R20.KMeans(n_clusters=R20.K_FINE16, random_state=0, n_init=10).fit(
            R20.fine16_label_comp(needed[0], S, save)[1])
    R22.PRIOR.clear(); R22.scan_maps(P, R22.PRIOR, log)   # bài học 42: PRIOR phải quét trước get_map
    W = S.setdefault("warnings", []); W.extend(W0)
    S["config"] = (f"45 mô={TISSUE} combos={COMBOS} r_max={R_MAX_UM} moran_min={MORAN_MIN} k={K_SPATIAL} "
                   f"phần={sorted(PARTS)} alphas={ALPHAS} alpha_tag={ALPHA_TAG} B={PART_B_ARMS}")
    try:
        # Mỗi phần bọc riêng: trước đây main CHỈ bắt OutOfTime, nên một lỗi bất kỳ trong một phần
        # giết cả lần chạy và KHÔNG ghi lại gì — đã mất nhiều giờ vì chuyện này. Giờ lỗi của một
        # phần/một cặp được ghi thành cảnh báo kèm traceback rút gọn, các phần còn lại vẫn chạy.
        import traceback
        for combo in COMBOS:
            cx = R22.ctx(combo)
            for ten, fn in (("D", part_d), ("M", part_m), ("A", part_a), ("B", part_b),
                            ("T", part_t), ("R", part_r), ("K", part_k), ("S", part_s),
                            ("V", part_v), ("W", part_w), ("X", part_x), ("Y", part_y)):
                if ten not in PARTS: continue
                try:
                    fn(combo, cx, S, save, W)
                except OutOfTime:
                    raise
                except Exception as e:
                    tb = traceback.format_exc().strip().splitlines()
                    W.append(f"{combo}/phần {ten}: LỖI {e!r} | {tb[-2].strip() if len(tb) > 1 else ''}")
                    log(f"[{combo}] phần {ten} LỖI:\n" + "\n".join(tb[-6:]))
                    save()
    except OutOfTime as e:
        W.append(f"HẾT GIỜ tại {e}"); log(f"HẾT GIỜ: {e}")
    S["warnings"] = W; save()
    report(S)


if __name__ == "__main__":
    main()
