"""Sinh 45 — chỉ số ST không chú giải (Moran's I trên nhãn, PAS) + chẩn đoán Moran's I của phần dư.
python build45.py            (kiểm run45.py)
python build45.py --write    (sinh notebook)

MỘT bước, CPU là đủ: không FGW, không huấn luyện, chỉ nạp map đã lưu rồi tính.
Kết quả: out<P>s/summary.json + summary.md.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build23 as B3
import build26d as B26
import build39 as B39
import build41 as B41
B = B3.B
NB_DIR = B3.NB_DIR
_NEW = ["run30.py", "run45.py"]
EXTRA = sorted(set(B26.EXTRA + _NEW), key=(B26.EXTRA + _NEW).index)

PFX = {"coad": "23", "ov": "26", "hcc": "27"}
COMBOS = {"coad": B26.COMBOS_COAD}
NB = {"coad": "45a_coad_st_metrics", "ov": "45b_ov_st_metrics", "hcc": "45c_hcc_st_metrics"}


def build(tissue="coad"):
    p = PFX[tissue]
    env = {"tissue": tissue, "mode": "report", "combos": COMBOS.get(tissue, ""), "p": p, "xmaps": "cmix"}
    c1 = (B26.C1_COAD % env).replace(
        'os.environ.setdefault("EXTRA_MAPS", "%s")' % "cmix",
        '\n'
        f'os.environ.setdefault("OUT_DIR_45", "/kaggle/working/out{p}s")\n'
        '# Phần cần chạy: "D" = chẩn đoán phần dư (cổng quyết định FROST 1.0.2),\n'
        '#               "M" = Moran\'s I trên nhãn + PAS,\n'
        '#               "A" = sàng α (chặn trên của FROST 1.0.2, RẺ),\n'
        '#               "B" = chấm ĐẦY ĐỦ R23.evaluate ở vài điểm α (ĐẮT, ~1-2 giờ),\n'
        '#               "T" = TRẦN của xfer_main (RẺ),\n'
        '#               "R" = CORAL QUAY theo không gian (CPU, ~1 giờ),\n'
        '#               "K" = CỔNG khử nhiễu không gian trên nhúng nguồn (CPU, ~15 phút),\n'
        '#               "S" = THANG ĐO: pas_true theo bán kính lân cận (CPU, ~10 phút),\n'
        '#               "V" = gộp BỀN VỮNG (trung vị) thay trung bình (CPU, ~20 phút),\n'
        '#               "W" = Moran theo CHIỀU: không gian làm bộ lọc chiều (CPU, ~25 phút).\n'
        '# CHẠY LẠI: Add output cũ của 45a làm Input rồi đặt ONLY_PART="A" — run45 nạp lại summary.json\n'
        '# từ Input và BỎ QUA mọi mục đã có, nên D/M không bị chấm lại.\n'
        'os.environ.setdefault("ONLY_PART", "D,M,A")\n'
        '# Cắt cạnh không gian dài hơn ngần này (µm). BẮT BUỘC: Q lấy từ tile giữ lại của bàn cờ 500µm,\n'
        '# không cắt thì hai tế bào ở hai tile khác nhau thành "láng giềng" qua một khe trống.\n'
        'os.environ.setdefault("R_MAX_UM", "100")\n'
        '# Chỉ chấm Moran\'s I trên nhãn CÓ tổ chức không gian (I của nhãn thật >= ngưỡng này).\n'
        'os.environ.setdefault("MORAN_MIN", "0.10")\n'
        'os.environ.setdefault("K_SPATIAL", "15")\n'
        '# Phần A: hệ số làm trơn không gian hậu kỳ, và map nền (cmix = FROST CHƯA CORAL).\n'
        'os.environ.setdefault("ALPHAS", "0.05,0.1,0.2,0.3,0.4,0.5")\n'
        'os.environ.setdefault("ALPHA_TAG", "cmix")\n'
        '# Phần B: các điểm α đem đi chấm đầy đủ. pre@0 = FROST 1.0, PHẢI giữ để có mốc so.\n'
        'os.environ.setdefault("PART_B_ARMS", "pre@0,pre@0.2,pre@0.3,pre@0.4,pre@0.5")\n'
        '# Phần T: cỡ tập Xenium giữ lại để đo trần.\n'
        'os.environ.setdefault("N_T_MAX", "20000")\n'
        '# Phần R: map nền (cmix_levns = FROST 1.0.1 leverage thuần, CHƯA CORAL), lưới t, và các t\n'
        '# được chấm ĐẦY ĐỦ bằng R23.evaluate (t khác chỉ có PAS/Moran, rẻ).\n'
        'os.environ.setdefault("ROT_TAG", "cmix_levns")\n'
        'os.environ.setdefault("ROT_TS", "0,0.25,0.5,0.75,1")\n'
        'os.environ.setdefault("ROT_FULL", "0,0.5,1")\n'
        '# Phần K: sigma^2 = c * trung vị(s^2). c không thứ nguyên nên khỏi ước lượng nhiễu tuyệt đối.\n'
        'os.environ.setdefault("KC", "0.25,0.5,1,2,4")\n'
        '# Phần S: các k để quét bán kính lân cận, trên CẢ đồ thị toàn bộ tế bào lẫn đồ thị Q.\n'
        'os.environ.setdefault("SK_KS", "5,10,15,30")\n'
        '# Phần V: ước lượng vị trí của lân cận. tb = trung bình (thứ mọi thứ đã dùng), trungvi = trung vị.\n'
        'os.environ.setdefault("AGGS", "tb,trungvi")\n'
        '# Phần W: số chiều giữ lại khi so chọn-theo-Moran với chọn-theo-phương-sai.\n'
        'os.environ.setdefault("WN_TOP", "10,25,50")\n'
        '# Phần X (FROST 1.0.4 Spatial-CORAL): map nền PHẢI là bản CHƯA CORAL — cmix cho t=0 = FROST.\n'
        'os.environ.setdefault("SPC_TAG", "cmix")\n'
        'os.environ.setdefault("SPC_TS", "0,0.25,0.5,0.75,1")\n'
        'os.environ.setdefault("SPC_FULL", "0,0.5,1")\n'
        '# cmix = FROST (out<P>j) · usher_orig = USHER BẢN GỐC 25 vòng, 1 seed (out<P>n, vòng 38 nhánh B1)\n'
        '# cmix_lev/cmix_levns = FROST 1.0.1 (out<P>v, vòng 46) -> cần Add out<P>v làm Input.\n'
        'os.environ.setdefault("EXTRA_MAPS", "cmix,usher_orig,cmix_lev,cmix_levns")') + B41.need_check(
            tissue, need_model=False, need_report=False)
    run = B26.RUN.replace("run26d.py", "run45.py").replace("run26d.log", "run45.log").replace(
        " WITH_PROC=$WITH_PROC", "").replace(
        "MODE26=$MODE26 ", "OUT_DIR_45=$OUT_DIR_45 ONLY_PART=$ONLY_PART R_MAX_UM=$R_MAX_UM "
                           "MORAN_MIN=$MORAN_MIN K_SPATIAL=$K_SPATIAL ALPHAS=$ALPHAS ALPHA_TAG=$ALPHA_TAG "
                           "PART_B_ARMS=$PART_B_ARMS N_T_MAX=$N_T_MAX "
                           "ROT_TAG=$ROT_TAG ROT_TS=$ROT_TS ROT_FULL=$ROT_FULL KC=$KC SK_KS=$SK_KS AGGS=$AGGS WN_TOP=$WN_TOP "
                           "SPC_TAG=$SPC_TAG SPC_TS=$SPC_TS SPC_FULL=$SPC_FULL "
                           "SKIP_INPUT_CHECK=1 ")
    md = (f"# 45 — chỉ số ST không chú giải + chẩn đoán phần dư ({tissue.upper()})\n\n"
          "Chỉ nạp map đã lưu rồi tính. **CPU là đủ, không cần GPU.**\n\n"
          "## Phần D — chẩn đoán, chạy TRƯỚC khi quyết định làm FROST 1.0.2\n\n"
          "Bản **liên tục** của `run20` Phần D. Vòng 20 đo η²(niche) của phần dư (cấu trúc theo **lớp**) "
          "và được 0.001–0.005. Nhưng phần dư có thể có tự tương quan **không gian** mà không có cấu trúc "
          "theo lớp — hai thứ khác nhau. Ở đây fit Θ trên T1, lấy phần dư trên T2, đo Moran's I của từng "
          "chiều phần dư theo toạ độ vật lý, so với mốc hoán vị toạ độ.\n\n"
          "- Chênh so với hoán vị **< 0.02** → phần dư không mang cấu trúc không gian liên tục; "
          "số hạng phạt Laplacian của 1.0.2 **không có gì để tác động** → đóng hướng đó bằng số liệu.\n"
          "- Chênh **≥ 0.02** → còn tín hiệu không gian chưa sửa → số hạng mục tiêu có cơ sở.\n\n"
          "## Phần M — hai chỉ số ST mới cho MỌI phương pháp trong bảng chính\n\n"
          "Nhãn chuyển sang = bỏ phiếu 15-NN nhãn tham chiếu (đúng `run22._knn_bacc`), rồi so **tổ chức "
          "không gian** của nhãn chuyển với nhãn **thật**:\n\n"
          "- **Moran's I trên nhãn** — từng loại tế bào, chỉ báo nhị phân, trọng số từ toạ độ.\n"
          "- **PAS** — tỉ lệ tế bào có nhãn khác đa số láng giềng **không gian**.\n\n"
          "Cả hai báo cáo dưới dạng **|Δ so với nhãn thật|**, không phải giá trị thô: gán tất cả cùng một "
          "nhãn cho PAS = 0 và Moran's I suy biến, nên \"thấp hơn là tốt hơn\" trên giá trị thô sẽ thưởng "
          "cho nhãn vô dụng.\n\n"
          "## Phần A — sàng α: chặn trên của FROST 1.0.2\n\n"
          "Làm trơn không gian **hậu kỳ** map FROST rồi chấm lại. Nghiệm của "
          "`min_Z ‖Z−Z₀‖² + α·tr(ZᵀLZ)` là `(I+αL)⁻¹Z₀ ≈ (1−α)Z₀ + α·W_sp·Z₀` — đúng **toán tử gần kề** "
          "của số hạng phạt Laplacian định thêm vào M-step.\n\n"
          "**Đây là CHẶN TRÊN, không phải 1.0.2.** Làm trơn hậu kỳ tác động theo **từng tế bào**; "
          "M-step phải thể hiện mọi thứ qua **một ma trận W tuyến tính duy nhất**, tự do ít hơn hẳn. Nên: "
          "đường α hỏng → 1.0.2 **chắc chắn** hỏng, dừng; đường α đẹp → 1.0.2 *có thể* được, lúc đó mới "
          "đáng viết code.\n\n"
          "Ba nhánh — α=0 của `pre` chính là **FROST 1.0**, α=0 của `nocoral` chính là **u_cmix**:\n\n"
          "| nhánh | công thức | nghĩa |\n|---|---|---|\n"
          "| `nocoral` | `smooth(F)` | tác dụng thuần của làm trơn |\n"
          "| `pre` | `coral(smooth(F))` | **khớp với Laplacian ở M-step** |\n"
          "| `post` | `smooth(coral(F))` | chữa triệu chứng trực tiếp |\n\n"
          "**Bẫy đọc `d_cov`:** ở nhánh `pre` nó ≈ 0 với **mọi** α một cách tầm thường, vì CORAL chạy sau "
          "cùng nên ép lại `Σ(Z[T]) = Σ_x`. Đừng đọc đó là \"an toàn\" — `d_cov` chỉ có nghĩa cho "
          "`nocoral` và `post`. Với `pre`, rủi ro trộn lô phải chấm bằng `R23.evaluate` (**chưa làm ở "
          "đây**): Σ khớp không bảo đảm iLISI khớp, vì iLISI phụ thuộc trộn lô **cục bộ**.\n\n"
          "## Phần T — trần của `xfer_main`\n\n"
          "Ở n=12, `xfer_main` **không phân biệt được** bốn can thiệp khác hẳn nhau về cấu trúc "
          "(bỏ CORAL p=0.79 · `usher_cc` p=0.85 · `usher_final` p=0.68 · `class_ridge_pl_cc` p=0.29), "
          "trong khi cùng chỉ số đó tách FROST khỏi harmony (+0.030) và scanorama (+0.099) rất rõ. "
          "Đó là dấu hiệu **bão hoà trong nội bộ họ USHER**. Phần T đo trần nằm ở đâu.\n\n"
          "Oracle theo nhãn là vô dụng (đặt tế bào nguồn cạnh tế bào Xenium cùng nhãn thì kNN đương "
          "nhiên đúng → ~1.0 tầm thường). Nên đo hai đại lượng không suy biến:\n\n"
          "- **`tran_dich`** — Xenium → Xenium (ref → phần giữ lại). Cùng chú giải, cùng nhúng, "
          "khoảng cách công nghệ **bằng 0**. Trần của chính thước đo.\n"
          "- **`tran_nguon`** — nguồn → nguồn (T → Q) trong không gian nguồn. M-step là ánh xạ "
          "**tuyến tính** nên không tạo ra khả năng phân tách vốn không có.\n"
          "- **`*_ngau`** — hoán vị nhãn dự đoán, cho biết sàn.\n\n"
          "Dùng `lab_tgt_ev`/`lab_src_ev` + `groups_ev`, tức **đúng nhãn và đúng nhóm** mà `xfer_main` "
          "chấm cho cặp đó (thô hay mịn), nếu không hai con số không so được.\n\n"
          "**Đọc:** `tran_dich` sát `xfer_main` của FROST (xem `out23e`) → `xfer_main` bão hoà, từ nay "
          "chỉ dùng làm ràng buộc *không được tụt*, không dùng xếp hạng biến thể.\n\n"
          "## Phần R — FROST 1.0.3: CORAL quay theo không gian\n\n"
          "Khuyết tật ST (nhãn chuyển **vụn** trong không gian: PAS vượt sự thật 4/4, Moran thấp hơn "
          "4/4) do **chính CORAL** gây ra — đo được trên hai bộ map độc lập. Nên sửa ở CORAL.\n\n"
          "CORAL là `Z = (F−μ_s)·Σ_s^(−1/2)·Σ_x^(1/2) + μ_x`. Nghiệm đó **không duy nhất**: với mọi "
          "`R` trực giao, `A_R = Σ_s^(−1/2)·R·Σ_x^(1/2)` cho `Cov(Z) = Σ_x^(1/2)·RᵀR·Σ_x^(1/2) = Σ_x`, "
          "y hệt nhau. Chọn `R = I` là **tuỳ tiện** — một bậc tự do chuẩn đang bị bỏ phí. Nhưng "
          "`Σ_s^(−1/2) R Σ_x^(1/2)` **không** trực giao nên `R` có đổi hình học kNN, tức đổi nhãn "
          "chuyển và tính liền mạch.\n\n"
          "Dùng bậc tự do đó để tối thiểu độ gồ ghề không gian. Với `B = (F−μ_s)Σ_s^(−1/2)` và "
          "`M = BᵀLB`:\n\n"
          "```\nmin_{RᵀR=I}  tr(ZᵀLZ) = tr(Rᵀ M R Σ_x)\n```\n\n"
          "**Có nghiệm đóng** — bất đẳng thức vết Ruhe: ghép trị riêng **nhỏ nhất** của `M` với trị "
          "riêng **lớn nhất** của `Σ_x`, tức `R* = U_M[:, ::-1] @ U_Xᵀ`. Đổi dấu cột không đổi mục "
          "tiêu nên ép `det = +1` được miễn phí. `R(t) = cực-phân-rã((1−t)I + t·R*)`, `t=0` cho đúng "
          "CORAL hiện hành.\n\n"
          "Đã kiểm bằng số (n=800, d=12): `det(R*) = +1`; `rough(R*)` thấp hơn `rough(I)` 25% **và "
          "thấp hơn điểm tốt nhất trong 4000 `R` ngẫu nhiên**; nội suy giữ trực giao tới 1e-15.\n\n"
          "**Vì sao hơn phần A:** phần A làm trơn đầu ra nên **bóp** phương sai (`d_cov` lên 0.22–0.49 "
          "ở nhánh `post`). Ở đây `Σ` bị **ghim cứng**, chỉ còn quay — ràng buộc chặt hơn hẳn, nên khả "
          "năng bào vào tín hiệu sinh học thấp hơn. Chi phí: vài giây mỗi `t`, **không dựng lại map**.\n\n"
          "**Tiêu chí 1.0.3:** `pas_abs_dev` giảm > 0.005 và `total_v2` không tụt quá 0.01 so với "
          "`t=0`. `xfer_main` in ra để biết giá phải trả, **không** dùng làm điều kiện.\n\n"
          "## Phần V — gộp bền vững thay cho trung bình\n\n"
          "`pas_true` = 0.20-0.38 không phải \"tín hiệu kém\". Nó là phát biểu chính xác rằng lân cận "
          "là mẫu **bị nhiễm bẩn 20-38%**. Điểm sụp đổ của **trung bình là 0** — một tế bào khác loại "
          "đủ kéo lệch; của **trung vị là 0.5**. Mức nhiễm bẩn đo được nằm gọn trong vùng trung vị còn "
          "sống mà trung bình đã chết.\n\n"
          "Mà **mọi thứ dự án đã thử đều là trung bình**: `cmix`, beta không gian, làm trơn alpha, phạt "
          "Laplacian, CORAL quay, co rút Phần K. Không cái nào dùng ước lượng bền vững.\n\n"
          "Phần V giữ **nguyên** cơ chế co rút của Phần K (`w` tính một lần, dùng chung mọi nhánh) và "
          "chỉ đổi đúng một thứ: ước lượng vị trí `m_i`. Chênh lệch quy thẳng về nó.\n\n"
          "**Dự đoán đặt TRƯỚC khi chạy:** nếu nguyên nhân là nhiễm bẩn thì mức hơn của trung vị phải "
          "**tăng theo `pas_true`** — cosmx6k .378 > visiumhd_ff .288 > visiumhd_ffpe .258 > stereoseq "
          ".197. Không theo thứ tự đó thì giả thuyết sai, **không được diễn giải lại sau khi nhìn số**. "
          "Báo cáo tự in phán quyết.\n\n"
          "Mốc phải vượt là `raw`, không phải `tb`.\n\n"
          "## Phần W — không gian làm bộ lọc CHIỀU, không phải bộ lọc tế bào\n\n"
          "Tín hiệu không gian đúng **62-80%** (`pas_true` 0.20-0.38). Dùng ở cấp **từng tế bào** thì "
          "vô dụng: mỗi lân cận là mẫu cỡ 15 với 1/3 nhiễm bẩn. Cùng tín hiệu đó ở cấp **quần thể** "
          "trên 45.000 tế bào có sai số chuẩn ~0.005 — **yếu ở cá thể, rất chắc ở tổng thể**. Sáu ý "
          "tưởng trước đều là MỘT ý tưởng (gộp láng giềng vào tế bào) dùng ở cấp sai.\n\n"
          "Ở đây không gian không nói *tế bào này giống hàng xóm*, nó nói **chiều này mới là sinh "
          "học**: nhiễu kỹ thuật (độ sâu, hiệu ứng lô, tràn tín hiệu giữa bin) không có tổ chức "
          "không gian; cấu trúc mô thì có.\n\n"
          "**Cổng là `SPEARMAN_ng_đích`** — tương quan giữa Moran theo chiều của nguồn và của Xenium. "
          "Hai bên cùng cơ sở nên chiều so được. Cao → *chiều có cấu trúc không gian* là tính chất "
          "chung, độc lập công nghệ, đánh trọng số theo nó có nghĩa. ~0 → đặc thù từng công nghệ, "
          "trọng số tuỳ tiện, ý tưởng chết.\n\n"
          "Và một phép thử **trực tiếp**, không suy luận gián tiếp: chấm với top-N chiều chọn theo "
          "Moran so với top-N chọn theo **phương sai**. Hơn → không gian chọn ra sinh học mà phương "
          "sai bỏ sót.\n\n"
          "## Phần X — FROST 1.0.4: Spatial-CORAL\n\n"
          "**Lập luận là một chứng minh, không phải một quan sát.** CORAL tính `Σ_s`, `Σ_x` từ ma "
          "trận tế_bào×chiều. Hoán vị thứ tự hàng không đổi hiệp phương sai. Vậy **CORAL bất biến "
          "với hoán vị tế bào**, nên nó mù thông tin không gian *về mặt cấu trúc* — không phải vì "
          "chưa tinh chỉnh đủ. Không α, không ridge, không lưới siêu tham số nào sửa được.\n\n"
          "Phần W đo chỗ mù đó tốn gì: Moran **trung vị theo chiều** lệch 40–50% giữa nguồn và "
          "Xenium (vhd_ff 0.410 vs 0.285; stereoseq 0.438 vs 0.285; cosmx6k 0.033 vs 0.131 — ngược "
          "chiều). Căn chỉnh hiện tại khớp `Σ` xong rồi để nguyên khoảng lệch đó.\n\n"
          "Thêm **mô-men bậc hai có trọng số không gian**, trong hệ đã tẩy trắng:\n\n"
          "```\nA = Bᵀ W̃_s B / n_s     B  = (F−μ_s)Σ_s^(−1/2)\n"
          "G = Bxᵀ W̃_x Bx / n_x    Bx = (X−μ_x)Σ_x^(−1/2)\n```\n\n"
          "`W̃ = (Wn+Wnᵀ)/2` vì `Wn` chuẩn hoá theo hàng nên không đối xứng. Bậc tự do còn lại sau "
          "CORAL **đúng là nhóm trực giao** (lập luận ở Phần R), và trong hệ tẩy trắng mô-men không "
          "gian của `Z` là `RᵀAR`. Nên\n\n"
          "```\nmin_{RᵀR=I} ‖RᵀAR − G‖_F   <=>   max_R tr(RᵀAR·G)\n```\n\n"
          "**Nghiệm đóng** — lại là Ruhe, nhưng ghép **thuận**: `max = Σ λ↑(A)λ↑(G)`, đạt tại "
          "`R* = U_A U_Gᵀ`. `R(t) = cực-phân-rã((1−t)I + t·R*)`; `t=0` cho **đúng FROST** khi "
          "`SPC_TAG=cmix`. Không huấn luyện, không siêu tham số ngoài `t`, cắm thẳng vào khe CORAL.\n\n"
          "**Tính mới:** scGPT-spatial và Nicheformer nhét không gian vào *encoder*. Chưa ai khớp "
          "mô-men không gian ở bước **căn chỉnh hậu kỳ** — mà đó đúng là chỗ FROST sống.\n\n"
          "**Ba cổng, đọc theo thứ tự; trượt cổng nào dừng tại đó:**\n\n"
          "1. `d_sp_san` so với `d_sp_t0` — sàn là chặn dưới của `‖RᵀAR−G‖/‖G‖` trên **toàn** nhóm "
          "trực giao, tính thẳng từ hai phổ. Sàn ≈ t0 → phép quay về nguyên tắc không sửa được gì, "
          "1.0.4 chết bất kể các số còn lại.\n"
          "2. `d_cov` phải **đứng yên** ở mức `RIDGE≈0.01` với mọi `t`. Tăng theo `t` là **code sai**, "
          "không phải phát hiện (bài học Phần R: ridge phá bất biến chuẩn ở d=512).\n"
          "3. `hoanvi@t1` so với `t1` — `R*` dựng từ đồ thị nguồn **đã hoán vị**. Ngang nhau → thứ "
          "mua được là bậc tự do quay nói chung, **không phải** thông tin không gian; hết tính mới. "
          "Phần K đã từng thua chính nhánh null này 4/4, nên cổng này không thừa.\n\n"
          "Chỉ khi qua cả ba mới đọc `total_v2`/`xfer_main`, và mốc so là `t0`, **không** phải bảng "
          "`u_cmix_cc` của vòng 23e (khác đường tính, lệch vài phần nghìn).\n\n"
          "## Chạy lại để bổ sung phần mới\n\n"
          "`run45` nạp lại `summary.json` từ Input khi thư mục làm việc chưa có, và **bỏ qua mọi mục đã "
          f"chấm**. Nên muốn thêm phần A vào kết quả cũ: Add output 45a cũ (`out{p}s`) làm Input, đặt "
          "`ONLY_PART=\"A\"`, chạy lại. D và M không bị chấm lại; `summary.md` cuối cùng vẫn có đủ ba phần.\n\n"
          "## Phải đọc trong log\n\n"
          "- `đồ thị: cắt X% cạnh` — nếu **> 30%** thì đồ thị không gian đã rời rạc, chỉ số ST đọc thận "
          "trọng; cân nhắc tăng `R_MAX_UM`.\n"
          "- `nhãn giữ k/n` — số loại tế bào qua được bộ lọc tự tương quan nền. Nếu `0/n` thì mô này không "
          "có loại nào đủ tổ chức không gian, phần M bị bỏ cho cặp đó.\n"
          "- `tiếp nối <- ...` — xác nhận đã nạp được summary cũ từ Input. KHÔNG thấy dòng này khi "
          "chạy lại nghĩa là D/M sẽ bị chấm lại từ đầu.\n\n"
          "**Add Input:**\n\n" + B41.add_input_md(tissue, need_model=False, need_report=False)
          + f"\n- output **{p}j** — map `cmix` → chấm **FROST**\n"
          f"- output **{p}n** — map `usher_orig` → chấm **USHER bản gốc 25 vòng** (vòng 38 nhánh B1, "
          "chỉ seed 0, chỉ 4 cặp COAD)\n\n"
          f"**Kết quả:** tải `out{p}s/summary.json` và `out{p}s/summary.md`.")
    cells = ([("markdown", md)] + B3.common_cells(c1, EXTRA)
             + [("markdown", "---\n## Chạy (CPU)"), ("code", run),
                ("markdown", f"Tải `/kaggle/working/out{p}s/summary.md` và `summary.json`.")])
    f_ = f"{NB_DIR}/{NB[tissue]}.ipynb"
    json.dump(B.nb(cells), open(f_, "w"), indent=1, ensure_ascii=False)
    print(len(cells), "cell ->", os.path.basename(f_))


if __name__ == "__main__":
    import py_compile
    B39.no_module_level_assert("run45.py")
    B26._check_names("run45.py")
    py_compile.compile("run45.py", doraise=True)
    print("kiểm run45.py OK (py_compile + tên + không assert cấp module). Sinh notebook: python build45.py --write")
    if "--write" in sys.argv:
        for t in [a for a in sys.argv[1:] if a in PFX] or ["coad"]:
            build(t)
