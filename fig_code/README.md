# Code vẽ 4 hình của báo cáo FROST

Toàn bộ code này sinh ra Hình 1, Hình 2, Hình 3 (vòng 36) và Hình 7 (vòng 39) trong `BaoCao_FROST.docx`.
Nó **chưa từng nằm trong repo FDS** — nó ở thư mục nháp của phiên Claude, nên hãy đặt vào
`research/figs/` hoặc chỗ nào bạn muốn.

## Chạy

```bash
python3 fig_grid.py     # -> figs/fig_grid_tech.png, figs/fig_grid_tissue.png  (bài báo: Hình S1, S2)
python3 fig3.py         # -> figs/fig_gaps.png  (bài báo: Hình 1)
```

**Cập nhật 2026-10-15 (bài báo LaTeX, `paper/`):** nhãn trong hai hình trên đã sang tiếng Anh (ghi đè
ngay trong script, `style.py` không đổi); **bỏ cột nội suy** (giao thức xuôi, rò rỉ gene — xem `paper/` §6);
bỏ tiêu đề và chú thích chân hình (chuyển vào caption LaTeX). Hai dòng đầu mỗi script ghi lệnh chạy và md5
`data.json`. Máy local không có scipy -> chạy ở môi trường có scipy rồi chép PNG vào `paper/figs/`.

```bash
python3 fig7_converge.py # -> figs/fig7_hoi_tu.png
python3 audit.py        # rà data.json so với notebook 36 + quét lệch dấu TB/trung vị
```

Cần `matplotlib`, `numpy`, `scipy`. **Đừng chạy với `python3 -I`** — cờ đó bỏ thư mục script
khỏi `sys.path` nên `import style` sẽ hỏng.

Thư mục `figs/` phải tồn tại trước khi chạy (`mkdir -p figs`).

## Các file

| file | là gì |
|---|---|
| `style.py` | bảng màu + thứ tự hàng/khối + nhóm chỉ số, **dùng chung cho cả 3 hình** |
| `fig_grid.py` | Hình 1 (trung bình theo công nghệ) và Hình 2 (trung bình theo mô) |
| `fig3.py` | Hình 3 (khoảng cách tới FROST, Wilcoxon signed-rank) |
| `fig7_converge.py` | Hình 7 (hội tụ bước E, ba cách dựng Cˢ + bản USHER công bố) |
| `converge.json` | đường hội tụ note 39 — 3 nhánh × 4 cặp COAD × 25 vòng + 5 mốc USHER gốc |
| `build_data.py` | sinh `data.json` từ notebook 36 + vá SD 3 seed cho COAD từ note 22 |
| `data.json` | **18 chỉ số × 12 cặp × 11 phương pháp**; 636 ô có SD; 12 ô rỗng (Symphony × nội suy) |
| `audit.py` | kiểm `data.json` so với notebook 36 (0/2.376 ô lệch) + quét lệch dấu |

## Bốn điều phải biết trước khi sửa

**1. `build_data.py` phụ thuộc đường dẫn notebook 36 đã biến mất.** Nó đọc
`/root/.claude/uploads/.../a87b6d47-36-usher.ipynb` — đường dẫn của phiên Claude cũ, giờ không
còn. `data.json` thì vẫn dùng được bình thường; chỉ cần sửa biến `NB` nếu muốn dựng lại từ đầu.
`audit.py` cũng vậy (biến `NB` ở đầu file).

**2. Hình 3 vẽ theo TRUNG VỊ của 12 hiệu số, không phải hiệu của hai trung bình.** Bản đầu vẽ
theo hiệu hai trung bình trong khi in trung vị, và `audit.py` tìm ra **3/64 ô lệch dấu**
(`knn_xen` vs `center`, `knn_xen` vs `Harmony`, `xfer_all` vs `ridge`). Đã sửa: vạch, thanh và
dòng số đều lấy từ `g = -med`, cùng một đại lượng mà Wilcoxon kiểm. **Đừng đổi lại.**

**3. Mọi tính toán bố cục trong `fig_grid.py` làm bằng INCH rồi mới đổi sang phân số hình**
(`fx = lambda v: v / FW`). Làm trực tiếp bằng phân số thì nhãn khối bị đẩy ra ngoài canvas khi
cột nhãn hẹp lại. `FW = 6.69 in = 17 cm` là bề rộng **thật** để Word không co hình.

## Quy ước tên trong code so với tên trong bài

| tên trong bài | khoá trong `data.json` / `style.py` |
|---|---|
| `USHER-ST` | `FROST_base` |
| `USHER-CC` | `FROST_cc` |
| **`FROST`** | `FROST` |

(`style.BLOCKS` giữ ánh xạ này; đây là tên hiển thị của `usher_final` / `usher_cc` / `u_cmix_cc`.)

## 16 cột của Hình 1/2

`style.GROUPS` định nghĩa: Trộn lô (iLISI · kBET · BRAS · graph conn · batch ASW · batch_v2) ·
Bảo toàn sinh học (NMI · ARI · ASW · bio_v2) · Tác vụ hạ nguồn (xfer_main · xfer_all ·
rare_purity · knn_xen) · Nội suy (Spearman) · Tổng hợp (total_v2).
**Bản bài báo (2026-10-15): cột Nội suy đã bỏ** — hình còn 15 cột; `fig_grid.py` dùng `BAND_A`/`BAND_B` riêng, không đọc `style.GROUPS`.

`cLISI` và `iso_labels` **có trong `data.json` nhưng không có trong hình** — đã xuống phụ lục.

## Hình 7 — ba điều riêng của nó

**4. Giá trị đầu tiên (v0 → v1) bị bỏ, có chủ ý.** Ở vòng 0 code đặt `gamma = 1` nên ghép hoàn
toàn theo `M_aux`; tỉ lệ ánh xạ đổi gần 100% là tất yếu chứ không phải dấu hiệu chưa hội tụ. Vẽ
cả điểm đó thì trục log bị kéo và ba đường bị ép sát nhau.

**Bản USHER công bố chỉ có 5 mốc đo** (v5/10/15/20/25) nên vẽ bằng điểm nối thẳng, KHÔNG nội
suy thành đường liền — đừng đổi thành `-` để "cho đẹp".

**Kết luận của §6.4 đọc ở ĐUÔI (3 vòng cuối), không đọc ở vòng 10.** Ba đường bám nhau tới
khoảng vòng 10 rồi mới tách: ở vòng 10 tỉ lệ `it25 / both25` chỉ 1,36× và sẽ dẫn tới kết luận
ngược với 5,65× ở đuôi.

### `converge.json`

```
{"nhanh": {"it25"|"both25"|"cmix25": {"<cặp>": [25 giá trị]}},
 "usher_goc_moc": {"<cặp>": {"5": .., "10": .., "15": .., "20": .., "25": ..}}}
```

Nguồn: `converge.json` của notebook 39 (ba nhánh a/b/c) và notebook 38 nhánh B1 (USHER bản công
bố). `fig7_converge.py` in ra các số của §6.4 để đối chiếu sau khi vẽ.
