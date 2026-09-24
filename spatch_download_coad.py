#!/usr/bin/env python3
"""
Tải dữ liệu SPATCH của bệnh nhân COAD.

SPATCH phân phối dữ liệu qua Aliyun Drive. Trang web lấy link tải theo 2 bước:
  1) POST /v2/share_link/get_share_token  với share_id  -> share_token
  2) POST /v2/file/list                   với token     -> download_url (hết hạn sau 2 giờ)
Script này làm đúng 2 bước đó, nên KHÔNG bị phụ thuộc link hết hạn.

Dùng trên Kaggle (Internet ON):
    !python spatch_download_coad.py --out /kaggle/working/spatch_coad --group core
Hoặc trên máy cá nhân:
    python spatch_download_coad.py --out ~/Downloads/spatch_coad --group core

Nhóm file:
  core  : 5 ma trận transcriptome + scRNA-seq COAD  (~2.1 GB nén)
  codex : 5 file proteome CODEX (16 kênh), dùng làm ground truth độc lập (~93 MB)
  supp  : file phụ trợ (toạ độ FOV của CosMx, slide ID của Visium HD)  (~43 MB)
  all   : cả ba nhóm trên
KHÔNG tải: ảnh H&E/DAPI, transcript thô, mask phân vùng, và supplement của Stereo-seq (2.75 GB).
"""
import argparse, hashlib, json, os, sys, time, zipfile
import urllib.request

API = "https://bj21400.api.aliyunfile.com"
UA = "Mozilla/5.0 (compatible; spatch-downloader/1.0)"

# share_id lấy từ bảng Download của http://spatch.pku-genomics.org (ngày 2026-09-24)
FILES = {
    # ---- core: ma trận biểu hiện (h5ad nén trong .zip) --------------------------------
    "xenium5k_tx":        ("Pdrp2CtWvxT", "core",  "Xenium 5K | 5,001 gene x 365,395 tế bào"),
    "cosmx6k_tx":         ("3YUkMGpF6kp", "core",  "CosMx 6K | 6,175 gene x 261,731 tế bào"),
    "visiumhd_ffpe_tx":   ("YTVzKjBmESy", "core",  "Visium HD FFPE | 18,042 gene x 520,382 bin 8um"),
    "stereoseq_tx":       ("h3tacUixPxw", "core",  "Stereo-seq v1.3 | 31,164 gene x 447,572 tế bào"),
    "visiumhd_ff_tx":     ("WDWdnhMTTNN", "core",  "Visium HD FF | 17,278 gene x 514,391 bin 8um"),
    "scrna_coad":         ("YT6hTCygnQV", "core",  "scRNA-seq COAD | 23,246 gene x 8,288 tế bào"),
    # ---- codex: proteome 16 kênh, ground truth độc lập --------------------------------
    "xenium5k_codex":     ("gbp4ofmJ4JU", "codex", "CODEX cạnh lát Xenium"),
    "cosmx6k_codex":      ("7i7gH5T7W9x", "codex", "CODEX cạnh lát CosMx"),
    "visiumhd_ffpe_codex":("jJxm21kQKyW", "codex", "CODEX cạnh lát Visium HD FFPE"),
    "stereoseq_codex":    ("RjKVRseppH7", "codex", "CODEX cạnh lát Stereo-seq"),
    "visiumhd_ff_codex":  ("66HrJg1qtho", "codex", "CODEX cạnh lát Visium HD FF"),
    # ---- supp -------------------------------------------------------------------------
    "cosmx6k_supp":       ("SYwVG1qGkvq", "supp",  "CosMx: fov_positions_file.csv"),
    "visiumhd_ffpe_supp": ("1KEC143BFhb", "supp",  "Visium HD FFPE: slide ID + capture area"),
    "visiumhd_ff_supp":   ("didkPZbRtuV", "supp",  "Visium HD FF: slide ID + capture area"),
}


def post(url, payload, headers=None):
    body = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json", "User-Agent": UA}
    hdr.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=hdr, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def resolve(share_id):
    """share_id -> danh sách {name, size, url}. url hết hạn sau 2 giờ."""
    tok = post(f"{API}/v2/share_link/get_share_token", {"share_id": share_id, "ignoreError": True})
    if "share_token" not in tok:
        raise RuntimeError(f"không lấy được share_token cho {share_id}: {tok}")
    ls = post(f"{API}/v2/file/list", {
        "limit": 100, "marker": "", "share_id": share_id, "parent_file_id": "root",
        "fields": "user_name,dir_size,url,content_type,upload_id,crc64_hash,revision_id,description",
        "url_expire_sec": 7200,
    }, {"x-share-token": tok["share_token"]})
    return [{"name": i["name"], "size": i.get("size", 0), "url": i.get("download_url") or i.get("url")}
            for i in ls.get("items", [])]


def download(url, dest, size_hint=0):
    if os.path.exists(dest) and size_hint and os.path.getsize(dest) == size_hint:
        print(f"    đã có, bỏ qua ({os.path.getsize(dest)/1e6:.0f} MB)")
        return
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    t0, done = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or size_hint or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if done % (50 << 20) < (1 << 20):
                pct = f"{100*done/total:.0f}%" if total else "?"
                print(f"    {done/1e6:.0f} MB ({pct}), {done/1e6/max(time.time()-t0,1e-9):.1f} MB/s", flush=True)
    os.replace(tmp, dest)
    print(f"    xong {os.path.getsize(dest)/1e6:.0f} MB trong {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/spatch_coad")
    ap.add_argument("--group", default="core", choices=["core", "codex", "supp", "all"])
    ap.add_argument("--only", default="", help="chỉ tải các key này, cách nhau bằng dấu phẩy")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê file và dung lượng, không tải")
    ap.add_argument("--no-unzip", action="store_true")
    args = ap.parse_args()

    keys = [k for k, (_, g, _) in FILES.items() if args.group in ("all", g)]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        keys = [k for k in keys if k in want]
    os.makedirs(args.out, exist_ok=True)

    manifest, total = {}, 0
    for k in keys:
        sid, grp, desc = FILES[k]
        print(f"\n[{k}] {desc}")
        try:
            items = resolve(sid)
        except Exception as e:
            print(f"    LỖI khi lấy link: {e}"); manifest[k] = {"error": str(e)}; continue
        if not items:
            print("    LỖI: share rỗng"); manifest[k] = {"error": "empty share"}; continue
        manifest[k] = {"share_id": sid, "desc": desc, "files": []}
        for it in items:
            print(f"    {it['name']}  {it['size']/1e6:.0f} MB")
            total += it["size"]
            manifest[k]["files"].append({"name": it["name"], "size": it["size"]})
            if args.list:
                continue
            sub = os.path.join(args.out, k); os.makedirs(sub, exist_ok=True)
            dest = os.path.join(sub, it["name"])
            download(it["url"], dest, it["size"])
            if dest.endswith(".zip") and not args.no_unzip:
                with zipfile.ZipFile(dest) as z:
                    names = z.namelist()
                    print(f"    giải nén {len(names)} file: {names[:5]}")
                    z.extractall(sub)
                manifest[k]["files"][-1]["unzipped"] = names

    print(f"\nTổng dung lượng: {total/1e9:.2f} GB")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Đã ghi", os.path.join(args.out, "manifest.json"))
    if not args.list:
        print("\nCây thư mục:")
        for root, _, files in os.walk(args.out):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                print(f"  {os.path.getsize(p)/1e6:9.1f} MB  {os.path.relpath(p, args.out)}")


if __name__ == "__main__":
    sys.exit(main())
