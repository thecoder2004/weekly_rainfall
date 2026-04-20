#!/usr/bin/env python3
"""
GSMaP FTP downloader for /realtime and other directories.

Usage examples (run in PowerShell):

  # Liệt kê thư mục con trong /realtime
  python get_data.py --list --remote-base /realtime

  # Tải toàn bộ thư mục product mức trên cùng
  python get_data.py --remote-path /realtime/GSMaP_NRT --out ./data

  # Tải theo thư mục ngày cụ thể (nếu cấu trúc dạng /YYYY/MM/DD/)
  python get_data.py --remote-path /realtime/GSMaP_NRT/2024/09/01 --out ./data --workers 4

  # Lọc theo tên file (regex), ví dụ chỉ tải file chứa "hourly"
  python get_data.py --remote-path /realtime/GSMaP_NRT --match ".*hourly.*" --out ./data

Ghi chú:
- Máy chủ: hokusai.eorc.jaxa.jp | UID: rainmap | PW: Niskur+1404
- Nếu bạn chưa chắc cấu trúc thư mục, dùng --list để khám phá.
"""

import argparse
import concurrent.futures
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from ftplib import FTP, error_perm
from typing import Iterable, List, Optional, Tuple


HOST = "hokusai.eorc.jaxa.jp"
USER = "rainmap"
PASSWORD = "Niskur+1404"


@dataclass
class RemoteFile:
    path: str
    size: Optional[int]


def connect_ftp() -> FTP:
    ftp = FTP()
    # Passive mode giúp vượt NAT/firewall tốt hơn trên Windows
    ftp.set_pasv(True)
    ftp.connect(HOST, 21, timeout=60)
    ftp.login(USER, PASSWORD)
    try:
        ftp.encoding = "utf-8"
    except Exception:
        pass
    return ftp


def ensure_local_dir(local_path: str) -> None:
    directory = local_path if os.path.isdir(local_path) else os.path.dirname(local_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def to_remote_path(path: str) -> str:
    """Chuẩn hóa mọi đường dẫn về dạng POSIX cho FTP (dùng '/')."""
    return path.replace("\\", "/")


def join_remote(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p is not None and p != ""]
    if not cleaned:
        return "/"
    prefix = "/" if parts and str(parts[0]).startswith("/") else ""
    return prefix + "/".join(cleaned)


def ftp_cwd_safe(ftp: FTP, path: str) -> bool:
    try:
        ftp.cwd(to_remote_path(path))
        return True
    except error_perm:
        return False


def list_dir(ftp: FTP, path: str) -> Tuple[List[str], List[RemoteFile]]:
    """Return (dirs, files) for given path."""
    dirs: List[str] = []
    files: List[RemoteFile] = []

    orig = ftp.pwd()
    path = to_remote_path(path)
    if not ftp_cwd_safe(ftp, path):
        raise FileNotFoundError(f"Remote path not found: {path}")

    try:
        # Try MLSD if available (machine-readable)
        try:
            for name, facts in ftp.mlsd():
                name = to_remote_path(name)
                ftype = facts.get("type")
                if ftype == "dir":
                    dirs.append(join_remote(path, name))
                elif ftype == "file":
                    size = None
                    try:
                        size = int(facts.get("size")) if facts.get("size") else None
                    except Exception:
                        size = None
                    files.append(RemoteFile(join_remote(path, name), size))
            return dirs, files
        except (error_perm, AttributeError):
            pass

        # Fallback: NLST + SIZE queries
        names = [to_remote_path(n) for n in ftp.nlst()]
        for name in names:
            # Make full POSIX path
            full = name if name.startswith("/") else join_remote(path, name)
            # Heuristic: try CWD into the item to detect dir
            if ftp_cwd_safe(ftp, full):
                dirs.append(full)
                ftp.cwd(path)
            else:
                fsize: Optional[int] = None
                try:
                    fsize = ftp.size(full)
                except Exception:
                    fsize = None
                files.append(RemoteFile(full, fsize))
        return dirs, files
    finally:
        ftp.cwd(orig)


def walk_remote(ftp: FTP, root: str) -> Iterable[RemoteFile]:
    stack = [root]
    seen = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            dirs, files = list_dir(ftp, current)
        except FileNotFoundError:
            continue
        for f in files:
            yield f
        for d in dirs:
            stack.append(d)


def should_download(local_path: str, remote_size: Optional[int], resume: bool) -> bool:
    if not os.path.exists(local_path):
        return True
    if remote_size is None:
        # Unknown remote size -> if exists and resume false, skip; if resume true, try re-download
        return resume
    local_size = os.path.getsize(local_path)
    if local_size == remote_size:
        return False
    # If resume requested and local smaller, we will append from offset
    return True


def download_file(ftp: FTP, remote_path: str, local_path: str, remote_size: Optional[int], resume: bool = True) -> Tuple[str, bool, Optional[str]]:
    try:
        remote_path = to_remote_path(remote_path)
        ensure_local_dir(local_path)
        mode = "ab" if resume and os.path.exists(local_path) else "wb"
        start_pos = os.path.getsize(local_path) if mode == "ab" else 0

        with open(local_path, mode) as f:
            if start_pos:
                try:
                    ftp.sendcmd(f"REST {start_pos}")
                except Exception:
                    # Server may not support REST for ASCII or certain files
                    f.seek(0)
                    f.truncate(0)
                    start_pos = 0

            def _callback(chunk: bytes) -> None:
                f.write(chunk)

            ftp.retrbinary(f"RETR {remote_path}", _callback)

        # Verify size if available
        if remote_size is not None and os.path.getsize(local_path) != remote_size:
            return remote_path, False, "size_mismatch"
        return remote_path, True, None
    except Exception as e:
        return remote_path, False, str(e)


def build_matcher(match_regex: Optional[str], glob: Optional[str]):
    regex = re.compile(match_regex) if match_regex else None
    def matcher(name: str) -> bool:
        if regex and not regex.search(name):
            return False
        if glob and not fnmatch.fnmatch(name, glob):
            return False
        return True
    return matcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GSMaP data from JAXA FTP.")
    parser.add_argument("--remote-base", default="/realtime", help="Thư mục gốc để liệt kê (khi --list). Mặc định: /realtime")
    parser.add_argument("--remote-path", default=None, help="Đường dẫn đầy đủ đến thư mục cần tải, ví dụ /realtime/GSMaP_NRT/2024/09/01")
    parser.add_argument("--remote-file", default=None, help="Đường dẫn đầy đủ tới MỘT file muốn tải trực tiếp, ví dụ /realtime/README.first.txt")
    parser.add_argument("--out", default="./data", help="Thư mục lưu file tải về")
    parser.add_argument("--out-file", default=None, help="Tên file đích (khi dùng --remote-file). Mặc định dùng basename từ remote")
    parser.add_argument("--list", action="store_true", help="Chỉ liệt kê nội dung thư mục --remote-base rồi thoát")
    parser.add_argument("--match", default=None, help="Regex để lọc file theo tên")
    parser.add_argument("--glob", default=None, help="Glob để lọc tên file (ví dụ: *.bin)")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng tải song song")
    parser.add_argument("--no-resume", action="store_true", help="Không resume các file dở, tải lại từ đầu")

    args = parser.parse_args()

    try:
        ftp = connect_ftp()
    except Exception as e:
        print(f"[ERR] Không thể kết nối FTP: {e}")
        return 2

    with ftp:
        if args.list:
            base = args.remote_base
            print(f"[INFO] Liệt kê: {base}")
            try:
                dirs, files = list_dir(ftp, base)
                if dirs:
                    print("[DIRS]")
                    for d in sorted(dirs):
                        print(d)
                if files:
                    print("[FILES]")
                    for f in sorted(files, key=lambda x: x.path):
                        print(f"{f.path}\t{f.size if f.size is not None else '-'}")
            except Exception as e:
                print(f"[ERR] Liệt kê thất bại: {e}")
                return 1
            return 0

        # Tải trực tiếp một file duy nhất nếu được chỉ định
        if args.remote_file:
            remote_file = to_remote_path(args.remote_file)
            # Thử lấy kích thước trước (không bắt buộc)
            size: Optional[int] = None
            try:
                size = ftp.size(remote_file)
            except Exception:
                size = None
            out_dir = args.out
            out_name = args.out_file if args.out_file else remote_file.rsplit("/", 1)[-1]
            local_path = os.path.join(out_dir, out_name)
            print(f"[INFO] Tải trực tiếp: {remote_file} -> {local_path}")
            rpath, ok, err = download_file(ftp, remote_file, local_path, size, resume=not args.no_resume)
            if ok:
                print(f"[OK]   {rpath}")
                print(f"[DONE] Thành công: 1 | Thất bại: 0")
                return 0
            else:
                print(f"[FAIL] {rpath} :: {err}")
                print(f"[DONE] Thành công: 0 | Thất bại: 1")
                return 1

        remote_root = to_remote_path(args.remote_path)
        if not remote_root:
            print("[ERR] Cần cung cấp --remote-path hoặc dùng --list để khám phá thư mục.")
            return 1

        print(f"[INFO] Quét tệp trong: {remote_root}")
        try:
            files_iter = list(walk_remote(ftp, remote_root))
        except Exception as e:
            print(f"[ERR] Không thể duyệt thư mục: {e}")
            return 1

        matcher = build_matcher(args.match, args.glob)
        files: List[RemoteFile] = [f for f in files_iter if matcher(f.path.rsplit("/", 1)[-1])]

        if not files:
            print("[WARN] Không tìm thấy tệp để tải sau khi lọc.")
            return 0

        print(f"[INFO] Số tệp cần tải: {len(files)}")
        results: List[Tuple[str, bool, Optional[str]]] = []

        def _task(f: RemoteFile) -> Tuple[str, bool, Optional[str]]:
            # Tạo đường dẫn tương đối POSIX từ remote_root
            prefix = remote_root.rstrip("/")
            if f.path.startswith(prefix + "/"):
                rel = f.path[len(prefix) + 1 :]
            elif f.path == prefix:
                rel = f.path.rsplit("/", 1)[-1]
            else:
                rel = f.path.lstrip("/")
            local_path = os.path.join(args.out, rel)
            if not should_download(local_path, f.size, resume=not args.no_resume):
                return f.path, True, "skipped"
            # Mỗi task mở kết nối riêng vì ftplib không thread-safe
            ftp_local = connect_ftp()
            try:
                return download_file(ftp_local, f.path, local_path, f.size, resume=not args.no_resume)
            finally:
                try:
                    ftp_local.quit()
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as exe:
            for res in exe.map(_task, files):
                results.append(res)
                rpath, ok, err = res
                if ok and err == "skipped":
                    print(f"[SKIP] {rpath}")
                elif ok:
                    print(f"[OK]   {rpath}")
                else:
                    print(f"[FAIL] {rpath} :: {err}")

        ok_count = sum(1 for _, ok, _ in results if ok)
        fail_count = len(results) - ok_count
        print(f"[DONE] Thành công: {ok_count} | Thất bại: {fail_count}")
        return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


