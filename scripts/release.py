"""發布新版到 GitHub Releases。由 ``release.cmd`` 呼叫。

做的事，照順序：

1. 從 ``src/mapleexp/__init__.py`` 讀版本號 —— 那是唯一的來源，tag 就是 ``v`` 加它。
2. 檢查 ``gh`` 裝了也登入了、工作目錄乾淨、這個 tag 還沒發過。
3. ``build.cmd`` 重新打包。
4. 算 ``dist/MapleExpTool.exe`` 的 SHA-256，寫成 ``MapleExpTool.exe.sha256``。
   程式更新時會拿它驗下載到的檔案（``mapleexp/update.py``）。
5. push 目前的分支，``gh release create`` 建 tag、上傳兩個檔案。

朋友那邊的程式下次啟動就會看到新版。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "mapleexp" / "__init__.py"
EXE = ROOT / "dist" / "MapleExpTool.exe"
SHA = EXE.with_name(EXE.name + ".sha256")
MIN_EXE_BYTES = 5 * 1024 * 1024


def read_version() -> str:
    match = re.search(r'__version__\s*=\s*"([^"]+)"', INIT.read_text(encoding="utf-8"))
    if not match:
        sys.exit(f"在 {INIT} 找不到 __version__")
    return match.group(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def output(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="發布新版到 GitHub Releases")
    notes = parser.add_mutually_exclusive_group()
    notes.add_argument("--notes", help="更新說明（會顯示在程式的更新提示裡）")
    notes.add_argument("--notes-file", help="從檔案讀更新說明")
    parser.add_argument("--skip-build", action="store_true", help="用現有的 dist\\MapleExpTool.exe")
    parser.add_argument("--draft", action="store_true", help="建成草稿，到網頁上確認後再發布")
    parser.add_argument("--dry-run", action="store_true", help="只打包與檢查，不 push、不發布")
    args = parser.parse_args()

    version = read_version()
    tag = f"v{version}"
    print(f"版本：{tag}")

    gh = shutil.which("gh")
    if gh is None:
        print(
            "找不到 gh（GitHub CLI）。安裝：winget install GitHub.cli，"
            "然後 gh auth login 登入一次。",
            file=sys.stderr,
        )
        return 1
    if subprocess.run([gh, "auth", "status"], cwd=ROOT, capture_output=True).returncode != 0:
        print("gh 還沒登入：先執行 gh auth login。", file=sys.stderr)
        return 1

    if output(["git", "status", "--porcelain"]):
        print("工作目錄有未 commit 的變更，先 commit 再發布。", file=sys.stderr)
        return 1
    branch = output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if output(["git", "tag", "-l", tag]) or output(["git", "ls-remote", "--tags", "origin", tag]):
        print(
            f"{tag} 已經發布過了。改 {INIT.relative_to(ROOT)} 裡的 __version__ 之後再來。",
            file=sys.stderr,
        )
        return 1

    if not args.skip_build:
        run(["cmd", "/c", str(ROOT / "build.cmd")])
    if not EXE.exists() or EXE.stat().st_size < MIN_EXE_BYTES:
        print(f"{EXE} 不存在或小得不像話，打包應該失敗了。", file=sys.stderr)
        return 1

    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    SHA.write_text(f"{digest}  {EXE.name}\n", encoding="ascii")
    print(f"SHA-256：{digest}")
    print(f"大小：{EXE.stat().st_size / 1_048_576:.1f} MB")

    if args.dry_run:
        print("dry-run：到此為止，沒有 push 也沒有發布。")
        return 0

    run(["git", "push", "origin", branch])
    cmd = [
        gh, "release", "create", tag, str(EXE), str(SHA),
        "--title", tag, "--target", branch,
    ]
    if args.notes:
        cmd += ["--notes", args.notes]
    elif args.notes_file:
        cmd += ["--notes-file", args.notes_file]
    else:
        cmd += ["--generate-notes"]
    if args.draft:
        cmd.append("--draft")
    run(cmd)
    print(f"\n完成：{tag} 已發布。朋友那邊下次開程式就會看到更新提示。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"指令失敗（exit {exc.returncode}）：{' '.join(map(str, exc.cmd))}", file=sys.stderr)
        sys.exit(exc.returncode or 1)
