"""自動更新：啟動時到 GitHub Releases 看有沒有新版，有就下載、換掉自己、重新啟動。

發布流程（見 ``release.cmd``）會把 ``MapleExpTool.exe`` 跟它的 ``.sha256`` 一起掛在
Release 上，tag 是 ``v0.2.0`` 這種格式。程式這邊：

1. 背景執行緒打 ``/releases/latest``，拿 tag 跟 ``__version__`` 比。網路慢或沒網路
   就當作沒有新版，浮窗照開 —— 更新永遠不能擋住主要功能。
2. 使用者同意後把新版下載到**同一個資料夾**的 ``MapleExpTool.exe.new``（放暫存目錄
   不行：暫存目錄常在別的磁碟，``os.replace`` 跨磁碟會失敗）。有 ``.sha256`` 就驗，
   另外檢查開頭是 ``MZ``，免得把 GitHub 的錯誤頁面當成執行檔裝進去。
3. 換檔。Windows 不准覆寫執行中的 exe，但**准改名**：把自己改成 ``.exe.old``、
   新檔搬到原名，然後啟動新的、自己退出。下次啟動再把 ``.old`` 清掉（舊行程可能
   還沒完全退出，所以在背景多試幾次）。

整段只用標準庫。``ssl`` 在 Windows 上會讀系統憑證庫，打包後不需要帶 certifi。
只有打包成執行檔的版本會做這些事；``python -m mapleexp`` 跑的是原始碼，更新請 ``git pull``。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from . import __version__

REPO = "andy971029/MapleExpTool"
ASSET_NAME = "MapleExpTool.exe"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
USER_AGENT = f"MapleExpTool/{__version__}"

# 查版本只是啟動時順手做的事，不值得讓人等太久。
CHECK_TIMEOUT = 4.0
DOWNLOAD_TIMEOUT = 60.0
OLD_SUFFIX = ".old"
NEW_SUFFIX = ".new"
CHUNK = 64 * 1024

# 設了這個環境變數就完全不碰網路（開發、測試時用）。
DISABLE_ENV = "MAPLEEXP_NO_UPDATE"

ProgressFn = Callable[[int, int | None], None]


class UpdateError(Exception):
    """更新失敗，訊息是給使用者看的。"""


@dataclass(frozen=True)
class Release:
    version: str  # "0.2.0"
    tag: str  # "v0.2.0"
    url: str  # 執行檔的下載網址
    sha256_url: str | None
    notes: str
    page: str  # Release 網頁，給「自己去下載」用


# --------------------------------------------------------------------------- #
# 版本比較
# --------------------------------------------------------------------------- #

_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


def parse_version(text: str) -> tuple[int, ...]:
    """``'v0.10.1'`` -> ``(0, 10, 1)``。不是版本號的字串回傳空 tuple。

    一定要拆成數字比：``"0.10.0" > "0.9.0"`` 用字串比是 False。
    """
    match = _VERSION_RE.match(text or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def normalize_version(text: str) -> str | None:
    parts = parse_version(text)
    return ".".join(str(p) for p in parts) if parts else None


def is_newer(candidate: str, current: str) -> bool:
    """``candidate`` 是否比 ``current`` 新。``0.2`` 與 ``0.2.0`` 視為相同。"""
    a, b = parse_version(candidate), parse_version(current)
    if not a or not b:
        return False
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


# --------------------------------------------------------------------------- #
# 查詢
# --------------------------------------------------------------------------- #


def parse_release(raw: object) -> Release | None:
    """把 GitHub API 的 release JSON 轉成 ``Release``。沒掛執行檔的 release 回傳 None。"""
    if not isinstance(raw, dict):
        return None
    tag = str(raw.get("tag_name") or "")
    version = normalize_version(tag)
    if version is None:
        return None
    urls: dict[str, str] = {}
    for asset in raw.get("assets") or []:
        if isinstance(asset, dict) and asset.get("name") and asset.get("browser_download_url"):
            urls[str(asset["name"])] = str(asset["browser_download_url"])
    url = urls.get(ASSET_NAME)
    if not url:
        return None
    return Release(
        version=version,
        tag=tag,
        url=url,
        sha256_url=urls.get(ASSET_NAME + ".sha256"),
        notes=str(raw.get("body") or "").strip(),
        page=str(raw.get("html_url") or RELEASES_PAGE),
    )


def _request(url: str) -> Request:
    # GitHub API 沒有 User-Agent 會直接回 403。
    return Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})


def fetch_latest(timeout: float = CHECK_TIMEOUT) -> Release | None:
    """最新的 release；還沒發布過任何版本（404）回傳 None。其他錯誤照常拋出。"""
    from urllib.error import HTTPError

    try:
        with urlopen(_request(API_LATEST), timeout=timeout) as response:
            raw = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return parse_release(raw)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def executable_path() -> Path:
    # 單檔打包時 sys.executable 指向真正的 .exe，不是解壓用的暫存目錄。
    return Path(sys.executable).resolve()


def disabled_by_env() -> bool:
    return bool(os.environ.get(DISABLE_ENV))


class UpdateCheck:
    """在背景執行緒查一次。

    ``release`` 查到比目前新的版本才會有值；``error`` 是查詢失敗的原因（沒網路之類，
    只拿來顯示在 ``update`` 子命令，浮窗不管它）；``done`` 表示查完了，不論成敗。
    """

    def __init__(
        self,
        current: str = __version__,
        fetch: Callable[[], Release | None] = fetch_latest,
    ) -> None:
        self.current = current
        self.release: Release | None = None
        self.error: str | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(fetch,), name="update-check", daemon=True
        )

    def start(self) -> "UpdateCheck":
        self._thread.start()
        return self

    def _run(self, fetch: Callable[[], Release | None]) -> None:
        try:
            latest = fetch()
            if latest is not None and is_newer(latest.version, self.current):
                self.release = latest
        except Exception as exc:  # noqa: BLE001 —— 任何網路錯誤都只是「這次沒查到」
            self.error = _describe(exc)
        finally:
            self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


def _describe(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    text = str(reason) if reason else str(exc)
    return text or exc.__class__.__name__


# --------------------------------------------------------------------------- #
# 下載
# --------------------------------------------------------------------------- #


def staging_path(target: Path | None = None) -> Path:
    target = target or executable_path()
    return target.with_name(target.name + NEW_SUFFIX)


def _fetch_text(url: str, timeout: float) -> str:
    with urlopen(_request(url), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download(
    release: Release,
    dest: Path,
    progress: ProgressFn | None = None,
    timeout: float = DOWNLOAD_TIMEOUT,
) -> Path:
    """把新版執行檔下載到 ``dest``。先寫 ``.part``，驗證通過才改名。"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        digest = hashlib.sha256()
        received = 0
        with urlopen(_request(release.url), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, total)
    except OSError as exc:
        raise UpdateError(f"下載失敗：{_describe(exc)}") from exc

    try:
        _verify(partial, digest.hexdigest(), release, timeout)
    except UpdateError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return dest


def _verify(path: Path, actual_sha256: str, release: Release, timeout: float) -> None:
    with open(path, "rb") as handle:
        head = handle.read(2)
    if head != b"MZ":
        raise UpdateError("下載到的不是執行檔（可能是網頁或錯誤訊息），已丟棄。")
    if release.sha256_url:
        try:
            expected = _fetch_text(release.sha256_url, timeout).split()[0].lower()
        except (OSError, IndexError) as exc:
            raise UpdateError(f"讀不到校驗碼：{_describe(exc)}") from exc
        if expected != actual_sha256.lower():
            raise UpdateError("下載的檔案校驗失敗（檔案不完整或被改過），已丟棄。")


# --------------------------------------------------------------------------- #
# 換檔與重啟
# --------------------------------------------------------------------------- #


def install(staged: Path, target: Path | None = None) -> Path:
    """把 ``staged`` 換到 ``target``（預設是自己）。回傳退役的舊檔路徑。

    順序：舊檔改名成 ``.old`` -> 新檔改成原名。第二步失敗就把舊檔改回去，
    不會留下一個沒有執行檔的狀態。
    """
    target = target or executable_path()
    old = target.with_name(target.name + OLD_SUFFIX)
    try:
        old.unlink(missing_ok=True)
    except OSError as exc:
        raise UpdateError(f"清不掉上次留下的舊檔 {old.name}：{_describe(exc)}") from exc
    try:
        target.rename(old)
    except OSError as exc:
        raise UpdateError(f"無法替換 {target.name}：{_describe(exc)}") from exc
    try:
        staged.replace(target)
    except OSError as exc:
        try:
            old.rename(target)
        except OSError:
            pass
        raise UpdateError(f"無法放入新版 {target.name}：{_describe(exc)}") from exc
    return old


def relaunch(target: Path | None = None, argv: list[str] | None = None) -> None:
    """啟動新版。呼叫端接著自己退出。"""
    target = target or executable_path()
    flags = 0
    if sys.platform == "win32":
        # 不要繼承我們的主控台（如果有的話），讓新行程自己決定要不要接。
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [str(target), *(argv or [])],
        cwd=str(target.parent),
        close_fds=True,
        creationflags=flags,
    )


def cleanup_old(
    target: Path | None = None, attempts: int = 20, delay: float = 0.5
) -> threading.Thread | None:
    """刪掉上次更新留下的 ``.old``。

    新版是在舊版還沒退出時啟動的（而且單檔打包還有個 bootloader 父行程要等子行程），
    刪不掉就隔一下再試，在背景做、不擋啟動。回傳那條執行緒（測試用）；沒東西要清
    回傳 None。
    """
    target = target or executable_path()
    old = target.with_name(target.name + OLD_SUFFIX)
    if not old.exists():
        return None

    def run() -> None:
        for _ in range(attempts):
            try:
                old.unlink()
                return
            except OSError:
                time.sleep(delay)

    thread = threading.Thread(target=run, name="update-cleanup", daemon=True)
    thread.start()
    return thread


def stage(release: Release, progress: ProgressFn | None = None) -> Path:
    """下載新版到自己旁邊，回傳暫存檔路徑。還沒換檔，可以安全地中止。"""
    return download(release, staging_path(), progress=progress)


def apply(staged: Path, relaunch_argv: list[str] | None) -> None:
    """換檔；``relaunch_argv`` 不是 None 就用那組參數啟動新版。"""
    target = executable_path()
    install(staged, target)
    if relaunch_argv is not None:
        relaunch(target, relaunch_argv)
