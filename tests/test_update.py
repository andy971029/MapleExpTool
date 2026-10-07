"""自動更新：版本比較、release 解析、下載驗證、換檔。全部不碰網路。"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401

from mapleexp import update
from mapleexp.update import Release, UpdateCheck, UpdateError


def _release(**overrides) -> Release:
    base = dict(
        version="0.3.0",
        tag="v0.3.0",
        url="https://example.invalid/MapleExpTool.exe",
        sha256_url=None,
        notes="",
        page="https://example.invalid/releases",
    )
    base.update(overrides)
    return Release(**base)


class FakeResponse:
    """假的 urlopen 回應：給 bytes 與標頭，支援 ``with`` 與分段 ``read``。"""

    def __init__(self, body: bytes, length: bool = True) -> None:
        self._stream = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body))} if length else {}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class TestVersion(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(update.parse_version("v0.10.1"), (0, 10, 1))
        self.assertEqual(update.parse_version("1.2"), (1, 2))
        self.assertEqual(update.parse_version("nightly"), ())
        self.assertEqual(update.parse_version(""), ())

    def test_numeric_not_lexical(self):
        """``"0.10.0" > "0.9.0"`` 用字串比是 False —— 這就是要拆成數字的原因。"""
        self.assertTrue(update.is_newer("0.10.0", "0.9.0"))
        self.assertFalse(update.is_newer("0.9.0", "0.10.0"))

    def test_equal_and_padding(self):
        self.assertFalse(update.is_newer("0.2.0", "0.2.0"))
        self.assertFalse(update.is_newer("0.2", "0.2.0"))
        self.assertTrue(update.is_newer("0.2.1", "0.2"))
        self.assertTrue(update.is_newer("v1.0.0", "0.99.99"))

    def test_garbage_is_never_newer(self):
        self.assertFalse(update.is_newer("latest", "0.1.0"))
        self.assertFalse(update.is_newer("0.2.0", "???"))


class TestParseRelease(unittest.TestCase):
    RAW = {
        "tag_name": "v0.3.0",
        "html_url": "https://github.com/x/y/releases/tag/v0.3.0",
        "body": "修了東西\n",
        "assets": [
            {"name": "MapleExpTool.exe", "browser_download_url": "https://dl/MapleExpTool.exe"},
            {"name": "MapleExpTool.exe.sha256", "browser_download_url": "https://dl/sum"},
        ],
    }

    def test_full(self):
        release = update.parse_release(self.RAW)
        self.assertIsNotNone(release)
        self.assertEqual(release.version, "0.3.0")
        self.assertEqual(release.tag, "v0.3.0")
        self.assertEqual(release.url, "https://dl/MapleExpTool.exe")
        self.assertEqual(release.sha256_url, "https://dl/sum")
        self.assertEqual(release.notes, "修了東西")
        self.assertEqual(release.page, self.RAW["html_url"])

    def test_without_checksum(self):
        raw = dict(self.RAW, assets=self.RAW["assets"][:1])
        self.assertIsNone(update.parse_release(raw).sha256_url)

    def test_without_exe_is_ignored(self):
        """只有原始碼壓縮檔的 release（例如打錯 tag）不能被當成可更新的版本。"""
        self.assertIsNone(update.parse_release(dict(self.RAW, assets=[])))

    def test_bad_tag_is_ignored(self):
        self.assertIsNone(update.parse_release(dict(self.RAW, tag_name="nightly")))

    def test_not_a_dict(self):
        self.assertIsNone(update.parse_release(None))
        self.assertIsNone(update.parse_release([]))


class TestUpdateCheck(unittest.TestCase):
    def test_newer_release_is_reported(self):
        check = UpdateCheck(current="0.2.0", fetch=lambda: _release(version="0.3.0")).start()
        self.assertTrue(check.wait(5))
        self.assertEqual(check.release.version, "0.3.0")
        self.assertIsNone(check.error)

    def test_same_or_older_is_silent(self):
        for latest in ("0.2.0", "0.1.9"):
            check = UpdateCheck(current="0.2.0", fetch=lambda: _release(version=latest)).start()
            self.assertTrue(check.wait(5))
            self.assertIsNone(check.release)

    def test_no_release_yet(self):
        check = UpdateCheck(current="0.2.0", fetch=lambda: None).start()
        self.assertTrue(check.wait(5))
        self.assertIsNone(check.release)
        self.assertIsNone(check.error)

    def test_network_failure_is_an_error_not_a_crash(self):
        def boom():
            raise OSError("no network")

        check = UpdateCheck(current="0.2.0", fetch=boom).start()
        self.assertTrue(check.wait(5))
        self.assertTrue(check.done)
        self.assertIsNone(check.release)
        self.assertIn("no network", check.error)


class TestDownload(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.dest = self.dir / "MapleExpTool.exe.new"

    def tearDown(self):
        self._tmp.cleanup()

    def test_download_and_progress(self):
        body = b"MZ" + bytes(range(256)) * 10
        seen = []
        with mock.patch.object(update, "urlopen", return_value=FakeResponse(body)):
            path = update.download(_release(), self.dest, progress=lambda got, total: seen.append((got, total)))
        self.assertEqual(path, self.dest)
        self.assertEqual(self.dest.read_bytes(), body)
        self.assertEqual(seen[-1], (len(body), len(body)))
        self.assertFalse(list(self.dir.glob("*.part")))

    def test_checksum_verified(self):
        body = b"MZ" + b"x" * 100
        import hashlib

        good = hashlib.sha256(body).hexdigest()
        release = _release(sha256_url="https://dl/sum")
        responses = [FakeResponse(body), FakeResponse(f"{good}  MapleExpTool.exe\n".encode())]
        with mock.patch.object(update, "urlopen", side_effect=responses):
            update.download(release, self.dest)
        self.assertTrue(self.dest.exists())

    def test_checksum_mismatch_discards_file(self):
        release = _release(sha256_url="https://dl/sum")
        responses = [FakeResponse(b"MZ" + b"x" * 100), FakeResponse(b"0" * 64 + "  MapleExpTool.exe".encode())]
        with mock.patch.object(update, "urlopen", side_effect=responses):
            with self.assertRaises(UpdateError):
                update.download(release, self.dest)
        self.assertFalse(self.dest.exists())
        self.assertFalse(list(self.dir.glob("*.part")))

    def test_html_page_is_rejected(self):
        """GitHub 出錯時回的是網頁；不能把它當執行檔裝進去。"""
        with mock.patch.object(update, "urlopen", return_value=FakeResponse(b"<html>oops</html>")):
            with self.assertRaises(UpdateError):
                update.download(_release(), self.dest)
        self.assertFalse(self.dest.exists())

    def test_network_error_is_update_error(self):
        with mock.patch.object(update, "urlopen", side_effect=OSError("timed out")):
            with self.assertRaises(UpdateError) as ctx:
                update.download(_release(), self.dest)
        self.assertIn("timed out", str(ctx.exception))


class TestInstall(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "MapleExpTool.exe"
        self.staged = self.dir / "MapleExpTool.exe.new"
        self.old = self.dir / "MapleExpTool.exe.old"
        self.target.write_bytes(b"old version")
        self.staged.write_bytes(b"new version")

    def tearDown(self):
        self._tmp.cleanup()

    def test_swap(self):
        retired = update.install(self.staged, self.target)
        self.assertEqual(retired, self.old)
        self.assertEqual(self.target.read_bytes(), b"new version")
        self.assertEqual(self.old.read_bytes(), b"old version")
        self.assertFalse(self.staged.exists())

    def test_previous_old_is_replaced(self):
        self.old.write_bytes(b"older")
        update.install(self.staged, self.target)
        self.assertEqual(self.old.read_bytes(), b"old version")

    def test_rollback_when_staged_missing(self):
        """第二步失敗不能留下「沒有執行檔」的狀態。"""
        self.staged.unlink()
        with self.assertRaises(UpdateError):
            update.install(self.staged, self.target)
        self.assertEqual(self.target.read_bytes(), b"old version")
        self.assertFalse(self.old.exists())

    def test_cleanup_old(self):
        self.old.write_bytes(b"leftover")
        thread = update.cleanup_old(self.target, attempts=3, delay=0)
        self.assertIsNotNone(thread)
        thread.join(5)
        self.assertFalse(self.old.exists())

    def test_cleanup_nothing(self):
        self.assertIsNone(update.cleanup_old(self.target))

    def test_staging_path_sits_next_to_exe(self):
        self.assertEqual(update.staging_path(self.target), self.staged)


if __name__ == "__main__":
    unittest.main()
