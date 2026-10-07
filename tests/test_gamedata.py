"""客戶端資料同步測試。

測試刻意**不碰真的遊戲檔案** —— 開發機上有不代表測試機上有，而且客戶端一改版
檔名就變。所以這裡自己組一包 UnityFS bundle 出來：這同時也是對格式理解的反向
驗證（寫得出來才算真的看懂讀的那一段）。
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from mapleexp import gamedata

try:
    import lz4.block as lz4_block
except Exception:  # pragma: no cover
    lz4_block = None


# --------------------------------------------------------------------------- #
# 組一包假的 bundle
# --------------------------------------------------------------------------- #


def serialized_file(tables: dict[str, bytes]) -> bytes:
    """模擬 SerializedFile 裡 TextAsset 的排法。

    前面塞一段雜訊，是為了確認掃描不是靠「剛好在開頭」才找到的。
    """
    out = bytearray(b"\x11\x22\x33\x44" * 9)
    for name, payload in tables.items():
        out += gamedata.table_pattern(name)
        out += struct.pack("<I", len(payload))
        out += payload
        out += b"\x00" * (-len(payload) % 4)
    return bytes(out)


def build_bundle(
    body: bytes,
    *,
    block_size: int = 4096,
    compress: bool = True,
    version: int = 8,
    pad_blocks_info: bool = True,
    info_at_end: bool = False,
) -> bytes:
    """把 *body* 包成一包 UnityFS。"""
    method = 3 if compress else 0

    def pack(chunk: bytes) -> bytes:
        if not compress:
            return chunk
        assert lz4_block is not None
        return lz4_block.compress(chunk, store_size=False)

    pieces = [body[i : i + block_size] for i in range(0, len(body), block_size)] or [b""]
    packed = [pack(piece) for piece in pieces]

    info = bytearray(b"\xab" * 16)
    info += struct.pack(">i", len(pieces))
    for piece, blob in zip(pieces, packed):
        info += struct.pack(">IIH", len(piece), len(blob), method)
    info += struct.pack(">i", 1)  # 一個節點
    info += struct.pack(">qqI", 0, len(body), 4)
    info += b"CAB-test\x00"
    info_blob = pack(bytes(info))

    flags = method
    if pad_blocks_info:
        flags |= 0x200
    if info_at_end:
        flags |= 0x80

    head = bytearray(gamedata.SIGNATURE + b"\x00")
    head += struct.pack(">I", version)
    head += b"6000.3.16f1\x00"
    head += b"6000.3.16f1\x00"
    size_at = len(head)
    head += struct.pack(">q", 0)  # 稍後回填
    head += struct.pack(">III", len(info_blob), len(info), flags)

    out = bytearray(head)
    if info_at_end:
        if version >= 8:
            out += b"\x00" * (-len(out) % 16)
        data_at = len(out)
        out += b"".join(packed)
        out += info_blob
    else:
        if pad_blocks_info:
            out += b"\x00" * (-len(out) % 16)
        out += info_blob
        if version >= 7:
            out += b"\x00" * (-len(out) % 16)
        data_at = len(out)
        out += b"".join(packed)

    struct.pack_into(">q", out, size_at, len(out))
    assert data_at >= 0
    return bytes(out)


MAP_JSON = json.dumps(
    {
        "100040110": {"streetName": "隱密之地", "mapName": "樹林底層"},
        "100040120": {"streetName": "隱密之地", "mapName": "猴子森林I"},
        "876009610": {"streetName": "隱藏地圖", "mapName": "廢棄的妖精的圖書館"},
        "999999999": {"streetName": "沒有名字的地方"},
        "888888888": {"mapName": "沒有地區的地圖"},
    },
    ensure_ascii=False,
).encode("utf-8")


def write_bundle(folder: Path, name: str, tables: dict[str, bytes], **kwargs) -> Path:
    path = folder / name
    path.write_bytes(build_bundle(serialized_file(tables), **kwargs))
    return path


@unittest.skipIf(lz4_block is None, "需要 lz4 才能組測試用的 bundle")
class TestBundleReader(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def test_roundtrip(self):
        path = write_bundle(self.folder, "json_a.bundle", {"Map": MAP_JSON})
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)

    def test_table_spanning_many_blocks(self):
        """地圖表有兩百 KB，一定會跨好幾個區塊 —— 跨塊拼接是這裡最容易寫錯的地方。"""
        big = json.dumps(
            {str(i): {"streetName": "地區", "mapName": f"地圖{i}"} for i in range(4000)},
            ensure_ascii=False,
        ).encode("utf-8")
        path = write_bundle(self.folder, "json_big.bundle", {"Map": big}, block_size=1024)
        self.assertEqual(gamedata.extract_table(path, "Map"), big)

    def test_signature_split_across_blocks(self):
        """簽名剛好被切在兩塊中間時也要找得到。"""
        body = serialized_file({"Map": MAP_JSON})
        offset = body.index(gamedata.table_pattern("Map"))
        path = self.folder / "json_split.bundle"
        # 讓區塊邊界落在簽名中間
        path.write_bytes(build_bundle(body, block_size=offset + 2))
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)

    def test_uncompressed_blocks(self):
        path = write_bundle(self.folder, "json_raw.bundle", {"Map": MAP_JSON}, compress=False)
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)

    def test_blocks_info_at_end(self):
        path = write_bundle(
            self.folder, "json_end.bundle", {"Map": MAP_JSON}, info_at_end=True
        )
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)

    def test_older_format_without_alignment(self):
        path = write_bundle(
            self.folder,
            "json_v6.bundle",
            {"Map": MAP_JSON},
            version=6,
            pad_blocks_info=False,
        )
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)

    def test_missing_table_returns_none(self):
        path = write_bundle(self.folder, "json_item.bundle", {"Item": MAP_JSON})
        self.assertIsNone(gamedata.extract_table(path, "Map"))

    def test_picks_the_right_table_among_several(self):
        tables = {"Item": b'{"item": 1}', "Map": MAP_JSON, "Mob": b'{"mob": 2}'}
        path = write_bundle(self.folder, "json_multi.bundle", tables)
        self.assertEqual(gamedata.extract_table(path, "Map"), MAP_JSON)
        self.assertEqual(gamedata.extract_table(path, "Mob"), b'{"mob": 2}')

    def test_name_is_matched_whole(self):
        """``MapName`` 不該被當成 ``Map``：簽名含長度，所以不會誤判。"""
        path = write_bundle(self.folder, "json_sim.bundle", {"MapName": MAP_JSON})
        self.assertIsNone(gamedata.extract_table(path, "Map"))

    def test_garbage_is_rejected(self):
        path = self.folder / "json_junk.bundle"
        path.write_bytes(b"not a unity bundle at all")
        with self.assertRaises(gamedata.BundleError):
            gamedata.extract_table(path, "Map")

    def test_truncated_bundle_is_rejected(self):
        full = build_bundle(serialized_file({"Map": MAP_JSON}))
        path = self.folder / "json_cut.bundle"
        path.write_bytes(full[: len(full) // 2])
        with self.assertRaises((gamedata.BundleError, struct.error)):
            gamedata.extract_table(path, "Map")


class TestTablePattern(unittest.TestCase):
    def test_map(self):
        self.assertEqual(gamedata.table_pattern("Map"), b"\x03\x00\x00\x00Map\x00")

    def test_already_aligned_name_gets_no_padding(self):
        self.assertEqual(gamedata.table_pattern("Item"), b"\x04\x00\x00\x00Item")


class TestParseMapTable(unittest.TestCase):
    def test_fields(self):
        records = gamedata.parse_map_table(MAP_JSON)
        by_id = {r.map_id: r for r in records}
        self.assertEqual(by_id["100040110"].street, "隱密之地")
        self.assertEqual(by_id["100040110"].name, "樹林底層")

    def test_entries_without_map_name_are_dropped(self):
        """缺 mapName 的留著沒有意義 —— 要比對的就是它。"""
        records = gamedata.parse_map_table(MAP_JSON)
        self.assertNotIn("999999999", {r.map_id for r in records})

    def test_entries_without_street_are_kept(self):
        """地區只是用來縮小範圍，缺了還是能比對地圖名。"""
        records = gamedata.parse_map_table(MAP_JSON)
        by_id = {r.map_id: r for r in records}
        self.assertEqual(by_id["888888888"].street, "")

    def test_bom_is_tolerated(self):
        records = gamedata.parse_map_table(b"\xef\xbb\xbf" + MAP_JSON)
        self.assertEqual(len(records), 4)

    def test_non_object_is_rejected(self):
        with self.assertRaises(ValueError):
            gamedata.parse_map_table(b"[1, 2, 3]")


class TestVocabulary(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = gamedata.MapVocabulary(gamedata.parse_map_table(MAP_JSON))

    def test_empty_is_falsy(self):
        self.assertFalse(gamedata.MapVocabulary())
        self.assertTrue(self.vocab)

    def test_streets_are_deduplicated_in_order(self):
        self.assertEqual(self.vocab.streets, ("隱密之地", "隱藏地圖"))

    def test_names_in_street(self):
        self.assertEqual(self.vocab.names_in("隱密之地"), ("樹林底層", "猴子森林I"))
        self.assertEqual(self.vocab.names_in("不存在"), ())

    def test_lengths(self):
        self.assertEqual(self.vocab.lengths()[9], 1)  # 廢棄的妖精的圖書館


@unittest.skipIf(lz4_block is None, "需要 lz4 才能組測試用的 bundle")
class TestSync(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.game = root / "game"
        self.folder = self.game / "Game_Data" / gamedata.BUNDLE_SUBPATH
        self.folder.mkdir(parents=True)
        self.home = root / "home"

        self._env = {k: os.environ.get(k) for k in ("MAPLEEXP_GAME_DIR", "MAPLEEXP_HOME")}
        os.environ["MAPLEEXP_GAME_DIR"] = str(self.game)
        os.environ["MAPLEEXP_HOME"] = str(self.home)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_finds_bundle_dir_through_data_folder(self):
        self.assertEqual(gamedata.find_bundle_dir(), self.folder)

    def test_sync_reads_the_client(self):
        write_bundle(self.folder, "json_aaa.bundle", {"Map": MAP_JSON})
        vocab = gamedata.sync()
        self.assertEqual(len(vocab), 4)
        self.assertEqual(vocab.source, "json_aaa.bundle")

    def test_sync_skips_bundles_without_the_table(self):
        write_bundle(self.folder, "json_item.bundle", {"Item": b'{"a": 1}'})
        write_bundle(self.folder, "json_map.bundle", {"Map": MAP_JSON})
        self.assertEqual(gamedata.sync().source, "json_map.bundle")

    def test_sync_ignores_unreadable_bundles(self):
        (self.folder / "json_broken.bundle").write_bytes(b"garbage")
        write_bundle(self.folder, "json_map.bundle", {"Map": MAP_JSON})
        self.assertEqual(gamedata.sync().source, "json_map.bundle")

    def test_sync_without_the_game_is_empty_not_an_error(self):
        os.environ["MAPLEEXP_GAME_DIR"] = str(self.game / "nope")
        self.assertFalse(gamedata.sync())

    def test_sync_with_no_bundles_is_empty(self):
        self.assertFalse(gamedata.sync())

    def test_cache_remembers_where_it_looked(self):
        write_bundle(self.folder, "json_aaa.bundle", {"Map": MAP_JSON})
        gamedata.sync()
        cache = json.loads(gamedata.cache_path().read_text(encoding="utf-8"))
        self.assertEqual(cache["bundle_dir"], str(self.folder))
        self.assertEqual(cache["bundle"], "json_aaa.bundle")

    def test_sync_follows_the_client_when_the_bundle_is_replaced(self):
        """客戶端改版後檔名（內容雜湊）會變；舊的提示不該讓我們讀到舊名單。"""
        write_bundle(self.folder, "json_old.bundle", {"Map": MAP_JSON})
        self.assertEqual(len(gamedata.sync()), 4)

        (self.folder / "json_old.bundle").unlink()
        updated = dict(json.loads(MAP_JSON))
        updated["111111111"] = {"streetName": "新地區", "mapName": "新地圖"}
        write_bundle(
            self.folder,
            "json_new.bundle",
            {"Map": json.dumps(updated, ensure_ascii=False).encode("utf-8")},
        )
        vocab = gamedata.sync()
        self.assertEqual(vocab.source, "json_new.bundle")
        self.assertIn("新地圖", vocab.names)

    def test_remembered_dir_is_used_when_the_game_is_not_running(self):
        write_bundle(self.folder, "json_aaa.bundle", {"Map": MAP_JSON})
        gamedata.sync()
        os.environ.pop("MAPLEEXP_GAME_DIR")
        self.assertEqual(gamedata.find_bundle_dir(), self.folder)


if __name__ == "__main__":
    unittest.main()
