"""跟遊戲客戶端的資料檔同步，取出官方的地圖名稱清單。

**為什麼需要這份清單。** 地圖名稱在遊戲裡的字級太小，筆畫在擷取的那一刻就已經
黏在一起了，放大也救不回來（實測最好的結果是「密之」「之地」這種殘缺片段）。
所以辨識策略改成：OCR 讀到幾個字就算幾個，再拿去跟一份**已知的地圖名稱清單**
比對，挑最接近的。有了清單，兩三個字就足以鎖定一張地圖。

**為什麼直接讀客戶端，而不是附一份快照。** 清單會隨改版增加，附檔案就會過期；
抓網站則是多一個會壞掉的外部相依。客戶端自己就有這份資料，而且玩家電腦上的
版本一定跟他正在玩的版本一致，所以每次啟動就跟客戶端同步一次。

**為什麼不用 UnityPy。** UnityPy 做得到，但會拉進十個相依套件（Pillow、各種
貼圖解碼器、fmod…），而這裡只需要「解壓 bundle + 掃位元組」兩件事，資料本身
就是純 JSON 字串，連 type tree 都不用解。所以自己走一遍 UnityFS 的格式，只多
依賴一個 ``lz4``。

格式細節（對照 Unity 6000.3，format version 8，實測驗證過）::

    UnityFS\\0  version:u32be  unityVersion:cstr  unityRevision:cstr
    size:i64be  compressedBlocksInfoSize:u32be  uncompressedBlocksInfoSize:u32be
    flags:u32be
    [flags & 0x200 -> 對齊到 16]
    <壓縮後的 blocksInfo>
    [version >= 7 -> 再對齊到 16]
    <資料區塊，逐塊壓縮>

blocksInfo 解壓後是 ``16 bytes hash`` + 區塊表 + 節點表；區塊全部串起來就是一個
標準的 SerializedFile。裡面每個 ``TextAsset`` 的位置長這樣（小端序）::

    <i32 名稱長度> <名稱> <對齊到 4 的填充> <i32 內容長度> <內容>

所以要拿 ``Map`` 這張表，只要掃 ``03 00 00 00 "Map" 00`` 就好。
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import data_dir

try:  # lz4 只在 Windows 上裝得到 wheel；缺了就退化成「沒有清單」
    import lz4.block as _lz4
except Exception:  # pragma: no cover - 取決於安裝環境
    _lz4 = None


# --------------------------------------------------------------------------- #
# UnityFS
# --------------------------------------------------------------------------- #

SIGNATURE = b"UnityFS"
_COMPRESSION_MASK = 0x3F
_BLOCKS_INFO_AT_END = 0x80
_BLOCKS_INFO_NEEDS_PADDING = 0x200
_ALIGNMENT = 16

_COMPRESSION_NONE = 0
_COMPRESSION_LZ4 = 3  # 2 = LZ4、3 = LZ4HC，解壓方式相同


class BundleError(Exception):
    """bundle 看起來不是我們認得的格式。"""


def _cstring(data: bytes, start: int) -> tuple[str, int]:
    end = data.find(b"\x00", start)
    if end < 0:
        raise BundleError("字串沒有結尾")
    return data[start:end].decode("utf-8", "replace"), end + 1


def _decompress(chunk: bytes, size: int, method: int) -> bytes:
    if method == _COMPRESSION_NONE:
        return chunk
    if method not in (2, _COMPRESSION_LZ4):
        raise BundleError(f"不支援的壓縮法 {method}")
    if _lz4 is None:
        raise BundleError("缺少 lz4")
    return _lz4.decompress(chunk, uncompressed_size=size)


@dataclass(frozen=True)
class _Block:
    uncompressed: int
    compressed: int
    flags: int

    @property
    def method(self) -> int:
        return self.flags & _COMPRESSION_MASK


def _read_header(data: bytes) -> tuple[int, list[_Block]]:
    """回傳 (資料區起點, 區塊表)。"""
    if not data.startswith(SIGNATURE):
        raise BundleError("不是 UnityFS")
    pos = len(SIGNATURE) + 1
    (version,) = struct.unpack_from(">I", data, pos)
    pos += 4
    _, pos = _cstring(data, pos)  # unityVersion
    _, pos = _cstring(data, pos)  # unityRevision
    pos += 8  # size:i64
    comp_size, uncomp_size, flags = struct.unpack_from(">III", data, pos)
    pos += 12

    limit = len(data)  # 資料區的結尾
    if flags & _BLOCKS_INFO_AT_END:
        info_at = len(data) - comp_size
        limit = info_at
    else:
        if flags & _BLOCKS_INFO_NEEDS_PADDING:
            pos = (pos + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
        info_at = pos
        pos += comp_size

    info = _decompress(
        data[info_at : info_at + comp_size], uncomp_size, flags & _COMPRESSION_MASK
    )
    if len(info) < 20:
        raise BundleError("blocksInfo 太短")

    cursor = 16  # 前面 16 bytes 是 hash
    (count,) = struct.unpack_from(">i", info, cursor)
    cursor += 4
    if not 0 < count <= 1_000_000:
        raise BundleError(f"區塊數不合理：{count}")
    blocks = []
    for _ in range(count):
        blocks.append(_Block(*struct.unpack_from(">IIH", info, cursor)))
        cursor += 10

    # version >= 7 的資料區也要對齊到 16。這裡不只照規則算，還順便驗證：
    # 壓縮後的區塊大小加起來應該剛好把檔案填滿，用這個條件挑起點最保險，
    # 改版動到對齊規則也不會悄悄讀出一堆垃圾。
    total = sum(b.compressed for b in blocks)
    aligned = (pos + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
    for start in ((aligned, pos) if version >= 7 else (pos, aligned)):
        if limit - start == total:
            return start, blocks
    raise BundleError(f"區塊大小總和 {total} 對不上資料區長度 {limit - pos}")


def iter_chunks(path: str | os.PathLike[str]) -> Iterator[bytes]:
    """逐塊解開 bundle 的資料區。

    刻意做成 generator：最大的那包 json bundle 壓縮前就 192 MB，一次全解會吃掉
    好幾百 MB 的記憶體，而我們只是要掃一段位元組而已。
    """
    data = Path(path).read_bytes()
    pos, blocks = _read_header(data)
    for block in blocks:
        chunk = data[pos : pos + block.compressed]
        pos += block.compressed
        yield _decompress(chunk, block.uncompressed, block.method)


def table_pattern(name: str) -> bytes:
    """``<i32 長度><名稱><對齊填充>``，也就是表格內容前面的那段簽名。"""
    raw = name.encode("utf-8")
    head = struct.pack("<i", len(raw)) + raw
    return head + b"\x00" * (-len(head) % 4)


_MAX_TABLE = 64 << 20


def extract_table(path: str | os.PathLike[str], name: str) -> bytes | None:
    """從 bundle 裡撈出名為 *name* 的 TextAsset 內容；沒有就回傳 None。"""
    pattern = table_pattern(name)
    tail = len(pattern) + 4  # 簽名可能跨在兩塊之間，要留一段重疊
    buf = bytearray()
    length: int | None = None
    hit = False

    for chunk in iter_chunks(path):
        buf += chunk
        while True:
            if not hit:
                index = buf.find(pattern)
                if index < 0:
                    if len(buf) > tail:
                        del buf[: len(buf) - tail]
                    break
                del buf[: index + len(pattern)]
                hit = True
            if length is None:
                if len(buf) < 4:
                    break
                (length,) = struct.unpack_from("<I", buf, 0)
                if not 2 <= length <= _MAX_TABLE:
                    # 撞到巧合的位元組而已，繼續往後找。
                    hit = False
                    length = None
                    continue
                del buf[:4]
            if len(buf) < length:
                break
            return bytes(buf[:length])
    return None


# --------------------------------------------------------------------------- #
# 找到客戶端
# --------------------------------------------------------------------------- #

BUNDLE_SUBPATH = Path("StreamingAssets") / "aa" / "w"
BUNDLE_GLOB = "json_*.bundle"

_FALLBACK_ROOTS = (
    r"C:\Program Files\Gamania\maplestory_classic",
    r"C:\Program Files (x86)\Gamania\maplestory_classic",
    r"C:\Gamania\maplestory_classic",
)


def bundle_dir_under(root: str | os.PathLike[str]) -> Path | None:
    """遊戲安裝目錄 -> 放 json bundle 的目錄。

    Unity 的版面是 ``<root>/<遊戲名>_Data/...``，但 ``_Data`` 的前綴跟執行檔名
    不一定完全一致（大小寫就常常不一樣），所以直接找 ``*_Data``。
    """
    base = Path(root)
    if base.is_file():
        base = base.parent
    candidates = [base / BUNDLE_SUBPATH]
    try:
        candidates += [d / BUNDLE_SUBPATH for d in base.glob("*_Data") if d.is_dir()]
    except OSError:
        pass
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_bundle_dir(hwnd: int | None = None) -> Path | None:
    """依序試：環境變數覆寫、正在跑的遊戲行程、上次記住的位置、常見安裝路徑。"""
    override = os.environ.get("MAPLEEXP_GAME_DIR")
    if override:
        # 明確指定了就以它為準。找不到就是找不到，不要偷偷掉回別的安裝位置 ——
        # 那會讓「我指到測試用的目錄」變成「其實讀到了另一套客戶端」。
        return bundle_dir_under(override)

    if hwnd:
        try:
            from .win32.windows import get_process_path

            path = get_process_path(hwnd)
        except Exception:
            path = ""
        if path:
            found = bundle_dir_under(path)
            if found:
                return found

    cached = _load_cache().get("bundle_dir")
    if cached and Path(cached).is_dir():
        return Path(cached)

    for root in _FALLBACK_ROOTS:
        found = bundle_dir_under(root)
        if found:
            return found
    return None


# --------------------------------------------------------------------------- #
# 地圖清單
# --------------------------------------------------------------------------- #

MAP_TABLE = "Map"


@dataclass(frozen=True)
class MapRecord:
    """客戶端的一筆地圖資料。``street`` 是上面那行（地區），``name`` 是下面那行。"""

    map_id: str
    street: str
    name: str


@dataclass(frozen=True)
class MapVocabulary:
    """比對用的地圖名稱清單。"""

    records: tuple[MapRecord, ...] = ()
    source: str = ""

    def __bool__(self) -> bool:
        return bool(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def streets(self) -> tuple[str, ...]:
        """所有地區名稱（去重，保留出現順序）。只有四十幾個，很好比對。"""
        return tuple(dict.fromkeys(r.street for r in self.records if r.street))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.name for r in self.records if r.name))

    def names_in(self, street: str) -> tuple[str, ...]:
        """某個地區底下的地圖名稱。先認出地區可以把候選數從上千壓到幾十。"""
        return tuple(
            dict.fromkeys(r.name for r in self.records if r.street == street and r.name)
        )

    def lengths(self) -> dict[int, int]:
        """字數 -> 有幾張地圖。排版的時候要估一行放得下幾個字會用到。"""
        counts: dict[int, int] = {}
        for record in self.records:
            counts[len(record.name)] = counts.get(len(record.name), 0) + 1
        return dict(sorted(counts.items()))


def parse_map_table(blob: bytes) -> tuple[MapRecord, ...]:
    """把 ``Map`` 表的 JSON 轉成 :class:`MapRecord`。

    客戶端的資料不保證每筆都齊全（實測 1671 筆裡有 7 筆缺欄位），缺 ``mapName``
    的直接跳過 —— 那是拿來比對的主體，缺了沒有意義；缺 ``streetName`` 還留著，
    地區只是用來縮小範圍的。
    """
    raw = json.loads(blob.decode("utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Map 表不是物件")
    records = []
    for map_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("mapName") or "").strip()
        if not name:
            continue
        street = str(entry.get("streetName") or "").strip()
        records.append(MapRecord(str(map_id), street, name))
    return tuple(records)


def read_map_table(
    directory: str | os.PathLike[str], prefer: str | None = None
) -> tuple[Path, bytes] | None:
    """在 bundle 目錄裡找出含 ``Map`` 表的那包。

    檔名是內容雜湊（``json_<hash>.bundle``），改版就會變，所以不能寫死，只能掃。
    *prefer* 是上次命中的檔名，先試它；改版後檔名會變，就自動回到掃描。

    掃描從小的掃起：地圖表在一包兩 MB 多的 bundle 裡，而最大那包有 192 MB ——
    由小往大掃通常解開十幾 MB 就命中，碰不到大的那幾包。
    """
    folder = Path(directory)
    try:
        bundles = sorted(folder.glob(BUNDLE_GLOB), key=lambda p: p.stat().st_size)
    except OSError:
        return None
    if prefer:
        bundles.sort(key=lambda p: p.name != prefer)
    for bundle in bundles:
        try:
            blob = extract_table(bundle, MAP_TABLE)
        except (BundleError, OSError, struct.error):
            continue
        if blob:
            return bundle, blob
    return None


# --------------------------------------------------------------------------- #
# 啟動時同步
# --------------------------------------------------------------------------- #


def cache_path() -> Path:
    return data_dir() / "gamedata.json"


def _load_cache() -> dict:
    try:
        with cache_path().open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(payload: dict) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        temp.replace(path)
    except OSError:
        pass  # 快取壞了頂多下次慢一點，不該讓程式起不來


def sync(hwnd: int | None = None) -> MapVocabulary:
    """每次啟動都直接從客戶端讀一次地圖清單。

    刻意不快取清單內容：整個流程（找目錄、解壓、解析 1671 筆）實測 43 毫秒，
    存一份會過期的副本省下的那幾十毫秒，不值得冒「客戶端改版了但我們還在用舊
    名單」的風險。快取裡只記上次在哪裡找到、命中哪一包，當作下次的起點 ——
    尤其是遊戲沒開的時候，沒有行程可以問路。

    整個過程不會讓程式起不來：遊戲沒裝、裝在找不到的地方、格式改掉、``lz4``
    沒裝，都只是回傳一份空清單，地圖名稱退回「使用者自己命名一次」的流程。
    """
    directory = find_bundle_dir(hwnd)
    if directory is None:
        return MapVocabulary()

    cache = _load_cache()
    prefer = cache.get("bundle") if cache.get("bundle_dir") == str(directory) else None

    found = read_map_table(directory, prefer=prefer)
    if found is None:
        return MapVocabulary()
    bundle, blob = found
    try:
        records = parse_map_table(blob)
    except (ValueError, UnicodeDecodeError):
        return MapVocabulary()
    if not records:
        return MapVocabulary()

    _save_cache(
        {
            "bundle_dir": str(directory),
            "bundle": bundle.name,
            "synced_at": int(time.time()),
            "count": len(records),
        }
    )
    return MapVocabulary(records, source=bundle.name)
