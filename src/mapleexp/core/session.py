"""把擷取、辨識、追蹤、儲存串成一條可以被驅動的管線。

這裡刻意不自己開迴圈跑到底，而是把一次取樣抽成 :meth:`TrackingSession.tick`。
原因是驅動者有兩種：主控台模式用一個簡單的 while，浮窗模式用 Tk 的 ``after``。
兩者共用同一個 tick，就不會出現「只有其中一種模式才正確」的狀況。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import config as config_module
from ..autosetup import AutoSetup, auto_configure, load_templates
from ..config import Config
from ..vision import ocr, pngio
from .. import gamedata
from . import jobvocab, mapvocab
from ..vision.identity import (
    Identity,
    LevelReader,
    MapWatcher,
    fingerprint,
    scan,
    text_bands,
)
from ..vision.panels import PanelTracker
from ..vision.reader import StatusReader
from ..vision.templates import TemplateSet
from ..win32.capture import CaptureError, WindowCapturer
from ..win32.windows import WindowInfo, find_window, window_exists
from .exptable import ExpTable
from .reading import StatusReading
from .stats import ExpSeries
from .tracker import PAUSED, Event, Snapshot, Tracker

# 地圖名稱沒有讀到滿分時，每隔這麼久重讀一次。
# 「重試是免費的，錯誤是黏著的」—— 地圖靠像素指紋認得，同一張圖會待很久，
# 多試幾次就有機會碰到 OCR 讀得完整的那一次；而讀錯的名字不會自己浮出來。
MAP_NAME_RETRY_SEC = 5.0

# 面板搜尋要對每個候選標題列跑 OCR（約 0.9 秒），不能常跑。
# 玩家不會一直搬視窗，所以「找到就記住、用壞了再找」就夠了。這個間隔只套用在
# **搜尋失敗之後**：成功找到的那次不算，否則面板一跟丟就得先白等 20 秒。
PANEL_REFRESH_INTERVAL = 20.0
MINIMAP_TITLE = "小地圖"
# 狀態列讀得到、但地圖名連續這麼多格讀不出來，才認定面板被搬走或關掉。
# 只看一格就忘掉的話，換圖時的讀取畫面會把面板踢掉，進新圖後要等上面那 20 秒
# 才會重找 —— 實際用起來就是「每次換圖都要二十幾秒才認得地圖」。
MAP_LOST_TICKS = 5

# 角色名與職業至少每這麼多秒重新確認一次。
# 真正要跑 OCR 的只有「那塊像素變了」的時候 —— 像素沒變，重跑必定得到
# 一樣的答案，白白讓浮窗卡半秒。換角色時像素會變，所以反應反而更快。
CHARACTER_RECHECK_SEC = 5.0

# 遊戲重開後會是**新的視窗 handle**，舊的永遠讀不到。
# 連續失敗這麼多次就回頭找一次視窗，但要有間隔，不要每格都掃。
REACQUIRE_AFTER_MISSES = 5
REACQUIRE_INTERVAL_SEC = 5.0


class SetupError(RuntimeError):
    """開始追蹤前就失敗（找不到視窗、沒有校準等等）。"""


@dataclass
class TickResult:
    reading: StatusReading
    events: list[Event]
    snapshot: Snapshot


# 段內峰值用的時間窗，以及窗內至少要跨多少活躍秒數才採計 —— 窗剛開始只有幾個
# 點時回歸斜率會亂跳，一隻怪就能噴出天文數字。
PEAK_WINDOW_SEC = 300
PEAK_MIN_SPAN_SEC = 60


@dataclass
class Segment:
    """同一隻角色在同一張地圖上連續練的一段。

    報表要的數字（練了多少、幾 %、多久、平均與峰值 /h）全部在這裡**線上算好**，
    結束時整筆寫進 ``segments`` 表。這樣就不需要把每秒的取樣存下來 —— 那每小時
    要 230 KB，而且除了事後算峰值之外沒有任何用途。
    """

    map_id: str
    started_at: float
    character: str = ""
    job: str = ""
    ended_at: float | None = None
    exp: int = 0
    active_sec: float = 0.0
    name: str = ""
    start_level: int | None = None
    end_level: int | None = None
    start_pct: float | None = None
    end_pct: float | None = None
    peak_exp_per_hour: float | None = None
    deaths: int = 0
    levelups: int = 0
    # 段內自己的取樣序列，只拿來算峰值。用段內的活躍秒數當 x 軸，窗就不會把上一段
    # 的點混進來。
    _series: ExpSeries = field(
        default_factory=lambda: ExpSeries([PEAK_WINDOW_SEC]), repr=False, compare=False
    )

    @property
    def exp_per_hour(self) -> float | None:
        if self.active_sec <= 0:
            return None
        return self.exp / (self.active_sec / 3600.0)

    @property
    def pct_gained(self) -> float | None:
        """這一段練了幾 %。升級過就把跨過的整級加回去（死亡掉的自然反映在終值上）。"""
        if self.start_pct is None or self.end_pct is None:
            return None
        return self.levelups * 100.0 + (self.end_pct - self.start_pct)

    def add(self, exp_delta: int, active_delta: float, snapshot: Snapshot, events: list[Event]) -> None:
        """把這一格的增量記進來。"""
        self.exp += exp_delta
        self.active_sec += max(0.0, active_delta)
        if snapshot.exp_pct is not None:
            if self.start_pct is None:
                self.start_pct = snapshot.exp_pct
                self.start_level = snapshot.level
            self.end_pct = snapshot.exp_pct
            self.end_level = snapshot.level
        self.deaths += sum(1 for e in events if e.kind == "death")
        self.levelups += sum(1 for e in events if e.kind == "levelup")

        self._series.add(self.active_sec, self.exp)
        rate = self._series.rate(PEAK_WINDOW_SEC)
        if rate.valid and rate.span_sec >= PEAK_MIN_SPAN_SEC:
            if self.peak_exp_per_hour is None or rate.exp_per_hour > self.peak_exp_per_hour:
                self.peak_exp_per_hour = rate.exp_per_hour

    def to_row(self) -> dict:
        return {
            "map_id": self.map_id,
            "character": self.character,
            "job": self.job,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exp": self.exp,
            "active_sec": self.active_sec,
            "start_level": self.start_level,
            "end_level": self.end_level,
            "start_pct": self.start_pct,
            "end_pct": self.end_pct,
            "peak_exp_per_hour": self.peak_exp_per_hour,
            "deaths": self.deaths,
            "levelups": self.levelups,
        }


class TrackingSession:
    """一次追蹤。"""

    def __init__(
        self,
        cfg: Config,
        label: str = "",
        store=None,
        templates_path: Path | None = None,
        exp_table_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.label = label
        self.store = store
        self._templates_path = templates_path or config_module.templates_dir() / "digits.json"
        self._level_templates_path = config_module.templates_dir() / "level.json"
        self._exp_table_path = exp_table_path or config_module.exp_table_path()

        self.window: WindowInfo | None = None
        self.capturer: WindowCapturer | None = None
        self.reader: StatusReader | None = None
        self.table = ExpTable(self._exp_table_path)
        self.tracker = Tracker(cfg.tracker, self.table)

        self.level_reader = LevelReader(TemplateSet.load(self._level_templates_path))
        self.map_watcher = MapWatcher()
        self.panels = PanelTracker()
        self._panel_refresh_at = None   # 上次搜尋失敗的時刻；None = 沒失敗過
        self._map_misses = 0
        self.identity = Identity()
        self.segments: list[Segment] = []
        # 角色名與職業是 OCR 讀出來的文字，給資料庫與報表用（浮窗顯示的是原始
        # 像素）。每 CHARACTER_RECHECK_SEC 秒確認一次，像素變了才真的重跑 OCR。
        self.character = ""
        self.job = ""
        self._character_checked_at = 0.0
        self._character_key = ""
        self._map_name_scores: dict[str, float] = {}
        self._map_name_checked_at = 0.0
        self._reacquired_at = 0.0
        # 身分掃描產生的事件（換圖、換角色），下一次 tick 併進追蹤器的事件裡回報。
        self._pending_events: list[Event] = []
        self._templates = None
        self.map_names: dict[str, str] = {}
        # 官方地圖名稱清單，setup() 時跟客戶端同步。
        self.map_vocab = gamedata.MapVocabulary()

        self._start_level_recorded = False
        self._next_tick = 0.0
        self._last_cum = 0
        self._last_active = 0.0
        # 使用者在結束視窗刪掉的段，要從這一場的總計裡扣掉。
        self._deducted_exp = 0
        self._deducted_active = 0.0
        self._deducted_deaths = 0
        self._deducted_levelups = 0
        # 升級後預期在等級方塊看到的數字，拿來把沒見過的字形教給模板。
        self._expected_level: int | None = None
        self.last_error = ""
        self.template_source = ""
        self.auto: AutoSetup | None = None

    # ------------------------------------------------------------------ #
    # 準備
    # ------------------------------------------------------------------ #

    def setup(self) -> WindowInfo:
        """準備開始追蹤。

        不需要事先校準：模板用內建的，ROI 由程式自己在畫面上找。
        只有在自動偵測也失敗時才會要求使用者跑校準精靈。
        """
        templates, self.template_source = load_templates(self._templates_path)
        if len(templates) == 0:
            raise SetupError("找不到任何字元模板（內建模板遺失？），請執行 calibrate")
        self._templates = templates

        window = find_window(
            self.cfg.capture.window_title_contains, self.cfg.capture.process_name
        )
        if window is None:
            raise SetupError(
                "找不到遊戲視窗（標題含 "
                f"{self.cfg.capture.window_title_contains!r} 或行程 "
                f"{self.cfg.capture.process_name!r}），請先啟動遊戲"
            )

        self.window = window
        self.capturer = WindowCapturer(window.hwnd, self.cfg.capture.backend)

        # 每次啟動都跟客戶端的資料檔對一次地圖名稱清單。要在找到視窗之後做，
        # 因為安裝位置是問遊戲行程要的。對不到只是少一個輔助，不影響追蹤。
        self.map_vocab = gamedata.sync(window.hwnd)

        self.auto = auto_configure(
            self.cfg, self.capturer, templates, template_source=self.template_source
        )
        if not self.auto.ok:
            self.capturer.close()
            self.capturer = None
            raise SetupError(self.auto.message)

        self.reader = StatusReader(
            self.capturer,
            self.cfg.reader,
            templates,
            debug_dir=config_module.debug_dir() if self.cfg.save_debug_frames else None,
        )

        if self.auto.roi_source == "自動偵測":
            try:
                self.cfg.save()
            except OSError:
                pass

        if self.store is not None and self.store.session_id is None:
            self.store.open_session(self.label)

        self._next_tick = time.perf_counter()
        return window

    @property
    def vocab_message(self) -> str:
        """地圖清單同步結果，給啟動訊息與 doctor 用。"""
        if not self.map_vocab:
            return "找不到客戶端的地圖資料（地圖名稱需要自己命名一次）"
        return (
            f"{len(self.map_vocab)} 張地圖、{len(self.map_vocab.streets)} 個地區"
            f"（來源 {self.map_vocab.source}）"
        )

    # ------------------------------------------------------------------ #
    # 執行
    # ------------------------------------------------------------------ #

    def tick(self) -> TickResult:
        """讀一次、更新一次。不會丟例外 —— 擷取失敗會變成一筆失敗的取樣。"""
        if self.reader is None:
            raise SetupError("尚未呼叫 setup()")

        try:
            reading = self.reader.read()
        except CaptureError as exc:
            self.last_error = str(exc)
            reading = StatusReading.failed(time.perf_counter(), time.time(), str(exc))
        else:
            self.last_error = reading.reason if not reading.ok else ""

        # 先辨識身分、再餵追蹤器。順序有意義：換角色必須在這一筆被算進統計**之前**
        # 知道，追蹤器才能把它當成新基準而不是「經驗暴跌」。
        #
        # 每一格都掃。實測 1920x1080 上 scan() 整包 2.2ms（locate_level_zone 1.3ms、
        # locate_map_zone 0.6ms），每秒一次負擔得起；唯一昂貴的 panels.refresh
        # （約 0.8 秒）只在小地圖面板跟丟時才跑，而且有 PANEL_REFRESH_INTERVAL 節流。
        # 這裡刻意不用「取樣計數取餘數」當閘門 —— 曾經寫成 ``count % N == 1`` 再把
        # N 改成 1，任何整數 % 1 都是 0，整個身分管線就悄悄停掉了，而所有單元測試
        # 照樣全綠。
        if reading.ok and self.tracker.state == PAUSED:
            # 畫面消失又回來（切頻、讀取、選角色）—— 回來的不一定是同一隻角色，
            # 不等 CHARACTER_RECHECK_SEC，這一格就重新確認。
            self._character_checked_at = 0.0
        level_before_scan = self.level_reader.level
        self._scan_identity(status_ok=reading.ok)
        self._maybe_refresh_map_name()

        # 等級是從橘色方塊那邊讀來的（狀態列 ROI 裡沒有等級）。
        # 在餵給追蹤器之前補上去，升級判定與等級經驗表才用得到它。
        if reading.ok and reading.level is None and self.level_reader.level is not None:
            reading = replace(reading, level=self.level_reader.level)

        events = list(self.tracker.feed(reading))
        snapshot = self.tracker.snapshot()
        if not reading.ok:
            # 放在 feed() 之後，這一次的失敗才會先計入 consecutive_misses。
            self._maybe_reacquire_window()

        if any(e.kind == "levelup" for e in events) and level_before_scan is not None:
            # 升級了。下一次掃描若模板與 OCR 都認不出方塊裡的新字形，就用「舊等級 + 1」
            # 教給模板。要用掃描**之前**的等級 —— 這一格的掃描可能已經把
            # level_reader.level 更新成新等級，再 +1 就教錯字了。
            self._expected_level = level_before_scan + 1
        events.extend(self._pending_events)
        self._pending_events.clear()

        self._account_segment(snapshot, reading.wall, events)

        if self.store is not None:
            if not self._start_level_recorded:
                level = reading.level if reading.level is not None else self.level_reader.level
                if level is not None:
                    self.store.set_start_level(level)
                    self._start_level_recorded = True
            self.store.add_events(events)

        return TickResult(reading=reading, events=events, snapshot=snapshot)

    def _maybe_reacquire_window(self) -> None:
        """遊戲重開之後要接回新的視窗。

        重開遊戲會換一個 handle，舊的那個 **永遠** 讀不到 —— 不重新找的話
        追蹤器就會一直卡在「連續 N 次讀不到經驗值」直到使用者自己發現。
        """
        if self.window is None or self._templates is None:
            return
        if self.tracker.snapshot().consecutive_misses < REACQUIRE_AFTER_MISSES:
            return
        now = time.monotonic()
        if now - self._reacquired_at < REACQUIRE_INTERVAL_SEC:
            return
        self._reacquired_at = now

        if window_exists(self.window.hwnd):
            return      # 視窗還在，讀不到是別的原因（被遮住、讀取畫面…）

        window = find_window(
            self.cfg.capture.window_title_contains, self.cfg.capture.process_name
        )
        if window is None or window.hwnd == self.window.hwnd:
            return

        if self.capturer is not None:
            self.capturer.close()
        self.window = window
        self.capturer = WindowCapturer(window.hwnd, self.cfg.capture.backend)
        self.reader = StatusReader(
            self.capturer,
            self.cfg.reader,
            self._templates,
            debug_dir=config_module.debug_dir() if self.cfg.save_debug_frames else None,
        )
        # 面板的位置跟著新視窗重算。
        self.panels = PanelTracker()
        self._panel_refresh_at = None   # 上次搜尋失敗的時刻；None = 沒失敗過
        self._map_misses = 0
        self.last_error = "遊戲視窗已重新連接"

    # ------------------------------------------------------------------ #
    # 身分與地圖
    # ------------------------------------------------------------------ #

    def _scan_identity(self, status_ok: bool = True) -> None:
        """掃一次畫面上的等級／角色／地圖。

        ``status_ok`` 是這一格狀態列讀不讀得到。讀不到（讀取畫面、切頻、選角色）
        時小地圖也不會在，那種畫面上搜尋面板只會把搜尋失敗的冷卻用掉、
        忘掉面板更是誤判 —— 所以那幾格只掃等級與角色，地圖的部分按兵不動。
        """
        if self.capturer is None:
            return
        try:
            frame = self.capturer.capture_client()
        except CaptureError:
            return

        width, height = frame.width, frame.height
        exp_rect = (
            self.cfg.reader.exp_roi.to_pixels(width, height)
            if self.cfg.reader.exp_roi.is_set()
            else None
        )
        identity = scan(
            frame.pixels,
            self.level_reader,
            exp_rect=exp_rect,
            map_rect=self._minimap_rect(frame.pixels, allow_search=status_ok),
        )

        # 升級時我們知道新等級 = 舊等級 + 1，可以把沒見過的字形學起來。
        # 等級的字型跟狀態列不同，而且一開始只看得到目前等級用到的數字，
        # 這是唯一不用人工標記就能補齊的方法。只在模板與 OCR 都認不出來時才教 ——
        # 認得出來就不需要；讀出來的若不是預期值，教下去會把錯的字形永久記住。
        if self._expected_level is not None and identity.zone is not None:
            if identity.level is None:
                self.level_reader.teach(identity.zone, self._expected_level)
                identity.level = self.level_reader.level
            self._expected_level = None

        self.identity = identity

        if identity.name_image is not None:
            self._maybe_read_character(identity.name_image)

        if identity.map_id:
            self._map_misses = 0
            if self.map_watcher.feed(identity.map_id, identity.map_image):
                self._on_map_change(self.map_watcher.map_id, self.map_watcher.image)
        elif status_ok and self.panels.get(MINIMAP_TITLE) is not None:
            # 面板還記著、狀態列也正常，卻讀不出地圖名 —— 連續幾格都這樣才算
            # 面板被搬走或關掉，忘掉它、下一格立刻重找。
            self._map_misses += 1
            if self._map_misses >= MAP_LOST_TICKS:
                self._map_misses = 0
                self.panels.forget(MINIMAP_TITLE)
                self._panel_refresh_at = None   # 上次搜尋失敗的時刻；None = 沒失敗過

    def _minimap_rect(
        self, frame, allow_search: bool = True
    ) -> tuple[int, int, int, int] | None:
        """小地圖面板的內容區。沒有就（有節制地）重新搜尋一次。

        冷卻只在搜尋**失敗**後開始算：成功找到之後面板會一直被記著，等到真的
        跟丟時應該馬上重找，而不是先白等 20 秒。
        """
        panel = self.panels.get(MINIMAP_TITLE)
        if panel is not None:
            return panel.body_rect
        if not allow_search:
            return None
        now = time.monotonic()
        if (
            self._panel_refresh_at is not None
            and now - self._panel_refresh_at < PANEL_REFRESH_INTERVAL
        ):
            return None
        self.panels.refresh(frame)
        panel = self.panels.get(MINIMAP_TITLE)
        if panel is None:
            self._panel_refresh_at = now
            return None
        return panel.body_rect

    def _maybe_read_character(self, image) -> None:
        """定期重新確認角色名與職業。

        用像素指紋當閘門：每 ``CHARACTER_RECHECK_SEC`` 秒檢查一次，但只有那塊
        像素真的變了才重跑 OCR。OCR 一次要幾百毫秒，每次都跑會讓浮窗週期性卡頓，
        而且像素沒變的話答案一定一樣。
        """
        now = time.monotonic()
        if now - self._character_checked_at < CHARACTER_RECHECK_SEC:
            return
        self._character_checked_at = now

        key = fingerprint(image)
        if key == self._character_key and self.character:
            return
        self._character_key = key
        self._read_character(image)

    def _read_character(self, image) -> None:
        """狀態列上第一行是職業、第二行是角色名。

        **分兩行各自辨識**，不要整塊丟給 OCR —— 兩行靠得很近，引擎會把它們
        併成一行（實測得到 ``獵人卍弘法艾德卍`` 這種黏在一起的結果）。
        我們本來就知道行在哪裡，自己切比較準。

        職業是封閉集合，OCR 結果再跟 :mod:`jobvocab` 的清單比對一次：資料庫裡就有
        ``槍騎兵`` 被讀成 ``搶騎兵`` 的實際紀錄，一個字的差距靠編輯距離就能修回來。
        角色名沒有清單可比（而且可能含 ``卍`` 這類符號），讀到什麼就是什麼。
        """
        bands = text_bands(image)
        if not bands:
            return
        texts = []
        for top, bottom in bands[:2]:
            texts.append(ocr.recognize_best(image[top:bottom]).replace("\n", ""))
        texts = [t for t in texts if t]
        if not texts:
            return      # 讀不到就保留上一次的結果，不要把已知的名字洗掉
        if len(texts) >= 2:
            guess = jobvocab.identify(texts[0])
            # 配不上清單（讀到的字不像任何職業、或兩個職業同分分不出來）時，
            # 不要拿它蓋掉已經讀對的職業 —— 每 5 秒重讀一次，偶爾一格讀壞很正常。
            # 還沒有職業時才照原文存，至少讓人看得到 OCR 讀到了什麼。
            job = guess.name if guess.matched or not self.job else self.job
            character = texts[1]
        else:
            job, character = self.job, texts[0]
        if self.character and character != self.character:
            self._on_character_change(self.character, character)
        self.job, self.character = job, character
        if self.store is not None:
            self.store.set_identity(self.character, self.job)

    def _on_character_change(self, previous: str, current: str) -> None:
        """換角色了：舊的一段到此為止，統計基準重設。

        兩隻角色的經驗值之間沒有任何可比較的關係，不重設的話追蹤器會把差值當成
        死亡或異常。地圖多半沒變，新的一段會在下一格由 ``_account_segment`` 開。
        """
        now = time.time()
        self._close_segment(now)
        self.tracker.mark_discontinuity()
        self._pending_events.append(
            Event(kind="character", wall=now, detail=f"換角色：{previous} -> {current}")
        )

    def _on_map_change(self, map_id: str, image) -> None:
        now = time.time()
        self._close_segment(now)

        name = self.map_names.get(map_id, "")
        if not name and self.store is not None:
            name = self.store.map_name(map_id)
        if not name and image is not None:
            name = self._name_from_image(map_id, image, log=True)
        self.map_names[map_id] = name

        self._open_segment(map_id, now)
        if self.store is not None:
            thumbnail = None
            if image is not None:
                try:
                    thumbnail = pngio.encode_png(pngio.bgra_to_rgb(image))
                except Exception:
                    thumbnail = None
            self.store.remember_map(map_id, thumbnail, name)

    def _name_from_image(self, map_id: str, image, log: bool) -> str:
        """把地圖名稱區的影像變成名字，順便記下這次比對有多像。

        地圖名的字很小，OCR 讀到的可能完整、也可能缺字，所以不直接用，而是拿去
        跟客戶端的官方清單比對、挑最接近的一張。

        ``log=True`` 會留一個 ``map`` 事件記下「讀到什麼 -> 選了什麼」：實測錯誤率時
        要看得到這個，光看結果判斷不了是 OCR 讀壞還是比對挑錯。重試的呼叫端只在
        結果變好時才記 —— 每 5 秒重讀一次、每次都記，一小時就是七百筆一樣的事件。
        """
        text = ocr.recognize_best(image)
        if not self.map_vocab:
            # 沒有清單（找不到客戶端）時只能用原始 OCR 結果，聊勝於無。
            return text.replace(chr(10), "")

        # 一定會挑出一個最接近的 —— 沒有門檻。猜錯的代價只是名字不對，使用者
        # 點一下就能改；真正用來區分地圖的是像素指紋，跟文字無關。
        guess = mapvocab.identify(text, self.map_vocab)
        self._map_name_scores[map_id] = guess.score
        if log:
            # 領先 0 代表這次是在同分裡硬挑一個。
            self._pending_events.append(
                Event(
                    kind="map",
                    wall=time.time(),
                    detail=(
                        f"OCR {guess.read!r} -> {guess.full_name or '（無）'}"
                        f"（相似度 {guess.score:.2f}、領先 {guess.lead:.2f}"
                        + (f"、第二名 {guess.runner_up}" if guess.runner_up else "")
                        + "）"
                    ),
                )
            )
        return guess.full_name

    def _maybe_refresh_map_name(self) -> None:
        """目前這張地圖還沒讀到滿分的話，隔一陣子重讀一次。

        OCR 在這個字級下每一幀的結果會跳動 —— 換圖當下剛好讀壞，名字就會一路錯
        下去。地圖待的時間通常以分鐘計，重試幾乎不花成本，而且只留更好的結果。
        """
        map_id = self.map_watcher.map_id
        image = self.map_watcher.image
        if not map_id or image is None or not self.map_vocab:
            return
        if self._map_name_scores.get(map_id, 0.0) >= 1.0:
            return      # 已經一字不差，不可能更好
        now = time.monotonic()
        if now - self._map_name_checked_at < MAP_NAME_RETRY_SEC:
            return
        self._map_name_checked_at = now

        previous = self._map_name_scores.get(map_id, -1.0)
        name = self._name_from_image(map_id, image, log=False)
        if self._map_name_scores.get(map_id, 0.0) <= previous:
            self._map_name_scores[map_id] = previous   # 這次沒有更好，保留原本的
            return
        self._pending_events.append(
            Event(
                kind="map",
                wall=time.time(),
                detail=f"重讀後改為 {name}（相似度 {self._map_name_scores[map_id]:.2f}）",
            )
        )
        self.rename_map(map_id, name)

    def _account_segment(
        self, snapshot: Snapshot, wall: float, events: list[Event]
    ) -> None:
        """把這一格的增量記到目前這一段（這隻角色、這張地圖）。"""
        exp_delta = snapshot.cum_net - self._last_cum
        active_delta = snapshot.active_sec - self._last_active
        self._last_cum = snapshot.cum_net
        self._last_active = snapshot.active_sec

        current = self.current_segment
        if current is None:
            if not self.map_watcher.map_id:
                return      # 還不知道在哪張地圖，這一格的增量不歸任何一段
            current = self._open_segment(self.map_watcher.map_id, wall)
        current.add(exp_delta, active_delta, snapshot, events)

    def _open_segment(self, map_id: str, wall: float) -> Segment:
        segment = Segment(
            map_id=map_id,
            started_at=wall,
            character=self.character,
            job=self.job,
            name=self.map_names.get(map_id, ""),
        )
        self.segments.append(segment)
        return segment

    def _close_segment(self, wall: float) -> None:
        current = self.current_segment
        if current is not None:
            current.ended_at = wall

    @property
    def current_map_id(self) -> str:
        return self.map_watcher.map_id

    @property
    def current_segment(self) -> Segment | None:
        """還沒結束的那一段；換圖或換角色之後、下一格開新段之前是 None。"""
        if self.segments and self.segments[-1].ended_at is None:
            return self.segments[-1]
        return None

    # ------------------------------------------------------------------ #

    def rename_map(self, map_id: str, name: str) -> None:
        """改掉某張地圖的名稱，立刻寫入資料庫。

        改名字不會影響統計 —— 真正用來區分地圖的是那塊像素的指紋，跟文字無關。
        """
        name = (name or "").strip()
        self.map_names[map_id] = name
        for segment in self.segments:
            if segment.map_id == map_id:
                segment.name = name
        if self.store is not None and name:
            self.store.name_map(map_id, name)

    def set_level(self, level: int) -> int:
        """手動告訴程式目前等級，順便把當下的字形學起來。

        只需要做這一次：之後每次升級都能自動學到新數字，十個數字很快就齊了。
        """
        if self.capturer is None:
            return 0
        try:
            frame = self.capturer.capture_client()
        except CaptureError:
            return 0
        width, height = frame.width, frame.height
        exp_rect = (
            self.cfg.reader.exp_roi.to_pixels(width, height)
            if self.cfg.reader.exp_roi.is_set()
            else None
        )
        identity = scan(frame.pixels, None, exp_rect=exp_rect)
        learned = self.level_reader.teach(identity.zone, level)
        self.level_reader.templates.save(self._level_templates_path)
        return learned

    def sleep_until_next_tick(self) -> float:
        """算出距離下一次取樣要等多久，並推進排程。

        用絕對時間排程（而不是每次都 sleep 固定秒數），取樣間隔才不會因為
        每次處理耗時而慢慢漂移。
        """
        interval = max(0.1, float(self.cfg.tracker.sample_interval))
        self._next_tick += interval
        now = time.perf_counter()
        if self._next_tick < now:
            # 落後太多（例如系統卡住）就重新對齊，不要狂補。
            self._next_tick = now + interval
            return interval
        return self._next_tick - now

    def set_interval(self, seconds: float) -> float:
        """改變取樣間隔並立刻生效。"""
        self.cfg.tracker.sample_interval = max(0.2, min(10.0, float(seconds)))
        self._next_tick = time.perf_counter()
        return self.cfg.tracker.sample_interval

    # ------------------------------------------------------------------ #

    def delete_segment(self, segment: Segment) -> bool:
        """把某一段從這一場拿掉（使用者在結束視窗按了垃圾桶）。

        段裡的經驗、時間、死亡、升級也要從**這一場的總計**扣掉，否則 ``sessions``
        表的數字跟 ``segments`` 加總對不上，報表的「各標記效率」會多算。
        """
        try:
            self.segments.remove(segment)
        except ValueError:
            return False
        self._deducted_exp += segment.exp
        self._deducted_active += segment.active_sec
        self._deducted_deaths += segment.deaths
        self._deducted_levelups += segment.levelups
        return True

    def _session_snapshot(self) -> Snapshot:
        """這一場的總計，已經扣掉被刪除的段。"""
        snapshot = self.tracker.snapshot()
        if not self._deducted_exp and not self._deducted_active:
            return snapshot
        return replace(
            snapshot,
            cum_net=max(0, snapshot.cum_net - self._deducted_exp),
            cum_gross=max(0, snapshot.cum_gross - self._deducted_exp),
            active_sec=max(0.0, snapshot.active_sec - self._deducted_active),
            deaths=max(0, snapshot.deaths - self._deducted_deaths),
            levelups=max(0, snapshot.levelups - self._deducted_levelups),
        )

    def finish(self, save: bool = True) -> None:
        """結束這一場。``save=False`` 就把這場紀錄整個丟掉。"""
        self._close_segment(time.time())
        if self.store is not None:
            if save:
                self.store.set_identity(self.character, self.job)
                for map_id, name in self.map_names.items():
                    if name:
                        self.store.name_map(map_id, name)
                self.store.write_segments([seg.to_row() for seg in self.segments])
                self.store.close_session(self._session_snapshot())
            else:
                self.store.discard_session()
        self.table.save()
        if self.level_reader.learned:
            self.level_reader.templates.save(self._level_templates_path)

    def restart(self) -> None:
        """歸零並開始新的一場（學到的地圖名稱、角色資訊都留著）。"""
        self.tracker.reset()
        self.segments.clear()
        self._last_cum = 0
        self._last_active = 0.0
        self._deducted_exp = 0
        self._deducted_active = 0.0
        self._deducted_deaths = 0
        self._deducted_levelups = 0
        self._start_level_recorded = False
        if self.store is not None:
            self.store.open_session(self.label)
            self.store.set_identity(self.character, self.job)
        if self.map_watcher.map_id:
            self._open_segment(self.map_watcher.map_id, time.time())

    def close(self, save: bool = True) -> None:
        self.finish(save)
        if self.capturer is not None:
            self.capturer.close()
            self.capturer = None

