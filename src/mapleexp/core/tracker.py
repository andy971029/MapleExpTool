"""經驗值追蹤狀態機。

輸入是一串 :class:`StatusReading`，輸出是一條單調的累積經驗曲線加上事件清單。
中間要處理的都是真實世界的麻煩事：

* **升級** —— 經驗歸零，必須補上「上一級剩餘的量」才能接續。
* **死亡掉經驗** —— 經典版會掉經驗，而且外觀上也是「經驗變少」，
  要跟升級區分開來。
* **掛機／切頻／讀取畫面** —— 這段時間不能算進分母，否則 EXP/hr 被稀釋。
* **單格辨識雜訊** —— 這是實務上最常見的失效模式。任何「意外」的轉換
  （下降、升級、暴增）都要等下一筆取樣確認才採信，代價是一次取樣的延遲。

累積經驗分成兩條：``gross``（只累加正增量）與 ``net``（扣掉死亡損失）。
速率與升級倒數用 net，因為那才是實際的升級進度；畫面上同時顯示 gross
讓人知道自己到底打了多少。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .exptable import ExpTable
from .reading import StatusReading
from .stats import ExpSeries, RateEstimate, eta_seconds
from ..config import TrackerConfig

# 狀態
WARMUP = "warmup"
ACTIVE = "active"
PAUSED = "paused"   # 辨識不到畫面
IDLE = "idle"       # 畫面讀得到，但經驗值停滯

# 轉換類型
GAIN = "gain"
DROP = "drop"
LEVELUP = "levelup"
REGRESS = "regress"
IMPLAUSIBLE = "implausible"

# 沒有校準等級 ROI 時，用來區分「升級」與「死亡」的啟發式門檻。
# 升級會讓經驗從接近滿值掉到接近零；死亡只掉本級所需經驗的一小部分。
LEVELUP_DROP_RATIO = 0.5
LEVELUP_MIN_PROGRESS = 0.5


@dataclass(frozen=True)
class Transition:
    """兩筆取樣之間發生了什麼。"""

    kind: str
    gain: int | None = None
    lost: int = 0
    levels: int = 0
    surprising: bool = False
    note: str = ""


@dataclass(frozen=True)
class Event:
    """值得記錄／顯示的事件。"""

    kind: str  # start | levelup | death | gap | glitch | anomaly | pause | resume | rebase
    wall: float
    level: int | None = None
    amount: int = 0
    detail: str = ""


@dataclass
class Snapshot:
    """給 UI 與儲存層用的即時狀態快照。"""

    state: str
    level: int | None
    exp_abs: int | None
    exp_pct: float | None
    need: int | None
    need_confidence: str
    remaining: int | None
    cum_net: int
    cum_gross: int
    exp_lost: int
    active_sec: float
    wall_sec: float
    skipped_sec: float
    idle_sec: float = 0.0
    rates: dict[int, RateEstimate] = field(default_factory=dict)
    eta_sec: float | None = None
    eta_window: int = 0
    deaths: int = 0
    levelups: int = 0
    glitches: int = 0
    misses: int = 0
    consecutive_misses: int = 0
    uncertain: int = 0
    gain_events: int = 0
    last_gain: int = 0
    samples: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def gains_per_min(self) -> float | None:
        if self.active_sec <= 0:
            return None
        return self.gain_events / (self.active_sec / 60.0)


class Tracker:
    """把取樣流轉成可用的統計數字。"""

    def __init__(
        self,
        cfg: TrackerConfig | None = None,
        table: ExpTable | None = None,
    ) -> None:
        self._cfg = cfg or TrackerConfig()
        self._table = table
        self._series = ExpSeries(self._cfg.rate_windows)
        self.reset()

    # ------------------------------------------------------------------ #
    # 生命週期
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._state = WARMUP
        self._committed: StatusReading | None = None
        self._suspect: StatusReading | None = None
        self._last: StatusReading | None = None
        self._cum_net = 0
        self._cum_gross = 0
        self._exp_lost = 0
        self._active_t = 0.0
        self._skipped_sec = 0.0
        self._start_mono: float | None = None
        self._start_wall: float | None = None
        self._last_gain_mono: float | None = None
        self._idle_sec = 0.0
        self._deaths = 0
        self._levelups = 0
        self._glitches = 0
        self._misses = 0
        self._uncertain = 0
        self._gain_events = 0
        self._last_gain = 0
        self._samples = 0
        self._consecutive_miss = 0
        self._discontinuity = False
        self._notes: list[str] = []
        self._series.clear()

    @property
    def state(self) -> str:
        return self._state

    def mark_discontinuity(self) -> None:
        """下一筆可用的取樣只當新基準，不跟上一筆比。

        換角色時用：兩隻角色的經驗值之間沒有任何可比較的關係，照常分析會把差值
        當成死亡或異常。累積值與活躍時間都不動，就像中斷太久的 gap 那樣接續。
        """
        self._discontinuity = True
        self._suspect = None

    # ------------------------------------------------------------------ #
    # 餵資料
    # ------------------------------------------------------------------ #

    def feed(self, reading: StatusReading) -> list[Event]:
        """處理一筆取樣，回傳這次產生的事件。"""
        self._last = reading
        self._samples += 1

        # 沒有絕對經驗值就無法追蹤：百分比只有兩位小數，高等級的解析度不夠。
        usable = reading.ok and reading.exp_abs is not None
        if not usable:
            return self._handle_miss(reading)

        self._consecutive_miss = 0
        if self._table is not None:
            self._table.observe_bounds(reading.level, reading.need_bounds)

        if self._committed is None:
            self._discontinuity = False
            self._start_mono = reading.mono
            self._start_wall = reading.wall
            self._committed = reading
            # 一開始是閒置：拿到第一筆經驗增量之前的時間不算。使用者常常先開程式
            # 再整理背包、走路到練功點，那段不該稀釋速率。
            self._state = IDLE if self._cfg.idle_pause_sec > 0 else ACTIVE
            return [
                Event(
                    kind="start",
                    wall=reading.wall,
                    level=reading.level,
                    amount=reading.exp_abs or 0,
                    detail="開始追蹤",
                )
            ]

        events: list[Event] = []
        if self._state == PAUSED:
            self._state = ACTIVE
            gap = reading.mono - self._committed.mono
            events.append(
                Event(
                    kind="resume",
                    wall=reading.wall,
                    level=reading.level,
                    detail=f"中斷 {gap:.0f} 秒後恢復",
                )
            )

        if self._discontinuity:
            self._discontinuity = False
            self._rebase(reading)
            events.append(
                Event(
                    kind="rebase",
                    wall=reading.wall,
                    level=reading.level,
                    detail="換角色，重設基準",
                )
            )
            return events

        events.extend(self._ingest(reading))
        return events

    def _handle_miss(self, reading: StatusReading) -> list[Event]:
        self._misses += 1
        self._consecutive_miss += 1
        if self._state != PAUSED and self._consecutive_miss >= self._cfg.miss_to_pause:
            self._state = PAUSED
            # 待確認的樣本已經過期，丟掉以免恢復後誤判。
            self._suspect = None
            return [
                Event(
                    kind="pause",
                    wall=reading.wall,
                    detail=f"連續 {self._consecutive_miss} 次辨識失敗，暫停計時",
                )
            ]
        return []

    # ------------------------------------------------------------------ #
    # 核心：雜訊過濾 + 提交
    # ------------------------------------------------------------------ #

    def _ingest(self, reading: StatusReading) -> list[Event]:
        assert self._committed is not None
        prev = self._committed
        transition = self.analyze(prev, reading)

        if not transition.surprising:
            events: list[Event] = []
            if self._suspect is not None:
                # 從 committed 可以正常接到新取樣，代表中間那筆是辨識雜訊。
                events.append(self._drop_suspect(reading.wall))
            events.extend(self._commit(prev, reading, transition))
            return events

        if not self._cfg.confirm_surprises:
            return self._commit(prev, reading, transition)

        if self._suspect is None:
            self._suspect = reading
            return []

        suspect = self._suspect
        if self._consistent(suspect, reading):
            # 下一筆佐證了意外轉換 —— 它是真的。
            self._suspect = None
            events = self._commit(prev, suspect, self.analyze(prev, suspect))
            follow = self.analyze(suspect, reading)
            if follow.surprising:
                self._rebase(reading)
            else:
                events.extend(self._commit(suspect, reading, follow))
            return events

        # 兩筆意外互相矛盾：舊的那筆是雜訊，改用新的當待確認樣本。
        events = [self._drop_suspect(reading.wall)]
        self._suspect = reading
        return events

    def _drop_suspect(self, wall: float) -> Event:
        suspect = self._suspect
        self._suspect = None
        self._glitches += 1
        raw = suspect.raw_exp if suspect is not None else ""
        return Event(kind="glitch", wall=wall, detail=f"丟棄疑似誤判取樣 {raw!r}")

    def _consistent(self, a: StatusReading, b: StatusReading) -> bool:
        """b 是否佐證了 a（同等級、經驗沒倒退、增量合理）。"""
        if a.exp_abs is None or b.exp_abs is None:
            return False
        delta = b.exp_abs - a.exp_abs
        if delta < 0:
            return False
        if a.level is not None and b.level is not None and a.level != b.level:
            # 等級顯示慢一拍：升級已經從經驗歸零看出來了，等級數字晚一格才跟上
            # （新字形要先學會才讀得到）。經驗沒倒退、等級剛好 +1，就是同一件事。
            if b.level != a.level + 1:
                return False
        need = self._need_for(b)
        if need is not None and delta > need * self._cfg.outlier_level_frac:
            return False
        return True

    def _commit(
        self, prev: StatusReading, cur: StatusReading, transition: Transition
    ) -> list[Event]:
        events: list[Event] = []
        dt = cur.mono - prev.mono

        if dt > self._cfg.max_bridge_sec:
            # 中斷太久，無法判斷這段時間是在打怪還是掛機 ——
            # 經驗差值與時間一起捨棄，兩邊都不偏。
            self._skipped_sec += dt
            events.append(
                Event(
                    kind="gap",
                    wall=cur.wall,
                    level=cur.level,
                    detail=f"中斷 {dt:.0f} 秒，該區間不列入統計",
                )
            )
            self._rebase(cur)
            return events

        if dt > 0:
            self._active_t += self._active_portion(prev.mono, cur.mono, transition)

        if transition.kind == LEVELUP:
            self._levelups += transition.levels
            events.append(
                Event(
                    kind="levelup",
                    wall=cur.wall,
                    level=cur.level,
                    amount=transition.levels,
                    detail=transition.note or "升級",
                )
            )
        elif transition.kind == DROP:
            self._deaths += 1
            self._exp_lost += transition.lost
            events.append(
                Event(
                    kind="death",
                    wall=cur.wall,
                    level=cur.level,
                    amount=transition.lost,
                    detail=f"經驗減少 {transition.lost:,}",
                )
            )
        elif transition.kind in (REGRESS, IMPLAUSIBLE):
            events.append(
                Event(
                    kind="anomaly",
                    wall=cur.wall,
                    level=cur.level,
                    detail=transition.note,
                )
            )

        if transition.gain is None:
            if transition.kind != DROP:
                self._uncertain += 1
                self._note(f"無法估算增量：{transition.note or transition.kind}")
        elif transition.gain > 0:
            self._cum_gross += transition.gain
            self._cum_net += transition.gain
            self._gain_events += 1
            self._last_gain = transition.gain

        if transition.lost > 0:
            self._cum_net -= transition.lost

        self._committed = cur
        self._series.add(self._active_t, self._cum_net)
        return events

    def _active_portion(
        self, previous_mono: float, current_mono: float, transition: Transition
    ) -> float:
        """這段區間有多少秒該算進「活躍時間」。

        經驗值停滯超過門檻就不再累計 —— 走路、補血、聊天、掛機都不該稀釋
        EXP/hr。但**結束停滯的那一筆要整段算進去**：那隻怪確實是在這段時間內
        打死的，不算的話分母會變成零，速率會噴成天文數字。
        """
        span = max(0.0, current_mono - previous_mono)
        limit = self._cfg.idle_pause_sec
        if limit <= 0:
            return span

        gained = transition.gain is not None and transition.gain > 0
        if gained:
            self._last_gain_mono = current_mono
            if self._state == IDLE:
                self._state = ACTIVE
            return span

        anchor = self._last_gain_mono
        if anchor is None:
            # 還沒有過任何一次增量：全部算閒置。第一隻怪打死的那一格會整段算進去
            # （上面 gained 的分支），所以時鐘是從真正開始練的那一刻起算。
            self._idle_sec += span
            if self._state != PAUSED:
                self._state = IDLE
            return 0.0

        deadline = anchor + limit
        counted = max(0.0, min(current_mono, deadline) - previous_mono)
        skipped = span - counted
        if skipped > 0:
            self._idle_sec += skipped
            if self._state == ACTIVE:
                self._state = IDLE
        return counted

    def _rebase(self, reading: StatusReading) -> None:
        """重設基準點，不動累積值也不推進時間。"""
        self._committed = reading
        self._suspect = None

    # ------------------------------------------------------------------ #
    # 轉換分析（純函式，方便測試）
    # ------------------------------------------------------------------ #

    def analyze(self, prev: StatusReading, cur: StatusReading) -> Transition:
        if prev.exp_abs is None or cur.exp_abs is None:
            return Transition(kind=IMPLAUSIBLE, surprising=True, note="缺少經驗值")

        need_prev = self._need_for(prev)
        need_cur = self._need_for(cur)

        if prev.level is not None and cur.level is not None:
            if cur.level < prev.level:
                return Transition(
                    kind=REGRESS,
                    surprising=True,
                    note=f"等級由 {prev.level} 降為 {cur.level}",
                )
            if cur.level > prev.level and cur.exp_abs < prev.exp_abs:
                return self._levelup_transition(prev, cur, need_prev, cur.level - prev.level)
            # 等級變大但經驗沒有歸零：升級本身已經靠經驗歸零算過了，這是等級數字
            # 晚一格才跟上（新字形學會之後才讀得到）。當成普通增量，不要算第二次。

        delta = cur.exp_abs - prev.exp_abs
        if delta >= 0:
            if need_cur is not None and delta > need_cur * self._cfg.outlier_level_frac:
                return Transition(
                    kind=IMPLAUSIBLE,
                    surprising=True,
                    note=f"單次增量 {delta:,} 超過本級 {self._cfg.outlier_level_frac:.0%}",
                )
            return Transition(kind=GAIN, gain=delta)

        # 經驗變少：可能是升級（等級 ROI 未校準時看不出來），也可能是死亡。
        if self._looks_like_levelup(prev, cur, need_prev):
            return self._levelup_transition(prev, cur, need_prev, 1)

        return Transition(
            kind=DROP,
            gain=0,
            lost=-delta,
            surprising=True,
            note=f"經驗減少 {-delta:,}",
        )

    def _levelup_transition(
        self,
        prev: StatusReading,
        cur: StatusReading,
        need_prev: int | None,
        levels: int,
    ) -> Transition:
        if levels != 1:
            # 一次取樣跳超過一級，幾乎都是辨識錯誤；就算是真的也補不出中間
            # 各級的所需經驗，不如明確標為不確定。
            return Transition(
                kind=LEVELUP,
                gain=None,
                levels=levels,
                surprising=True,
                note=f"單次取樣跳 {levels} 級",
            )
        if need_prev is None:
            return Transition(
                kind=LEVELUP,
                gain=None,
                levels=1,
                surprising=True,
                note="升級但不知道上一級所需經驗",
            )
        gain = need_prev - (prev.exp_abs or 0) + (cur.exp_abs or 0)
        if gain < 0:
            return Transition(
                kind=LEVELUP,
                gain=None,
                levels=1,
                surprising=True,
                note="升級接續結果為負值，所需經驗估計可能有誤",
            )
        return Transition(
            kind=LEVELUP,
            gain=gain,
            levels=1,
            surprising=True,
            note=f"升級，補上上一級剩餘 {need_prev - (prev.exp_abs or 0):,}",
        )

    def _looks_like_levelup(
        self, prev: StatusReading, cur: StatusReading, need_prev: int | None
    ) -> bool:
        """經驗變少時，用「掉多少」區分升級與死亡。

        升級會讓經驗從接近滿值掉到接近零；死亡只掉本級所需經驗的一小部分
        （經典版的死亡懲罰是本級所需經驗的某個百分比）。

        就算兩筆的等級都讀到了而且相同，也**不能**因此斷定不是升級：等級數字是
        從橘色方塊的字形讀的，新等級用到沒學過的數字時會晚幾格才跟上（甚至要靠
        這裡判定出升級之後才學得到）。那幾格看起來就是「等級沒變、經驗歸零」——
        照死亡算的話，一次升級會被記成一次死亡加上一整級的損失。
        """
        prev_exp = prev.exp_abs or 0
        cur_exp = cur.exp_abs or 0
        if prev_exp <= 0:
            return False
        if cur_exp >= prev_exp * LEVELUP_DROP_RATIO:
            return False
        if need_prev is not None and prev_exp < need_prev * LEVELUP_MIN_PROGRESS:
            # 本級進度還很低就掉了一大半，比較像死亡而不是升級。
            return False
        return True

    def _need_for(self, reading: StatusReading) -> int | None:
        """本級所需經驗。

        優先用這筆取樣自己推導的值（最新、不受等級 ROI 是否校準影響），
        其次才用跨場次累積的經驗表。
        """
        derived = reading.derived_need
        if derived is not None:
            return derived
        if self._table is not None:
            return self._table.reliable_need(reading.level)
        return None

    def _note(self, text: str) -> None:
        if text and (not self._notes or self._notes[-1] != text):
            self._notes.append(text)
            del self._notes[:-10]

    # ------------------------------------------------------------------ #
    # 輸出
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Snapshot:
        last = self._last
        committed = self._committed
        ref = committed if committed is not None else last

        level = ref.level if ref is not None else None
        exp_abs = ref.exp_abs if ref is not None else None
        exp_pct = ref.exp_pct if ref is not None else None

        need = self._need_for(ref) if ref is not None else None
        remaining = None
        if need is not None and exp_abs is not None:
            remaining = max(0, need - exp_abs)

        rates = self._series.rates(self._active_t)
        eta_window = self._cfg.eta_window
        eta_rate = rates.get(eta_window)
        if eta_rate is None or not eta_rate.valid:
            # 指定視窗還沒有足夠資料時，退而用最短的可用視窗。
            for window in sorted(rates):
                candidate = rates[window]
                if candidate.valid:
                    eta_rate = candidate
                    eta_window = window
                    break

        eta = eta_seconds(remaining, eta_rate.exp_per_hour if eta_rate else None)

        wall_sec = 0.0
        if self._start_wall is not None and last is not None:
            wall_sec = max(0.0, last.wall - self._start_wall)

        confidence = "derived" if (ref is not None and ref.derived_need is not None) else "unknown"
        if confidence == "unknown" and self._table is not None:
            confidence = self._table.confidence(level)

        return Snapshot(
            state=self._state,
            level=level,
            exp_abs=exp_abs,
            exp_pct=exp_pct,
            need=need,
            need_confidence=confidence,
            remaining=remaining,
            cum_net=self._cum_net,
            cum_gross=self._cum_gross,
            exp_lost=self._exp_lost,
            active_sec=self._active_t,
            wall_sec=wall_sec,
            skipped_sec=self._skipped_sec,
            idle_sec=self._idle_sec,
            rates=rates,
            eta_sec=eta,
            eta_window=eta_window,
            deaths=self._deaths,
            levelups=self._levelups,
            glitches=self._glitches,
            misses=self._misses,
            consecutive_misses=self._consecutive_miss,
            uncertain=self._uncertain,
            gain_events=self._gain_events,
            last_gain=self._last_gain,
            samples=self._samples,
            notes=list(self._notes),
        )
