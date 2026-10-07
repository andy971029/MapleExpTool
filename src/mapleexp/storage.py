"""SQLite 儲存層。

即時數字看完就沒了；真正有長期價值的是「哪隻角色、哪張地圖、哪個時段、哪套配裝
的效率最好」。所以每一場、每一段（角色 × 地圖）與每一個事件都落地，之後用
``report`` 指令比較。

**每秒的取樣刻意不存。** 報表要的數字（練了多少、幾 %、多久、平均與峰值 /h）在段
結束時就算好了；原始取樣存下來每小時長 230 KB，而且除了「事後重算峰值」之外沒有
任何用途 —— 峰值改成追蹤時線上算就不需要它了。

地圖是自動偵測的（像素指紋 + 客戶端名稱清單，落在 ``segments``／``maps`` 兩張表）；
``label`` 是另一個維度的標記，拿來比較「同一張圖、不同打法或配裝」。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .core.tracker import Event, Snapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT    NOT NULL DEFAULT '',
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    start_level INTEGER,
    end_level   INTEGER,
    cum_net     INTEGER NOT NULL DEFAULT 0,
    cum_gross   INTEGER NOT NULL DEFAULT 0,
    exp_lost    INTEGER NOT NULL DEFAULT 0,
    active_sec  REAL    NOT NULL DEFAULT 0,
    wall_sec    REAL    NOT NULL DEFAULT 0,
    skipped_sec REAL    NOT NULL DEFAULT 0,
    deaths      INTEGER NOT NULL DEFAULT 0,
    levelups    INTEGER NOT NULL DEFAULT 0,
    glitches    INTEGER NOT NULL DEFAULT 0,
    misses      INTEGER NOT NULL DEFAULT 0,
    samples     INTEGER NOT NULL DEFAULT 0,
    character   TEXT    NOT NULL DEFAULT '',
    job         TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    wall       REAL    NOT NULL,
    kind       TEXT    NOT NULL,
    level      INTEGER,
    amount     INTEGER NOT NULL DEFAULT 0,
    detail     TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS segments (
    session_id        INTEGER NOT NULL REFERENCES sessions(id),
    map_id            TEXT    NOT NULL,
    character         TEXT    NOT NULL DEFAULT '',
    job               TEXT    NOT NULL DEFAULT '',
    started_at        REAL    NOT NULL,
    ended_at          REAL,
    exp               INTEGER NOT NULL DEFAULT 0,
    active_sec        REAL    NOT NULL DEFAULT 0,
    start_level       INTEGER,
    end_level         INTEGER,
    start_pct         REAL,
    end_pct           REAL,
    peak_exp_per_hour REAL,
    deaths            INTEGER NOT NULL DEFAULT 0,
    levelups          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS maps (
    map_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    thumbnail  BLOB
);

CREATE INDEX IF NOT EXISTS idx_segments_map ON segments(map_id);
CREATE INDEX IF NOT EXISTS idx_segments_started ON segments(started_at);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id, wall);
CREATE INDEX IF NOT EXISTS idx_sessions_label  ON sessions(label);
"""

# 後來才加上的欄位。``CREATE TABLE IF NOT EXISTS`` 不會動既有的表，
# 所以舊資料庫升級時要自己補 —— 少了這段，舊使用者一跑就會
# ``no such column`` 當掉（實測踩過）。
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("sessions", "character", "TEXT NOT NULL DEFAULT ''"),
    ("sessions", "job", "TEXT NOT NULL DEFAULT ''"),
    ("sessions", "skipped_sec", "REAL NOT NULL DEFAULT 0"),
    ("segments", "character", "TEXT NOT NULL DEFAULT ''"),
    ("segments", "job", "TEXT NOT NULL DEFAULT ''"),
    ("segments", "start_level", "INTEGER"),
    ("segments", "end_level", "INTEGER"),
    ("segments", "start_pct", "REAL"),
    ("segments", "end_pct", "REAL"),
    ("segments", "peak_exp_per_hour", "REAL"),
    ("segments", "deaths", "INTEGER NOT NULL DEFAULT 0"),
    ("segments", "levelups", "INTEGER NOT NULL DEFAULT 0"),
)

# 以前存過的每秒取樣表。沒有任何功能讀它，升級時直接丟掉並回收空間。
DROPPED_TABLES = ("samples",)


@dataclass
class SessionRow:
    id: int
    label: str
    started_at: float
    ended_at: float | None
    start_level: int | None
    end_level: int | None
    cum_net: int
    cum_gross: int
    exp_lost: int
    active_sec: float
    wall_sec: float
    deaths: int
    levelups: int
    samples: int

    @property
    def exp_per_hour(self) -> float | None:
        if self.active_sec <= 0:
            return None
        return self.cum_net / (self.active_sec / 3600.0)


class Store:
    """追蹤資料的持久化。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        # WAL 讓讀取（例如另一個視窗在跑 report）不會被寫入擋住。
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        self._session_id: int | None = None

    def _migrate(self) -> None:
        """把舊資料庫缺的欄位補上、不再用的表丟掉。"""
        for table, column, ddl in MIGRATIONS:
            try:
                existing = {
                    row[1]
                    for row in self._conn.execute(f"PRAGMA table_info({table})")
                }
            except sqlite3.DatabaseError:
                continue
            if not existing or column in existing:
                continue
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            except sqlite3.DatabaseError:
                pass

        dropped = False
        for table in DROPPED_TABLES:
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists:
                self._conn.execute(f"DROP TABLE {table}")
                dropped = True
        if dropped:
            # 取樣表通常佔掉整個檔案的九成以上，丟掉之後不 VACUUM 檔案不會變小。
            # VACUUM 不能在交易裡跑，所以先 commit。
            self._conn.commit()
            self._conn.execute("VACUUM")

    # ------------------------------------------------------------------ #
    # 寫入
    # ------------------------------------------------------------------ #

    @property
    def session_id(self) -> int | None:
        return self._session_id

    def open_session(self, label: str = "", started_at: float | None = None) -> int:
        cursor = self._conn.execute(
            "INSERT INTO sessions (label, started_at) VALUES (?, ?)",
            (label, started_at if started_at is not None else time.time()),
        )
        self._conn.commit()
        self._session_id = int(cursor.lastrowid)
        return self._session_id

    def add_events(self, events: list[Event]) -> None:
        if self._session_id is None or not events:
            return
        self._conn.executemany(
            "INSERT INTO events (session_id, wall, kind, level, amount, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (self._session_id, e.wall, e.kind, e.level, e.amount, e.detail)
                for e in events
            ],
        )
        # 事件（升級、死亡、換圖）稀疏但重要，每次都立刻寫入。
        self._conn.commit()

    def close_session(self, snapshot: Snapshot, start_level: int | None = None) -> None:
        if self._session_id is None:
            return
        self._conn.execute(
            "UPDATE sessions SET ended_at=?, start_level=COALESCE(?, start_level), "
            "end_level=?, cum_net=?, cum_gross=?, exp_lost=?, active_sec=?, wall_sec=?, "
            "skipped_sec=?, deaths=?, levelups=?, glitches=?, misses=?, samples=? "
            "WHERE id=?",
            (
                time.time(),
                start_level,
                snapshot.level,
                snapshot.cum_net,
                snapshot.cum_gross,
                snapshot.exp_lost,
                snapshot.active_sec,
                snapshot.wall_sec,
                snapshot.skipped_sec,
                snapshot.deaths,
                snapshot.levelups,
                snapshot.glitches,
                snapshot.misses,
                snapshot.samples,
                self._session_id,
            ),
        )
        self._conn.commit()
        self._session_id = None

    def set_identity(self, character: str = "", job: str = "") -> None:
        """記下這一場是誰在打。空字串不覆蓋既有值。"""
        if self._session_id is None:
            return
        self._conn.execute(
            "UPDATE sessions SET character=COALESCE(NULLIF(?, ''), character), "
            "job=COALESCE(NULLIF(?, ''), job) WHERE id=?",
            (character, job, self._session_id),
        )
        self._conn.commit()

    def discard_session(self) -> None:
        """整場丟掉 —— 使用者在結束時選「不要儲存」。"""
        if self._session_id is None:
            return
        for table in ("events", "segments"):
            self._conn.execute(f"DELETE FROM {table} WHERE session_id=?", (self._session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id=?", (self._session_id,))
        self._conn.commit()
        self._session_id = None

    def set_start_level(self, level: int | None) -> None:
        if self._session_id is None or level is None:
            return
        self._conn.execute(
            "UPDATE sessions SET start_level=COALESCE(start_level, ?) WHERE id=?",
            (level, self._session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # 查詢
    # ------------------------------------------------------------------ #

    # ---------------- 地圖 ----------------

    def remember_map(
        self, map_id: str, thumbnail: bytes | None = None, name: str = ""
    ) -> bool:
        """第一次看到某張地圖時記下來（含縮圖，之後取名時可以對照）。回傳是否為新地圖。"""
        if not map_id:
            return False
        row = self._conn.execute(
            "SELECT map_id FROM maps WHERE map_id=?", (map_id,)
        ).fetchone()
        if row is not None:
            self._conn.execute(
                "UPDATE maps SET thumbnail=COALESCE(thumbnail, ?), "
                "name=COALESCE(NULLIF(name, ''), ?) WHERE map_id=?",
                (thumbnail, name, map_id),
            )
            self._conn.commit()
            return False
        self._conn.execute(
            "INSERT INTO maps (map_id, name, first_seen, thumbnail) VALUES (?, ?, ?, ?)",
            (map_id, name, time.time(), thumbnail),
        )
        self._conn.commit()
        return True

    def name_map(self, map_id: str, name: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE maps SET name=? WHERE map_id=?", (name, map_id)
        )
        self._conn.commit()
        return bool(cursor.rowcount)

    def map_name(self, map_id: str) -> str:
        row = self._conn.execute(
            "SELECT name FROM maps WHERE map_id=?", (map_id,)
        ).fetchone()
        return (row["name"] if row else "") or ""

    def list_maps(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT m.map_id, m.name, m.first_seen, "
            "       COALESCE(SUM(s.exp), 0) AS exp, "
            "       COALESCE(SUM(s.active_sec), 0) AS active "
            "FROM maps m LEFT JOIN segments s ON s.map_id = m.map_id "
            "GROUP BY m.map_id ORDER BY active DESC"
        ).fetchall()
        result = []
        for row in rows:
            active = float(row["active"] or 0.0)
            exp = int(row["exp"] or 0)
            result.append(
                {
                    "map_id": row["map_id"],
                    "name": row["name"] or "",
                    "first_seen": float(row["first_seen"]),
                    "exp": exp,
                    "active_sec": active,
                    "exp_per_hour": exp / (active / 3600.0) if active > 0 else None,
                }
            )
        return result

    def prune_maps(self, min_active_sec: float = 60.0) -> int:
        """清掉沒命名、而且累計時間很短的地圖。

        定位偶爾會誤判（抓到背包或角色資料視窗），每個誤判都會變成一張
        「待命名的地圖」。待久的那些是真的，待幾秒就沒了的多半是雜訊。
        """
        rows = self._conn.execute(
            "SELECT m.map_id FROM maps m "
            "LEFT JOIN segments s ON s.map_id = m.map_id "
            "WHERE COALESCE(m.name, '') = '' "
            "GROUP BY m.map_id HAVING COALESCE(SUM(s.active_sec), 0) < ?",
            (min_active_sec,),
        ).fetchall()
        ids = [row["map_id"] for row in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM segments WHERE map_id IN ({placeholders})", ids)
        self._conn.execute(f"DELETE FROM maps WHERE map_id IN ({placeholders})", ids)
        self._conn.commit()
        return len(ids)

    def map_thumbnail(self, map_id: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT thumbnail FROM maps WHERE map_id=?", (map_id,)
        ).fetchone()
        return row["thumbnail"] if row else None

    def write_segments(self, segments: list[dict]) -> None:
        """把這一場依「角色 × 地圖」切出來的分段寫入（欄位見 Segment.to_row）。"""
        if self._session_id is None or not segments:
            return
        self._conn.executemany(
            "INSERT INTO segments (session_id, map_id, character, job, started_at, ended_at, "
            "exp, active_sec, start_level, end_level, start_pct, end_pct, "
            "peak_exp_per_hour, deaths, levelups) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    self._session_id,
                    seg.get("map_id", ""),
                    seg.get("character", "") or "",
                    seg.get("job", "") or "",
                    seg.get("started_at", 0.0),
                    seg.get("ended_at"),
                    int(seg.get("exp", 0)),
                    float(seg.get("active_sec", 0.0)),
                    seg.get("start_level"),
                    seg.get("end_level"),
                    seg.get("start_pct"),
                    seg.get("end_pct"),
                    seg.get("peak_exp_per_hour"),
                    int(seg.get("deaths", 0)),
                    int(seg.get("levelups", 0)),
                )
                for seg in segments
            ],
        )
        self._conn.commit()

    def list_segments(self) -> list[dict]:
        """所有已儲存的段，給報表視窗用。只算正常結束的場次，最新的在前。"""
        rows = self._conn.execute(
            "SELECT s.session_id, s.map_id, COALESCE(m.name, '') AS map_name, "
            "       s.character, s.job, s.started_at, s.ended_at, s.exp, s.active_sec, "
            "       s.start_pct, s.end_pct, s.peak_exp_per_hour, s.deaths, s.levelups, "
            "       COALESCE(ss.label, '') AS label "
            "FROM segments s "
            "LEFT JOIN maps m ON m.map_id = s.map_id "
            "JOIN sessions ss ON ss.id = s.session_id "
            "WHERE ss.ended_at IS NOT NULL "
            "ORDER BY s.started_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            pct = None
            if row["start_pct"] is not None and row["end_pct"] is not None:
                pct = int(row["levelups"] or 0) * 100.0 + float(row["end_pct"]) - float(row["start_pct"])
            result.append(
                {
                    "session_id": int(row["session_id"]),
                    "map_id": row["map_id"],
                    "map_name": row["map_name"] or "",
                    "character": row["character"] or "",
                    "job": row["job"] or "",
                    "label": row["label"] or "",
                    "started_at": float(row["started_at"]),
                    "ended_at": float(row["ended_at"]) if row["ended_at"] is not None else None,
                    "exp": int(row["exp"] or 0),
                    "active_sec": float(row["active_sec"] or 0.0),
                    "pct_gained": pct,
                    "peak_exp_per_hour": row["peak_exp_per_hour"],
                    "deaths": int(row["deaths"] or 0),
                    "levelups": int(row["levelups"] or 0),
                }
            )
        return result

    def daily_report(self, days: int = 30) -> list[dict]:
        """「哪一天、哪隻角色、在哪張地圖」練了多少。

        同一天同一隻角色在同一張圖的幾段合併成一列：經驗與時間相加、峰值取最大、
        百分比把每段的 ``levelups*100 + end - start`` 加起來。
        """
        since = time.time() - max(0, days) * 86400.0 if days > 0 else 0.0
        rows = self._conn.execute(
            "SELECT date(s.started_at, 'unixepoch', 'localtime') AS day, "
            "       s.character, MAX(s.job) AS job, s.map_id, "
            "       COALESCE(MAX(m.name), '') AS map_name, "
            "       COUNT(*) AS runs, SUM(s.exp) AS exp, SUM(s.active_sec) AS active, "
            "       SUM(s.deaths) AS deaths, SUM(s.levelups) AS levelups, "
            "       MAX(s.peak_exp_per_hour) AS peak, "
            "       SUM(CASE WHEN s.start_pct IS NOT NULL AND s.end_pct IS NOT NULL "
            "                THEN s.levelups * 100.0 + s.end_pct - s.start_pct END) AS pct, "
            "       MIN(s.start_level) AS start_level, MAX(s.end_level) AS end_level "
            "FROM segments s LEFT JOIN maps m ON m.map_id = s.map_id "
            "WHERE s.started_at >= ? AND s.active_sec > 0 "
            "GROUP BY day, s.character, s.map_id "
            "ORDER BY day DESC, s.character, exp DESC",
            (since,),
        ).fetchall()
        result = []
        for row in rows:
            active = float(row["active"] or 0.0)
            exp = int(row["exp"] or 0)
            result.append(
                {
                    "day": row["day"],
                    "character": row["character"] or "",
                    "job": row["job"] or "",
                    "map_id": row["map_id"],
                    "map_name": row["map_name"] or "",
                    "runs": int(row["runs"]),
                    "exp": exp,
                    "active_sec": active,
                    "exp_per_hour": exp / (active / 3600.0) if active > 0 else None,
                    "peak_exp_per_hour": row["peak"],
                    "pct_gained": row["pct"],
                    "start_level": row["start_level"],
                    "end_level": row["end_level"],
                    "deaths": int(row["deaths"] or 0),
                    "levelups": int(row["levelups"] or 0),
                }
            )
        return result

    # ---------------- 查詢 ----------------

    def list_sessions(self, limit: int = 20) -> list[SessionRow]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._to_row(r) for r in rows]

    def label_summary(self) -> list[dict]:
        """按 label 匯總，用來比較不同地圖／打法的效率。"""
        rows = self._conn.execute(
            "SELECT label, COUNT(*) AS runs, SUM(cum_net) AS exp, "
            "SUM(active_sec) AS active, SUM(deaths) AS deaths, SUM(levelups) AS levelups "
            "FROM sessions WHERE ended_at IS NOT NULL AND active_sec > 0 "
            "GROUP BY label ORDER BY (SUM(cum_net) / SUM(active_sec)) DESC"
        ).fetchall()
        result = []
        for row in rows:
            active = float(row["active"] or 0.0)
            exp = int(row["exp"] or 0)
            result.append(
                {
                    "label": row["label"] or "(未標記)",
                    "runs": int(row["runs"]),
                    "exp": exp,
                    "active_sec": active,
                    "exp_per_hour": exp / (active / 3600.0) if active > 0 else None,
                    "deaths": int(row["deaths"] or 0),
                    "levelups": int(row["levelups"] or 0),
                }
            )
        return result

    def events_for(self, session_id: int, kinds: tuple[str, ...] | None = None) -> list[dict]:
        query = "SELECT * FROM events WHERE session_id=?"
        params: list = [session_id]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            query += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        query += " ORDER BY wall"
        return [dict(r) for r in self._conn.execute(query, params).fetchall()]

    def prune_incomplete(self) -> int:
        """清掉沒有正常結束、而且什麼都沒發生的場次（例如找不到視窗就退出）。

        「有發生事情」的判準是 events 表：第一筆讀到的取樣就會寫一個 ``start`` 事件。
        """
        cursor = self._conn.execute(
            "DELETE FROM sessions WHERE ended_at IS NULL "
            "AND id NOT IN (SELECT DISTINCT session_id FROM events)"
        )
        self._conn.commit()
        return cursor.rowcount or 0

    @staticmethod
    def _to_row(row: sqlite3.Row) -> SessionRow:
        return SessionRow(
            id=int(row["id"]),
            label=row["label"] or "",
            started_at=float(row["started_at"]),
            ended_at=float(row["ended_at"]) if row["ended_at"] is not None else None,
            start_level=row["start_level"],
            end_level=row["end_level"],
            cum_net=int(row["cum_net"]),
            cum_gross=int(row["cum_gross"]),
            exp_lost=int(row["exp_lost"]),
            active_sec=float(row["active_sec"]),
            wall_sec=float(row["wall_sec"]),
            deaths=int(row["deaths"]),
            levelups=int(row["levelups"]),
            samples=int(row["samples"]),
        )

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
