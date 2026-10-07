"""測試共用工具。"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapleexp.core.reading import StatusReading  # noqa: E402


def reading(
    mono: float,
    exp: int | None,
    *,
    need: int | None = None,
    level: int | None = None,
    pct: float | None = None,
    ok: bool = True,
) -> StatusReading:
    """建立一筆取樣。

    給 ``need`` 時會自動算出對應的百分比（四捨五入到兩位小數），模擬遊戲
    狀態列同時顯示絕對值與百分比的樣子。
    """
    if pct is None and need and exp is not None:
        pct = round(100.0 * exp / need, 2)
    return StatusReading(
        mono=mono,
        wall=1_700_000_000.0 + mono,
        ok=ok,
        level=level,
        exp_abs=exp,
        exp_pct=pct,
        raw_exp="" if exp is None else f"{exp}[{pct}%]",
    )


def miss(mono: float) -> StatusReading:
    return StatusReading.failed(mono, 1_700_000_000.0 + mono, reason="test-miss")
