"""命令列入口。

    python -m mapleexp track                  # 直接開始追蹤，不需要任何設定
    python -m mapleexp track --label 遺跡之墓II # 順便標記這一場
    python -m mapleexp track --console        # 不開浮窗，印在終端機
    python -m mapleexp report                 # 歷史場次與各地圖效率比較
    python -m mapleexp doctor                 # 讀不到時用這個查原因
    python -m mapleexp windows                # 列出可擷取的視窗
    python -m mapleexp selftest               # 不需要遊戲的自我測試
    python -m mapleexp calibrate              # 自動設定失敗時才需要
    python -m mapleexp update                 # 檢查並安裝新版（打包版才有東西可更新）
"""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__, update
from . import config as config_module
from .config import Config


def _setup_console(wanted: bool) -> bool:
    """確保有地方可以印字。回傳是否是我們自己開的主控台（結束時要等人看完）。

    打包成免安裝版時是**視窗程式**（沒有黑色的 cmd 視窗），所以 ``sys.stdout`` 是
    None。三種情況：

    * 從終端機帶參數執行（``MapleExpTool.exe doctor``）—— 用 ``AttachConsole`` 接回
      那個終端機，輸出就印在原本的視窗裡。
    * 雙擊、或從捷徑執行一個需要輸出的子命令 —— 沒有終端機可接，就自己開一個，
      結束前停住等人看。
    * 雙擊開浮窗（``track``）—— 不需要主控台，所有訊息都在浮窗與對話框上。
    """
    if sys.stdout is not None and sys.stderr is not None:
        # 一般的 python 執行、或輸出被導向到檔案／管線。
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        return False
    if sys.platform != "win32":
        return False

    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ATTACH_PARENT_PROCESS = ctypes.c_uint32(-1).value
    allocated = False
    attached = bool(kernel32.AttachConsole(ATTACH_PARENT_PROCESS))
    if not attached and wanted:
        attached = allocated = bool(kernel32.AllocConsole())
    if not attached:
        return False
    try:
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not allocated:
        # 接回別人的終端機時，提示字元已經印出去了，先換行免得跟我們的輸出黏在一起。
        print()
    return allocated


def _fatal(message: str) -> None:
    """開始追蹤前就失敗。有主控台就印在那裡，沒有就跳對話框 —— 不能無聲地消失。"""
    if sys.stderr is not None:
        print(message, file=sys.stderr)
        return
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("MS EXP TOOL", message)
        root.destroy()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #


def cmd_calibrate(_args) -> int:
    from .ui.calibrate import run_calibration

    run_calibration(Config.load())
    return 0


# --------------------------------------------------------------------------- #
# track
# --------------------------------------------------------------------------- #


def cmd_track(args) -> int:
    from .core.session import SetupError, TrackingSession
    from .core.stats import format_duration, format_exp, format_rate
    from .storage import Store

    cfg = Config.load()
    if args.interval:
        cfg.tracker.sample_interval = args.interval
    if args.debug_frames:
        cfg.save_debug_frames = True

    # 查更新在背景做，跟 setup 的時間重疊；結果由浮窗在開好之後處理。
    update_check = None
    if _updates_enabled(cfg) and not args.no_update_check:
        update_check = update.UpdateCheck().start()

    store = None if args.no_db else Store(config_module.db_path())
    if store is not None:
        store.prune_incomplete()

    session = TrackingSession(cfg, label=args.label, store=store)
    try:
        window = session.setup()
    except SetupError as exc:
        message = f"無法開始：{exc}"
        # 如果問題出在這一版本身，至少讓人知道有新版可以換。
        if update_check is not None and update_check.wait(1.0) and update_check.release:
            message += (
                f"\n\n另外：有新版本 v{update_check.release.version} 可以下載"
                f"\n{update_check.release.page}"
            )
        _fatal(message)
        if store is not None:
            store.close()
        return 2

    print(f"目標視窗：{window.label}")
    if session.auto is not None:
        print(f"模板來源：{session.template_source}")
        print(f"ROI 來源：{session.auto.roi_source} — {session.auto.message}")
    print(f"地圖清單：{session.vocab_message}")
    print(f"取樣間隔：{cfg.tracker.sample_interval:g} 秒")
    if args.label:
        print(f"標記：{args.label}")

    from .win32.windows import covers_whole_screen

    if not args.console and covers_whole_screen(window.hwnd):
        print(
            "提示：遊戲目前鋪滿整個螢幕。若是**獨占全螢幕**，浮窗會被蓋住看不到"
            "（經驗統計照常運作）。想看浮窗請把遊戲切成視窗模式或無邊框視窗。"
        )

    pending_update = None
    try:
        if args.console:
            _run_console(
                session, format_rate, format_exp, format_duration, args.seconds
            )
        else:
            from .ui.overlay import run_overlay

            print("浮窗已開啟。可直接拖曳移動；按 × 結束。")
            pending_update = run_overlay(session, update_check)
    except KeyboardInterrupt:
        print()
    finally:
        snapshot = session.tracker.snapshot()
        session.close()
        if store is not None:
            store.close()
        _print_summary(snapshot, format_rate, format_exp, format_duration)

    if pending_update is not None:
        # 資料庫、擷取器都關了，現在換檔並用同樣的參數把新版叫起來。
        try:
            update.apply(pending_update, relaunch_argv=sys.argv[1:])
        except update.UpdateError as exc:
            _fatal(f"更新失敗：{exc}\n\n可以到這裡手動下載：\n{update.RELEASES_PAGE}")
            return 3
        print("已更新，新版啟動中。")
    return 0


def _updates_enabled(cfg: Config) -> bool:
    """只有打包版會自動查更新；原始碼版本用 git pull。環境變數可整個關掉。"""
    return update.is_frozen() and cfg.check_updates and not update.disabled_by_env()


def _run_console(
    session, format_rate, format_exp, format_duration, seconds: float = 0.0
) -> None:
    print("按 Ctrl+C 結束。")
    deadline = time.perf_counter() + seconds if seconds and seconds > 0 else None
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            return
        result = session.tick()
        snapshot = result.snapshot

        for event in result.events:
            if event.kind in ("levelup", "death", "gap", "anomaly", "map", "character"):
                stamp = time.strftime("%H:%M:%S", time.localtime(event.wall))
                print(f"\r[{stamp}] {event.kind}: {event.detail}".ljust(100))

        windows = sorted(snapshot.rates)
        rates = "  ".join(
            f"{format_duration(w)} {format_rate(snapshot.rates[w].exp_per_hour)}"
            for w in windows
        )
        line = (
            f"{snapshot.state:7s} Lv{snapshot.level if snapshot.level else '--':>4} "
            f"{format_exp(snapshot.exp_abs):>12} "
            f"| {rates} "
            f"| 升級 {format_duration(snapshot.eta_sec)} "
            f"| 本場 {format_exp(snapshot.cum_net)} / {format_duration(snapshot.active_sec)}"
        )
        print(f"\r{line[:160]:<160}", end="", flush=True)
        time.sleep(session.sleep_until_next_tick())


def _print_summary(snapshot, format_rate, format_exp, format_duration) -> None:
    print("\n--- 本場結果 ---")
    print(f"活躍時間   : {format_duration(snapshot.active_sec)}")
    print(f"排除時間   : {format_duration(snapshot.skipped_sec)}")
    print(f"淨得經驗   : {format_exp(snapshot.cum_net)}")
    print(f"總得經驗   : {format_exp(snapshot.cum_gross)}")
    print(f"死亡損失   : {format_exp(snapshot.exp_lost)}（{snapshot.deaths} 次）")
    print(f"升級       : {snapshot.levelups} 次")
    if snapshot.active_sec > 0:
        overall = snapshot.cum_net / (snapshot.active_sec / 3600.0)
        print(f"整場效率   : {format_rate(overall)}")
    for window in sorted(snapshot.rates):
        rate = snapshot.rates[window]
        if rate.valid:
            print(f"  最近 {format_duration(window):>6}: {format_rate(rate.exp_per_hour)}")
    print(f"辨識失敗   : {snapshot.misses} 次")
    print(f"誤判丟棄   : {snapshot.glitches} 次")
    if snapshot.notes:
        print("備註       : " + "；".join(snapshot.notes))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def cmd_report(args) -> int:
    from .core.stats import format_duration, format_exp, format_rate
    from .storage import Store

    path = config_module.db_path()
    if not path.exists():
        print("還沒有任何紀錄。先跑一次 track 吧。")
        return 0

    with Store(path) as store:
        sessions = store.list_sessions(args.limit)
        if not sessions:
            print("還沒有任何紀錄。")
            return 0

        print("=== 最近場次 ===")
        header = f"{'ID':>4} {'開始時間':16} {'標記':14} {'活躍':>8} {'淨得經驗':>14} {'效率':>12} {'死':>3} {'升':>3}"
        print(header)
        print("-" * len(header))
        for row in sessions:
            started = time.strftime("%m-%d %H:%M", time.localtime(row.started_at))
            print(
                f"{row.id:>4} {started:16} {(row.label or '-')[:14]:14} "
                f"{format_duration(row.active_sec):>8} "
                f"{format_exp(row.cum_net):>14} "
                f"{format_rate(row.exp_per_hour):>12} "
                f"{row.deaths:>3} {row.levelups:>3}"
            )

        maps = [m for m in store.list_maps() if m["active_sec"] > 0]
        if maps:
            print("\n=== 各地圖效率（高到低）===")
            header = f"{'地圖':16} {'累計活躍':>10} {'累計經驗':>14} {'效率':>12}"
            print(header)
            print("-" * len(header))
            for entry in sorted(
                maps, key=lambda m: m["exp_per_hour"] or 0, reverse=True
            ):
                shown = entry["name"] or f"(未命名 {entry['map_id'][:6]})"
                print(
                    f"{shown[:16]:16} {format_duration(entry['active_sec']):>10} "
                    f"{format_exp(entry['exp']):>14} "
                    f"{format_rate(entry['exp_per_hour']):>12}"
                )

        summary = store.label_summary()
        if summary:
            print("\n=== 各標記效率（高到低）===")
            header = f"{'標記':18} {'次數':>4} {'累計活躍':>10} {'累計經驗':>14} {'效率':>12} {'死':>4}"
            print(header)
            print("-" * len(header))
            for entry in summary:
                print(
                    f"{entry['label'][:18]:18} {entry['runs']:>4} "
                    f"{format_duration(entry['active_sec']):>10} "
                    f"{format_exp(entry['exp']):>14} "
                    f"{format_rate(entry['exp_per_hour']):>12} "
                    f"{entry['deaths']:>4}"
                )

        daily = store.daily_report(args.days)
        if daily:
            print(f"\n=== 各角色 × 地圖（最近 {args.days} 天，按日）===")
            header = (
                f"{'日期':10} {'角色':12} {'職業':8} {'地圖':16} {'經驗':>12} "
                f"{'練了':>9} {'活躍':>8} {'平均':>12} {'峰值':>12} {'死':>3} {'升':>3}"
            )
            print(header)
            print("-" * len(header))
            for entry in daily:
                pct = entry["pct_gained"]
                shown_map = entry["map_name"] or f"(未命名 {entry['map_id'][:6]})"
                print(
                    f"{entry['day']:10} {(entry['character'] or '-')[:12]:12} "
                    f"{(entry['job'] or '-')[:8]:8} {shown_map[:16]:16} "
                    f"{format_exp(entry['exp']):>12} "
                    f"{(f'{pct:+.2f}%' if pct is not None else '--'):>9} "
                    f"{format_duration(entry['active_sec']):>8} "
                    f"{format_rate(entry['exp_per_hour']):>12} "
                    f"{format_rate(entry['peak_exp_per_hour']):>12} "
                    f"{entry['deaths']:>3} {entry['levelups']:>3}"
                )
    return 0


# --------------------------------------------------------------------------- #
# maps
# --------------------------------------------------------------------------- #


def cmd_maps(args) -> int:
    """列出偵測到的地圖，或幫某張地圖取名。

    區分地圖靠的是地圖名那塊像素的雜湊（同一張地圖每次渲染完全相同），名字則是
    OCR 結果跟客戶端的官方清單比對出來的；比錯了就用 ``--name`` 改一次，之後那張
    地圖都認得。縮圖會匯出成 PNG，照著看就知道那是哪張圖。
    """
    import sqlite3

    from .core.stats import format_duration, format_rate
    from .storage import Store

    path = config_module.db_path()
    if not path.exists():
        print("還沒有任何紀錄。先跑一次 track 吧。")
        return 0

    with Store(path) as store:
        try:
            if args.prune:
                removed = store.prune_maps(args.prune)
                print(f"清掉 {removed} 筆沒命名、累計不到 {args.prune:g} 秒的地圖。")
                return 0
            if args.name:
                map_id, name = args.name
                if store.name_map(map_id, name):
                    print(f"已命名：{map_id} -> {name}")
                    return 0
                print(f"找不到地圖 {map_id}", file=sys.stderr)
                return 1
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc):
                raise
            # 追蹤中的那個行程握著資料庫；與其吐一串 traceback，不如講清楚。
            print(
                "資料庫正在被使用中 —— 請先結束正在執行的 track（浮窗按 ×）再試一次。",
                file=sys.stderr,
            )
            return 2

        maps = store.list_maps()
        if not maps:
            print("還沒有偵測到任何地圖。")
            return 0

        out_dir = config_module.data_dir() / "map_thumbs"
        out_dir.mkdir(parents=True, exist_ok=True)
        header = f"{'ID':14} {'名稱':16} {'累計活躍':>10} {'效率':>12}  縮圖"
        print(header)
        print("-" * (len(header) + 20))
        for entry in maps:
            thumb = store.map_thumbnail(entry["map_id"])
            thumb_path = ""
            if thumb:
                target = out_dir / f"{entry['map_id']}.png"
                try:
                    target.write_bytes(thumb)
                    thumb_path = str(target)
                except OSError:
                    thumb_path = ""
            print(
                f"{entry['map_id']:14} {(entry['name'] or '(未命名)')[:16]:16} "
                f"{format_duration(entry['active_sec']):>10} "
                f"{format_rate(entry['exp_per_hour']):>12}  {thumb_path}"
            )
        print()
        print("命名方式： run.cmd maps --name <ID> <名稱>")
    return 0


# --------------------------------------------------------------------------- #
# level
# --------------------------------------------------------------------------- #


def cmd_level(args) -> int:
    """告訴程式目前等級，順便把等級字型學起來。

    等級用的是跟狀態列不同的粗體字型，而且畫面上一次只看得到目前等級用到的
    那幾個數字 —— 沒辦法像狀態列那樣一次採集完。做過這一次之後，每次升級都能
    自動學到新數字（升級時新等級必定是舊等級 +1），十個數字很快就補齊。
    """
    from .core.session import SetupError, TrackingSession

    cfg = Config.load()
    session = TrackingSession(cfg)
    try:
        session.setup()
    except SetupError as exc:
        print(f"無法開始：{exc}", file=sys.stderr)
        return 2
    try:
        learned = session.set_level(args.level)
    finally:
        session.close()
    if learned:
        print(f"已記住等級 {args.level}，新學會 {learned} 個數字字形。")
    else:
        print(f"已記住等級 {args.level}（這些字形之前就學過了）。")
    print(f"目前認得的等級數字：{session.level_reader.known_digits or '（無）'}")
    return 0


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #


def cmd_windows(_args) -> int:
    from .win32.api import enable_dpi_awareness
    from .win32.windows import list_windows

    enable_dpi_awareness()
    windows = list_windows()
    if not windows:
        print("找不到任何可擷取的視窗。")
        return 1
    for info in windows:
        print(f"  {info.hwnd:>10}  {info.label}")
    print(f"\n共 {len(windows)} 個。")
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(_args) -> int:
    from .autosetup import auto_configure, builtin_templates_path, load_templates
    from .vision.reader import StatusReader
    from .win32.api import enable_dpi_awareness
    from .win32.capture import WindowCapturer
    from .win32.wgc import WGC_AVAILABLE, WGC_IMPORT_ERROR
    from .win32.windows import find_window

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"[{'OK  ' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not passed:
            ok = False

    print(f"DPI 模式          : {enable_dpi_awareness()}")
    print(f"資料目錄          : {config_module.data_dir()}")
    print(
        "擷取後端          : "
        + (
            "WGC（視窗/全螢幕/被遮住都可擷取）"
            if WGC_AVAILABLE
            else f"僅螢幕 BitBlt —— 未安裝 windows-capture（{WGC_IMPORT_ERROR}）"
        )
    )

    cfg = Config.load()
    templates, source = load_templates()
    missing = sorted(set("0123456789") - set("".join(templates.chars())))
    check(
        "字元模板",
        not missing and len(templates) > 0,
        f"{source}，{len(templates)} 張"
        + (f"，缺 {''.join(missing)}" if missing else "")
        + f"；內建檔 {builtin_templates_path().name}"
        + ("（存在）" if builtin_templates_path().exists() else "（遺失）"),
    )

    window = find_window(cfg.capture.window_title_contains, cfg.capture.process_name)
    check(
        "找到遊戲視窗",
        window is not None,
        window.label if window else f"找不到 {cfg.capture.window_title_contains!r}",
    )
    if window is None:
        return 0 if ok else 1

    from . import gamedata

    vocab = gamedata.sync(window.hwnd)
    check(
        "客戶端地圖清單",
        bool(vocab),
        f"{len(vocab)} 張、{len(vocab.streets)} 個地區，來源 {vocab.source}"
        if vocab
        else f"找不到（找過 {gamedata.find_bundle_dir(window.hwnd) or '無目錄'}）"
        "；地圖名稱會變成要自己命名一次，其餘功能不受影響",
    )

    capturer = WindowCapturer(window.hwnd, cfg.capture.backend)
    try:
        try:
            frame = capturer.capture_client()
            check("擷取畫面", True, f"{frame.width}x{frame.height} 後端={frame.backend}")
        except Exception as exc:
            check("擷取畫面", False, str(exc))
            return 1

        from .win32.windows import covers_whole_screen

        if covers_whole_screen(window.hwnd):
            print(
                "浮窗顯示          : 遊戲鋪滿整個螢幕。若是獨占全螢幕，"
                "浮窗會被蓋住（擷取不受影響）"
            )

        auto = auto_configure(cfg, capturer, templates, template_source=source)
        check("定位經驗值欄位", auto.ok, f"{auto.roi_source} — {auto.message}")
        if not auto.ok:
            return 1
        print(f"ROI               : {cfg.reader.exp_roi}")
        print(f"二值化            : 門檻 {cfg.reader.threshold}, invert={cfg.reader.invert}")

        reader = StatusReader(capturer, cfg.reader, templates)
        reading = reader.read()
        if reading.ok:
            check(
                "讀取狀態列",
                True,
                f"{reading.raw_exp!r} 經驗={reading.exp_abs:,} 百分比={reading.exp_pct}",
            )
            need = reading.derived_need
            check(
                "可推導所需經驗",
                need is not None,
                f"{need:,}" if need else "ROI 需要同時包含絕對值與百分比",
            )
        else:
            check("讀取狀態列", False, reading.reason)
    finally:
        capturer.close()

    print("\n全部通過，可以直接 `track`。" if ok else "\n有項目未通過，請依上面訊息處理。")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #


def cmd_selftest(_args) -> int:
    """不需要遊戲的端對端檢查：合成畫面 -> 辨識 -> 追蹤統計。"""
    from . import testfont
    from .config import TrackerConfig
    from .core.reading import StatusReading, parse_exp_text
    from .core.tracker import Tracker
    from .vision.preprocess import binarize
    from .vision.segment import segment

    failures: list[str] = []

    def expect(label: str, condition: bool, detail: str = "") -> None:
        print(f"[{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not condition:
            failures.append(label)

    templates = testfont.build_template_set()

    # 視覺管線
    source = "1234567[30.25%]"
    image = testfont.render_bgra(source, noise=4)
    mask = binarize(image, threshold=128, invert=False)
    boxes = segment(mask, expected_width=templates.median_width())
    text = "".join(templates.match(b.extract(mask)).char or "?" for b in boxes)
    expect("視覺管線辨識", text == source, f"讀到 {text!r}")

    exp_abs, exp_pct = parse_exp_text(text)
    expect("數值解析", (exp_abs, exp_pct) == (1234567, 30.25), f"{exp_abs} / {exp_pct}")

    reading = StatusReading(mono=0, wall=0, ok=True, exp_abs=exp_abs, exp_pct=exp_pct)
    need = reading.derived_need
    expect(
        "所需經驗推導",
        need is not None and abs(need - 4_081_213) / 4_081_213 < 0.01,
        f"{need:,}" if need else "失敗",
    )

    # 追蹤邏輯：穩定成長 -> 升級 -> 死亡 -> 掛機
    tracker = Tracker(TrackerConfig(rate_windows=[60], eta_window=60))
    need_value = 10_000

    def sample(mono: float, exp: int, lvl: int) -> StatusReading:
        return StatusReading(
            mono=mono,
            wall=mono,
            ok=True,
            level=lvl,
            exp_abs=exp,
            exp_pct=round(100.0 * exp / need_value, 2),
        )

    for i in range(11):
        tracker.feed(sample(i, 1000 + 100 * i, 30))
    snapshot = tracker.snapshot()
    expect("累積經驗", snapshot.cum_gross == 1000, f"{snapshot.cum_gross}")
    expect(
        "速率估算",
        snapshot.rates[60].exp_per_hour is not None
        and abs(snapshot.rates[60].exp_per_hour - 360_000) < 1,
        f"{snapshot.rates[60].exp_per_hour}",
    )

    for mono in range(11, 311):
        tracker.feed(StatusReading.failed(mono, mono, "selftest-afk"))
    for i in range(5):
        tracker.feed(sample(320 + i, 3000 + 100 * i, 30))
    snapshot = tracker.snapshot()
    expect(
        "掛機不稀釋速率",
        snapshot.rates[60].exp_per_hour is not None
        and abs(snapshot.rates[60].exp_per_hour - 360_000) < 1,
        f"{snapshot.rates[60].exp_per_hour}",
    )
    expect("中斷時間被排除", snapshot.skipped_sec > 290, f"{snapshot.skipped_sec:.0f} 秒")

    print()
    if failures:
        print(f"{len(failures)} 項失敗：" + "、".join(failures))
        return 1
    print("全部通過。")
    return 0


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #


def cmd_update(args) -> int:
    """手動檢查並安裝新版。浮窗那條自動路徑用的是同一套函式，這裡只是印出來。"""
    print(f"目前版本：v{__version__}")
    if update.disabled_by_env():
        print(f"已設定 {update.DISABLE_ENV}，不檢查更新。")
        return 0
    check = update.UpdateCheck().start()
    print("正在查詢 GitHub Releases…", end="", flush=True)
    check.wait()
    print()
    if check.error:
        print(f"查詢失敗：{check.error}", file=sys.stderr)
        return 1
    release = check.release
    if release is None:
        print("已經是最新版。")
        return 0
    print(f"有新版本：v{release.version}（{release.page}）")
    if release.notes:
        print()
        print(release.notes)
        print()
    if args.check:
        return 0
    if not update.is_frozen():
        print("這是原始碼版本，不會自動替換；請用 git pull 更新。")
        return 0

    def progress(received: int, total: int | None) -> None:
        if total:
            print(f"\r下載中 {received / 1_048_576:6.1f} / {total / 1_048_576:.1f} MB", end="")
        else:
            print(f"\r下載中 {received / 1_048_576:6.1f} MB", end="")

    try:
        staged = update.stage(release, progress=progress)
        print()
        update.apply(staged, relaunch_argv=None)
    except update.UpdateError as exc:
        print()
        print(f"更新失敗：{exc}", file=sys.stderr)
        return 1
    print(f"已更新到 v{release.version}。下次啟動就是新版。")
    return 0


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapleexp",
        description="楓之谷：經典版 經驗值計算器（截圖 + 模板比對，不接觸遊戲程序）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "calibrate", help="開啟校準精靈（通常不需要；自動設定失敗時才用）"
    ).set_defaults(func=cmd_calibrate)

    track = sub.add_parser("track", help="開始追蹤")
    track.add_argument("--label", default="", help="場次標記，建議填地圖名稱")
    track.add_argument("--console", action="store_true", help="不開浮窗，輸出到終端機")
    track.add_argument("--no-db", action="store_true", help="不寫入資料庫")
    track.add_argument("--interval", type=float, default=None, help="取樣間隔（秒）")
    track.add_argument(
        "--seconds", type=float, default=0.0, help="跑幾秒後自動結束（僅 --console）"
    )
    track.add_argument(
        "--debug-frames", action="store_true", help="把辨識失敗的畫面存成 PNG"
    )
    track.add_argument(
        "--no-update-check", action="store_true", help="這次啟動不檢查新版本"
    )
    track.set_defaults(func=cmd_track)

    report = sub.add_parser("report", help="歷史場次與效率比較")
    report.add_argument("--limit", type=int, default=20, help="列出幾場")
    report.add_argument(
        "--days", type=int, default=30, help="角色 × 地圖的按日報表要回溯幾天（0 = 全部）"
    )
    report.set_defaults(func=cmd_report)

    maps = sub.add_parser("maps", help="列出偵測到的地圖 / 幫地圖命名")
    maps.add_argument(
        "--name", nargs=2, metavar=("ID", "名稱"), help="幫某張地圖取名"
    )
    maps.add_argument(
        "--prune", nargs="?", type=float, const=60.0, default=None,
        metavar="秒數", help="清掉沒命名且累計時間不到 N 秒的地圖（預設 60 秒）",
    )
    maps.set_defaults(func=cmd_maps)

    level = sub.add_parser("level", help="告訴程式目前等級（只需做一次）")
    level.add_argument("level", type=int, help="目前的等級")
    level.set_defaults(func=cmd_level)

    sub.add_parser("windows", help="列出可擷取的視窗").set_defaults(func=cmd_windows)
    sub.add_parser("doctor", help="診斷為什麼讀不到").set_defaults(func=cmd_doctor)
    sub.add_parser("selftest", help="不需要遊戲的自我測試").set_defaults(func=cmd_selftest)

    upd = sub.add_parser("update", help="檢查並安裝新版本")
    upd.add_argument("--check", action="store_true", help="只檢查，不下載")
    upd.set_defaults(func=cmd_update)
    return parser


# 直接點兩下執行檔時沒有任何參數。對打包好的免安裝版來說，那個情境就是
# 「開始追蹤」，不該丟一個 usage 錯誤給使用者看。
DEFAULT_COMMAND = ["track"]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # argparse 的 usage／錯誤訊息也要有地方印，所以主控台要在 parse 之前準備好。
    # 只有「雙擊開浮窗」這一種情況不需要主控台。
    wants_console = bool(argv) and not (argv[0] == "track" and "--console" not in argv)
    allocated = _setup_console(wanted=wants_console)
    if update.is_frozen():
        # 上次更新把舊版改名成 .old 留著；現在輪到我們把它清掉。
        update.cleanup_old()
    try:
        args = build_parser().parse_args(argv or DEFAULT_COMMAND)
        return args.func(args)
    finally:
        if allocated:
            try:
                input("\n按 Enter 關閉。")
            except (EOFError, OSError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
