"""網頁版原型的本機伺服器。

標準的 ``http.server`` 只送 Last-Modified，Chrome 會靠啟發式快取，改了 bridge.py
或重打包 mapleexp.zip 之後一般重新整理仍可能拿到舊檔，看起來就像「修了沒效」。
開發原型時每次都要拿最新的，所以一律送 no-store。
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
PORT = 8765


class NoStoreHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    handler = partial(NoStoreHandler, directory=str(WEB))
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as server:
        print(f"http://localhost:{PORT}/  (Ctrl+C 停止)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
