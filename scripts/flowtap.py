#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""flowtap: watch requests travel client -> platform gateway -> plugin backend(s).

Hop 1 is a logging reverse proxy you point curl / the nemo CLI / a browser at.
Hop 2 is the gateway re-issuing the call to a plugin's own backend. flowtap can
observe it two ways, per plugin:

  --adapter-log [NAME=]PATH   tail the backend's access log (e.g. uvicorn)
  --tap NAME=LISTEN:TARGET    run a real proxy in front of the backend; the plugin
                              must be registered with its upstream at LISTEN
                              (works for any language, even one that logs nothing)

In a terminal it draws a live flow diagram (one box per plugin, edges light up as
traffic moves) above a rolling request log. Piped or with --plain it prints a
scrolling log instead. Keys: r = reset, q = quit, 1-7 = run a sample `nemo` call through the tap (a spinner
shows it running, then its output appears under the diagram), ? = list the samples
(also: --list-samples to print them). Up/Down, PgUp/PgDn, Home/End (or j/k) scroll long output.

    # Python plugin that logs requests (Phoenix on :6007)
    python scripts/flowtap.py --adapter-log phoenix=/path/to/phoenix-adapter.log

    # Rust plugin on :6010, registered with PUBLIC_URL=http://127.0.0.1:6011
    python scripts/flowtap.py --tap is-it-christmas=6011:http://127.0.0.1:6010

    curl http://127.0.0.1:8081/apis/plugins/phoenix/v2/workspaces/default/projects
    NHX_BASE_URL=http://127.0.0.1:8081 nemo plugins list
    open http://127.0.0.1:8081/studio/      # browser traffic goes through the tap

Stdlib only. Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import http.client
import itertools
import json
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}  # fmt: skip
ANSI = {
    "dim": "\033[2m", "bold": "\033[1m", "red": "\033[31m", "green": "\033[32m",
    "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "off": "\033[0m",
}  # fmt: skip
ACTIVE_SECONDS = 1.5
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def c(color: str, text: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['off']}" if USE_COLOR else text


def client_kind(ua: str) -> str:
    ua = (ua or "").lower()
    if "curl" in ua:
        return "curl"
    if "mozilla" in ua or "safari" in ua:
        return "browser"
    if "python" in ua or "httpx" in ua or "nemo" in ua:
        return "cli/sdk"
    return ua.split("/")[0][:8] or "client"


def classify(path: str) -> tuple[str, str, str]:
    """Return (lane, label, color): which gateway lane the request takes."""
    p = urlsplit(path).path
    if p.startswith("/apis/plugins/-/") or p == "/apis/plugins":
        return "registry", "gateway: plugin registry", "yellow"
    m = re.match(r"^/apis/plugins/([^/]+)/", p)
    if m:
        return f"plugin:{m.group(1)}", f"gateway -> PLUGIN '{m.group(1)}' (external backend)", "magenta"
    m = re.match(r"^/apis/([^/]+)/", p)
    if m:
        return "core", f"gateway -> service '{m.group(1)}'", "cyan"
    if p.startswith("/plugin-ui/"):
        return "static", "gateway: trusted plugin UI bundle", "magenta"
    return "static", "gateway: Studio / static assets", "dim"


class Stats:
    """Thread-safe counters + recent events that feed both renderers."""

    def __init__(self, tap_port: int, gateway: str, hop2_sources: dict[str, str]) -> None:
        self.lock = threading.Lock()
        self.tap_port = tap_port
        self.gateway = gateway
        self.hop2_sources = hop2_sources  # plugin name -> how hop 2 is observed
        self.plugins: set[str] = set(hop2_sources)
        self.clients: dict[str, tuple[int, float]] = {}
        self.edges: dict[str, tuple[int, float]] = {}
        self.events: collections.deque[str] = collections.deque(maxlen=200)
        self.samples: list[tuple[str, list[str]]] = []
        self.show_samples = False
        self.running: dict[int, tuple[str, float]] = {}
        self.last_run: tuple[str, int, float, list[str]] | None = None  # cmd, exit code, seconds, output
        self._run_ids = itertools.count(1)
        self.scroll = 0  # first visible line of the last command's output
        self.visible = 0  # how many output lines the panel showed last frame

    def reset(self) -> None:
        with self.lock:
            self.clients.clear()
            self.edges.clear()
            self.events.clear()
            self.last_run = None
            self.scroll = 0

    def start_run(self, cmd: str) -> int:
        with self.lock:
            run_id = next(self._run_ids)
            self.running[run_id] = (cmd, time.time())
            return run_id

    def finish_run(self, run_id: int, cmd: str, code: int, seconds: float, output: list[str]) -> None:
        with self.lock:
            self.running.pop(run_id, None)
            self.last_run = (cmd, code, seconds, output)
            self.scroll = 0

    def scroll_output(self, key: str) -> None:
        """Move the CLI output panel's window. ``key`` is up/down/pgup/pgdn/home/end."""
        with self.lock:
            total = len(self.last_run[3]) if self.last_run else 0
            visible = max(self.visible, 1)
            top = max(total - visible, 0)
            step = {"up": -1, "down": 1, "pgup": -(visible - 1), "pgdn": visible - 1}.get(key)
            if key == "home":
                self.scroll = 0
            elif key == "end":
                self.scroll = top
            elif step is not None:
                self.scroll = min(max(self.scroll + step, 0), top)

    def note(self, event: str) -> None:
        with self.lock:
            self.events.append(event)

    def bump_edge(self, name: str) -> None:
        n, _ = self.edges.get(name, (0, 0.0))
        self.edges[name] = (n + 1, time.time())

    def hop1(self, client: str, lane: str, event: str, error: bool = False) -> None:
        with self.lock:
            n, _ = self.clients.get(client, (0, 0.0))
            self.clients[client] = (n + 1, time.time())
            if lane.startswith("plugin:"):
                self.plugins.add(lane.split(":", 1)[1])
            for e in ("in", "gw", lane):
                self.bump_edge(e)
            if error:
                self.bump_edge(f"err:{lane}")
            self.events.append(event)

    def hop2(self, plugin: str, event: str) -> None:
        with self.lock:
            self.plugins.add(plugin)
            self.bump_edge(f"hop2:{plugin}")
            self.events.append(event)


def edge(stats: Stats, name: str, width: int = 9) -> str:
    n, ts = stats.edges.get(name, (0, 0.0))
    live = time.time() - ts < ACTIVE_SECONDS and n > 0
    arrow = ("━" * (width - 1) + "▶") if live else ("─" * (width - 1) + "▶")
    return c("green", arrow) if live else c("dim", arrow)


def count(stats: Stats, name: str) -> str:
    n, ts = stats.edges.get(name, (0, 0.0))
    live = time.time() - ts < ACTIVE_SECONDS and n > 0
    return c("green" if live else "dim", f"{n:>4}")


def box(lines: list[str], width: int, color: str = "dim") -> list[str]:
    out = [c(color, "┌" + "─" * width + "┐")]
    for ln in lines:
        pad = width - len(re.sub(r"\033\[[0-9;]*m", "", ln))
        out.append(c(color, "│") + ln + " " * max(pad, 0) + c(color, "│"))
    out.append(c(color, "└" + "─" * width + "┘"))
    return out


def live(stats: Stats, name: str) -> bool:
    n, ts = stats.edges.get(name, (0, 0.0))
    return n > 0 and time.time() - ts < ACTIVE_SECONDS


def render_diagram(stats: Stats) -> list[str]:
    """Clients -> tap -> gateway -> one box per lane. Rows are composed side by side."""
    now = time.time()
    client_lines = [" clients"]
    for name, (n, ts) in sorted(stats.clients.items())[:4]:
        client_lines.append(c("green" if now - ts < ACTIVE_SECONDS else "dim", f" {name[:7]:<7} {n:>3}"))
    client_lines += [""] * (5 - len(client_lines))
    cbox = box(client_lines, 13, "bold")
    tap = box([f" tap :{stats.tap_port}", " logs every", " request", "", ""], 13, "bold")

    def count(key: str) -> str:
        return c("green" if live(stats, key) else "dim", f"{stats.edges.get(key, (0, 0.0))[0]:>3} req")

    def lane(key: str, title: str, sub: str, color: str, hot: bool) -> list[str]:
        first = f" {title[:19]}".ljust(21) + count(key) + " "
        lines = [first, c("dim", f" {sub}")] if sub else [first]
        errors = stats.edges.get(f"err:{key}", (0, 0.0))[0]
        if errors:
            lines.append(c("red", f" ✗ {errors} error{'s' if errors != 1 else ''}"))
        border = "red" if live(stats, f"err:{key}") else (color if hot else "dim")
        return box(lines, 29, border)

    lanes: list[tuple[str, list[str]]] = []
    for name in sorted(stats.plugins):
        key, hop2 = f"plugin:{name}", f"hop2:{name}"
        if name in stats.hop2_sources:
            seen = stats.edges.get(hop2, (0, 0.0))[0]
            sub = c("green" if live(stats, hop2) else "dim", f"hop 2 seen: {seen}  ({stats.hop2_sources[name]})")
        else:
            sub = c("dim", "hop 2: not observed")
        lanes.append((key, lane(key, name, sub, "magenta", live(stats, key) or live(stats, hop2))))
    lanes += [
        ("core", lane("core", "core services", "/apis/* in-process, no hop", "cyan", live(stats, "core"))),
        ("registry", lane("registry", "plugin registry", "", "yellow", live(stats, "registry"))),
        ("static", lane("static", "Studio / UI assets", "", "dim", live(stats, "static"))),
    ]
    lane_rows: list[str] = []
    arrow_rows: dict[int, str] = {}
    for key, bx in lanes:
        arrow_rows[len(lane_rows) + 1] = key
        lane_rows.extend(bx)
    total = len(lane_rows)

    gw_inner = [" gateway", f" {stats.gateway.replace('http://', '')}", " routes by path"]
    gw_inner += [""] * (total - 2 - len(gw_inner))
    gw = box(gw_inner, 17, "bold")

    def at(rows: list[str], row: int, start: int, width: int) -> str:
        i = row - start
        return rows[i] if 0 <= i < len(rows) else " " * width

    mid = (total - len(cbox)) // 2
    out: list[str] = []
    for r in range(total):
        a_in = edge(stats, "in", 5) if r == mid + 2 else " " * 5
        a_gw = edge(stats, "gw", 5) if r == mid + 2 else " " * 5
        key = arrow_rows.get(r)
        a_ln = edge(stats, key, 6) if key else " " * 6
        out.append(
            " " + at(cbox, r, mid, 15) + a_in + at(tap, r, mid, 15) + a_gw + at(gw, r, 0, 19) + a_ln + lane_rows[r]
        )
    return out


def build_samples(project: str) -> list[tuple[str, list[str]]]:
    """Sample `nemo` calls worth showing: each exercises a different lane of the diagram."""
    return [
        ("list plugins (built-in + external)", ["plugins", "list"]),
        ("what do plugins expose?", ["plugins", "endpoints"]),
        ("is it Christmas? (Rust backend)", ["plugins", "call", "is-it-christmas"]),
        ("phoenix projects (Python backend)", ["plugins", "call", "phoenix", "projects"]),
        (f"phoenix spans in '{project}'", ["plugins", "call", "phoenix", "spans", "-q", f"project={project}"]),
        ("registry: installed plugins", ["plugins", "installed"]),
        ("a Rust panic, on purpose", ["plugins", "call", "is-it-christmas", "panic"]),
    ]


def find_nemo(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which("nemo")
    if found:
        return found
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "nemo")
    return local if os.path.exists(local) else None


SCROLL_KEYS = {"A": "up", "B": "down", "H": "home", "F": "end", "5~": "pgup", "6~": "pgdn", "1~": "home", "4~": "end"}


def parse_keys(data: bytes) -> list[str]:
    """Split raw terminal input into keys: plain characters, or up/down/pgup/pgdn/home/end."""
    keys: list[str] = []
    i = 0
    while i < len(data):
        if data[i : i + 1] == b"\x1b" and data[i + 1 : i + 2] in (b"[", b"O"):
            j = i + 2
            while j < len(data) and not 0x40 <= data[j] <= 0x7E:  # parameter bytes, then the final byte
                j += 1
            keys.append(SCROLL_KEYS.get(data[i + 2 : j + 1].decode(errors="ignore"), "?"))
            i = j + 1
        else:
            keys.append(data[i : i + 1].decode(errors="ignore").lower())
            i += 1
    return keys


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def run_in_pty(argv: list[str], env: dict[str, str], cols: int, timeout: float = 60) -> tuple[int, str]:
    """Run a command as if typed in a terminal of width ``cols``.

    The nemo CLI prints tables only to a terminal (it falls back to JSON in a pipe), so
    the sample output is captured through a pty to match what you would see typing it.
    """
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, cols, 0, 0))
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    chunks: list[bytes] = []
    deadline = time.time() + timeout
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    data = os.read(master, 65536)
                except OSError:  # EIO once the child has closed its end
                    break
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                break
            if time.time() > deadline:
                proc.kill()
                chunks.append(f"\n(timed out after {timeout:.0f}s)".encode())
                break
    finally:
        os.close(master)
    return proc.wait(), b"".join(chunks).decode(errors="replace")


def clean_output(text: str) -> list[str]:
    text = ANSI_ESCAPE.sub("", text).replace("\r\n", "\n").replace("\r", "")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines[:200]


def run_sample(stats: Stats, index: int, nemo: str | None, config: str | None) -> None:
    """Run one sample through the tap in the background, showing progress and then its output."""
    _, argv = stats.samples[index]
    cmd = "nemo " + " ".join(argv)
    run_id = stats.start_run(cmd)
    cols = max(shutil.get_terminal_size((100, 40)).columns - 6, 40)

    def work() -> None:
        started = time.time()
        if nemo is None:
            code, output = 127, ["could not find the nemo CLI (use --nemo PATH)"]
        else:
            env = {**os.environ, "NHX_BASE_URL": f"http://127.0.0.1:{stats.tap_port}", "NO_COLOR": "1"}
            if config:
                env["NHX_CONFIG_FILE"] = config
            try:
                code, raw = run_in_pty([nemo, *argv], env, cols)
                output = clean_output(raw)
            except Exception as exc:  # noqa: BLE001
                code, output = 1, [f"could not run: {exc}"]
        stats.finish_run(run_id, cmd, code, time.time() - started, output)
        note = "exit 0" if code == 0 else f"exit {code}: {(output or [''])[0][:90]}"
        stats.note(
            f"{c('dim', time.strftime('%H:%M:%S'))} {c('cyan', 'sample'.ljust(7))} {c('bold', '$ ' + cmd)}\n"
            f"         {c('cyan', '└─ ' + note)}"
        )

    threading.Thread(target=work, daemon=True).start()


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def render_run_panel(stats: Stats, width: int, budget: int) -> list[str]:
    """Spinner lines for commands in flight, then the last finished command's output."""
    lines: list[str] = []
    now = time.time()
    for cmd, started in stats.running.values():
        frame = SPINNER[int(now * 10) % len(SPINNER)]
        lines.append(f" {c('cyan', frame)} {c('bold', 'running')}  $ {cmd}  {c('dim', f'{now - started:.1f}s')}")
    if stats.last_run is not None:
        cmd, code, seconds, output = stats.last_run
        mark = c("green", "✓") if code == 0 else c("red", "✗")
        header_at = len(lines)
        lines.append("")  # status line, filled in once we know how much fits
        visible = max(budget - len(lines), 2)
        stats.visible = visible
        total = len(output)
        start = min(stats.scroll, max(total - visible, 0))
        shown = output[start : start + visible]
        status = f"{seconds:.1f}s · exit {code}"
        if total > visible:
            status += f" · lines {start + 1}-{start + len(shown)} of {total} · ↑/↓ PgUp/PgDn scroll"
        lines[header_at] = f" {mark} $ {cmd}  {c('dim', status)}"
        lines += [c("dim", " │ ") + ln[: width - 6] for ln in shown]
    if not lines:
        return []
    return [c("dim", " ── cli output " + "─" * max(width - 16, 0)), *lines, ""]


def render(stats: Stats) -> str:
    width = shutil.get_terminal_size((100, 40)).columns
    height = shutil.get_terminal_size((100, 40)).lines
    with stats.lock:
        diagram = render_diagram(stats)
        header = [
            c("bold", " flowtap")
            + c(
                "dim",
                f"  point curl / NHX_BASE_URL / browser at http://127.0.0.1:{stats.tap_port}   r = reset   q = quit",
            ),
            c(
                "dim",
                f" 1-{len(stats.samples)} run a sample nemo call through the tap   ? list them   ↑/↓ scroll output",
            )
            if stats.samples
            else "",
        ]
        if stats.show_samples:
            header += [c("dim", f"   {i}  nemo {' '.join(argv)}") for i, (_, argv) in enumerate(stats.samples, 1)]
            header.append("")
        panel = render_run_panel(stats, width, max(4, min(16, height - len(header) - len(diagram) - 12)))
        room = max(height - len(header) - len(diagram) - len(panel) - 3, 4)
        recent = list(stats.events)[-room:]
        log = [c("dim", " ── recent requests " + "─" * max(width - 22, 0))]
        for ev in recent:
            for ln in ev.split("\n"):
                log.append(" " + ln)
    lines = header + diagram + [""] + panel + log[-room * 2 :]
    # Overwrite in place (clear to end of line, then to end of screen) so the spinner does not flicker.
    return "\033[H" + "".join(ln + "\033[K\n" for ln in lines) + "\033[J"


def make_handler(target, on_done):
    """Reverse-proxy handler. ``on_done(handler, status, size_bytes, ms)`` runs after each request."""
    host, port = target.hostname, target.port or 80

    class Tap(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:  # we log ourselves
            pass

        def _proxy(self) -> None:
            start = time.time()
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            headers["Host"] = f"{host}:{port}"
            conn = http.client.HTTPConnection(host, port, timeout=300)
            status, size = 502, 0
            head = b""
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                self.send_response_only(status, resp.reason)
                streaming = False
                for k, v in resp.getheaders():
                    kl = k.lower()
                    if kl in HOP_BY_HOP:
                        continue
                    if kl == "location" and v.startswith(f"http://{host}:{port}"):
                        v = v.replace(f"http://{host}:{port}", f"http://{self.headers.get('Host')}")
                    self.send_header(k, v)
                    if kl == "content-type" and "text/event-stream" in v:
                        streaming = True
                if resp.getheader("Content-Length") is None:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                while True:
                    chunk = resp.read1(8192) if streaming else resp.read(8192)
                    if not chunk:
                        break
                    size += len(chunk)
                    if status >= 400 and len(head) < 4096:
                        head += chunk[: 4096 - len(head)]
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as exc:  # noqa: BLE001
                if status == 502:
                    msg = f"tap could not reach {host}:{port}: {exc}".encode()
                    try:
                        self.send_response(502)
                        self.send_header("Content-Length", str(len(msg)))
                        self.end_headers()
                        self.wfile.write(msg)
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                conn.close()
                self.error_head = head
                on_done(self, status, size, (time.time() - start) * 1000)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _proxy

    return Tap


def error_summary(head: bytes, lines: int = 2) -> list[str]:
    """First lines of an error response: the `detail` of a JSON error body, else the raw text."""
    try:
        body = json.loads(head)
        text = body.get("detail") if isinstance(body, dict) else None
        text = text if isinstance(text, str) else (head.decode(errors="replace") if text is None else json.dumps(text))
    except ValueError:
        text = head.decode(errors="replace")
    return [ln[:100] for ln in text.splitlines() if ln.strip()][:lines]


def status_color(status: int) -> str:
    return "green" if status < 400 else ("yellow" if status < 500 else "red")


def hop1_logger(stats: Stats, plain: bool, show_static: bool):
    def on_done(h: BaseHTTPRequestHandler, status: int, size: int, ms: float) -> None:
        lane, label, color = classify(h.path)
        if lane == "static" and not show_static:
            return
        kind = client_kind(h.headers.get("User-Agent", ""))
        event = (
            f"{c('dim', time.strftime('%H:%M:%S'))} {c('bold', kind.ljust(7))} "
            f"{c('blue', h.command.ljust(6))}{h.path[:80]}\n"
            f"         {c(color, '└─ ' + label)}  {c(status_color(status), str(status))} "
            f"{c('dim', f'{ms:.0f}ms {size}B')}"
        )
        if status >= 400:
            for line in error_summary(getattr(h, "error_head", b"")):
                event += f"\n         {c('red', '✗ ' + line)}"
        stats.hop1(kind, lane, event, error=status >= 400)
        if plain:
            print(event, flush=True)

    return on_done


def hop2_logger(stats: Stats, plain: bool, plugin: str, listen: int, target: str):
    def on_done(h: BaseHTTPRequestHandler, status: int, size: int, ms: float) -> None:
        event = (
            f"{c('dim', time.strftime('%H:%M:%S'))} {c('magenta', 'gateway'.ljust(7))} "
            f"{c('blue', h.command.ljust(6))}{h.path[:80]}\n"
            f"         {c('magenta', f'└─ HOP 2: {plugin} via tap :{listen} -> {target}')}  "
            f"{c(status_color(status), str(status))} {c('dim', f'{ms:.0f}ms {size}B')}"
        )
        stats.hop2(plugin, event)
        if plain:
            print(event, flush=True)

    return on_done


ACCESS = re.compile(r'"(?P<m>[A-Z]+) (?P<p>\S+) HTTP/[\d.]+" (?P<s>\d+)')


def tail_adapter(plugin: str, path: str, stats: Stats, plain: bool) -> None:
    while not os.path.exists(path):
        time.sleep(0.5)
    with open(path, errors="replace") as fh:
        fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.1)
                continue
            m = ACCESS.search(line)
            if not m:
                continue
            s = int(m["s"])
            event = (
                f"{c('dim', time.strftime('%H:%M:%S'))} {c('magenta', 'gateway'.ljust(7))} "
                f"{c('blue', m['m'].ljust(6))}{m['p'][:80]}\n"
                f"         {c('magenta', f'└─ HOP 2: {plugin} (from access log)')}  {c(status_color(s), str(s))}"
            )
            stats.hop2(plugin, event)
            if plain:
                print(event, flush=True)


def serve(port: int, handler) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="http://127.0.0.1:8080", help="platform gateway to forward to")
    ap.add_argument("--listen", type=int, default=8081, help="port the tap listens on")
    ap.add_argument(
        "--adapter-log",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="tail a plugin backend's access log for hop 2 (NAME defaults to 'phoenix')",
    )
    ap.add_argument(
        "--tap",
        action="append",
        default=[],
        metavar="NAME=LISTEN:TARGET",
        help="proxy LISTEN port to TARGET url as a real hop-2 tap, e.g. mine=6011:http://127.0.0.1:6010",
    )
    ap.add_argument("--plugin", action="append", default=[], metavar="NAME", help="show a plugin box before traffic")
    ap.add_argument("--project", default="nemo-pr-review", help="Phoenix project used by the spans sample")
    ap.add_argument("--nemo", metavar="PATH", help="nemo CLI used for the sample calls (default: PATH, then .venv/bin)")
    ap.add_argument("--nemo-config", metavar="FILE", help="NHX_CONFIG_FILE for sample calls, e.g. an empty file")
    ap.add_argument("--list-samples", action="store_true", help="print the sample commands and exit")
    ap.add_argument("--show-static", action="store_true", help="also log Studio static asset requests")
    ap.add_argument("--plain", action="store_true", help="scrolling log only, no live diagram")
    args = ap.parse_args()

    samples = build_samples(args.project)
    if args.list_samples:
        print(f"# run these from any terminal; they go through the tap on :{args.listen}")
        print(f"export NHX_BASE_URL=http://127.0.0.1:{args.listen}")
        for label, argv in samples:
            print(f"nemo {' '.join(argv)}    # {label}")
        return

    plain = args.plain or not sys.stdout.isatty()
    sources: dict[str, str] = {}
    taps: list[tuple[str, int, str]] = []
    logs: list[tuple[str, str]] = []
    for spec in args.tap:
        try:
            name, rest = spec.split("=", 1)
            listen, target = rest.split(":", 1)
            taps.append((name, int(listen), target))
        except ValueError:
            ap.error(f"--tap expects NAME=LISTEN:TARGET, got {spec!r}")
        sources[name] = f"tap :{listen}"
    for spec in args.adapter_log:
        name, _, path = spec.rpartition("=") if "=" in spec else ("phoenix", "", spec)
        logs.append((name, path))
        sources[name] = "access log"

    stats = Stats(args.listen, args.target, sources)
    stats.plugins.update(args.plugin)
    stats.samples = samples
    nemo_bin = find_nemo(args.nemo)
    if plain:
        print(f"flowtap listening on http://127.0.0.1:{args.listen} -> {args.target}", flush=True)
    for name, path in logs:
        threading.Thread(target=tail_adapter, args=(name, path, stats, plain), daemon=True).start()
    for name, listen, target in taps:
        serve(listen, make_handler(urlsplit(target), hop2_logger(stats, plain, name, listen, target)))
    serve(args.listen, make_handler(urlsplit(args.target), hop1_logger(stats, plain, args.show_static)))
    fd = sys.stdin.fileno() if (not plain and sys.stdin.isatty()) else None
    saved = termios.tcgetattr(fd) if fd is not None else None
    try:
        if fd is not None:
            tty.setcbreak(fd)  # single-key input; Ctrl-C still raises KeyboardInterrupt
        while True:
            if not plain:
                sys.stdout.write(render(stats))
                sys.stdout.flush()
            if fd is None:
                time.sleep(0.25)
                continue
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                quit_requested = False
                for key in parse_keys(os.read(fd, 64)):
                    if key == "q":
                        quit_requested = True
                    elif key == "r":
                        stats.reset()
                    elif key == "?":
                        stats.show_samples = not stats.show_samples
                    elif key in ("up", "down", "pgup", "pgdn", "home", "end"):
                        stats.scroll_output(key)
                    elif key in ("k", "j"):
                        stats.scroll_output("up" if key == "k" else "down")
                    elif key.isdigit() and 1 <= int(key) <= len(stats.samples):
                        run_sample(stats, int(key) - 1, nemo_bin, args.nemo_config)
                if quit_requested:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if fd is not None and saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        if not plain:
            sys.stdout.write("\033[H\033[J")


if __name__ == "__main__":
    main()
