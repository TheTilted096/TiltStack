#!/usr/bin/env python3
"""
gpu_monitor.py — Live terminal GPU monitor with rolling time graph.

Usage:
    python gpu_monitor.py [--interval SECS] [--history SECS]

Press q or Ctrl-C to quit.
"""

import argparse
import curses
import subprocess
import time
from collections import deque


# ---------------------------------------------------------------------------
# RAM polling
# ---------------------------------------------------------------------------

def poll_ram():
    """System-wide RAM usage."""
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':')
                info[k.strip()] = int(v.split()[0])  # kB
        total = info['MemTotal']     / 1024**2   # GB
        avail = info['MemAvailable'] / 1024**2   # GB
        used  = total - avail
        return {'used': used, 'total': total, 'pct': 100.0 * used / total,
                'label': f'{used:.1f}/{total:.1f}GB'}
    except Exception:
        return None


def poll_process_ram(pid, total_gb):
    """Resident set size for a specific PID, as a fraction of total system RAM."""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    rss_gb = rss_kb / 1024**2
                    return {'used': rss_gb, 'total': total_gb,
                            'pct': 100.0 * rss_gb / total_gb,
                            'label': f'{rss_gb:.2f}GB  (PID {pid})'}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# nvidia-smi polling
# ---------------------------------------------------------------------------

FIELDS = [
    'index', 'name',
    'utilization.gpu', 'memory.used', 'memory.total',
    'temperature.gpu', 'power.draw', 'power.limit',
    'clocks.current.graphics',
]

def poll():
    try:
        raw = subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={",".join(FIELDS)}',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []

    gpus = []
    for line in raw.strip().splitlines():
        tok = [t.strip() for t in line.split(',')]
        def f(i):
            try: return float(tok[i])
            except: return 0.0
        gpus.append({
            'index':       int(f(0)),
            'name':        tok[1],
            'util':        f(2),
            'mem_used':    f(3),
            'mem_total':   f(4),
            'temp':        f(5),
            'power':       f(6),
            'power_limit': f(7),
            'clock':       f(8),
        })
    return gpus


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

BLOCKS = ' ▁▂▃▄▅▆▇█'
SOLID  = '█'
SHADE  = '▒'
EMPTY  = '░'


def safe_put(win, row, col, text, attr=0):
    h, w = win.getmaxyx()
    if row >= h - 1 or col >= w:
        return
    text = text[:w - col]
    try:
        win.addstr(row, col, text, attr)
    except curses.error:
        pass


def bar(val, width, lo=0.0, hi=100.0):
    frac  = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    full  = round(frac * width)
    return SOLID * full + EMPTY * (width - full)


def val_color(val, pairs, thresholds=(70, 90)):
    lo, mid, hi = pairs
    if val >= thresholds[1]: return hi
    if val >= thresholds[0]: return mid
    return lo


def sparkline(hist, width):
    n = len(hist)
    out = []
    for i in range(width):
        j   = int(i * n / width)
        val = hist[j] if j < n else 0.0
        out.append(BLOCKS[int(max(0.0, min(1.0, val / 100.0)) * 8)])
    return ''.join(out)


def single_graph_rows(hist, width, height):
    """Fractional-block graph for a single series (values 0–100)."""
    n = len(hist)
    col_vals = []
    for ci in range(width):
        lo = int(ci * n / width)
        hi = max(lo + 1, int((ci + 1) * n / width))
        col_vals.append(max((hist[s] for s in range(lo, min(hi, n))), default=0.0))

    rows = [[(' ', None)] * width for _ in range(height)]
    for ci in range(width):
        h_val = max(0.0, min(float(height), col_vals[ci] / 100.0 * height))
        for ri in range(height):
            rb = height - 1 - ri
            fill = (1.0          if rb < int(h_val) else
                    h_val % 1.0  if rb == int(h_val) else 0.0)
            if fill > 0:
                rows[ri][ci] = (BLOCKS[max(1, round(fill * 8))], 'val')
    return rows


def graph_rows(hist_u, hist_m, width, height):
    """
    Return (height) rows of (char, kind) pairs using fractional block chars.

    Each column maps to a contiguous slice of history; the max value in that
    slice is used so spikes are never lost during compression.  Vertical
    resolution is 8× the number of rows via the ▁▂▃▄▅▆▇█ block characters.
    """
    n = len(hist_u)
    # Pre-compute per-column max values.
    col_u = []
    col_m = []
    for ci in range(width):
        lo = int(ci * n / width)
        hi = max(lo + 1, int((ci + 1) * n / width))
        col_u.append(max((hist_u[s] for s in range(lo, min(hi, n))), default=0.0))
        col_m.append(max((hist_m[s] for s in range(lo, min(hi, n))), default=0.0))

    # Build rows top-to-bottom.
    rows = [[(' ', None)] * width for _ in range(height)]
    for ci in range(width):
        uh = max(0.0, min(float(height), col_u[ci] / 100.0 * height))
        mh = max(0.0, min(float(height), col_m[ci] / 100.0 * height))
        for ri in range(height):
            rb = height - 1 - ri          # row index counted from the bottom
            u_fill = (1.0          if rb < int(uh) else
                      uh - int(uh) if rb == int(uh) else 0.0)
            m_fill = (1.0          if rb < int(mh) else
                      mh - int(mh) if rb == int(mh) else 0.0)
            if u_fill > 0 and m_fill > 0:
                char = BLOCKS[max(1, round(max(u_fill, m_fill) * 8))]
                kind = 'both'
            elif u_fill > 0:
                char = BLOCKS[max(1, round(u_fill * 8))]
                kind = 'util'
            elif m_fill > 0:
                char = BLOCKS[max(1, round(m_fill * 8))]
                kind = 'mem'
            else:
                char = ' '
                kind = None
            rows[ri][ci] = (char, kind)
    return rows


def fmt_duration(secs):
    if secs >= 3600: return f"{secs/3600:.1f}h"
    if secs >= 60:   return f"{secs/60:.0f}m"
    return f"{secs:.0f}s"


# ---------------------------------------------------------------------------
# Main curses loop
# ---------------------------------------------------------------------------

def draw(stdscr, gpus, histories, args, tick, ram_now, ram_hist):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN,  -1)   # util / good
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # warn
    curses.init_pair(3, curses.COLOR_RED,    -1)   # critical
    curses.init_pair(4, curses.COLOR_CYAN,   -1)   # header
    curses.init_pair(5, curses.COLOR_WHITE,  -1)   # dim labels
    curses.init_pair(6, curses.COLOR_BLUE,   -1)   # mem
    curses.init_pair(7, curses.COLOR_MAGENTA,-1)   # both

    GREEN  = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    RED    = curses.color_pair(3)
    CYAN   = curses.color_pair(4)
    DIM    = curses.color_pair(5)
    BLUE   = curses.color_pair(6)
    BOTH   = curses.color_pair(7)
    BOLD   = curses.A_BOLD

    h, w = stdscr.getmaxyx()
    put   = lambda r, c, txt, attr=0: safe_put(stdscr, r, c, txt, attr)
    hline = lambda r: put(r, 0, '─' * (w - 1), DIM)

    stdscr.erase()
    row = 0

    # ── Title bar ────────────────────────────────────────────────────────────
    title = ' TiltStack Monitor'
    hint  = f'interval={args.interval}s   r to refresh   q to quit   {time.strftime("%H:%M:%S")}'
    put(row, 0,       title, CYAN | BOLD)
    put(row, w - len(hint) - 1, hint, DIM)
    row += 1
    hline(row); row += 1

    graph_w = w - 8                                # leave room for y-axis

    # ── System RAM ───────────────────────────────────────────────────────────
    ram = ram_now
    if ram is not None:
        gauge_w = w - 22
        rc = val_color(ram['pct'], (GREEN, YELLOW, RED))
        put(row, 0, ' RAM ', BOLD)
        put(row, 5, f'[{bar(ram["pct"], gauge_w)}]', rc)
        put(row, 7 + gauge_w, f' {ram["label"]}  {ram["pct"]:.1f}%', BOLD)
        row += 1

        if args.pid and len(ram_hist) > 1:
            ram_graph_h = 3
            rows = single_graph_rows(list(ram_hist), graph_w, ram_graph_h)
            for gi, cells in enumerate(rows):
                pct = 100 - int(gi * 100 / max(ram_graph_h - 1, 1))
                put(row, 0, f'{pct:3d}┤', DIM)
                for ci, (ch, kind) in enumerate(cells):
                    safe_put(stdscr, row, 4 + ci, ch, rc if kind else 0)
                row += 1
            put(row, 0, f'    └{"─" * graph_w}', DIM); row += 1

        hline(row); row += 1

    graph_h = max(4, (h - row - 3) // max(len(gpus), 1) - 7)

    for gpu in gpus:
        idx   = gpu['index']
        hu    = histories[idx]['util']
        hm    = histories[idx]['mem']
        mem_p = 100.0 * gpu['mem_used'] / gpu['mem_total'] if gpu['mem_total'] > 0 else 0.0
        gauge_w = w - 22

        # ── GPU header ───────────────────────────────────────────────────────
        put(row, 0, f' GPU {idx}  {gpu["name"]}', CYAN | BOLD); row += 1

        # ── Util gauge ───────────────────────────────────────────────────────
        uc = val_color(gpu['util'], (GREEN, YELLOW, RED))
        put(row, 2, 'Util ', BOLD)
        put(row, 7, f'[{bar(gpu["util"], gauge_w)}]', uc)
        put(row, 9 + gauge_w, f' {gpu["util"]:5.1f}%', BOLD)
        row += 1

        # ── Mem gauge ────────────────────────────────────────────────────────
        mc = val_color(mem_p, (BLUE, YELLOW, RED))
        put(row, 2, 'Mem  ', BOLD)
        put(row, 7, f'[{bar(mem_p, gauge_w)}]', mc)
        put(row, 9 + gauge_w,
            f' {gpu["mem_used"]/1024:.1f}/{gpu["mem_total"]/1024:.1f}GB', BOLD)
        row += 1

        # ── Scalars ──────────────────────────────────────────────────────────
        tc = val_color(gpu['temp'],  (GREEN, YELLOW, RED), (75, 85))
        pc = val_color(100.0 * gpu['power'] / gpu['power_limit']
                       if gpu['power_limit'] > 0 else 0, (GREEN, YELLOW, RED))
        put(row, 2,
            f'Temp ', BOLD)
        put(row, 7,
            f'{gpu["temp"]:.0f}°C', tc | BOLD)
        put(row, 14,
            f'Power  {gpu["power"]:.0f}/{gpu["power_limit"]:.0f}W'
            f'   Clock  {gpu["clock"]:.0f}MHz', DIM)
        row += 1

        # ── Time graph ───────────────────────────────────────────────────────
        if graph_h >= 2 and len(hu) > 1:
            rows = graph_rows(list(hu), list(hm), graph_w, graph_h)

            for gi, cells in enumerate(rows):
                pct = 100 - int(gi * 100 / max(graph_h - 1, 1))
                put(row, 0, f'{pct:3d}┤', DIM)
                for ci, (ch, kind) in enumerate(cells):
                    attr = (GREEN if kind == 'util' else
                            BLUE  if kind == 'mem'  else
                            BOTH  if kind == 'both' else 0)
                    safe_put(stdscr, row, 4 + ci, ch, attr)
                row += 1

            dur = fmt_duration(len(hu) * args.interval)
            put(row, 0, f'    └{"─" * graph_w}', DIM); row += 1
            legend = f'      {SOLID} util   {SHADE} mem   ({dur} of history)'
            put(row, 0, legend, DIM); row += 1

        row += 1
        hline(row); row += 1

        if row >= h - 1:
            break

    stdscr.refresh()


def monitor(stdscr, args):
    stdscr.nodelay(True)
    max_samples = max(10, int(args.history / args.interval))

    histories = {}
    ram_hist  = deque(maxlen=max_samples)
    ram_now   = None
    tick = 0

    while True:
        gpus    = poll()
        sys_ram = poll_ram()
        total_gb = sys_ram['total'] if sys_ram else 0.0
        ram_now = (poll_process_ram(args.pid, total_gb)
                   if args.pid else sys_ram)

        if ram_now is not None:
            ram_hist.append(ram_now['pct'])

        if gpus:
            for gpu in gpus:
                idx = gpu['index']
                if idx not in histories:
                    histories[idx] = {
                        'util': deque(maxlen=max_samples),
                        'mem':  deque(maxlen=max_samples),
                    }
                mem_p = (100.0 * gpu['mem_used'] / gpu['mem_total']
                         if gpu['mem_total'] > 0 else 0.0)
                histories[idx]['util'].append(gpu['util'])
                histories[idx]['mem'] .append(mem_p)

            draw(stdscr, gpus, histories, args, tick, ram_now, ram_hist)
        else:
            h, w = stdscr.getmaxyx()
            safe_put(stdscr, h // 2, w // 2 - 10,
                     'nvidia-smi not found', curses.color_pair(3))
            stdscr.refresh()

        tick += 1

        # Sleep in small increments so q/r is responsive
        elapsed = 0.0
        step    = 0.05
        while elapsed < args.interval:
            ch = stdscr.getch()
            if ch in (ord('q'), ord('Q'), 27):
                return
            if ch in (ord('r'), ord('R')):
                histories.clear()
                ram_hist.clear()
            time.sleep(step)
            elapsed += step


def main():
    p = argparse.ArgumentParser(description='Live GPU monitor with time graph')
    p.add_argument('--interval', type=float, default=0.5,
                   help='Polling interval in seconds  (default: 0.5)')
    p.add_argument('--history',  type=float, default=300.0,
                   help='Rolling history window in seconds  (default: 300)')
    p.add_argument('--pid',      type=int,   default=None,
                   help='PID to track for process RAM usage (shows graph)')
    args = p.parse_args()

    try:
        curses.wrapper(monitor, args)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
