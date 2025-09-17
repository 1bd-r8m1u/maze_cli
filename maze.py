#!/usr/bin/env python3
"""
maze.py — generate, style, solve, and play mazes in the terminal.

Features:
 - recursive backtracker maze generation
 - BFS and A* solvers
 - animated carving and animated path drawing
 - rich-powered colored output (if available)
 - interactive play mode (curses preferred, termios fallback)
 - argparse flags: --width, --height, --animate, --delay, --solve, --astar, --animate-solve, --play, --walk, --interactive, --seed

Usage examples:
  python3 maze.py --width 51 --height 21 --solve
  python3 maze.py --width 81 --height 31 --animate --delay 0.005
  python3 maze.py --width 41 --height 101 --astar --animate-solve --delay 0.003
  python3 maze.py --width 31 --height 81 --play
"""
from __future__ import annotations
import sys, random, time, collections, shutil, argparse, heapq, os
from typing import List, Tuple, Deque, Optional, Dict

# Try to import rich for nicer colors — optional
try:
    from rich.console import Console
    from rich.text import Text
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

# Try to import curses for nicer interactive play — optional; we'll fallback if unavailable
try:
    import curses
    HAVE_CURSES = True
except Exception:
    HAVE_CURSES = False

# For termios fallback walker
try:
    import termios, tty, select
    HAVE_TERM = True
except Exception:
    HAVE_TERM = False

# ------------------------
# Maze data & helpers
# ------------------------
def make_empty_grid(rows: int, cols: int) -> List[List[str]]:
    return [['#' for _ in range(cols)] for _ in range(rows)]

def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

# cell step directions (two-step for carving)
DIRS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

def carve_maze(rows: int, cols: int, animate: bool=False, delay: float=0.02):
    grid = make_empty_grid(rows, cols)
    start_r, start_c = 1, 1
    grid[start_r][start_c] = ' '
    stack = [(start_r, start_c)]
    visited = set(stack)

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and (nr, nc) not in visited:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            wall_r, wall_c = (r + nr) // 2, (c + nc) // 2
            grid[wall_r][wall_c] = ' '
            grid[nr][nc] = ' '
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

        if animate:
            render_grid(grid)
            time.sleep(delay)
    return grid

# ------------------------
# Solvers: BFS and A*
# ------------------------
def solve_maze_bfs(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    rows, cols = len(grid), len(grid[0])
    q: Deque[Tuple[int,int]] = collections.deque([start])
    parent = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if in_bounds(nr, nc, rows, cols) and grid[nr][nc] == ' ' and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def solve_maze_astar(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    rows, cols = len(grid), len(grid[0])
    open_heap = []
    heapq.heappush(open_heap, (0 + manhattan(start, goal), 0, start))
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    gscore = {start: 0}
    closed = set()
    while open_heap:
        f, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            break
        closed.add(node)
        r, c = node
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if not in_bounds(nr, nc, rows, cols) or grid[nr][nc] != ' ':
                continue
            tentative_g = g + 1
            neighbor = (nr, nc)
            if neighbor in gscore and tentative_g >= gscore[neighbor]:
                continue
            parent[neighbor] = node
            gscore[neighbor] = tentative_g
            heapq.heappush(open_heap, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor))
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

# ------------------------
# Rendering (rich if available)
# ------------------------
def render_grid(grid: List[List[str]], path: List[Tuple[int,int]] = None, player: Tuple[int,int]=None, styled: bool=True):
    rows, cols = len(grid), len(grid[0])
    show = [row[:] for row in grid]
    if path:
        for (r,c) in path:
            if show[r][c] == ' ':
                show[r][c] = '.'
    if player:
        pr, pc = player
        if 0 <= pr < rows and 0 <= pc < cols:
            show[pr][pc] = '@'

    # positions for start and goal
    start = (1,1)
    goal = (rows-2, cols-2)

    if RICH and styled:
        t = Text()
        for r in range(rows):
            for c in range(cols):
                ch = show[r][c]
                pos = (r,c)
                if pos == start:
                    t.append('S', style="bold green on #071725")
                elif pos == goal:
                    t.append('G', style="bold red on #071725")
                elif ch == '#':
                    t.append('█', style="on #0f2540")
                elif ch == ' ':
                    t.append(' ', style="on #071725")
                elif ch == '.':
                    t.append('·', style="bold yellow on #071725")
                elif ch == '@':
                    t.append('@', style="bold magenta on #071725")
                else:
                    t.append(ch)
            t.append('\n')
        console.print(t, end='')
    else:
        # plain ASCII/emoji fallback
        out_lines = []
        for r in range(rows):
            line = []
            for c in range(cols):
                ch = show[r][c]
                pos = (r,c)
                if pos == start:
                    line.append('S')
                elif pos == goal:
                    line.append('G')
                elif ch == '#':
                    line.append('█')
                elif ch == ' ':
                    line.append(' ')
                elif ch == '.':
                    line.append('.')
                elif ch == '@':
                    line.append('@')
                else:
                    line.append(ch)
            out_lines.append(''.join(line))
        print('\n'.join(out_lines))

def clear_screen():
    if RICH:
        console.clear()
    else:
        # ANSI clear
        print("\033[H\033[J", end='')

# ------------------------
# Fallback termios walker (keeps previous behavior)
# ------------------------
def _getch_termios(timeout: Optional[float]=None) -> str:
    """Read single keypress. If arrow key, returns escape sequence like '\x1b[A'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if not rlist:
            return ''
        ch1 = os.read(fd, 1)
        if ch1 == b'\x1b':
            # possible escape sequence (arrow)
            rlist, _, _ = select.select([fd], [], [], 0.01)
            if rlist:
                ch2 = os.read(fd, 2)
                return (ch1 + ch2).decode('utf-8', errors='ignore')
            else:
                return ch1.decode()
        else:
            return ch1.decode('utf-8', errors='ignore')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def interactive_walker_termios(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]):
    rows, cols = len(grid), len(grid[0])
    player = list(start)
    clear_screen()
    render_grid(grid, player=tuple(player))
    print("\nUse arrow keys to move, 'r' to reset, 'q' to quit. Reach the exit to win.")
    try:
        while True:
            key = _getch_termios(timeout=None)
            if not key:
                continue
            if key in ('q', 'Q'):
                break
            if key in ('r','R'):
                player = list(start)
            if key == '\x1b[A':
                dr, dc = -1, 0
            elif key == '\x1b[B':
                dr, dc = 1, 0
            elif key == '\x1b[C':
                dr, dc = 0, 1
            elif key == '\x1b[D':
                dr, dc = 0, -1
            else:
                dr, dc = 0, 0
            if dr or dc:
                nr, nc = player[0] + dr, player[1] + dc
                if in_bounds(nr, nc, rows, cols) and grid[nr][nc] == ' ':
                    player[0], player[1] = nr, nc
            clear_screen()
            render_grid(grid, player=tuple(player))
            if tuple(player) == goal:
                print("\nYou reached the exit! 🎉")
                time.sleep(1.2)
                break
    except KeyboardInterrupt:
        pass
    finally:
        clear_screen()
        render_grid(grid)
        print("\nExited walker.")

# ------------------------
# Curses-based interactive player (clean, recommended if available)
# ------------------------
def _curses_play_wrapper(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]):
    def _draw(stdscr, grid, player, goal):
        stdscr.erase()
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if (r,c) == player:
                    stdscr.addstr(r, c, '@')
                elif (r,c) == start:
                    stdscr.addstr(r, c, 'S')
                elif (r,c) == goal:
                    stdscr.addstr(r, c, 'G')
                elif ch == '#':
                    stdscr.addstr(r, c, '█')
                else:
                    stdscr.addstr(r, c, ' ')
        stdscr.refresh()

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        player = list(start)
        _draw(stdscr, grid, tuple(player), goal)
        stdscr.addstr(len(grid), 0, "Arrow keys to move, q to quit, r to reset")
        while True:
            key = stdscr.getch()
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                player = list(start)
            dr, dc = 0, 0
            if key == curses.KEY_UP: dr, dc = -1, 0
            elif key == curses.KEY_DOWN: dr, dc = 1, 0
            elif key == curses.KEY_LEFT: dr, dc = 0, -1
            elif key == curses.KEY_RIGHT: dr, dc = 0, 1
            if dr or dc:
                nr, nc = player[0] + dr, player[1] + dc
                if in_bounds(nr, nc, len(grid), len(grid[0])) and grid[nr][nc] == ' ':
                    player[0], player[1] = nr, nc
            _draw(stdscr, grid, tuple(player), goal)
            if tuple(player) == goal:
                stdscr.addstr(len(grid)+1, 0, "You reached the exit! Press any key.")
                stdscr.refresh()
                stdscr.getch()
                break

    curses.wrapper(_inner)

# ------------------------
# Utilities & CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Maze generator + solver (terminal)")
    p.add_argument('--height', type=int, help='maze height in characters (odd recommended)')
    p.add_argument('--width', type=int, help='maze width in characters (odd recommended)')
    p.add_argument('--animate', action='store_true', help='animate carving')
    p.add_argument('--delay', type=float, default=0.01, help='animation delay (seconds)')
    p.add_argument('--solve', action='store_true', help='solve and show shortest path using BFS')
    p.add_argument('--astar', action='store_true', help='solve using A* (Manhattan heuristic)')
    p.add_argument('--animate-solve', action='store_true', help='animate solving/path draw')
    p.add_argument('--play', '--walk', '--interactive', action='store_true', dest='play', help='interactive play mode (arrow keys)')
    p.add_argument('--no-rich', action='store_true', help='disable rich colors even if installed')
    p.add_argument('--seed', type=int, default=None, help='random seed for reproducible maze')
    p.add_argument('--styled', action='store_true', help='force styled output (uses rich if available)')
    p.add_argument('pos_rows', nargs='?', type=int, help=argparse.SUPPRESS)
    p.add_argument('pos_cols', nargs='?', type=int, help=argparse.SUPPRESS)
    return p.parse_args()

def center_in_terminal(rows, cols):
    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    return cols <= term_cols and rows <= term_rows

def animate_path(grid: List[List[str]], path: List[Tuple[int,int]], delay: float):
    # Draw the path incrementally for nicer effect
    drawn = []
    for cell in path:
        drawn.append(cell)
        render_grid(grid, path=drawn)
        time.sleep(delay)

# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    if args.no_rich:
        global RICH, console
        RICH = False
        console = None

    if args.seed is not None:
        random.seed(args.seed)

    # choose rows/cols: flags override positional
    rows = args.pos_rows if args.pos_rows else None
    cols = args.pos_cols if args.pos_cols else None
    if args.height: rows = args.height
    if args.width: cols = args.width
    if rows is None: rows = 21
    if cols is None: cols = 51

    # ensure odd sizes
    if rows % 2 == 0: rows += 1
    if cols % 2 == 0: cols += 1

    if not center_in_terminal(rows, cols):
        term_w, term_h = shutil.get_terminal_size((80,24))
        rows = min(rows, term_h-2 if term_h>5 else rows)
        cols = min(cols, term_w-2 if term_w>10 else cols)

    clear_screen()
    grid = carve_maze(rows, cols, animate=args.animate, delay=args.delay if args.animate else 0)
    clear_screen()
    start = (1,1)
    goal = (rows-2, cols-2)

    # interactive play (curses preferred, fallback termios)
    if args.play:
        if HAVE_CURSES:
            try:
                _curses_play_wrapper(grid, start, goal)
            except Exception:
                # fallback to termios walker if something odd happens
                if HAVE_TERM:
                    interactive_walker_termios(grid, start, goal)
                else:
                    print("Interactive play not available on this platform.")
        else:
            if HAVE_TERM:
                interactive_walker_termios(grid, start, goal)
            else:
                print("Interactive play not available on this platform.")
        return

    # normal rendering + solving
    render_grid(grid, styled=args.styled or True)
    if args.solve or args.astar:
        if args.astar:
            path = solve_maze_astar(grid, start, goal)
        else:
            path = solve_maze_bfs(grid, start, goal)
        time.sleep(0.15)
        if path:
            if args.animate_solve:
                animate_path(grid, path, args.delay if args.delay else 0.01)
            else:
                render_grid(grid, path=path, styled=args.styled or True)
            print("\nSolved! Path length:", len(path))
        else:
            print("\nNo path found.")
    else:
        print("\nMaze generated. Use --solve or --astar to display solution, or --play to play.")

if __name__ == "__main__":
    main()
