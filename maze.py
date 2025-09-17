#!/usr/bin/env python3
"""
maze.py — generate and (optionally) solve a maze in the terminal.

New features:
 - --astar : use A* to compute a shortest path (Manhattan heuristic)
 - --walk  : interactive walker: use arrow keys to move '@', q to quit, r to reset

Usage:
  python maze.py             # default 21x51 maze, no animation, no solve
  python maze.py 31 81 --animate
  python maze.py 41 101 --solve
  python maze.py 41 101 --astar
  python maze.py 41 101 --walk
"""
from __future__ import annotations
import sys, random, time, collections, shutil, argparse, heapq, termios, tty, os, select
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

# ------------------------
# Maze data & helpers
# ------------------------
def make_empty_grid(rows: int, cols: int) -> List[List[str]]:
    return [['#' for _ in range(cols)] for _ in range(rows)]

def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

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
# Rendering
# ------------------------
def render_grid(grid: List[List[str]], path: List[Tuple[int,int]] = None, player: Tuple[int,int]=None):
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
    if RICH:
        t = Text()
        for r in range(rows):
            for c in range(cols):
                ch = show[r][c]
                if ch == '#':
                    t.append('█')
                elif ch == ' ':
                    t.append(' ')
                elif ch == '.':
                    t.append('.', style="bold magenta")
                elif ch == '@':
                    t.append('@', style="bold yellow")
                else:
                    t.append(ch)
            t.append('\n')
        console.print(t, end='')
    else:
        out = '\n'.join(''.join(row) for row in show)
        print(out)

def clear_screen():
    if RICH:
        console.clear()
    else:
        print("\033[H\033[J", end='')

# ------------------------
# Interactive walker helpers
# ------------------------
def _getch(timeout: Optional[float]=None) -> str:
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
            # read next two bytes with small timeout
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

def interactive_walker(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]):
    rows, cols = len(grid), len(grid[0])
    player = list(start)
    clear_screen()
    render_grid(grid, player=tuple(player))
    print("\nUse arrow keys to move, 'r' to reset, 'q' to quit. Reach the exit to win.")
    try:
        while True:
            key = _getch(timeout=None)
            if not key:
                continue
            if key in ('q', 'Q'):
                break
            if key in ('r','R'):
                player = list(start)
            # arrow sequences
            if key == '\x1b[A':  # up
                dr, dc = -1, 0
            elif key == '\x1b[B':  # down
                dr, dc = 1, 0
            elif key == '\x1b[C':  # right
                dr, dc = 0, 1
            elif key == '\x1b[D':  # left
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
                _getch(timeout=2)
                break
    except KeyboardInterrupt:
        pass
    finally:
        clear_screen()
        render_grid(grid)
        print("\nExited walker.")

# ------------------------
# CLI & main
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Maze generator + solver (terminal)")
    p.add_argument('rows', nargs='?', type=int, default=21, help='maze height in characters (odd recommended)')
    p.add_argument('cols', nargs='?', type=int, default=51, help='maze width in characters (odd recommended)')
    p.add_argument('--animate', action='store_true', help='animate carving')
    p.add_argument('--delay', type=float, default=0.01, help='animation delay (seconds)')
    p.add_argument('--solve', action='store_true', help='solve and show shortest path after generation (BFS)')
    p.add_argument('--astar', action='store_true', help='solve using A* (Manhattan heuristic)')
    p.add_argument('--walk', action='store_true', help='interactive walker to navigate the maze')
    p.add_argument('--seed', type=int, default=None, help='random seed for reproducible maze')
    return p.parse_args()

def center_in_terminal(rows, cols):
    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    return cols <= term_cols and rows <= term_rows

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    rows = args.rows
    cols = args.cols
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

    if args.walk:
        interactive_walker(grid, start, goal)
        return

    render_grid(grid)
    if args.solve or args.astar:
        if args.astar:
            path = solve_maze_astar(grid, start, goal)
        else:
            path = solve_maze_bfs(grid, start, goal)
        time.sleep(0.2)
        if path:
            render_grid(grid, path=path)
            print("\nSolved! Path length:", len(path))
        else:
            print("\nNo path found.")
    else:
        print("\nMaze generated. Use --solve or --astar to display solution, or --walk to play.")

if __name__ == "__main__":
    main()
