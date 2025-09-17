#!/usr/bin/env python3
"""
maze.py — generate and (optionally) solve a maze in the terminal.

Usage:
  python maze.py             # default 21x51 maze, no animation, no solve
  python maze.py 31 81       # custom rows cols (odd numbers recommended)
  python maze.py 31 81 --animate  # show carving animation
  python maze.py 31 81 --solve    # show solution after generation
  python maze.py --help

Notes:
- Uses only Python stdlib. If 'rich' is installed it will color the output.
- Works in Termux / Pydroid's terminal.
"""
from __future__ import annotations
import sys, random, time, collections, shutil, argparse
from typing import List, Tuple, Deque

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
# We represent the maze as a grid of chars where:
#  '#' = wall
#  ' ' = passage
#  '.' = solution path
# The internal cell grid dimensions are odd numbers: cells at odd indices.
# rows x cols refer to char grid size.
def make_empty_grid(rows: int, cols: int) -> List[List[str]]:
    return [['#' for _ in range(cols)] for _ in range(rows)]

def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

# Neighbors in four cardinal directions (dr,dc)
DIRS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

def carve_maze(rows: int, cols: int, animate: bool=False, delay: float=0.02):
    """
    Use recursive backtracker iterative (stack) to carve passages.
    grid uses odd indices for cells so rows/cols should be odd for symmetrical mazes.
    """
    grid = make_empty_grid(rows, cols)
    start_r, start_c = 1, 1
    grid[start_r][start_c] = ' '
    stack = [(start_r, start_c)]
    visited = set(stack)

    while stack:
        r, c = stack[-1]
        # find unvisited neighbors two steps away
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and (nr, nc) not in visited:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            # knock down wall between (r,c) and (nr,nc)
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
# Solve with BFS (shortest path)
# ------------------------
def solve_maze(grid: List[List[str]], start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    rows, cols = len(grid), len(grid[0])
    q: Deque[Tuple[int,int]] = collections.deque()
    q.append(start)
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
    # reconstruct
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
def render_grid(grid: List[List[str]], path: List[Tuple[int,int]] = None):
    rows, cols = len(grid), len(grid[0])
    # overlay path
    show = [row[:] for row in grid]
    if path:
        for (r,c) in path:
            if show[r][c] == ' ':
                show[r][c] = '.'
    if RICH:
        t = Text()
        for r in range(rows):
            for c in range(cols):
                ch = show[r][c]
                if ch == '#':
                    t.append('█')  # thick wall
                elif ch == ' ':
                    t.append(' ')
                else:  # '.'
                    t.append('.', style="bold magenta")
            t.append('\n')
        console.print(t, end='')
    else:
        # fallback plain print
        out = '\n'.join(''.join(row) for row in show)
        print(out)
    # move cursor up to allow in-place animation
    if sys.stdout.isatty():
        if not RICH:
            # for plain, clear screen after a short pause
            pass

def clear_screen():
    if RICH:
        console.clear()
    else:
        print("\033[H\033[J", end='')

# ------------------------
# Utilities & CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Maze generator + solver (terminal)")
    p.add_argument('rows', nargs='?', type=int, default=21, help='maze height in characters (odd recommended)')
    p.add_argument('cols', nargs='?', type=int, default=51, help='maze width in characters (odd recommended)')
    p.add_argument('--animate', action='store_true', help='animate carving')
    p.add_argument('--delay', type=float, default=0.01, help='animation delay (seconds)')
    p.add_argument('--solve', action='store_true', help='solve and show shortest path after generation')
    p.add_argument('--seed', type=int, default=None, help='random seed for reproducible maze')
    return p.parse_args()

def center_in_terminal(rows, cols):
    # optional: try to center; returns whether it fits
    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    return cols <= term_cols and rows <= term_rows

# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    rows = args.rows
    cols = args.cols
    # ensure odd sizes so walls/cells align nicely
    if rows % 2 == 0: rows += 1
    if cols % 2 == 0: cols += 1

    if not center_in_terminal(rows, cols):
        # Try to shrink to terminal if too big
        term_w, term_h = shutil.get_terminal_size((80,24))
        rows = min(rows, term_h-2 if term_h>5 else rows)
        cols = min(cols, term_w-2 if term_w>10 else cols)

    clear_screen()
    grid = carve_maze(rows, cols, animate=args.animate, delay=args.delay if args.animate else 0)
    clear_screen()
    # Entrance and exit (top-left inside cell and bottom-right inside cell)
    start = (1,1)
    goal = (rows-2, cols-2)
    render_grid(grid)
    if args.solve:
        path = solve_maze(grid, start, goal)
        time.sleep(0.2)
        if path:
            # mark path and show it
            if RICH:
                # show colored path overlayed
                render_grid(grid, path)
            else:
                # overlay '.' characters and print
                for r,c in path:
                    grid[r][c] = '.'
                render_grid(grid)
            print("\nSolved! Path length:", len(path))
        else:
            print("\nNo path found.")
    else:
        print("\nMaze generated. Use --solve to display solution.")
    # done
if __name__ == "__main__":
    main()
