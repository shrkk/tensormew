"""OFFLINE DIAGNOSTIC ONLY (never fed to the agent): where does the ROM-derived route leave Route 29's corridor?"""
import sys
sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from types import SimpleNamespace
from src import route_graph as rg

g = rg._graph()
print("graph loaded:", g is not None)
# potential (negative remaining steps to the goal) at tiles of Route 29: find where it increases towards the goal
rows = []
for y in range(0, 24):
    row = []
    for x in range(0, 60, 1):
        st = SimpleNamespace(map_group=24, map_number=3, x=x, y=y)
        try:
            p = rg.goal_remaining(st)
        except Exception:
            p = None
        row.append(p)
    rows.append(row)
# print remaining distance for a few columns around the pocket x=10..30 and all y with any value
print("remaining steps to the story goal (None = not on graph / unreachable):")
print("   x:", " ".join(f"{x:4d}" for x in range(8, 34, 2)))
for y in range(8, 22):
    print(f"y={y:2d}", " ".join(("  . " if rows[y][x] is None else f"{rows[y][x]:4.0f}") for x in range(8, 34, 2)))
