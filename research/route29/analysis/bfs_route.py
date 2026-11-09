"""DIAGNOSTIC ONLY (never used in training or as an agent input): breadth-first search over REAL emulator tile moves from the Route 29
entry state to the first tile outside Route 29, to see what the game actually requires.  Repel (wRepelEffect=255) is written in this
diagnostic copy only so random encounters do not interrupt the search.  Reports the shortest move sequence and the single-tile
bottlenecks (tiles whose removal disconnects start from goal)."""
import sys
from collections import deque

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.emulator import Emulator  # noqa: E402
from src.ram_parser import RamParser  # noqa: E402
from src.states import StateRegistry  # noqa: E402
from src.step import TileStepper  # noqa: E402

DIRS = ("up", "down", "left", "right")
emu = Emulator(headless=True, speed=0)
reader = RamParser(emu)
stepper = TileStepper(emu, reader)
bank, addr = emu.symbol("wRepelEffect")


def repel():
    emu.write(bank, addr, bytes([255]))


emu.load_state(StateRegistry().load_bytes("route29_entry"))
repel()
st = reader.parse(emu.frame)
start = (st.map_group, st.map_number, st.x, st.y)
print("start tile:", start, "| battle:", st.battle_mode)
states = {start: emu.save_state()}
edges = {}                 # tile -> {dir: next tile}
parent = {start: None}
goal = None
q = deque([start])
expanded = 0
while q and expanded < 4000:
    cur = q.popleft()
    expanded += 1
    for d in DIRS:
        emu.load_state(states[cur], apply_options=False)
        repel()
        res = stepper.step(d)
        after = reader.parse(emu.frame)
        if after.battle_mode:
            continue
        nxt = (after.map_group, after.map_number, after.x, after.y)
        if nxt == cur:
            continue
        edges.setdefault(cur, {})[d] = nxt
        if nxt not in parent:
            parent[nxt] = (cur, d)
            if (nxt[0], nxt[1]) != (start[0], start[1]):
                if (nxt[0], nxt[1]) == (24, 4):
                    continue    # back into New Bark: the wrong way, ignore
                if goal is None and nxt[0] == 26:
                    goal = nxt
                    print("FIRST TILE IN CHERRYGROVE-SIDE MAPS:", nxt, "after expanding", expanded, "tiles")
                continue    # do not expand beyond Route 29
            states[nxt] = emu.save_state()
            q.append(nxt)
print(f"explored {len(states)} Route 29 tiles reachable from the entry; goal found: {goal}")
if goal is None:
    sys.exit(0)
path, cur = [], goal
while parent[cur] is not None:
    cur, d = parent[cur]
    path.append((cur, d))
path.reverse()
moves = [d for _, d in path]
print(f"shortest path: {len(moves)} moves")
compact, last, n = [], None, 0
for d in moves:
    if d == last:
        n += 1
    else:
        if last:
            compact.append(f"{last}x{n}")
        last, n = d, 1
compact.append(f"{last}x{n}")
print("moves:", " ".join(compact))
tiles = [p for p, _ in path] + [goal]


def reachable_without(block):
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for v in edges.get(u, {}).values():
            if v in seen or v == block:
                continue
            seen.add(v)
            dq.append(v)
    return any((t[0], t[1]) != (start[0], start[1]) for t in seen) or goal in seen


bottlenecks = []
for t in tiles[1:-1]:
    seen, dq = {start}, deque([start])
    ok = False
    while dq:
        u = dq.popleft()
        for v in edges.get(u, {}).values():
            if v == t or v in seen:
                continue
            if v[0] == 26:
                ok = True
                break
            seen.add(v)
            dq.append(v)
        if ok:
            break
    if not ok:
        bottlenecks.append(t)
print(f"single-tile bottlenecks on the route ({len(bottlenecks)}):", [(b[2], b[3]) for b in bottlenecks])
print("route tiles (x,y) in order, first 90:", [(t[2], t[3]) for t in tiles][:90])
emu.close()
