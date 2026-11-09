"""ASCII map of Route 29 around the wall at (36,6): emulator-reachable tiles (diagnostic BFS with Repel), the verified route, and where
agents actually spent time (all decisions of a few traces).  Read-only diagnostic."""
import json
import sys
from collections import Counter, deque
from pathlib import Path

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
emu.load_state(StateRegistry().load_bytes("route29_entry"))
emu.write(bank, addr, bytes([255]))
st = reader.parse(emu.frame)
start = (st.map_group, st.map_number, st.x, st.y)
states = {start: emu.save_state()}
tileinfo = {}
q = deque([start])
while q:
    cur = q.popleft()
    emu.load_state(states[cur], apply_options=False)
    emu.write(bank, addr, bytes([255]))
    s0 = reader.parse(emu.frame)
    tileinfo[cur] = dict(s0.tiles)          # collision ids of the 4 neighbours as the game reports them
    for d in DIRS:
        emu.load_state(states[cur], apply_options=False)
        emu.write(bank, addr, bytes([255]))
        stepper.step(d)
        a = reader.parse(emu.frame)
        if a.battle_mode:
            continue
        nxt = (a.map_group, a.map_number, a.x, a.y)
        if nxt in states or (nxt[0], nxt[1]) != (start[0], start[1]):
            continue
        states[nxt] = emu.save_state()
        q.append(nxt)
emu.close()
reach = {(t[2], t[3]) for t in states}
ROUTE = [(31, 10), (32, 10), (33, 10), (34, 10), (35, 10), (36, 10), (36, 9), (36, 8), (36, 7), (36, 6), (35, 6), (34, 6), (33, 6), (32, 6),
         (31, 6), (30, 6), (30, 5), (29, 5), (28, 5), (27, 5), (26, 5), (25, 5), (24, 5), (23, 5), (22, 5)]
visits = Counter()
X = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/outputs/experiment-log/claude-route29-20261109")
for arm in sys.argv[1:]:
    with open(X / arm / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"event": "step"' not in line:
                continue
            o = json.loads(line)["outcome"]
            if tuple(o["map"]) == (24, 3):
                visits[(o["x"], o["y"])] += 1
print(f"{len(reach)} Route 29 tiles reachable. Legend: '#' not reachable | '.' reachable, never visited | digit = log10(visits)+1 | 'R' = verified route tile (shown in place of a digit if visited it is lowercase 'r' + count class)")
print("      " + "".join(f"{x % 100:3d}" for x in range(20, 41)))
for y in range(2, 13):
    row = []
    for x in range(20, 41):
        c = (x, y)
        if c not in reach:
            ch = " #"
        else:
            v = visits.get(c, 0)
            ch = "." if v == 0 else str(min(9, len(str(v))))
            ch = (" R" if v == 0 else " r") if c in ROUTE and v == 0 else (" " + ("r" + ch if False else ch))
            if c in ROUTE:
                ch = " R" if v == 0 else " " + ch
                ch = ch.replace(" ", "*", 1) if v else ch
        row.append(ch.rjust(3))
    print(f"y={y:2d} " + "".join(row))
print("(R = route tile never visited; '*d' = route tile visited, d = number of digits of its visit count; plain digit = non-route tile visited)")
