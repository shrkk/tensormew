"""ASSISTED-TRACK tool: save start states at chosen tiles along the emulator-searched Route 29 path (offline diagnostic search; the
route is never an agent input).  The BFS copy had Repel written; it is CLEARED in every exported state so no hidden assist is baked in.
Usage: python export_route_states.py OUTDIR idx idx ..."""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main")
from src.emulator import Emulator  # noqa: E402
from src.ram_parser import RamParser  # noqa: E402
from src.states import StateRegistry  # noqa: E402
from src.step import TileStepper  # noqa: E402

out = Path(sys.argv[1]); want = [int(a) for a in sys.argv[2:]]
DIRS = ("up", "down", "left", "right")
emu = Emulator(headless=True, speed=0); reader = RamParser(emu); stepper = TileStepper(emu, reader)
bank, addr = emu.symbol("wRepelEffect")
repel = lambda v=255: emu.write(bank, addr, bytes([v]))
emu.load_state(StateRegistry().load_bytes("route29_entry")); repel()
st = reader.parse(emu.frame); start = (st.map_group, st.map_number, st.x, st.y)
states = {start: emu.save_state()}; parent = {start: None}; goal = None; q = deque([start])
while q:
    cur = q.popleft()
    for d in DIRS:
        emu.load_state(states[cur], apply_options=False); repel()
        stepper.step(d); a = reader.parse(emu.frame)
        if a.battle_mode:
            continue
        nxt = (a.map_group, a.map_number, a.x, a.y)
        if nxt == cur or nxt in parent:
            continue
        parent[nxt] = (cur, d)
        if (nxt[0], nxt[1]) != (start[0], start[1]):
            if nxt[0] == 26 and goal is None:
                goal = nxt
            continue
        states[nxt] = emu.save_state(); q.append(nxt)
    if goal is not None:
        break
tiles, cur = [goal], goal
while parent[cur] is not None:
    cur = parent[cur][0]; tiles.append(cur)
tiles.reverse()
print("route length (tiles):", len(tiles), "start", tiles[0], "goal", tiles[-1])
out.mkdir(parents=True, exist_ok=True)
for i in want:
    t = tiles[i]
    emu.load_state(states[t], apply_options=False); repel(0)
    for _ in range(4):           # let any pending frame effects settle without moving (no input)
        pass
    s = reader.parse(emu.frame)
    assert (s.map_group, s.map_number, s.x, s.y) == t and not s.battle_mode, (i, t)
    (out / f"a29_idx{i:02d}.state").write_bytes(emu.save_state())
    print(f"idx {i:2d} -> tile ({t[2]},{t[3]}) saved a29_idx{i:02d}.state")
emu.close()
