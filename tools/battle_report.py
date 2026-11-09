"""Battle outcome report from a training trace (``steps.jsonl``): win / caught / escape / whiteout per window of updates.

Reads the ``battle_end`` records the worker attaches to the step where a battle ends (src.battle_record).  Traces written before
that field existed have none, and the report says so instead of inventing numbers.  Read-only.

    python -m tools.battle_report --trace <steps.jsonl> [--window 40] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.battle_record import summarize


def read_battles(path):
    """(records with their update, number of step rows).  Entries a later ``resume`` marker made obsolete are dropped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event") == "resume":
                cut = int(e["from_update"])
                rows = [r for r in rows if r[0] < cut]
            elif e.get("event") == "step":
                be = (e.get("outcome") or {}).get("battle_end")
                rows.append((int(e["update"]), be))
    return [(u, b) for u, b in rows if b], len(rows)


def windows(records, window):
    by = {}
    for u, b in records:
        by.setdefault(u // window, []).append(b)
    return [{"first_update": k * window, **summarize(v)} for k, v in sorted(by.items())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    recs, n_rows = read_battles(a.trace)
    report = {"trace": a.trace, "step_rows": n_rows, "battle_records": len(recs),
              "note": None if recs else "no battle_end records: this trace predates battle records (or no battle finished)",
              "overall": summarize([b for _, b in recs]), "windows": windows(recs, a.window)}
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
