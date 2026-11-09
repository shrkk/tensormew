"""Entropy / KL / clip / value-loss over training for several arms (per-arm metrics jsonl).  Read-only."""
import json
import sys
from pathlib import Path

L = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight/work/tensormew-main/logs")
for arm in sys.argv[1:]:
    p = L / f"training_metrics-{arm}.jsonl"
    if not p.exists():
        print(arm, "no metrics file")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"\n{arm}: {len(rows)} updates; keys sample: {sorted(rows[0].keys())[:14]}")
    for u in (0, 25, 50, 100, 200, 300, len(rows) - 1):
        if u >= len(rows):
            continue
        r = rows[u]
        print(f"  u{r.get('update', u):3d} entropy={r.get('entropy', float('nan')):.3f} kl={r.get('approx_kl', float('nan')):.4f} "
              f"clip={r.get('clip_frac', float('nan')):.3f} vloss={r.get('value_loss', float('nan')):.3f} "
              f"ev={r.get('explained_variance')} grad={r.get('grad_norm')} lr={r.get('lr')}")
