"""Append reproducible experiment facts and concise decision summaries (never hidden reasoning)."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / 'work/tensormew-main'
OUT = ROOT / 'outputs/experiment-log'

def record(kind, summary, details):
    OUT.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO).decode('utf-8').strip()
    diff = git('diff', '--binary')
    event = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'event': kind, 'summary': summary, 'details': details,
        'code_commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
        'working_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
    }
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    with (OUT / 'decisions.md').open('a', encoding='utf-8') as f:
        f.write(f"\n- {event['timestamp_utc']} — **{kind}**: {summary}\n")
    print(json.dumps({'event': kind, 'summary': summary}))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind')
    p.add_argument('summary')
    p.add_argument('--details', default='{}')
    a = p.parse_args()
    record(a.kind, a.summary, json.loads(a.details))
