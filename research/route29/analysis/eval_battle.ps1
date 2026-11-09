param([Parameter(Mandatory=$true)][string[]]$Names)
# Re-evaluate the FINAL (update 384) snapshot of each named arm with the current evaluator (adds battle_summary).
# Same dev seeds as the original evaluations; writes to <arm>/eval-battle/.  Evaluation only: no training.
$W='C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight'
$R="$W/work/tensormew-main"; $PY="$W/work/audit-env/Scripts/python.exe"; $O="$W/outputs"; $X="$O/experiment-log/claude-route29-20261109"
Set-Location $R
$env:PYTHONIOENCODING='utf-8'
foreach ($n in $Names) {
  foreach ($seed in 10101,20001) {
    & $PY -m tools.evaluate_policy --sampled --checkpoint "$X/$n/snapshots/update-000384.pt" --config "$O/route29-phase2/$n.json" --episodes 2 --max-steps 4096 --seed $seed --device cuda --out-dir "$X/$n/eval-battle" --name "u000384-seed$seed" 2>&1 | Out-File "$X/$n/eval-battle-seed$seed.log" -Encoding utf8
  }
  "$n done"
}
