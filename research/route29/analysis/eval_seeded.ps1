param([Parameter(Mandatory=$true)][string[]]$Names, [string[]]$States = @('a29_idx30','a29_idx50'))
# EVALUATION ONLY: final (update 384) checkpoint of each arm, started from route-seeded states (ASSISTED yardstick), dev seed 10101.
$null = (Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$W='C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight'
$R="$W/work/tensormew-main"; $PY="$W/work/audit-env/Scripts/python.exe"; $O="$W/outputs"; $X="$O/experiment-log/claude-route29-20261109"
Set-Location $R; $env:PYTHONIOENCODING='utf-8'
foreach ($n in $Names) { foreach ($st in $States) {
  & $PY -m tools.evaluate_policy --sampled --checkpoint "$X/$n/snapshots/update-000384.pt" --config "$O/route29-phase2/$n.json" --start-state $st --episodes 2 --max-steps 3000 --seed 10101 --device cuda --out-dir "$X/$n/eval-seeded" --name "u384-$st" 2>&1 | Out-File "$X/$n/eval-seeded-$st.log" -Encoding utf8
} ; "$n done" }
