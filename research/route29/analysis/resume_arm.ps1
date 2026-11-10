param(
  [Parameter(Mandatory=$true)][string]$Name,
  [Parameter(Mandatory=$true)][string]$Repo     # a code tree byte-identical (source fingerprint) to the one the arm started on
)
# EXACT resume of a paused arm from its newest verified generation, then its standard isolated evaluations.
# Removes the arm's STOP file (the user lifted the pause).  Refuses (the trainer refuses) if the source fingerprint differs.
$null = (Get-Process -Id $PID).PriorityClass = 'BelowNormal'   # keep the user's PC responsive: experiments run below normal priority
$W='C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight'
$PY="$W/work/audit-env/Scripts/python.exe"; $O="$W/outputs"; $X="$O/experiment-log/claude-route29-20261109"
Set-Location $Repo
$env:PYTHONIOENCODING='utf-8'
$st = Get-Content "$X/$Name/status.json" -Raw | ConvertFrom-Json
# count from the newest SAVED GENERATION (a crash can leave status.last_completed_update ahead of it); fall back to the status value
$gu = $st.last_generation.update; if ($null -eq $gu) { $gu = $st.last_completed_update }
$remaining = 384 - ($gu + 1)
Remove-Item "$X/$Name/STOP" -ErrorAction SilentlyContinue
"resuming $Name from update $($st.last_completed_update); $remaining updates remain"
& $PY -m tools.train_deep_ppo --config "$O/route29-phase2/$Name.json" --resume-generation auto --updates $remaining --live-dir "logs/live-$Name" --metrics-path "logs/training_metrics-$Name.jsonl" 2>&1 | Out-File "$X/$Name/train-resume.log" -Encoding utf8
$code = $LASTEXITCODE
"exit $code" | Out-File "$X/$Name/exit-resume.txt"
if ($code -ne 0) { "$Name resume failed ($code); no evaluation run"; exit $code }
foreach ($u in '000000','000192','000384') { foreach ($seed in 10101,20001) {
  & $PY -m tools.evaluate_policy --sampled --checkpoint "$X/$Name/snapshots/update-$u.pt" --config "$O/route29-phase2/$Name.json" --episodes 2 --max-steps 4096 --seed $seed --device cuda --out-dir "$X/$Name/eval" --name "u$u-seed$seed" 2>&1 | Out-File "$X/$Name/eval-u$u-seed$seed.log" -Encoding utf8
} }
"$Name resumed and done"
