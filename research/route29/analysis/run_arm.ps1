param(
  [Parameter(Mandatory=$true)][string]$Name,
  [Parameter(Mandatory=$true)][string]$Init,   # absolute path of the init weights, or the literal 'fresh'
  [string]$Repo = ''   # code tree to run from (default: the main repo)
)
# Train one pre-declared arm for its configured 384 updates, then run its isolated evaluations (3 snapshots x 2 seed pairs).
# No retries, no extension. Stops (status/exit file) on any nonzero exit.
$W='C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight'
$R="$W/work/tensormew-main"; if ($Repo) { $R = $Repo }; $PY="$W/work/audit-env/Scripts/python.exe"; $O="$W/outputs"; $X="$O/experiment-log/claude-route29-20261109"
Set-Location $R
$env:PYTHONIOENCODING='utf-8'
New-Item -ItemType Directory -Force "$X/$Name" | Out-Null
$trainArgs = @('-m','tools.train_deep_ppo','--config',"$O/route29-phase2/$Name.json",'--live-dir',"logs/live-$Name",'--metrics-path',"logs/training_metrics-$Name.jsonl")
if ($Init -ne 'fresh') { $trainArgs += @('--init-weights',$Init) }
& $PY @trainArgs 2>&1 | Out-File "$X/$Name/train.log" -Encoding utf8
$code = $LASTEXITCODE
"exit $code" | Out-File "$X/$Name/exit.txt"
if ($code -ne 0) { "$Name training failed ($code); no evaluation run"; exit $code }
foreach ($u in '000000','000192','000384') { foreach ($seed in 10101,20001) {
  & $PY -m tools.evaluate_policy --sampled --checkpoint "$X/$Name/snapshots/update-$u.pt" --config "$O/route29-phase2/$Name.json" --episodes 2 --max-steps 4096 --seed $seed --device cuda --out-dir "$X/$Name/eval" --name "u$u-seed$seed" 2>&1 | Out-File "$X/$Name/eval-u$u-seed$seed.log" -Encoding utf8
} }
"$Name done"
