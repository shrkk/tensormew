"""Build the 'transfer' arm's starting weights: vision trunk from the user's deep_policy.pt, everything else freshly initialized.

Only ``vision_trunk.*`` (ResNet-18; input-schema independent) is copied.  The RAM trunk, fusion, GRU and actor/critic heads
depend on the old 21-input/5-action layout and are NOT transferred.  The new policy is created exactly as the trainer
creates a fresh one (same seed, same constructor), so its non-trunk tensors equal the fresh arm's initial tensors; this is
verified against the fresh arm's update-000000 snapshot when it exists.

Run with:  python -I work/make_transfer_init.py
The source file is loaded with weights_only=True (tensors/plain types only); it is never modified.
"""
import hashlib
import json
import sys
from pathlib import Path

WS = Path("C:/Users/shrey/Documents/Codex/2026-11-09/https-rikiyat-github-io-blog-weight")
REPO = WS / "work" / "tensormew-main"
SRC = Path("C:/Users/shrey/Downloads/deep_policy.pt")
SRC_SHA = "a0ab5382c51c1a0bf9849fb2b05cc823218aa78ea2a13673a4106ff5f9d9bdcb"
SEED = 20261110
OUT_DIR = WS / "outputs" / "experiment-log" / "claude-route29-20261109" / "transfer"
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.deep_policy import SpatialVisionPolicy  # noqa: E402
from src.rl_env import ACTIONS, deep_obs_dim  # noqa: E402
from src import story_path  # noqa: E402

sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
assert sha == SRC_SHA, f"source file changed: {sha}"
blob = torch.load(str(SRC), map_location="cpu", weights_only=True)
src_sd = blob["state_dict"]
print("source: obs_dim", blob["obs_dim"], "actions", blob["num_actions"], "gru", blob["use_gru"])

story_path.set_profile(True)                      # egg profile, as long_v9 trains with
obs_dim = deep_obs_dim(True, time_aware=True)
assert obs_dim == 81 and len(ACTIONS) == 6
torch.manual_seed(SEED)                            # same order as train(): seed, then construct
policy = SpatialVisionPolicy(obs_dim=obs_dim, num_actions=6, use_gru=True, gru_hidden_dim=512, aux_heads=False)
fresh_sd = {k: v.clone() for k, v in policy.state_dict().items()}

trunk_keys = [k for k in fresh_sd if k.startswith("vision_trunk.")]
missing = [k for k in trunk_keys if k not in src_sd]
bad = [k for k in trunk_keys if k in src_sd and tuple(src_sd[k].shape) != tuple(fresh_sd[k].shape)]
assert not missing and not bad, (missing, bad)
new_sd = dict(fresh_sd)
for k in trunk_keys:
    new_sd[k] = src_sd[k].clone()
policy.load_state_dict(new_sd, strict=True)

n_copied = sum(v.numel() for k, v in new_sd.items() if k.startswith("vision_trunk."))
n_total = sum(v.numel() for v in new_sd.values())
OUT_DIR.mkdir(parents=True, exist_ok=True)
target = OUT_DIR / "init-weights.pt"
assert not target.exists(), "refusing to overwrite"
policy.save(target)

report = {"source": str(SRC), "source_sha256": SRC_SHA, "seed": SEED, "copied_keys": len(trunk_keys),
          "copied_elements": n_copied, "total_elements": n_total, "copied_fraction": round(n_copied / n_total, 4),
          "not_transferred": sorted({k.split(".")[0] for k in fresh_sd if not k.startswith("vision_trunk.")}),
          "init_weights": str(target), "init_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
          "bn_note": "BatchNorm running statistics were copied with the trunk; they reflect the source model's training pixels."}

ref = OUT_DIR.parent / "fresh" / "snapshots" / "update-000000.pt"
if ref.exists():
    f = torch.load(str(ref), map_location="cpu", weights_only=False)["state_dict"]
    heads_equal = all(torch.equal(f[k], new_sd[k]) for k in f if not k.startswith("vision_trunk."))
    trunk_differs = any(not torch.equal(f[k], new_sd[k]) for k in f if k.startswith("vision_trunk.") and f[k].is_floating_point())
    report["non_trunk_tensors_equal_fresh_arm_init"] = bool(heads_equal)
    report["trunk_differs_from_fresh_arm_init"] = bool(trunk_differs)
(OUT_DIR / "init-provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
