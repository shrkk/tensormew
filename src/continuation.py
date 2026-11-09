"""Transactional continuation checkpoints ("generations").

A generation is a directory of component files (policy, optimizer + trainer state, world/emulator state) plus a
``manifest.json`` holding every component's SHA256 and one shared ``generation_id``.  Individually atomic files are NOT a
transactional set, so the generation only becomes current when ``CURRENT.json`` (the commit pointer) is replaced, last.
Readers verify: pointer -> manifest hash -> every component hash.  Components also embed the generation id themselves
(``check_embedded``), so a file swapped in from another generation is rejected even if someone also edits the manifest.

Component files are produced by this experiment on this machine and are loaded with trusted deserialization
(``torch.load(weights_only=False)`` / ``pickle``); never load a generation received from elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable

from src.run_control import atomic_write_json, utc_now

POINTER = "CURRENT.json"
MANIFEST = "manifest.json"
SCHEMA_VERSION = 1


class GenerationError(RuntimeError):
    """A generation is missing, incomplete, or internally inconsistent; nothing is silently repaired."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_generation_id() -> str:
    return uuid.uuid4().hex[:16]


def write_generation(root, update: int, components: dict[str, Callable[[str, str], None]], meta: dict) -> Path:
    """Write every component into ``root/gen-<update>-<id>/`` then commit the pointer.  ``components`` maps a name to
    ``writer(path, generation_id)``.  An exception before the pointer swap leaves the previous generation current."""
    root = Path(root)
    gid = new_generation_id()
    gdir = root / f"gen-{int(update):08d}-{gid}"
    gdir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, writer in components.items():
        target = gdir / f"{name}.pt"
        tmp = gdir / f"{name}.pt.tmp"
        writer(str(tmp), gid)
        os.replace(tmp, target)
        entries[name] = {"file": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size}
    manifest = {"schema_version": SCHEMA_VERSION, "generation_id": gid, "update": int(update),
                "created_utc": utc_now(), "components": entries, "meta": meta}
    atomic_write_json(gdir / MANIFEST, manifest)
    atomic_write_json(root / POINTER, {"generation_dir": gdir.name, "generation_id": gid, "update": int(update),
                                       "manifest_sha256": sha256_file(gdir / MANIFEST)})
    return gdir


def read_generation(root_or_dir, expect_components: tuple[str, ...] = ()) -> tuple[Path, dict]:
    """Resolve + verify a generation.  ``root_or_dir`` may be the generations root (follows ``CURRENT.json``) or one
    ``gen-*`` directory.  Raises ``GenerationError`` on any mismatch (missing file, hash, id, component set)."""
    p = Path(root_or_dir)
    pointer = None
    if (p / POINTER).exists():
        pointer = json.loads((p / POINTER).read_text(encoding="utf-8"))
        p = p / pointer["generation_dir"]
    if not (p / MANIFEST).exists():
        raise GenerationError(f"no {MANIFEST} in {p}: incomplete or not a generation")
    if pointer is not None and sha256_file(p / MANIFEST) != pointer["manifest_sha256"]:
        raise GenerationError(f"manifest hash differs from the commit pointer for {p.name}")
    manifest = json.loads((p / MANIFEST).read_text(encoding="utf-8"))
    if pointer is not None and manifest.get("generation_id") != pointer.get("generation_id"):
        raise GenerationError("generation id differs between commit pointer and manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GenerationError(f"unsupported generation schema {manifest.get('schema_version')}")
    for name in expect_components:
        if name not in manifest["components"]:
            raise GenerationError(f"generation {p.name} lacks component '{name}'")
    for name, entry in manifest["components"].items():
        f = p / entry["file"]
        if not f.exists():
            raise GenerationError(f"component '{name}' missing: {f}")
        if sha256_file(f) != entry["sha256"]:
            raise GenerationError(f"component '{name}' hash mismatch in {p.name} (swapped or corrupt file)")
    return p, manifest


def check_embedded(component_name: str, obj: dict, manifest: dict) -> None:
    """The deserialized component must carry the manifest's generation id."""
    got = obj.get("generation_id") if isinstance(obj, dict) else None
    if got != manifest["generation_id"]:
        raise GenerationError(
            f"component '{component_name}' belongs to generation {got!r}, manifest is {manifest['generation_id']!r}")
