from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_manifest(paths: Iterable[Path]) -> Dict[str, Any]:
    rows = []
    for path in sorted({Path(item).resolve() for item in paths}, key=lambda item: str(item).lower()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return {"files": rows, "signature": sha256_json(rows)}


def dataset_signature(paths: Iterable[Path], settings: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    manifest = file_manifest(paths)
    payload = {"file_manifest": manifest, "settings": dict(settings)}
    return sha256_json(payload), payload


def environment_manifest(project_root: Optional[Path] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import numpy as np
        result["numpy"] = np.__version__
    except Exception:
        pass
    for package_name in ("pandas", "scipy", "sklearn", "matplotlib", "tensorflow", "nptdms"):
        try:
            module = __import__(package_name)
            result[package_name] = getattr(module, "__version__", "installed")
        except Exception:
            result[package_name] = None
    if project_root is not None:
        result["project_root"] = str(Path(project_root))
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root) if project_root is not None and Path(project_root).exists() else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["git_commit"] = completed.stdout.strip()
    except Exception:
        result["git_commit"] = None
    return result


def checkpoint_signature(
    model_spec: Any,
    training_spec: Any,
    split_signature: str,
    data_signature: str,
) -> str:
    model_payload = model_spec.to_dict() if hasattr(model_spec, "to_dict") else dict(model_spec)
    training_payload = training_spec.to_dict() if hasattr(training_spec, "to_dict") else dict(training_spec)
    return sha256_json(
        {
            "model_spec": model_payload,
            "training_spec": training_payload,
            "split_signature": split_signature,
            "data_signature": data_signature,
        }
    )


def validate_checkpoint_manifest(manifest_path: Path, expected_signature: str, weights_path: Path) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("checkpoint_signature") != expected_signature:
        raise RuntimeError("Checkpoint manifest signature mismatch: {0}".format(manifest_path))
    if not Path(weights_path).is_file():
        raise FileNotFoundError("Checkpoint weights not found: {0}".format(weights_path))
    expected_hash = manifest.get("weights_sha256")
    if expected_hash and sha256_file(Path(weights_path)) != expected_hash:
        raise RuntimeError("Checkpoint weights hash mismatch: {0}".format(weights_path))
    return manifest
