"""Portable safetensors/JSON/NumPy bundles for local and HF inference."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .model import UPM, ModelConfig

ASSET_FILES = ("config.json", "genes.json", "perturbations.json",
               "embeddings.npz", "model.safetensors")
BUNDLE_FILES = ASSET_FILES + ("manifest.json",)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_bundle(model, destination, *, metadata=None):
    """Save inference weights and fixed embeddings, excluding optimizer state.

    Destination must not exist. Writes are staged in a sibling directory so a
    failed export cannot appear to be a complete bundle. Aligned models must be
    reloaded from their original source before export.
    """
    if model._aligned:
        raise ValueError("Export the original checkpoint before gene alignment")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".upm-export-", dir=destination.parent))
    try:
        write_json(staging / "config.json", asdict(model.config))
        write_json(staging / "genes.json", model.genes)
        write_json(staging / "perturbations.json", model.perturbations)
        np.savez(staging / "embeddings.npz",
                 genes=model.gene_embeddings.cpu().numpy(),
                 molecules=model.molecule_embeddings.cpu().numpy(),
                 cytokines=model.cytokine_embeddings.cpu().numpy())
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                  staging / "model.safetensors")
        write_json(staging / "manifest.json", {
            "format_version": 1,
            "sha256": {name: sha256(staging / name) for name in ASSET_FILES},
            "metadata": metadata or {},
        })
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def resolve_bundle(source, *, subfolder=None, revision=None, local_files_only=False):
    source = str(source)
    subfolder = subfolder or ""
    folder = PurePosixPath(subfolder)
    if folder.is_absolute() or ".." in folder.parts or "\\" in subfolder:
        raise ValueError("subfolder must be a relative path inside the checkpoint repository")
    local = Path(source).expanduser()
    if local.is_dir():
        root = local.joinpath(*folder.parts)
        return root, {"source": str(local.resolve()), "subfolder": subfolder, "revision": "local"}
    if local.is_absolute() or source.startswith(("./", "../", "~")):
        raise FileNotFoundError(local)
    from huggingface_hub import snapshot_download
    prefix = subfolder.rstrip("/") + "/" if subfolder else ""
    snapshot = Path(snapshot_download(
        repo_id=source, revision=revision, local_files_only=local_files_only,
        allow_patterns=[prefix + name for name in BUNDLE_FILES],
    ))
    return snapshot.joinpath(*folder.parts), {
        "source": source, "subfolder": subfolder, "revision": snapshot.name,
    }


def load_bundle(source, *, subfolder=None, revision=None, device="cpu", local_files_only=False):
    root, provenance = resolve_bundle(source, subfolder=subfolder, revision=revision,
                                     local_files_only=local_files_only)
    for name in BUNDLE_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    manifest = read_json(root / "manifest.json")
    if manifest.get("format_version") != 1 or set(manifest.get("sha256", {})) != set(ASSET_FILES):
        raise ValueError("Invalid bundle manifest")
    for name in ASSET_FILES:
        if sha256(root / name) != manifest["sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    try:
        config = ModelConfig(**read_json(root / "config.json"))
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid UPM configuration") from error
    with np.load(root / "embeddings.npz", allow_pickle=False) as assets:
        if set(assets.files) != {"genes", "molecules", "cytokines"}:
            raise ValueError("Invalid embedding assets")
        embeddings = {name: assets[name] for name in assets.files}
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
        if any(weights.get_slice(key).get_dtype() != "F32" for key in weights.keys()):
            raise ValueError("UPM inference weights must be float32")
    # Meta construction avoids allocating a second full set of multi-GB weights.
    with torch.device("meta"):
        model = UPM(config, read_json(root / "genes.json"),
                       read_json(root / "perturbations.json"), embeddings)
    model.load_state_dict(load_file(root / "model.safetensors", device="cpu"), strict=True, assign=True)
    model.source = {**provenance, "metadata": manifest.get("metadata", {})}
    return model.to(device).eval().requires_grad_(False)
