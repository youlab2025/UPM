import importlib
import shutil

import numpy as np
import pytest
from huggingface_hub import constants
from huggingface_hub.utils import LocalEntryNotFoundError

from upm import UPM, Perturbation
from upm.bundle import BUNDLE_FILES, save_bundle


REVISION = "a" * 40


@pytest.fixture
def cached_bundle(model, tmp_path, monkeypatch):
    """Use a real revision-specific HF cache; no network download doubles."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    return save_bundle(model, cache / "models--youlab2025--UPM" / "snapshots" / REVISION)


def arguments(destination, revision=REVISION):
    return ["--output", str(destination), "--revision", revision, "--local-files-only"]


def test_downloaded_checkpoint_runs_inference(model, controls, tmp_path, cached_bundle):
    destination = tmp_path / "checkpoints" / "upm"
    main = importlib.import_module("upm.download").main
    main(arguments(destination))
    assert {p.name for p in destination.iterdir()} == set(BUNDLE_FILES)
    downloaded = UPM.from_pretrained(destination)
    np.testing.assert_array_equal(
        downloaded.predict(controls, Perturbation("crispri", "TP53")),
        model.predict(controls, Perturbation("crispri", "TP53")),
    )


@pytest.mark.parametrize("broken", ["corrupt", "missing", "manifest"])
def test_download_rejects_incomplete_or_modified_checkpoint(
        tmp_path, cached_bundle, broken, capsys):
    source = cached_bundle
    if broken == "corrupt":
        (source / "model.safetensors").write_bytes(b"modified weights")
        error, message = ValueError, "checksum"
    elif broken == "missing":
        (source / "embeddings.npz").unlink()
        error, message = FileNotFoundError, "embeddings.npz"
    else:
        (source / "manifest.json").write_text('{"format_version": 99}')
        error, message = ValueError, "manifest"
    main = importlib.import_module("upm.download").main
    with pytest.raises(error, match=message):
        main(arguments(tmp_path / "download"))
    assert "Verified" not in capsys.readouterr().out


def test_offline_download_reuses_checkpoint_without_modifying_cache(
        model, tmp_path, cached_bundle):
    main = importlib.import_module("upm.download").main
    destination = tmp_path / "download"
    main(arguments(destination))
    main(arguments(destination))
    assert UPM.from_pretrained(destination).config == model.config
    (destination / "config.json").write_text("modified destination")
    assert UPM.from_pretrained(cached_bundle).config == model.config


def test_offline_uncached_revision_does_not_accept_existing_directory(
        tmp_path, cached_bundle, capsys):
    destination = tmp_path / "download"
    shutil.copytree(cached_bundle, destination)
    main = importlib.import_module("upm.download").main
    with pytest.raises(LocalEntryNotFoundError):
        main(arguments(destination, revision="b" * 40))
    assert "Verified" not in capsys.readouterr().out
