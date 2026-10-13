import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from upm import UPM, Perturbation
from upm.bundle import save_bundle, sha256, write_json
from upm.cli import main
from upm.export import export_checkpoint


def test_safe_bundle_roundtrip(model, controls, tmp_path):
    root = tmp_path / "bundle"
    save_bundle(model, root)
    assert {p.name for p in root.iterdir()} == {
        "config.json", "genes.json", "perturbations.json", "embeddings.npz",
        "model.safetensors", "manifest.json"}
    loaded = UPM.from_pretrained(root)
    p = Perturbation("crispri", "TP53")
    np.testing.assert_array_equal(loaded.predict(controls, p), model.predict(controls, p))
    with pytest.raises(FileExistsError):
        save_bundle(model, root)


def test_corrupt_bundle_fails_before_loading_weights(model, tmp_path):
    root = tmp_path / "bundle"
    save_bundle(model, root)
    (root / "genes.json").write_text('["WRONG"]')
    with pytest.raises(ValueError, match="checksum"):
        UPM.from_pretrained(root)


def test_legacy_bundle_keeps_identical_predictions(model, controls, tmp_path):
    root = tmp_path / "legacy_bundle"
    save_bundle(model, root)
    config = json.loads((root / "config.json").read_text())
    config["architecture"] = "DataV4-L10"
    write_json(root / "config.json", config)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sha256"]["config.json"] = sha256(root / "config.json")
    write_json(root / "manifest.json", manifest)
    loaded = UPM.from_pretrained(root)
    assert loaded.config.architecture == "UPM"
    np.testing.assert_array_equal(
        loaded.predict(controls, Perturbation("crispri", "TP53")),
        model.predict(controls, Perturbation("crispri", "TP53")),
    )


def test_legacy_pickle_requires_explicit_trust(tmp_path):
    with pytest.raises(ValueError, match="trust"):
        export_checkpoint(tmp_path / "absent", tmp_path / "absent", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_strips_training_state_and_keeps_inference(model, controls, tmp_path):
    checkpoint = tmp_path / "legacy"
    embedding = tmp_path / "emb"
    checkpoint.mkdir(); embedding.mkdir()
    args = argparse.Namespace(in_dim=3, hidden_dim=4, num_mlp_layers_in_module=10,
                              log_dir="model#layer10#datav4")
    torch.save(args, checkpoint / "args.pt")
    torch.save(["ENSG1", "ENSG2", "ENSG3"], checkpoint / "ensgids_all.pt")
    torch.save({"state_dict": model.state_dict(), "optimizer_states": [{"private": 1}],
                "epoch": 123}, checkpoint / "best.ckpt")
    torch.save([("TP53", ["P1"]), ("MYC", ["P2", "P3"])], embedding / "genes2uniprots.pt")
    torch.save(["P1", "P2", "P3"], embedding / "uniprots.pt")
    np.save(embedding / "uniprot_embeddings.npy", [[1., 2.], [2., 3.], [4., 5.]])
    for kind, singular in [("molecules", "molecule"), ("cytokines", "cytokine")]:
        torch.save(model.perturbations[kind], embedding / (kind + ".pt"))
        np.save(embedding / (singular + "_embeddings.npy"),
                getattr(model, singular + "_embeddings").numpy())
    export_checkpoint(checkpoint, embedding, tmp_path / "export", trust_legacy_pickle=True)
    loaded = UPM.from_pretrained(tmp_path / "export")
    assert not hasattr(loaded, "optimizer_states")
    np.testing.assert_array_equal(loaded.predict(controls, Perturbation("crispra", "MYC")),
                                  model.predict(controls, Perturbation("crispra", "MYC")))


def test_cli_h5ad_preserves_axes_and_conditions(model, tmp_path):
    save_bundle(model, tmp_path / "upm")
    counts = np.array([[1., 3., 2.], [0., 2., 3.]])
    data = ad.AnnData(sparse.csr_matrix(counts),
                     obs=pd.DataFrame({"cell_type": ["A", "B"]}, index=["c1", "c2"]),
                     var=pd.DataFrame({"ensembl": ["ENSG2", "ENSG1", "UNKNOWN"]},
                                      index=["name2", "name1", "missing"]))
    input_path = tmp_path / "controls.h5ad"
    data.write_h5ad(input_path)
    conditions = [{"id": "TP53i", "perturbations": [{"type": "crispri", "name": "TP53"}]},
                  {"id": "combo", "perturbations": [{"type": "crispri", "name": "TP53"},
                   {"type": "molecule", "name": "DrugA", "dose": 100}]}]
    condition_path = tmp_path / "conditions.json"
    condition_path.write_text(json.dumps(conditions))
    out = tmp_path / "predictions.h5ad"
    main(["--model", str(tmp_path / "upm"),
          "--control", str(input_path), "--conditions", str(condition_path),
          "--input-space", "counts", "--gene-id-column", "ensembl",
          "--output", str(out), "--device", "cpu", "--batch-size", "1"])
    result = ad.read_h5ad(out)
    assert result.shape == (4, 3)
    assert result.var_names.tolist() == data.var_names.tolist()
    assert result.obs["condition_id"].tolist() == ["TP53i", "TP53i", "combo", "combo"]
    assert result.obs["control_id"].tolist() == ["c1", "c2", "c1", "c2"]
    assert result.obs["cell_type"].tolist() == ["A", "B", "A", "B"]
    np.testing.assert_array_equal(result.var["upm_covered"].to_numpy(), [True, True, False])
    normalized = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    np.testing.assert_array_equal(result.X[:, 2], np.tile(normalized[:, 2], 2))
    m1 = UPM.from_pretrained(tmp_path / "upm").align_genes(["ENSG2", "ENSG1", "UNKNOWN"])
    expected = m1.predict(normalized, Perturbation("crispri", "TP53"), batch_size=1)
    np.testing.assert_array_equal(result.X[:2], expected)
    assert result.uns["upm"]["model_name"] == "UPM"
    with pytest.raises(FileExistsError):
        main(["--model", str(tmp_path / "upm"), "--control", str(input_path),
              "--conditions", str(condition_path), "--input-space", "counts", "--output", str(out)])


def test_malformed_conditions_do_not_load_weights(tmp_path):
    from upm.cli import read_conditions
    for conditions in [[], [{"id": "x", "perturbations": []}],
                       [{"id": "x", "perturbations": [{"type": "molecule", "name": "A"}]}],
                       [{"id": "x", "perturbations": [{"type": "crispri", "name": "TP53"}]}] * 2]:
        path = tmp_path / "conditions.json"
        path.write_text(json.dumps(conditions))
        with pytest.raises(ValueError):
            read_conditions(path)
