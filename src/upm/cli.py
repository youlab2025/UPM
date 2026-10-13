"""UPM inference on control cells in H5AD format.

Example:
  upm-infer --model checkpoints/upm --control controls.h5ad
    --conditions conditions.json --input-space counts --output predicted.h5ad

conditions.json is a nonempty list, for example:
  [{"id": "TP53i", "perturbations": [{"type": "crispri", "name": "TP53"}]},
   {"id": "drug", "perturbations": [{"type": "molecule", "name": "drug-name",
                                     "dose": 1000}]}]
Use multiple perturbations in one condition for a combination. Molecule doses
are in nM. Names must match the bundle vocabulary. All supplied cells must be
controls from the intended cell context; each condition uses that same pool.
Output X contains log-expression after native normalization and missing-gene
control padding. Missing-gene padding can change its total count sum.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import os

import anndata as ad
import numpy as np
import pandas as pd
import torch

from . import __version__
from .bundle import read_json
from .model import UPM, Perturbation, gene_channels
from .preprocessing import prepare_controls


def read_conditions(path):
    values = read_json(path)
    if not isinstance(values, list) or not values:
        raise ValueError("Conditions must be a nonempty JSON list")
    result, seen = [], set()
    for record in values:
        if not isinstance(record, dict) or set(record) != {"id", "perturbations"}:
            raise ValueError("Each condition requires id and perturbations")
        name, items = record["id"], record["perturbations"]
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("Condition IDs must be unique nonempty strings")
        if not isinstance(items, list) or not items or any(not isinstance(p, dict) for p in items):
            raise ValueError("Each condition requires at least one perturbation")
        try:
            perturbations = [Perturbation(**p) for p in items]
        except TypeError as error:
            raise ValueError("Perturbations require type, name, and optional molecule dose") from error
        result.append((name, perturbations))
        seen.add(name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Local UPM bundle or Hugging Face owner/repo")
    parser.add_argument("--subfolder", help="Bundle subfolder inside the local directory or HF repository")
    parser.add_argument("--revision", help="HF revision; use a commit hash for reproducibility")
    parser.add_argument("--local-files-only", action="store_true", help="Load from disk/HF cache without downloading")
    parser.add_argument("--control", required=True, help="Control-only AnnData .h5ad")
    parser.add_argument("--conditions", required=True, help="JSON list of perturbation conditions")
    parser.add_argument("--input-space", required=True, choices=["counts", "log1p-cp10k"])
    parser.add_argument("--layer", help="AnnData layer to use; default: X")
    identifiers = parser.add_mutually_exclusive_group()
    identifiers.add_argument("--gene-id-column", help="var column with Ensembl IDs; default: var_names")
    identifiers.add_argument("--gene-aliases", help="JSON list of alias lists, one per input column")
    parser.add_argument("--unknown-gene", choices=["control", "error"], default="control",
                        help="Missing-gene behavior; control reproduces Figure 2D padding")
    parser.add_argument("--unknown-perturbation", choices=["error", "control"], default="error",
                        help="control explicitly returns controls for an unsupported whole condition")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True, help="New .h5ad file; existing files are never overwritten")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    conditions = read_conditions(args.conditions)
    data = ad.read_h5ad(args.control)
    matrix = data.layers[args.layer] if args.layer else data.X
    controls = prepare_controls(matrix, args.input_space)
    if args.gene_aliases:
        genes = gene_channels(read_json(args.gene_aliases))
    else:
        ids = data.var[args.gene_id_column] if args.gene_id_column else data.var_names
        if pd.isna(ids).any() or any(not isinstance(g, str) for g in ids):
            raise ValueError("Gene identifiers must be nonmissing strings")
        genes = gene_channels(list(ids))
    if len(genes) != data.n_vars:
        raise ValueError("Gene alias count must match the control gene columns")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = UPM.from_pretrained(args.model, subfolder=args.subfolder, revision=args.revision,
                                     device=device, local_files_only=args.local_files_only)
    model.align_genes(genes, unknown_gene=args.unknown_gene)
    predictions, annotations = [], []
    for condition, perturbations in conditions:
        predictions.append(model.predict(controls, perturbations, batch_size=args.batch_size,
                                          unknown_perturbation=args.unknown_perturbation))
        obs = data.obs.copy()
        obs["condition_id"] = condition
        obs["control_id"] = data.obs_names.astype(str)
        obs["control_row"] = np.arange(data.n_obs)
        obs.index = pd.Index([f"{condition}:{i}" for i in range(data.n_obs)])
        annotations.append(obs)
        print(f"Predicted {condition}: {data.n_obs} cells; "
              f"{int(model.coverage.sum())}/{data.n_vars} genes covered", flush=True)
    var = data.var.copy()
    var["upm_covered"] = model.coverage
    result = ad.AnnData(np.concatenate(predictions), obs=pd.concat(annotations), var=var)
    result.uns["upm"] = {
        "version": __version__, "model_name": "UPM",
        "model_source": json.dumps(model.source, ensure_ascii=False),
        "conditions": json.dumps([{"id": name, "perturbations": [asdict(p) for p in ps]}
                                  for name, ps in conditions], ensure_ascii=False),
        "input_space": args.input_space, "prediction_mode": "legacy deterministic VAE mean",
        "normalization": "native log1p(CP10K), followed by missing-gene control padding",
        "unknown_gene": args.unknown_gene, "unknown_perturbation": args.unknown_perturbation,
        "batch_size": args.batch_size,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".upm-", suffix=".h5ad", dir=output.parent)
    os.close(descriptor)
    try:
        result.write_h5ad(temporary)
        # Hard-link installation refuses a concurrent writer's existing output.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
