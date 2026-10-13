"""One-time conversion of trusted UPM Lightning checkpoints.

Usage: upm-export --checkpoint-dir PATH --embedding-dir PATH --output PATH
                 --trust-legacy-pickle
The directory must contain args.pt, ensgids_all.pt and best.ckpt. Embedding
assets must be the ESM/fingerprint assets used when training that checkpoint.
Only run this on trusted files: legacy PyTorch pickle can execute Python code.
The resulting bundle can be loaded by inference without pickle or Lightning.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .bundle import save_bundle
from .model import UPM, ModelConfig


def _legacy_load(path, *, mmap=False):
    return torch.load(str(path), map_location="cpu", weights_only=False, mmap=mmap)


def export_checkpoint(checkpoint_dir, embedding_dir, output, *, trust_legacy_pickle=False):
    if not trust_legacy_pickle:
        raise ValueError("Legacy pickle export requires explicit trust_legacy_pickle=True")
    checkpoint_dir, embedding_dir, output = map(Path, (checkpoint_dir, embedding_dir, output))
    if output.exists():
        raise FileExistsError(output)
    args = _legacy_load(checkpoint_dir / "args.pt")
    if "#datav4" not in str(getattr(args, "log_dir", "")).lower():
        raise ValueError("Checkpoint training metadata does not match the supported UPM release")
    if args.num_mlp_layers_in_module != 10:
        raise ValueError("Checkpoint depth does not match the supported UPM architecture")
    genes = _legacy_load(checkpoint_dir / "ensgids_all.pt")
    proteins = _legacy_load(embedding_dir / "uniprots.pt")
    protein_index = {}
    for index, name in enumerate(proteins):
        protein_index.setdefault(name, index)
    protein_embeddings = np.load(embedding_dir / "uniprot_embeddings.npy", allow_pickle=False,
                                  mmap_mode="r")
    mapping = _legacy_load(embedding_dir / "genes2uniprots.pt")
    gene_embeddings = np.zeros((len(mapping), protein_embeddings.shape[1]), dtype=np.float32)
    for index, (_, identifiers) in enumerate(mapping):
        if not identifiers:
            raise ValueError("Gene embedding mapping contains an empty protein list")
        rows = [protein_index[name] for name in identifiers]
        gene_embeddings[index] = protein_embeddings[rows].mean(axis=0)
    molecules = _legacy_load(embedding_dir / "molecules.pt")
    molecule_embeddings = np.load(embedding_dir / "molecule_embeddings.npy", allow_pickle=False)
    cytokine_path = embedding_dir / "cytokines.pt"
    cytokine_embedding_path = embedding_dir / "cytokine_embeddings.npy"
    if cytokine_path.exists() != cytokine_embedding_path.exists():
        raise ValueError("Cytokine vocabulary and embeddings must both be present or absent")
    cytokines = _legacy_load(cytokine_path) if cytokine_path.exists() else []
    cytokine_embeddings = (np.load(cytokine_embedding_path, allow_pickle=False)
                          if cytokine_path.exists() else np.zeros((0, gene_embeddings.shape[1])))
    config = ModelConfig(in_dim=args.in_dim, hidden_dim=args.hidden_dim,
                         num_layers=args.num_mlp_layers_in_module,
                         gene_embedding_dim=gene_embeddings.shape[1],
                         molecule_embedding_dim=molecule_embeddings.shape[1])
    with torch.device("meta"):
        model = UPM(config, genes, {"genes": [name for name, _ in mapping],
                       "molecules": molecules, "cytokines": cytokines},
                       {"genes": gene_embeddings, "molecules": molecule_embeddings,
                        "cytokines": cytokine_embeddings})
    checkpoint = _legacy_load(checkpoint_dir / "best.ckpt", mmap=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    del checkpoint  # optimizer and training state never enter the exported bundle
    return save_bundle(model, output, metadata={
        "model_name": "UPM",
        "checkpoint": "best.ckpt", "prediction_mode": "legacy deterministic VAE mean",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-legacy-pickle", action="store_true",
                        help="Allow unpickling trusted training checkpoints and metadata")
    args = parser.parse_args(argv)
    result = export_checkpoint(args.checkpoint_dir, args.embedding_dir, args.output,
                               trust_legacy_pickle=args.trust_legacy_pickle)
    print(f"Exported inference bundle: {result}")


if __name__ == "__main__":
    main()
