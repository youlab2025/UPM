"""Deterministic UPM inference without training-framework dependencies."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .layers import MLP, MLPBiasFree
from .preprocessing import validate_expression


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int
    hidden_dim: int
    num_layers: int
    gene_embedding_dim: int
    molecule_embedding_dim: int
    architecture: str = "UPM"
    format_version: int = 1

    def __post_init__(self):
        for value in (self.in_dim, self.hidden_dim, self.num_layers,
                      self.gene_embedding_dim, self.molecule_embedding_dim):
            if type(value) is not int or value <= 0:
                raise ValueError("Model dimensions must be positive integers")
        # Existing portable bundles retain their original architecture tag.
        if self.architecture == "DataV4-L10":
            object.__setattr__(self, "architecture", "UPM")
        if self.num_layers != 10 or self.architecture != "UPM" or self.format_version != 1:
            raise ValueError("Unsupported UPM architecture or bundle version")


@dataclass(frozen=True)
class Perturbation:
    """A gene edit, molecule (dose in nM), or cytokine.

    CRISPRi negates the learned gene embedding; CRISPRa keeps its sign.
    Cytokines use their fixed embedding without dose scaling. Molecules require
    an explicit nonnegative dose and use log1p(dose_nM / 100).
    """
    type: str
    name: str
    dose: float | None = None

    def __post_init__(self):
        if self.type not in ("crispri", "crispra", "molecule", "cytokine"):
            raise ValueError("Unknown perturbation type")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Perturbation name must be a nonempty string")
        if self.type == "molecule":
            if (not isinstance(self.dose, (int, float)) or isinstance(self.dose, bool)
                    or not math.isfinite(self.dose) or self.dose < 0):
                raise ValueError("Molecule dose must be finite, nonnegative, and in nM")
        elif self.dose is not None:
            raise ValueError("Dose is supported only for molecule perturbations")


def gene_channels(genes):
    if not isinstance(genes, (list, tuple)) or not genes:
        raise ValueError("Gene channels must be a nonempty list of IDs or alias lists")
    channels = []
    for item in genes:
        aliases = [item] if isinstance(item, str) else item
        if not isinstance(aliases, (list, tuple)) or any(
            not isinstance(g, str) or not g.strip() for g in aliases
        ):
            raise ValueError("Each gene channel must contain string identifiers")
        channels.append(list(aliases))
    return channels


class UPM(nn.Module):
    """Inference architecture matching the UPM checkpoint.

    Gene alignment replaces input/output weights once, using mean weights for
    aliases that match multiple source channels. Missing genes receive zero
    weights and their output is restored to control AFTER native normalization,
    as in Figure 2D. Use unknown_gene='error' to require complete coverage.
    """
    def __init__(self, config, genes, perturbations, embeddings):
        super().__init__()
        self.config = config
        self.genes = gene_channels(genes)
        if len(self.genes) != config.in_dim:
            raise ValueError("Gene channels do not match in_dim")
        if set(perturbations) != {"genes", "molecules", "cytokines"}:
            raise ValueError("Vocabulary must contain genes, molecules and cytokines")
        self.perturbations = {}
        self._indices = {}
        for kind, dim in (("genes", config.gene_embedding_dim),
                          ("molecules", config.molecule_embedding_dim),
                          ("cytokines", config.gene_embedding_dim)):
            names = perturbations[kind]
            if not isinstance(names, list) or any(
                not isinstance(n, str) or not n.strip() for n in names
            ) or len(set(names)) != len(names):
                raise ValueError("Vocabulary names must be unique nonempty strings")
            self.perturbations[kind] = list(names)
            self._indices[kind] = {name: i for i, name in enumerate(names)}
            tensor = torch.as_tensor(embeddings[kind], dtype=torch.float32, device="cpu")
            if tensor.shape != (len(names), dim) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid {kind} embeddings")
            buffer_name = {"genes": "gene", "molecules": "molecule", "cytokines": "cytokine"}[kind]
            self.register_buffer(buffer_name + "_embeddings", tensor, persistent=False)

        h, layers = config.hidden_dim, config.num_layers
        self.gene_encoder = MLPBiasFree(config.in_dim, h, h, layers)
        self.mask_cell_decoder = MLPBiasFree(2 * h, h, h, layers)
        self.mask_cell_expression = MLPBiasFree(h, config.in_dim, h, layers)
        self.vae_logvar_decoder = MLPBiasFree(2 * h, h, h, layers)
        self.gene_embeddings_mlp = MLP(config.gene_embedding_dim, h, h, 2)
        self.molecule_embeddings_mlp = MLP(config.molecule_embedding_dim, h, h, 2)
        self.perturbation_embeddings_mlp = MLP(h, h, h, layers)
        self.coverage = np.ones(config.in_dim, dtype=bool)
        self._aligned = False
        self.source = {}
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, source, *, subfolder=None, revision=None, device="cpu", local_files_only=False):
        """Load a local bundle or HF repository; revision may pin an HF commit.

        Bundles contain config.json, genes.json, perturbations.json,
        embeddings.npz and model.safetensors. No pickle is loaded by inference.
        """
        from .bundle import load_bundle
        return load_bundle(source, subfolder=subfolder, revision=revision, device=device,
                           local_files_only=local_files_only)

    def align_genes(self, target, *, unknown_gene="control"):
        if unknown_gene not in ("control", "error"):
            raise ValueError("unknown_gene must be control or error")
        if self._aligned:
            raise RuntimeError("Gene channels have already been aligned; load a fresh model to realign")
        target = gene_channels(target)
        lookup = {}
        for i, aliases in enumerate(self.genes):
            for alias in dict.fromkeys(aliases):
                lookup.setdefault(alias, []).append(i)
        matches = [[i for alias in aliases for i in lookup.get(alias, [])] for aliases in target]
        coverage = np.array([bool(indices) for indices in matches])
        if not coverage.any():
            raise ValueError("No input gene channels match the checkpoint; use Ensembl IDs")
        if unknown_gene == "error" and not coverage.all():
            raise ValueError(f"{int((~coverage).sum())} unmatched input gene channels")
        original_in = self.gene_encoder.layers[0].weight
        original_out = self.mask_cell_expression.layers[-1].weight
        linear_in = nn.Linear(len(target), self.config.hidden_dim, bias=False,
                             device=original_in.device, dtype=original_in.dtype)
        linear_out = nn.Linear(self.config.hidden_dim, len(target), bias=False,
                              device=original_out.device, dtype=original_out.dtype)
        with torch.no_grad():
            linear_in.weight.zero_()
            linear_out.weight.zero_()
            for index, sources in enumerate(matches):
                if sources:
                    linear_in.weight[:, index] = torch.stack(
                        [original_in[:, s] for s in sources]).mean(0)
                    linear_out.weight[index] = torch.stack(
                        [original_out[s] for s in sources]).mean(0)
        self.gene_encoder.layers[0] = linear_in.requires_grad_(False)
        self.mask_cell_expression.layers[-1] = linear_out.requires_grad_(False)
        self.genes, self.coverage, self._aligned = target, coverage, True
        return self

    def perturbation_embedding(self, perturbations):
        """Combinations average their component embeddings before decoding."""
        items = [perturbations] if isinstance(perturbations, Perturbation) else list(perturbations)
        if not items or any(not isinstance(p, Perturbation) for p in items):
            raise ValueError("Provide at least one Perturbation")
        outputs = []
        for p in items:
            kind = "genes" if p.type in ("crispri", "crispra") else p.type + "s"
            if p.name not in self._indices[kind]:
                raise ValueError(f"Unknown {p.type} perturbation: {p.name}")
            raw = getattr(self, {"genes": "gene", "molecules": "molecule",
                                "cytokines": "cytokine"}[kind] + "_embeddings")[self._indices[kind][p.name]]
            encoder = self.molecule_embeddings_mlp if kind == "molecules" else self.gene_embeddings_mlp
            latent = self.perturbation_embeddings_mlp(encoder(raw))
            if p.type == "crispri":
                latent = -latent
            elif p.type == "molecule":
                latent = latent * np.log1p(p.dose / 100.)
            outputs.append(latent)
        return torch.stack(outputs).mean(0)

    @torch.inference_mode()
    def predict(self, controls, perturbations, *, batch_size=256, unknown_perturbation="error"):
        """Predict float32 log-expression for each supplied control cell.

        Inference is deterministic (VAE mean), uses native CP10K normalization,
        and preserves cell/gene order. Unknown perturbations raise by default;
        'control' explicitly returns the whole control matrix for an unsupported
        condition, including a combination with an unsupported component.
        """
        controls = np.asarray(controls, dtype=np.float32)
        validate_expression(controls, log_space=True)
        if controls.shape[1] != len(self.genes):
            raise ValueError("Control columns must match the aligned gene channels")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if unknown_perturbation not in ("error", "control"):
            raise ValueError("unknown_perturbation must be error or control")
        try:
            perturbation = self.perturbation_embedding(perturbations)
        except ValueError as error:
            if unknown_perturbation == "control" and str(error).startswith("Unknown "):
                return controls.copy()
            raise
        output = np.empty_like(controls)
        device = self.gene_encoder.layers[0].weight.device
        for start in range(0, len(controls), batch_size):
            chunk = controls[start:start + batch_size]
            x = torch.as_tensor(chunk, device=device)
            encoded = self.gene_encoder(x)
            latent = torch.cat([encoded, perturbation.reshape(1, -1).repeat(len(x), 1)], 1)
            delta = self.mask_cell_expression(self.mask_cell_decoder(latent))
            delta = delta.clamp(-np.log1p(1e4), np.log1p(1e4))
            expression = torch.relu(x + delta).cpu().numpy()
            counts = np.expm1(expression)
            totals = counts.sum(1, keepdims=True)
            totals[totals == 0] = 1
            predicted = np.log1p(counts / totals * 1e4)
            predicted[:, ~self.coverage] = chunk[:, ~self.coverage]
            output[start:start + len(chunk)] = predicted
        return output


# Compatibility with the pre-release model class name.
DataV4L10 = UPM

