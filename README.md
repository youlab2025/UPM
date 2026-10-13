<div align="center">

<h1>UPM</h1>
<h3>Unified Perturbation Modeling</h3>

<p><b>From a cell's starting state to its response to an intervention.</b></p>

<p>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10–3.12"></a>
  <a href="#python-api"><img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.1 or later"></a>
  <a href="#checkpoint"><img src="https://img.shields.io/badge/Model-UPM-8054B3?style=flat-square" alt="UPM"></a>
  <a href="#inputs-and-outputs"><img src="https://img.shields.io/badge/AnnData-H5AD-339A94?style=flat-square" alt="AnnData H5AD"></a>
  <a href="#checkpoint"><img src="https://img.shields.io/badge/Hugging%20Face-weights%20coming%20soon-777777?style=flat-square&logo=huggingface&logoColor=FFD21E" alt="Hugging Face weights coming soon"></a>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#checkpoint">Checkpoint</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#python-api">Python API</a> ·
  <a href="#citation">Citation</a>
</p>

<p><i>How Much Can AI Virtual Cells Gain from Unified Perturbational Modeling?</i></p>

</div>

## Overview

**UPM predicts single-cell transcriptomic responses to genetic, chemical, and cytokine interventions within a shared framework.** It combines the starting cell state with a perturbation representation and decodes the resulting expression profile. Combined interventions supply multiple perturbation inputs to the same model.

This repository provides the **UPM inference implementation** for the standalone checkpoint used in the paper. It includes a PyTorch model, an AnnData command-line interface, and portable checkpoint loading on CPU or CUDA.

<p align="center">
  <a href="assets/fig1_ab.pdf">
    <img src="assets/fig1_ab.png" width="1000" alt="Figure 1, panels A and B. UPM intervention inputs and model architecture.">
  </a>
</p>

| Intervention | Input | Inference behavior |
| :--- | :--- | :--- |
| **Gene editing** | A gene name and `crispri` or `crispra` | Predict repression or activation responses. |
| **Chemical** | A molecule name and dose in **nM** | Predict dose-conditioned responses. |
| **Cytokine** | A cytokine name | Predict stimulation responses using its fixed embedding. |
| **Combination** | Multiple perturbations in one condition | Average component perturbation representations before decoding. |

Perturbation names must be present in the checkpoint vocabulary. The bundle includes the fixed embeddings needed for inference.

## Installation

Use **Python 3.10–3.12**. For CUDA inference, install a compatible PyTorch build using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) before installing UPM.

```bash
conda create -n upm python=3.11 -y
conda activate upm

git clone https://github.com/zongtingwei/UPM.git
cd UPM
python -m pip install -e .

upm-infer --help
```

<details>
<summary><b>Install directly from GitHub</b></summary>

```bash
python -m pip install git+https://github.com/zongtingwei/UPM.git
```

The examples below assume a repository checkout. With a direct installation, supply your own conditions JSON file in place of `examples/conditions.json`.

</details>

## Checkpoint

> [!NOTE]
> **Hugging Face weight release is pending.** The inference code is available in this repository; the UPM download link will be added here when the checkpoint is published. The examples below require a local checkpoint bundle.

Code is distributed through GitHub, and model weights will be hosted on Hugging Face. The release is scoped to **UPM inference**.

A portable checkpoint contains these six files:

```text
checkpoints/upm/
├── config.json           # UPM architecture and dimensions
├── genes.json            # Expression gene channels and aliases
├── perturbations.json    # Gene, molecule, and cytokine vocabularies
├── embeddings.npz        # Fixed perturbation embeddings
├── model.safetensors     # Inference weights
└── manifest.json         # File checksums and bundle metadata
```

Load the whole bundle with `UPM.from_pretrained("checkpoints/upm")`. The loader verifies file checksums and architecture compatibility. Inference loads Safetensors and NumPy arrays without pickle or a training framework.

The same loader also accepts a Hugging Face repository ID through `from_pretrained(source, revision=...)`, or `--model` and `--revision` on the CLI. Once weights are published, use an explicit Hugging Face commit revision for reproducible runs. `--local-files-only` loads an existing local bundle or cached snapshot without downloading.

## Quickstart

### 1. Prepare control cells

Use a **control-only** AnnData file from the cell context you want to perturb. By default, UPM reads expression from `.X` and uses Ensembl identifiers in `.var_names`. Keep all measured gene columns when normalizing raw counts.

### 2. Define interventions

Each condition has a unique `id` and a nonempty list of `perturbations`. The provided [examples/conditions.json](examples/conditions.json) contains:

```json
[
  {
    "id": "TP53_CRISPRi",
    "perturbations": [{"type": "crispri", "name": "TP53"}]
  },
  {
    "id": "MYC_CRISPRa",
    "perturbations": [{"type": "crispra", "name": "MYC"}]
  }
]
```

For a combination, put its components in the **same** `perturbations` list. For a chemical intervention, also provide a numeric `dose` in nM; cytokines do not take a dose field. See [Python API](#python-api) for examples.

### 3. Predict

With raw counts in `.X` and a local bundle at `checkpoints/upm`:

```bash
upm-infer \
  --model checkpoints/upm \
  --control controls.h5ad \
  --conditions examples/conditions.json \
  --input-space counts \
  --device cpu \
  --batch-size 256 \
  --output predictions.h5ad
```

Use `--device cuda` for GPU inference, or `--device auto` to select CUDA when available. If `.X` is already `log1p(CP10K)`, set `--input-space log1p-cp10k`. If raw counts are in `.layers["counts"]`, add `--layer counts`. If Ensembl IDs are stored in `.var["ensembl_id"]`, add `--gene-id-column ensembl_id`.

Each condition uses the same supplied control-cell pool. With two conditions and 1,000 controls, the output contains 2,000 predicted cells in the input gene order.

## Python API

Load, align, and predict. Gene alignment is performed **once per model instance**; load a fresh instance to use a different gene panel.

```python
import anndata as ad

from upm import UPM, Perturbation
from upm.preprocessing import prepare_controls

adata = ad.read_h5ad("controls.h5ad")
controls = prepare_controls(adata.X, input_space="counts")

model = UPM.from_pretrained("checkpoints/upm", device="cpu")
model.align_genes(adata.var_names.tolist())  # Ensembl IDs, in input column order

predicted = model.predict(
    controls,
    Perturbation(type="crispri", name="TP53"),
    batch_size=256,
)

# Float32 log-expression, with the same cells × genes shape as controls.
print(predicted.shape)
```

Inspect available perturbations before choosing a condition:

```python
print(model.perturbations["genes"][:10])
print(model.perturbations["molecules"][:10])
print(model.perturbations["cytokines"][:10])
```

<details>
<summary><b>Gene × gene, gene × drug, and cytokine examples</b></summary>

These examples reuse `model` and `controls` from above. Choose names from the loaded bundle's vocabulary.

```python
# Gene × gene: two interventions in one prediction.
gene_pair = model.predict(controls, [
    Perturbation("crispri", "TP53"),
    Perturbation("crispra", "MYC"),
])

# Gene × drug: 1,000 nM = 1 µM.
drug_name = model.perturbations["molecules"][0]
gene_drug = model.predict(controls, [
    Perturbation("crispri", "TP53"),
    Perturbation("molecule", drug_name, dose=1000),
])

# Cytokine stimulation: no dose field.
cytokine_name = model.perturbations["cytokines"][0]
cytokine = model.predict(controls, Perturbation("cytokine", cytokine_name))
```

</details>

## Inputs and outputs

| Item | Contract |
| :--- | :--- |
| **Control expression** | Nonempty cells × genes matrix. The CLI accepts counts or `log1p(CP10K)`; the Python `predict` method expects `log1p(CP10K)`. `prepare_controls` handles dense and sparse inputs and returns dense float32 arrays. |
| **Gene identifiers** | Exact checkpoint identifiers, normally Ensembl IDs. The CLI uses `.var_names` or a specified `.var` column. For multiple aliases per input column, use `--gene-aliases` with a JSON list of alias lists, or pass alias lists to `align_genes`. |
| **Molecule dose** | Explicit, finite, nonnegative dose in **nM**. |
| **Missing genes** | By default, unmatched genes retain their control expression after the model's native normalization. `--unknown-gene error` requires complete coverage. A panel with no matching genes is rejected. |
| **Unknown perturbations** | Rejected by default. `--unknown-perturbation control` explicitly returns controls for an unsupported whole condition, including a combination with any unsupported component. |
| **Prediction mode** | Float32 inference using the deterministic VAE mean. |

The CLI writes a new `.h5ad` file with:

- **`.X`** — predicted log-expression, grouped by condition and preserving control-cell order within each condition.
- **`.obs`** — input cell annotations plus `condition_id`, `control_id`, and `control_row`.
- **`.var`** — input gene annotations plus the `upm_covered` coverage flag.
- **`.uns["upm"]`** — model source, conditions, normalization, and inference settings.

Output genes remain in the input order. Predictions undergo native CP10K normalization followed by control-value restoration for uncovered genes, matching the Figure 2D inference contract. Consequently, the final count-space row sum can differ from 10,000 when uncovered genes are present. The CLI refuses to overwrite an existing output file.

## Advanced: exporting a legacy checkpoint

<details>
<summary><b>For owners of a trusted UPM training checkpoint</b></summary>

Convert `args.pt`, `ensgids_all.pt`, and `best.ckpt` together with the exact perturbation embedding assets used for that checkpoint:

```bash
upm-export \
  --checkpoint-dir /path/to/upm/checkpoint \
  --embedding-dir /path/to/matching/embedding-assets \
  --output checkpoints/upm \
  --trust-legacy-pickle
```

The trust flag authorizes loading legacy pickle files, which can execute Python code; use it only with trusted checkpoint and embedding files. The exporter retains inference weights and embedding assets and discards optimizer state. Regular inference uses the resulting portable bundle.

</details>

## Repository guide

| Path | Contents |
| :--- | :--- |
| [src/upm/model.py](src/upm/model.py) | UPM model and perturbation interface. |
| [src/upm/cli.py](src/upm/cli.py) | Control-only H5AD inference. |
| [src/upm/bundle.py](src/upm/bundle.py) | Local and Hugging Face bundle loading. |
| [src/upm/preprocessing.py](src/upm/preprocessing.py) | Expression validation and count normalization. |
| [src/upm/export.py](src/upm/export.py) | Conversion of trusted legacy UPM checkpoints. |
| [examples/](examples/) | CLI entry point and example conditions. |
| [tests/](tests/) | Model, preprocessing, checkpoint, and CLI checks. |

For usage questions or reproducible problems, [open a GitHub issue](https://github.com/zongtingwei/UPM/issues) with the command, package versions, and error message.

## Citation

If you use this code, please cite our paper:

> **How Much Can AI Virtual Cells Gain from Unified Perturbational Modeling?**

Publication details and BibTeX will be added here when available.
