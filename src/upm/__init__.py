"""DataV4-L10 inference from portable local or Hugging Face checkpoints.

Load DataV4L10.from_pretrained(), align Ensembl gene channels with
align_genes(), then predict from log1p(CP10K) control cells.
"""
from .model import DataV4L10, ModelConfig, Perturbation

__all__ = ["DataV4L10", "ModelConfig", "Perturbation"]
__version__ = "0.1.0"
