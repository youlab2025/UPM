"""UPM inference from portable local or Hugging Face checkpoints.

Load UPM.from_pretrained(), align Ensembl gene channels with
align_genes(), then predict from log1p(CP10K) control cells.
"""
from .model import UPM, ModelConfig, Perturbation

# Compatibility with scripts written before the public UPM interface.
DataV4L10 = UPM

__all__ = ["UPM", "ModelConfig", "Perturbation"]
__version__ = "0.1.0"
