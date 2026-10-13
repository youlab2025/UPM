import numpy as np
import pytest
import torch

from upm import UPM, ModelConfig


@pytest.fixture
def model():
    torch.manual_seed(13)
    config = ModelConfig(in_dim=3, hidden_dim=4, num_layers=10,
                         gene_embedding_dim=2, molecule_embedding_dim=3)
    return UPM(
        config, genes=[["ENSG1"], ["ENSG2"], ["ENSG3"]],
        perturbations={"genes": ["TP53", "MYC"], "molecules": ["DrugA"],
                       "cytokines": ["IFNG"]},
        embeddings={"genes": torch.tensor([[1., 2.], [3., 4.]]),
                    "molecules": torch.tensor([[1., 3., 2.]]),
                    "cytokines": torch.tensor([[2., 1.]])},
    ).eval()


@pytest.fixture
def controls():
    return np.log1p(np.array([[3000, 6000, 1000], [5000, 2000, 3000]],
                           dtype=np.float32))
