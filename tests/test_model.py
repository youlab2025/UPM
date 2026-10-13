import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from upm import Perturbation
from upm.layers import MLP, MLPBiasFree


@pytest.mark.parametrize("cls", [MLP, MLPBiasFree])
def test_residual_layers_match_independent_reference(cls):
    torch.manual_seed(5)
    mlp = cls(3, 2, 4, 4)
    x = torch.randn(2, 3)
    expected = F.relu(F.linear(x, mlp.layers[0].weight, mlp.layers[0].bias))
    for i in range(3):
        norm = mlp.layernorms[i]
        if i:
            expected = expected + F.relu(F.linear(expected, mlp.layers[i].weight,
                                                   mlp.layers[i].bias))
        expected = F.layer_norm(expected, (4,), norm.weight, norm.bias, 1e-5)
    expected = F.linear(expected, mlp.layers[-1].weight, mlp.layers[-1].bias)
    torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)


def test_sign_dose_and_combination_mean(model):
    activation = model.perturbation_embedding(Perturbation("crispra", "TP53"))
    inhibition = model.perturbation_embedding(Perturbation("crispri", "TP53"))
    drug1 = model.perturbation_embedding(Perturbation("molecule", "DrugA", 100))
    drug2 = model.perturbation_embedding(Perturbation("molecule", "DrugA", 1000))
    torch.testing.assert_close(inhibition, -activation, rtol=0, atol=0)
    torch.testing.assert_close(drug2, drug1 * (np.log1p(10) / np.log1p(1)))
    torch.testing.assert_close(model.perturbation_embedding(
        [Perturbation("crispri", "TP53"), Perturbation("molecule", "DrugA", 100)]
    ), (inhibition + drug1) * .5, rtol=0, atol=0)
    assert torch.count_nonzero(model.perturbation_embedding(
        Perturbation("molecule", "DrugA", 0))) == 0


def test_prediction_matches_legacy_formula(model, controls):
    p = Perturbation("crispri", "TP53")
    with torch.no_grad():
        x = torch.from_numpy(controls)
        latent = -model.perturbation_embeddings_mlp(model.gene_embeddings_mlp(
            model.gene_embeddings[0])).reshape(1, -1).repeat(len(x), 1)
        decoded = model.mask_cell_decoder(torch.cat([model.gene_encoder(x), latent], 1))
        delta = model.mask_cell_expression(decoded).clamp(-np.log1p(1e4), np.log1p(1e4))
        counts = np.expm1(torch.relu(x + delta).numpy())
        expected = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    np.testing.assert_array_equal(model.predict(controls, p), expected)


def test_alias_average_and_unmatched_control_padding(model, controls):
    original_in = model.gene_encoder.layers[0].weight.detach().clone()
    original_out = model.mask_cell_expression.layers[-1].weight.detach().clone()
    model.align_genes([["ENSG2", "ENSG1"], ["ENSG3"], ["UNKNOWN"]])
    torch.testing.assert_close(model.gene_encoder.layers[0].weight[:, 0],
                               original_in[:, [1, 0]].mean(1))
    torch.testing.assert_close(model.mask_cell_expression.layers[-1].weight[0],
                               original_out[[1, 0]].mean(0))
    np.testing.assert_array_equal(model.coverage, [True, True, False])
    predicted = model.predict(controls, Perturbation("crispri", "TP53"))
    np.testing.assert_array_equal(predicted[:, 2], controls[:, 2])


def test_bad_gene_alignment_is_atomic(model):
    old = model.gene_encoder.layers[0].weight.detach().clone()
    with pytest.raises(ValueError, match="unmatched"):
        model.align_genes(["ENSG1", "MISSING"], unknown_gene="error")
    torch.testing.assert_close(model.gene_encoder.layers[0].weight, old)
    with pytest.raises(ValueError, match="match"):
        model.align_genes(["MISSING"])


def test_batch_size_preserves_predictions(model, controls):
    p = Perturbation("cytokine", "IFNG")
    np.testing.assert_allclose(model.predict(controls, p, batch_size=1),
                               model.predict(controls, p, batch_size=256), atol=1e-6)


def test_unknown_perturbation_is_explicit(model, controls):
    p = Perturbation("crispri", "UNKNOWN")
    with pytest.raises(ValueError, match="Unknown"):
        model.predict(controls, p)
    np.testing.assert_array_equal(model.predict(controls, p, unknown_perturbation="control"),
                                  controls)
    with pytest.raises(ValueError, match="Unknown"):
        model.predict(controls, [p, Perturbation("crispra", "TP53")])


@pytest.mark.parametrize("kind,name,dose", [
    ("bad", "TP53", None), ("molecule", "DrugA", None),
    ("molecule", "DrugA", -1), ("molecule", "DrugA", float("nan")),
    ("crispri", "TP53", 10), ("crispri", "", None),
])
def test_invalid_conditions_fail(kind, name, dose):
    with pytest.raises(ValueError):
        Perturbation(kind, name, dose)


@pytest.mark.parametrize("value", [float("nan"), -1., float("inf")])
def test_invalid_controls_fail(model, controls, value):
    controls[0, 0] = value
    with pytest.raises(ValueError):
        model.predict(controls, Perturbation("crispri", "TP53"))


def test_empty_or_wrong_shape_controls_fail(model):
    for x in [np.zeros((0, 3)), np.zeros((2, 4)), np.zeros(3)]:
        with pytest.raises(ValueError):
            model.predict(x, Perturbation("crispri", "TP53"))


def test_no_training_dependencies(model):
    assert isinstance(model, nn.Module)
    assert not hasattr(model, "training_step")
