import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnnpf import data, models  # noqa: E402

warnings.simplefilter("ignore")


def test_small_dataset_shapes_and_physics():
    d = data.generate(n_samples=6, outage_fraction=0.5, seed=1, verbose=False)
    assert d["x"].shape == (6, 30, 7)
    assert d["y"].shape == (6, 30, 2)
    slack = d["x"][0, :, 4].argmax()
    assert np.allclose(d["y"][:, slack, 1], 0.0)          # slack angle is the reference
    assert ((d["y"][..., 0] > 0.85) & (d["y"][..., 0] < 1.15)).all()  # plausible voltages
    assert np.allclose(d["x"][..., 4:].sum(-1), 1)        # exactly one bus type per bus


def test_outage_removes_exactly_one_branch():
    d = data.generate(n_samples=10, outage_fraction=1.0, seed=2, verbose=False)
    assert (d["status"].sum(1) == d["status"].shape[1] - 1).all()


def test_adjacency_is_symmetric_and_normalised():
    net = data.base_network()
    edges = np.array([(f, t) for _, _, f, t in data.branch_list(net)])
    status = np.ones((1, len(edges)))
    a = data.normalised_adjacency(edges, status, len(net.bus))[0]
    assert np.allclose(a, a.T)
    assert np.all(np.linalg.eigvalsh(a) <= 1 + 1e-9)


def test_gcn_is_permutation_equivariant():
    torch.manual_seed(0)
    g = models.GCN(layers=2)
    x = torch.randn(1, 5, 7)
    a = torch.rand(1, 5, 5); a = (a + a.transpose(1, 2)) / 2
    perm = torch.randperm(5)
    out = g(x, a)
    out_p = g(x[:, perm], a[:, perm][:, :, perm])
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)
