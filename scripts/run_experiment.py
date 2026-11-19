"""Train and compare GCN, MLP and MLP+status models; evaluate against DC power flow.

Usage:
    python scripts/run_experiment.py            # uses data/dataset.npz, generating it if missing

Test conditions
---------------
* intact network
* seen outages: single-line outages that also appear in training
* unseen outages: outages of lines that NEVER appear out of service in training
  (tests whether a model generalises to a new topology)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnnpf import data, models, train  # noqa: E402

FIG, TAB, DATA = ROOT / "results" / "figures", ROOT / "results" / "tables", ROOT / "data" / "dataset.npz"
SEED = 0


def load_dataset():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        d = data.generate(n_samples=6000, seed=SEED)
        np.savez_compressed(DATA, **d)
    return dict(np.load(DATA))


def split(d, rng):
    """Hold out a quarter of the outage lines entirely for testing."""
    outage = d["outage"]
    lines = np.unique(outage[outage >= 0])
    unseen = rng.choice(lines, size=len(lines) // 4, replace=False)
    is_unseen = np.isin(outage, unseen)
    pool = np.where(~is_unseen)[0]
    rng.shuffle(pool)
    n_test = 1000
    test_idx = np.concatenate([pool[:n_test], np.where(is_unseen)[0]])
    val_idx, train_idx = pool[n_test:n_test + 800], pool[n_test + 800:]
    return train_idx, val_idx, test_idx, unseen


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    d = load_dataset()
    n_bus = d["x"].shape[1]
    tr, va, te, unseen = split(d, rng)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  (unseen outage lines: {sorted(unseen.tolist())})")

    xs = train.Standardiser().fit(d["x"][tr])
    ys = train.Standardiser().fit(d["y"][tr])
    a_hat = data.normalised_adjacency(d["edges"], d["status"], n_bus)

    def tensors(idx):
        return train.to_tensors(xs.transform(d["x"][idx]), a_hat[idx], d["status"][idx], ys.transform(d["y"][idx]))

    T_tr, T_va, T_te = tensors(tr), tensors(va), tensors(te)

    candidates = {
        "MLP (topology-blind)": models.MLP(n_bus=n_bus),
        "MLP + branch status": models.MLP(n_bus=n_bus, n_status=d["status"].shape[1]),
        "GCN": models.GCN(),
    }
    preds, info = {}, {}
    for name, m in candidates.items():
        hist = train.fit(m, T_tr, T_va, seed=SEED)
        preds[name] = train.predict(m, *T_te[:3], ys)
        info[name] = {"parameters": models.count_parameters(m), "epochs": len(hist)}
        print(f"{name:22s} params={info[name]['parameters']:7d}  epochs={len(hist)}")
    preds["DC power flow (physics baseline)"] = d["y_dc"][te]

    y_true = d["y"][te]
    out = d["outage"][te]
    groups = {
        "Intact network": out < 0,
        "Seen outages": (out >= 0) & ~np.isin(out, unseen),
        "Unseen outages": np.isin(out, unseen),
    }
    rows = []
    for name, p in preds.items():
        for g, mask in groups.items():
            e = train.errors(y_true, p, mask)
            rows.append({"model": name, "test_condition": g, "n": int(mask.sum()), **e})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "test_errors.csv", index=False)
    pd.DataFrame(info).T.to_csv(TAB / "model_info.csv")
    (TAB / "run_info.json").write_text(json.dumps({
        "n_train": len(tr), "n_val": len(va), "n_test": len(te), "unseen_outage_lines": sorted(unseen.tolist()),
        "failed_power_flows_skipped": int(d["n_failed"]),
    }, indent=2))

    pivot_va = res.pivot(index="model", columns="test_condition", values="va_MAE_deg")
    pivot_vm = res.pivot(index="model", columns="test_condition", values="vm_MAE_pu")
    print("\nAngle MAE (degrees):\n", pivot_va.round(3), "\n\nVoltage MAE (p.u.):\n", pivot_vm.round(4), sep="")
    _figures(pivot_va, pivot_vm, preds, y_true, groups)


def _figures(pva, pvm, preds, y_true, groups):
    order = list(groups)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pva[order].plot.bar(ax=axes[0], rot=15, logy=True)
    axes[0].set(title="Voltage angle error (lower is better)", ylabel="MAE (degrees, log scale)", xlabel="")
    pvm[order].plot.bar(ax=axes[1], rot=15, logy=True)
    axes[1].set(title="Voltage magnitude error (lower is better)", ylabel="MAE (p.u., log scale)", xlabel="")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(FIG / "errors_by_condition.png", dpi=150); plt.close(fig)

    # per-bus angle error on unseen outages
    mask = groups["Unseen outages"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, p in preds.items():
        ax.plot(np.abs(y_true[mask] - p[mask])[..., 1].mean(0), marker="o", ms=3, label=name)
    ax.set(title="Angle error by bus, unseen outages", xlabel="Bus number", ylabel="MAE (degrees)", yscale="log")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "angle_error_by_bus.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
