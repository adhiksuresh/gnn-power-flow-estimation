"""Training loop with early stopping, and evaluation in physical units."""

import copy

import numpy as np
import torch


class Standardiser:
    def fit(self, arr):
        flat = arr.reshape(-1, arr.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0) + 1e-8
        return self

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse(self, arr):
        return arr * self.std + self.mean


def to_tensors(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


def fit(model, train, val, epochs=300, lr=1e-3, batch=128, patience=40, seed=0):
    """train/val are tuples (x, a_hat, status, y) of tensors with standardised x and y."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best, best_state, wait, history = np.inf, None, 0, []
    n = len(train[0])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x, a, s, y = (t[idx] for t in train)
            opt.zero_grad()
            loss = loss_fn(model(x, a, s), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(*val[:3]), val[3]).item()
        history.append(v)
        if v < best:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return np.array(history)


def predict(model, x, a, s, y_scaler):
    model.eval()
    with torch.no_grad():
        out = model(x, a, s).numpy()
    return y_scaler.inverse(out)


def errors(y_true, y_pred, mask=None):
    """Mean absolute error of voltage magnitude (p.u.) and angle (degrees)."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    err = np.abs(y_true - y_pred)
    return {"vm_MAE_pu": float(err[..., 0].mean()), "va_MAE_deg": float(err[..., 1].mean()),
            "va_max_err_deg": float(err[..., 1].max())}
