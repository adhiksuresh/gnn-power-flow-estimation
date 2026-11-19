"""Generate power-flow samples on the IEEE 30-bus test network with pandapower.

Each sample:
* every load is scaled by a system-wide factor (0.7-1.3) times a local
  factor (0.85-1.15), with reactive power scaled to keep the power factor;
* the main generator's active power output is varied;
* in a fraction of samples one transmission line is taken out of service
  (only outages that keep the network connected and solvable are used);
* a full AC power flow is solved (the "truth") and, for comparison, a
  DC power flow (the classic linear approximation).

Per-bus input features (7):
    P_load, Q_load, P_gen_setpoint, V_setpoint, is_slack, is_PV, is_PQ
Per-bus targets (2):
    voltage magnitude (p.u.), voltage angle (degrees)
"""

import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

FEATURES = ["p_load", "q_load", "p_gen", "v_set", "is_slack", "is_pv", "is_pq"]
TARGETS = ["vm_pu", "va_degree"]


def base_network():
    return pn.case_ieee30()


def branch_list(net):
    """All branches as (kind, index, from_bus, to_bus)."""
    lines = [("line", i, int(r.from_bus), int(r.to_bus)) for i, r in net.line.iterrows()]
    trafos = [("trafo", i, int(r.hv_bus), int(r.lv_bus)) for i, r in net.trafo.iterrows()]
    return lines + trafos


def safe_outage_lines(net):
    """Lines whose single outage keeps every bus connected."""
    safe = []
    for i in net.line.index:
        net.line.at[i, "in_service"] = False
        if len(list(top.connected_components(top.create_nxgraph(net)))) == 1:
            safe.append(int(i))
        net.line.at[i, "in_service"] = True
    return safe


def generate(n_samples=5000, outage_fraction=0.5, seed=0, verbose=True):
    """Return a dict of arrays: x (n, buses, 7), y (n, buses, 2), y_dc (n, buses, 2),
    status (n, branches) with 1 = in service, outage (n,) line index or -1."""
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    net = base_network()
    base_p, base_q = net.load.p_mw.values.copy(), net.load.q_mvar.values.copy()
    base_gen = net.gen.p_mw.values.copy()
    safe = safe_outage_lines(net)
    branches = branch_list(net)
    n_bus = len(net.bus)
    slack_bus = int(net.ext_grid.bus.iloc[0])
    pv_buses = set(net.gen.bus.astype(int))

    xs, ys, ydc, status, outage = [], [], [], [], []
    attempts = 0
    while len(xs) < n_samples:
        attempts += 1
        net.line["in_service"] = True
        scale = rng.uniform(0.7, 1.3) * rng.uniform(0.85, 1.15, len(base_p))
        net.load["p_mw"] = base_p * scale
        net.load["q_mvar"] = base_q * scale
        net.gen["p_mw"] = base_gen * rng.uniform(0.6, 1.4, len(base_gen))
        out = -1
        if rng.random() < outage_fraction:
            out = int(rng.choice(safe))
            net.line.at[out, "in_service"] = False
        try:
            pp.runpp(net, numba=False)
        except Exception:
            continue  # skip cases where the AC power flow does not converge
        y = net.res_bus[TARGETS].values.copy()
        pp.rundcpp(net, numba=False)
        y_dc = np.column_stack([np.ones(n_bus), net.res_bus["va_degree"].values])

        x = np.zeros((n_bus, len(FEATURES)))
        np.add.at(x[:, 0], net.load.bus.values, net.load.p_mw.values)
        np.add.at(x[:, 1], net.load.bus.values, net.load.q_mvar.values)
        np.add.at(x[:, 2], net.gen.bus.values, net.gen.p_mw.values)
        x[net.gen.bus.values, 3] = net.gen.vm_pu.values
        x[slack_bus, 3] = net.ext_grid.vm_pu.iloc[0]
        x[slack_bus, 4] = 1
        for b in range(n_bus):
            if b in pv_buses:
                x[b, 5] = 1
            elif b != slack_bus:
                x[b, 6] = 1

        st = np.array([0 if (k == "line" and i == out) else 1 for k, i, _, _ in branches], dtype=float)
        xs.append(x); ys.append(y); ydc.append(y_dc); status.append(st); outage.append(out)
        if verbose and len(xs) % 1000 == 0:
            print(f"  {len(xs)} samples")

    return {
        "x": np.array(xs), "y": np.array(ys), "y_dc": np.array(ydc),
        "status": np.array(status), "outage": np.array(outage),
        "edges": np.array([(f, t) for _, _, f, t in branches]), "n_failed": attempts - n_samples,
    }


def normalised_adjacency(edges, status, n_bus):
    """Batch of GCN propagation matrices D^-1/2 (A + I) D^-1/2, one per sample,
    built from the branches that are in service in that sample."""
    n = len(status)
    a = np.zeros((n, n_bus, n_bus))
    for k, (f, t) in enumerate(edges):
        a[:, f, t] = np.maximum(a[:, f, t], status[:, k])
        a[:, t, f] = a[:, f, t]
    a += np.eye(n_bus)[None]
    d = a.sum(-1) ** -0.5
    return d[:, :, None] * a * d[:, None, :]
