"""Compare TracksterGraphAlgo (via testTracksterGraphAlgo) with the Python graph builder.

Random events -> both builders -> graph-by-graph comparison of the node sets and the edge lists, plus layout checks
(ascending nodes, edges sorted by (localA, localB), consistent local indices).
The windows are those of TracksterGraphProducer: polynomials in |eta_seed|, mirrored for negative eta.

Usage: BUILDER=path/to/testTracksterGraphAlgoSerialSync python3 compare_graph_builder.py [seed]
"""
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

F = np.float32
# ---- configuration: the defaults of TracksterGraphProducer -----------------------------------------------------
SEED_ET = F(1.0)
DETA_LOW = [-0.08]
DETA_HIGH = [0.08, 0.0, 0.1 / (1.48 * 1.48)]
DPHI_MAX = [1.20, 0.25 / 1.5]


def polynomial(coefficients, x):
    """sum_k c[k] x^k with the Horner method in float32, as TracksterGraphParams.h"""
    c = [F(v) for v in coefficients]
    value = np.full_like(x, c[-1], dtype=F)
    for ck in reversed(c[:-1]):
        value = (value * x + ck).astype(F)
    return value


def windows_for_eta(seed_eta):
    a = np.abs(seed_eta).astype(F)
    return polynomial(DETA_LOW, a), polynomial(DETA_HIGH, a), polynomial(DPHI_MAX, a)


def mirrored_deta(eta_t, eta_seed):
    """(eta_t - eta_seed) * sign(eta_seed), with the same float operations as TracksterGraphAlgo"""
    return np.where(eta_seed < 0, eta_seed - eta_t, eta_t - eta_seed).astype(F)


def delta_phi(a, b):
    d = (a - b).astype(F)
    d = np.where(d > F(np.pi), d - F(2 * np.pi), d)
    d = np.where(d <= -F(np.pi), d + F(2 * np.pi), d)
    return d


def _connected_components(n, edges_i, edges_j):
    adj = [[] for _ in range(n)]
    for a, b in zip(edges_i, edges_j):
        adj[a].append(b)
        adj[b].append(a)
    labels = np.full(n, -1, dtype=np.int64)
    current = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if labels[y] == -1:
                    labels[y] = current
                    stack.append(y)
        current += 1
    return labels


def build_graphs(eta, phi, et):
    """The graph builder of the training, returning {label: (nodes set, edge list)}."""
    n = len(eta)
    if n == 0:
        return {}
    seed_mask = et >= SEED_ET
    seed_idx = np.nonzero(seed_mask)[0]
    n_seeds = len(seed_idx)
    if n_seeds == 0:
        return {}
    seed_eta = eta[seed_idx]
    seed_phi = phi[seed_idx]
    lo, hi, dphi_max = windows_for_eta(seed_eta)

    a, b = np.triu_indices(n_seeds, k=1)
    connected = np.zeros(len(a), bool)
    if len(a):
        deta_a = mirrored_deta(seed_eta[b], seed_eta[a])  # b in the window of a
        deta_b = mirrored_deta(seed_eta[a], seed_eta[b])  # a in the window of b
        abs_dphi_ab = np.abs(delta_phi(seed_phi[b], seed_phi[a]))
        a_sees_b = (deta_a > lo[a]) & (deta_a < hi[a]) & (abs_dphi_ab < dphi_max[a])
        b_sees_a = (deta_b > lo[b]) & (deta_b < hi[b]) & (abs_dphi_ab < dphi_max[b])
        connected = a_sees_b | b_sees_a
    comp = _connected_components(n_seeds, a[connected], b[connected])

    deta_mat = mirrored_deta(eta[np.newaxis, :], seed_eta[:, np.newaxis])
    dphi_mat = np.abs(delta_phi(phi[np.newaxis, :], seed_phi[:, np.newaxis]))
    in_win = (deta_mat > lo[:, np.newaxis]) & (deta_mat < hi[:, np.newaxis]) & (dphi_mat < dphi_max[:, np.newaxis])
    in_win &= ~seed_mask[np.newaxis, :]
    s_local, foll_global = np.nonzero(in_win)

    g_nodes = defaultdict(set)
    g_edges = defaultdict(list)
    comp_list = comp.tolist()
    seed_idx_list = seed_idx.tolist()
    for s in range(n_seeds):
        g_nodes[comp_list[s]].add(seed_idx_list[s])
    ca, cb = a[connected], b[connected]
    for k in range(len(ca)):
        lab = comp_list[ca[k]]
        si, sj = seed_idx_list[ca[k]], seed_idx_list[cb[k]]
        g_edges[lab].append((si, sj))
        g_edges[lab].append((sj, si))
    s_comp = comp[s_local].tolist()
    s_global = seed_idx[s_local].tolist()
    foll_list = foll_global.tolist()
    for i in range(len(s_local)):
        lab, seed, foll = s_comp[i], s_global[i], foll_list[i]
        g_edges[lab].append((seed, foll))
        g_edges[lab].append((foll, seed))
        g_nodes[lab].add(foll)
    return {lab: (g_nodes[lab], g_edges[lab]) for lab in g_nodes}


def make_event(rng, n, mode):
    eta = rng.uniform(-1.5, 1.5, n).astype(F)
    phi = rng.uniform(-np.pi, np.pi, n).astype(F)
    et = rng.exponential(1.0, n).astype(F)
    selected = (rng.uniform(0, 1, n) < 0.8).astype(np.int32)
    if mode == "allseeds":
        et[:] = 10
    if mode == "noseeds":
        et[:] = 0.1
    if mode == "dense":  # everything in a small eta-phi region -> big graphs
        eta = rng.uniform(0.8, 1.1, n).astype(F)
        phi = rng.uniform(0.0, 0.3, n).astype(F)
    if mode == "phiwrap":  # cluster across phi = +-pi
        phi = (np.pi + rng.uniform(-0.2, 0.2, n)).astype(F)
        phi = np.where(phi > np.pi, phi - 2 * np.pi, phi).astype(F)
    if mode == "etazero":  # seeds on both sides of eta = 0: the windows are mirrored
        eta = rng.uniform(-0.15, 0.15, n).astype(F)
    if mode == "forward":  # beyond the eta bins of TracksterGraphAlgo: the edge bins collect them
        eta = rng.uniform(-4.0, 4.0, n).astype(F)
    if mode == "unselected":
        selected[:] = 0
    return eta, phi, et, selected



def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    events = []
    for n in [0, 1, 2, 5]:
        events.append(make_event(rng, n, "random"))
    events += [make_event(rng, 50, "allseeds"), make_event(rng, 50, "noseeds"), make_event(rng, 50, "unselected"),
               make_event(rng, 300, "dense"), make_event(rng, 200, "phiwrap"), make_event(rng, 200, "etazero"),
               make_event(rng, 300, "forward")]
    for _ in range(40):
        events.append(make_event(rng, int(rng.integers(10, 600)), "random"))

    # ---- run the C++ builder ----------------------------------------------------------------------------------------
    def coefficients_line(c):
        return " ".join([str(len(c))] + [repr(float(F(v))) for v in c])


    lines = [repr(float(SEED_ET)), coefficients_line(DETA_LOW), coefficients_line(DETA_HIGH), coefficients_line(DPHI_MAX),
             str(len(events))]
    for eta, phi, et, selected in events:
        lines.append(str(len(eta)))
        lines += [f"{float(a)!r} {float(b)!r} {float(c)!r} {int(d)}" for a, b, c, d in zip(eta, phi, et, selected)]
    proc = subprocess.run([os.environ.get("BUILDER", "testTracksterGraphAlgoSerialSync")],
                          input="\n".join(lines), capture_output=True, text=True)
    out = proc.stdout
    if proc.returncode:
        sys.exit(f"builder failed ({proc.returncode}): {proc.stderr[-500:]}")

    blocks = out.split("EVENT ")[1:]
    assert len(blocks) == len(events), (len(blocks), len(events))

    n_graphs_tot = n_edges_tot = max_nodes = shared = 0
    for ev, (blk, (eta, phi, et, selected)) in enumerate(zip(blocks, events)):
        rows = blk.strip().split("\n")
        nG, nN, nE = map(int, rows[0].split())
        G = [list(map(int, r.split()[1:])) for r in rows[1:] if r.startswith("G ")]
        N = [list(map(int, r.split()[1:])) for r in rows[1:] if r.startswith("N ")]
        E = [list(map(int, r.split()[1:])) for r in rows[1:] if r.startswith("E ")]
        assert (len(G), len(N), len(E)) == (nG, nN, nE)

        # reference on the selected tracksters, with the indices mapped back to the full collection
        sel = np.nonzero(selected)[0]
        ref = build_graphs(eta[sel], phi[sel], et[sel])
        ref = {lab: ({int(sel[i]) for i in nodes}, sorted((int(sel[x]), int(sel[y])) for x, y in edges))
               for lab, (nodes, edges) in ref.items()}

        assert nG == len(ref), f"event {ev}: {nG} graphs vs {len(ref)} in Python"
        for g, (nodeOff, nNodes, edgeOff, nEdges) in enumerate(G):
            nodes = [t for t, gi in N[nodeOff:nodeOff + nNodes]]
            assert all(gi == g for _, gi in N[nodeOff:nodeOff + nNodes])
            assert nodes == sorted(nodes), "nodes not ascending"
            edges = E[edgeOff:edgeOff + nEdges]
            assert all(e[2] == g for e in edges)
            for tA, tB, _, lA, lB in edges:  # local indices consistent with the node rows
                assert nodes[lA] == tA and nodes[lB] == tB
            assert [(e[3], e[4]) for e in edges] == sorted((e[3], e[4]) for e in edges), "edges not sorted"
            ref_nodes, ref_edges = ref[g]
            assert set(nodes) == ref_nodes, f"event {ev} graph {g}: nodes differ"
            assert sorted((e[0], e[1]) for e in edges) == ref_edges, f"event {ev} graph {g}: edges differ"
            max_nodes = max(max_nodes, nNodes)
        assert sum(g[1] for g in G) == nN and sum(g[3] for g in G) == nE
        counts = defaultdict(int)
        for t, _ in N:
            counts[t] += 1
        shared += sum(1 for c in counts.values() if c > 1)
        n_graphs_tot += nG
        n_edges_tot += nE

    print(f"OK: {len(events)} events, {n_graphs_tot} graphs, {n_edges_tot} edges, "
          f"largest graph {max_nodes} nodes, {shared} tracksters in more than one graph")


if __name__ == "__main__":
    main()
