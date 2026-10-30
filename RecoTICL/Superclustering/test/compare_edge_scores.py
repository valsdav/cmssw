"""Compare the edge score chain of TICLBarrelGraphEdgeScoreProducer (via testTracksterGraphEdgeScore) with Python.

Random barrel events -> graphs of the Python graph builder -> both chains:
  C++    : fillNodeFeatures + model inputs (alpaka) -> TorchScript model (PyTorchAlpaka) -> gather + sigmoid
  Python : node_features() (numpy float32 mirror of fillNodeFeatures) -> dense inputs -> the same TorchScript model
and an edge-by-edge comparison of the scores, also of which edges get -1 (graphs with more than MAX_NODES nodes).
It checks the C++ features, the input layout, the tensor wrapping and the gather; the export itself is validated by
ticl-egm-linking/scripts/export_cmssw.py.

Usage: BUILDER=path/to/testTracksterGraphEdgeScoreSerialSync MODEL=model.pt python3 compare_edge_scores.py [seed]
"""
import os
import subprocess
import sys

import numpy as np
import torch

from compare_graph_builder import F, build_graphs, eta_phi_et

MAX_NODES = 50  # = kMaxNodes in TracksterGraphEdgeModelSoA.h
PLACEHOLDER = F(-1.0)
PROPERTY_COLUMNS = ["nEcal", "nHcal", "ecalX", "ecalY", "ecalZ", "ecalEta", "ecalPhi",
                    "hcalX", "hcalY", "hcalZ", "hcalEta", "hcalPhi"]


def node_features(ev):
    """Python mirror of fillNodeFeatures() in RecoTICL/Superclustering/interface/TracksterGraphEdgeFeatures.h"""
    eta, _, et, rho = eta_phi_et(ev["rawEnergy"], ev["x"], ev["y"], ev["z"])
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_phi = np.where(rho > 0, ev["y"] / rho, F(0)).astype(F)
        cos_phi = np.where(rho > 0, ev["x"] / rho, F(1)).astype(F)
    hcal_energy = (ev["rawEnergy"] - ev["rawEmEnergy"]).astype(F)
    has_hcal = (ev["nHcal"] > 0) & (hcal_energy > 0)
    dphi = (ev["hcalPhi"] - ev["ecalPhi"]).astype(F)
    dphi = np.where(dphi >= F(np.pi), dphi - F(2 * np.pi), np.where(dphi < -F(np.pi), dphi + F(2 * np.pi), dphi))
    deta = (ev["hcalEta"] - ev["ecalEta"]).astype(F)
    dz = np.abs(ev["ecalZ"] - ev["hcalZ"]).astype(F)
    return np.stack([ev["rawEmEnergy"], et, eta, sin_phi, cos_phi, ev["nHcal"].astype(F), hcal_energy,
                     np.where(has_hcal, deta, PLACEHOLDER), np.where(has_hcal, dphi.astype(F), PLACEHOLDER),
                     np.where(has_hcal, dz, PLACEHOLDER)], axis=1).astype(F)


def _xyz_eta_phi(rho, eta, phi):
    x, y, z = (rho * np.cos(phi)).astype(F), (rho * np.sin(phi)).astype(F), (rho * np.sinh(eta)).astype(F)
    return x, y, z, eta.astype(F), np.arctan2(y, x).astype(F)


def make_event(rng, n, mode):
    eta = rng.uniform(-1.48, 1.48, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    if mode == "dense":  # everything in a small region -> graphs above MAX_NODES
        eta = rng.uniform(0.2, 0.5, n)
        phi = rng.uniform(0.0, 0.3, n)
    ev = {"rawEmEnergy": rng.exponential(5.0, n).astype(F)}
    ev["x"], ev["y"], ev["z"], _, _ = _xyz_eta_phi(np.full(n, 140.0), eta, phi)
    ev["nEcal"] = (rng.uniform(0, 1, n) < 0.9).astype(np.int64)
    ev["nHcal"] = np.where(rng.uniform(0, 1, n) < 0.5, rng.integers(1, 4, n), 0)
    ev["ecalX"], ev["ecalY"], ev["ecalZ"], ev["ecalEta"], ev["ecalPhi"] = _xyz_eta_phi(
        np.full(n, 130.0), eta + rng.normal(0, 0.01, n), phi + rng.normal(0, 0.01, n))
    hcal_energy = np.where(ev["nHcal"] > 0, rng.exponential(2.0, n), 0.0).astype(F)
    hcal_energy[rng.uniform(0, 1, n) < 0.05] = 0  # HCAL clusters with no energy: placeholder too
    ev["rawEnergy"] = (ev["rawEmEnergy"] + hcal_energy).astype(F)  # the HCAL energy is rawEnergy - rawEmEnergy
    hcal_phi = phi + rng.normal(0, 0.05, n)
    hcal_phi = np.where(rng.uniform(0, 1, n) < 0.05, np.pi - phi, hcal_phi)  # some differences across +-pi
    ev["hcalX"], ev["hcalY"], ev["hcalZ"], ev["hcalEta"], ev["hcalPhi"] = _xyz_eta_phi(
        np.full(n, 190.0), eta + rng.normal(0, 0.05, n), hcal_phi)
    for c in ("hcalX", "hcalY", "hcalZ", "hcalEta", "hcalPhi"):  # as TracksterBarrelPropertiesAlgoWrapper
        ev[c] = np.where(hcal_energy > 0, ev[c], F(0)).astype(F)
    return ev


def graph_blocks(ev):
    """The graphs of the Python builder in the TracksterGraphSoA layout: (G rows, N rows, E rows)"""
    eta, phi, et, _ = eta_phi_et(ev["rawEnergy"], ev["x"], ev["y"], ev["z"])
    sel = np.nonzero(ev["nEcal"] >= 1)[0]
    G, N, E = [], [], []
    graphs = build_graphs(eta[sel], phi[sel], et[sel])
    for g in sorted(graphs):
        nodes = sorted(int(sel[i]) for i in graphs[g][0])
        local = {t: i for i, t in enumerate(nodes)}
        edges = sorted((local[int(sel[a])], local[int(sel[b])]) for a, b in graphs[g][1])
        G.append((len(N), len(nodes), len(E), len(edges)))
        N += [(t, g) for t in nodes]
        E += [(nodes[a], nodes[b], g, a, b) for a, b in edges]
    return G, N, E


def python_scores(model, features, G, N, E):
    nf = np.zeros((len(G), MAX_NODES, features.shape[1]), np.float32)
    adj = np.zeros((len(G), MAX_NODES, MAX_NODES), np.float32)
    for g, (node_off, n_nodes, _, _) in enumerate(G):
        if n_nodes > MAX_NODES:
            adj[g, 0, 0] = 1
            continue
        nf[g, :n_nodes] = features[[t for t, _ in N[node_off:node_off + n_nodes]]]
        adj[g, np.arange(n_nodes), np.arange(n_nodes)] = 1
    for _, _, g, a, b in E:
        if G[g][1] <= MAX_NODES:
            adj[g, a, b] = 1
    if not E:
        return np.zeros(0, np.float32)
    with torch.no_grad():
        # lists, not torch.from_numpy: the CMSSW torch is built without NumPy support
        logits = np.array(model(torch.tensor(nf.tolist()), torch.tensor(adj.tolist())).tolist(), np.float32)
    return np.array([1 / (1 + np.exp(-logits[g, a, b])) if G[g][1] <= MAX_NODES else -1.0
                     for _, _, g, a, b in E], np.float32)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = np.random.default_rng(seed)
    torch.backends.mha.set_fastpath_enabled(False)
    model = torch.jit.load(os.environ["MODEL"]).eval()

    events = [make_event(rng, n, "random") for n in (0, 1, 3)] + [make_event(rng, 300, "dense")]
    events += [make_event(rng, int(rng.integers(20, 400)), "random") for _ in range(12)]

    lines = [str(len(events))]
    expected = []
    for ev in events:
        n = len(ev["x"])
        lines.append(str(n))
        cols = [ev["rawEnergy"], ev["rawEmEnergy"], ev["x"], ev["y"], ev["z"]] + [ev[c] for c in PROPERTY_COLUMNS]
        lines += [" ".join(repr(float(v)) if isinstance(v, np.floating) else str(int(v)) for v in row)
                  for row in zip(*cols)]
        G, N, E = graph_blocks(ev)
        lines.append(f"{len(G)} {len(N)} {len(E)}")
        lines += [" ".join(map(str, row)) for row in G + N + E]
        expected.append(python_scores(model, node_features(ev), G, N, E))

    for batch in ("0", "32"):
        proc = subprocess.run([os.environ.get("BUILDER", "testTracksterGraphEdgeScoreSerialSync"), os.environ["MODEL"],
                               batch], input="\n".join(lines) + "\n", capture_output=True, text=True)
        if proc.returncode:
            sys.exit(f"builder failed ({proc.returncode}): {proc.stderr[-2000:]}")
        got = [np.asarray(b.split(), np.float32) for b in proc.stdout.split("EVENT")[1:]]
        assert len(got) == len(expected), (len(got), len(expected))

        n_edges = n_skipped = 0
        worst = 0.0
        for i, (e, c) in enumerate(zip(expected, got)):
            assert len(e) == len(c), f"event {i}: {len(c)} scores from C++, {len(e)} expected"
            assert np.array_equal(e == -1, c == -1), f"event {i}: different edges with score -1"
            if len(e):
                worst = max(worst, float(np.abs(e - c).max()))
            n_edges += len(e)
            n_skipped += int((e == -1).sum())
        print(f"graphsPerBatch {batch}: {len(events)} events, {n_edges} edges, {n_skipped} in graphs above "
              f"{MAX_NODES} nodes (-1); max |score C++ - score Python| = {worst:.2e}")
        assert n_skipped > 0 and n_edges > n_skipped, "the events must have scored and not scored edges"
        assert worst < 1e-5, "C++ and Python scores differ"
    print("OK")


if __name__ == "__main__":
    main()
