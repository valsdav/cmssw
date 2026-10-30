"""Compare TracksterGroupAlgo (the superclustering on the graphs, via testTracksterGroupAlgo) with a Python mirror.

Random barrel events -> graphs of the Python graph builder -> random symmetric link probabilities -> both
resolutions, in the "direct" and the "transitive" mode, and a comparison of the groups (exact member lists, the seed
first) and of the layer cluster -> group map.

Usage: BUILDER=path/to/testTracksterGroupAlgoSerialSync python3 compare_graph_superclusters.py [seed]
"""
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

from compare_graph_builder import F, SEED_ET, build_graphs, eta_phi_et

LINKING_THRESHOLD = 0.5


def resolve_groups(et, node_tracksters, edges, p, seed_et, threshold, transitive):
    """Python mirror of KernelResolve + KernelFillGroups in TracksterGroupAlgo.dev.cc.

    et: float32 per trackster; node_tracksters: the trackster of each graph node; edges: (a, b) per edge row, in the
    row order; p: probability per edge row. Returns the groups, the seed first, then the others ascending."""
    can_seed = {t for t in node_tracksters if et[t] >= F(seed_et)}
    seeds = sorted(can_seed, key=lambda t: (-et[t], t))  # descending et, equal et: smaller index first
    out = defaultdict(list)
    for (a, b), prob in zip(edges, p):
        out[a].append((b, prob))
    owner = {}
    groups = []
    for seed in seeds:
        if seed in owner:
            continue
        g = len(groups)
        members = [seed]
        owner[seed] = g
        c = 0
        while c < len(members):
            if c > 0 and not transitive:
                break
            m = members[c]
            c += 1
            if m not in can_seed:  # the chain goes only through seeds
                continue
            for b, prob in out[m]:
                if b not in owner and prob > F(threshold):
                    owner[b] = g
                    members.append(b)
        groups.append([members[0]] + sorted(members[1:]))
    return groups


def graph_blocks(eta, phi, et, selected):
    """The graphs of the Python builder in the TracksterGraphSoA layout: (G rows, N rows, E rows)"""
    sel = np.nonzero(selected)[0]
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


def make_event(rng, n, mode):
    eta = rng.uniform(-1.48, 1.48, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    if mode == "dense":  # one busy region: large graphs, many seed-seed links
        eta = rng.uniform(0.2, 0.4, n)
        phi = rng.uniform(0.0, 0.2, n)
    raw_energy = rng.exponential(3.0, n)
    if mode == "ties" and n >= 4:  # tracksters with the same energy and position: equal et
        for i in range(0, n - 1, 4):
            eta[i + 1], phi[i + 1], raw_energy[i + 1] = eta[i], phi[i], raw_energy[i]
    rho = np.full(n, 140.0)
    ev = {"rawEnergy": raw_energy.astype(F),
          "x": (rho * np.cos(phi)).astype(F), "y": (rho * np.sin(phi)).astype(F), "z": (rho * np.sinh(eta)).astype(F)}
    ev["selected"] = rng.uniform(0, 1, n) < 0.9
    # disjoint layer clusters, 1 to 4 per trackster, in a shuffled order
    sizes = rng.integers(1, 5, n)
    lcs = rng.permutation(int(sizes.sum()) + 5)
    ev["vertices"] = np.split(lcs[:int(sizes.sum())], np.cumsum(sizes)[:-1])
    ev["nLayerClusters"] = len(lcs)
    eta32, phi32, et32, _ = eta_phi_et(ev["rawEnergy"], ev["x"], ev["y"], ev["z"])
    ev["et"] = et32
    G, N, E = graph_blocks(eta32, phi32, et32, ev["selected"])
    # symmetric probabilities: one value per pair, some exactly at the threshold, the large graphs not evaluated
    pair_p = {}
    p = []
    for a, b, g, _, _ in E:
        key = (min(a, b), max(a, b))
        if key not in pair_p:
            u = rng.uniform(0, 1)
            pair_p[key] = F(LINKING_THRESHOLD) if u < 0.05 else F(u)
        p.append(F(-1) if G[g][1] > 50 else pair_p[key])
    if mode == "nolinks":
        p = [F(0.1)] * len(E)
    ev["G"], ev["N"], ev["E"], ev["p"] = G, N, E, p
    return ev


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    events = [make_event(rng, n, "random") for n in (0, 1, 3)]
    events += [make_event(rng, 200, "dense"), make_event(rng, 100, "ties"), make_event(rng, 100, "nolinks")]
    events += [make_event(rng, int(rng.integers(20, 400)), "random") for _ in range(20)]

    lines = [str(len(events))]
    for ev in events:
        lines.append(f"{len(ev['x'])} {ev['nLayerClusters']}")
        for i in range(len(ev["x"])):
            v = ev["vertices"][i]
            lines.append(" ".join([repr(float(ev[c][i])) for c in ("rawEnergy", "x", "y", "z")] + [str(len(v))] +
                                  [str(int(x)) for x in v]))
        lines.append(f"{len(ev['G'])} {len(ev['N'])} {len(ev['E'])}")
        lines += [" ".join(map(str, row)) for row in ev["G"] + ev["N"] + ev["E"]]
        lines.append(" ".join(repr(float(x)) for x in ev["p"]))

    for mode in ("direct", "transitive"):
        proc = subprocess.run([os.environ.get("BUILDER", "testTracksterGroupAlgoSerialSync"), repr(float(SEED_ET)),
                               repr(LINKING_THRESHOLD), mode], input="\n".join(lines) + "\n",
                              capture_output=True, text=True)
        if proc.returncode:
            sys.exit(f"builder failed ({proc.returncode}): {proc.stderr[-2000:]}")
        blocks = proc.stdout.split("EVENT ")[1:]
        assert len(blocks) == len(events), (len(blocks), len(events))

        n_groups = n_multi = n_members = 0
        for i, (blk, ev) in enumerate(zip(blocks, events)):
            rows = blk.strip().split("\n")
            n_g, n_m = map(int, rows[0].split())
            got = [list(map(int, r.split()[1:])) for r in rows[1:1 + n_g]]
            assert all(r.startswith("G") for r in rows[1:1 + n_g])
            map_counts = list(map(int, rows[1 + n_g].split()[1:]))
            lc_map = list(map(int, rows[2 + n_g].split())) if len(rows) > 2 + n_g else []

            expected = resolve_groups(ev["et"], [t for t, _ in ev["N"]], [(a, b) for a, b, *_ in ev["E"]], ev["p"],
                                      SEED_ET, LINKING_THRESHOLD, mode == "transitive")
            assert got == expected, f"{mode}, event {i}: groups differ\nC++    {got}\nPython {expected}"
            assert n_m == sum(len(g) for g in got)

            ref_map = [-1] * ev["nLayerClusters"]
            for g, members in enumerate(expected):
                for t in members:
                    for lc in ev["vertices"][t]:
                        ref_map[lc] = g
            assert map_counts == [len(expected), sum(len(ev["vertices"][t]) for g in expected for t in g)], \
                f"{mode}, event {i}: map counts {map_counts}"
            assert lc_map == ref_map, f"{mode}, event {i}: layer cluster map differs"
            n_groups += n_g
            n_members += n_m
            n_multi += sum(1 for g in got if len(g) > 1)
        print(f"{mode}: OK, {len(events)} events, {n_groups} groups ({n_multi} with more than one trackster), "
              f"{n_members} members")


if __name__ == "__main__":
    main()
