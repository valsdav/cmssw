"""Compare the trackster graphs of real events with the Python graph builder.

Inputs:
  - a file with the legacy barrel tracksters (ticlTrackstersCLUE3DBarrel) and the barrel properties
    (ticlTracksterBarrelProperties), as written by the barrel SoA test job;
  - the log of the same job, with the graphs printed by TracksterGraphValidator (dump = True).

For each event, eta, phi, et and selected are computed from the legacy tracksters with the float32 operations of
TracksterGraphAlgo::prepareInputs, the Python builder of compare_graph_builder.py makes the graphs, and they are
compared graph by graph with the graphs of the job.

Usage: python3 compare_graph_events.py barrelSoA.root job.log
"""
import re
import sys

import numpy as np
from DataFormats.FWLite import Events, Handle

from compare_graph_builder import F, build_graphs


def read_job_blocks(path):
    """{(run, lumi, event): (G rows, N rows, E rows)} from the TracksterGraphValidator dump, in the row order"""
    blocks = {}
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        m = re.search(r"event (\d+):(\d+):(\d+): \d+ graphs", lines[i])
        if not m or i + 1 >= len(lines) or not lines[i + 1].startswith("EVENT "):
            i += 1
            continue
        key = tuple(int(x) for x in m.groups())
        nG, nN, nE = map(int, lines[i + 1].split()[1:])
        rows = lines[i + 2:i + 2 + nG + nN + nE]
        blocks[key] = ([list(map(int, r.split()[1:])) for r in rows[:nG]],
                       [list(map(int, r.split()[1:])) for r in rows[nG:nG + nN]],
                       [list(map(int, r.split()[1:])) for r in rows[nG + nN:]])
        i += 2 + nG + nN + nE
    return blocks


def read_job_graphs(path):
    """{(run, lumi, event): {graph: (nodes, sorted edges)}} from the TracksterGraphValidator dump"""
    return {key: {g: ({t for t, _ in N[no:no + nn]}, sorted((e[0], e[1]) for e in E[eo:eo + ne]))
                  for g, (no, nn, eo, ne) in enumerate(G)}
            for key, (G, N, E) in read_job_blocks(path).items()}


def inputs_from_tracksters(tracksters, properties):
    """eta, phi, et, selected with the float32 operations of KernelInputs"""
    n = tracksters.size()
    x = np.array([tracksters[i].barycenter().x() for i in range(n)], F)
    y = np.array([tracksters[i].barycenter().y() for i in range(n)], F)
    z = np.array([tracksters[i].barycenter().z() for i in range(n)], F)
    energy = np.array([tracksters[i].raw_energy() for i in range(n)], F)
    rho = np.sqrt(x * x + y * y).astype(F)
    r = np.sqrt(rho * rho + z * z).astype(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(rho > 0, np.where(z >= 0, np.log((z + r) / rho), -np.log((r - z) / rho)), 0).astype(F)
        et = np.where(r > 0, energy * rho / r, 0).astype(F)
    phi = np.arctan2(y, x).astype(F)
    selected = np.array([properties[i].nEcalClusters() >= 1 for i in range(n)], bool)
    return eta, phi, et, selected


def main():
    job = read_job_graphs(sys.argv[2])
    hT = Handle("std::vector<ticl::io_v1::Trackster>")
    hP = Handle("ticl::TracksterBarrelPropertiesHostCollection")
    n_events = n_same = n_graphs = 0
    for ev in Events(sys.argv[1]):
        aux = ev.eventAuxiliary()
        key = (aux.run(), aux.luminosityBlock(), aux.event())
        if key not in job:
            print(f"event {key}: no graphs in the log")
            continue
        ev.getByLabel("ticlTrackstersCLUE3DBarrel", hT)
        ev.getByLabel("ticlTracksterBarrelProperties", hP)
        tracksters, properties = hT.product(), hP.product().const_view()
        eta, phi, et, selected = inputs_from_tracksters(tracksters, properties)
        sel = np.nonzero(selected)[0]
        ref = build_graphs(eta[sel], phi[sel], et[sel])
        ref = [({int(sel[i]) for i in nodes}, sorted((int(sel[a]), int(sel[b])) for a, b in edges))
               for _, (nodes, edges) in sorted(ref.items())]
        got = [job[key][g] for g in sorted(job[key])]
        n_events += 1
        n_graphs += len(got)
        if got == ref:
            n_same += 1
        else:
            print(f"event {key}: {len(got)} graphs in the job, {len(ref)} in Python; "
                  f"{sum(1 for a, b in zip(got, ref) if a == b)} identical")
        print(f"event {key}: {tracksters.size()} tracksters, {len(sel)} selected, "
              f"{int((et[sel] >= F(1.0)).sum())} seeds, {len(got)} graphs")
    print(f"{n_same} of {n_events} events identical, {n_graphs} graphs in the job")
    sys.exit(0 if n_same == n_events and n_events > 0 else 1)


if __name__ == "__main__":
    main()
