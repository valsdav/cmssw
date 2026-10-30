"""Compare the edge scores of real events (TICLBarrelGraphEdgeScoreProducer) with the TorchScript model in Python.

Inputs:
  - a file with the legacy barrel tracksters (ticlTrackstersCLUE3DBarrel), the barrel properties
    (ticlTracksterBarrelProperties) and the edge scores (ticlBarrelGraphEdgeScore), as written by the barrel SoA test
    job;
  - the log of the same job, with the graphs printed by TracksterGraphValidator (dump = True): the scores are in the
    order of its edge rows;
  - the TorchScript model of the job.

For each event, the node features are computed from the file with node_features() of compare_edge_scores.py, the
model runs in Python on the graphs of the job, and the scores are compared edge by edge.

Usage: python3 compare_edge_scores_events.py barrelSoA.root job.log model.pt
"""
import sys

import numpy as np
import torch
from DataFormats.FWLite import Events, Handle

from compare_edge_scores import MAX_NODES, node_features, python_scores
from compare_graph_builder import F
from compare_graph_events import read_job_blocks

PROPERTIES = {"nEcal": "nEcalClusters", "nHcal": "nHcalClusters", "ecalX": "ecalX",
              "ecalY": "ecalY", "ecalZ": "ecalZ", "ecalEta": "ecalEta", "ecalPhi": "ecalPhi",
              "hcalX": "hcalBarycenterX", "hcalY": "hcalBarycenterY",
              "hcalZ": "hcalBarycenterZ", "hcalEta": "hcalBarycenterEta", "hcalPhi": "hcalBarycenterPhi"}


def event_columns(tracksters, properties):
    n = tracksters.size()
    ev = {"rawEnergy": np.array([tracksters[i].raw_energy() for i in range(n)], F),
          "rawEmEnergy": np.array([tracksters[i].raw_em_energy() for i in range(n)], F),
          "x": np.array([tracksters[i].barycenter().x() for i in range(n)], F),
          "y": np.array([tracksters[i].barycenter().y() for i in range(n)], F),
          "z": np.array([tracksters[i].barycenter().z() for i in range(n)], F)}
    for name, column in PROPERTIES.items():
        ev[name] = np.array([getattr(properties[i], column)() for i in range(n)],
                            np.int64 if name.startswith("n") else F)
    return ev


def main():
    job = read_job_blocks(sys.argv[2])
    torch.backends.mha.set_fastpath_enabled(False)
    model = torch.jit.load(sys.argv[3]).eval()
    hT = Handle("std::vector<ticl::io_v1::Trackster>")
    hP = Handle("ticl::TracksterBarrelPropertiesHostCollection")
    hS = Handle("ticl::TracksterGraphEdgeScoresHostCollection")
    n_events = n_good = 0
    for ev in Events(sys.argv[1]):
        aux = ev.eventAuxiliary()
        key = (aux.run(), aux.luminosityBlock(), aux.event())
        if key not in job:
            print(f"event {key}: no graphs in the log")
            continue
        ev.getByLabel("ticlTrackstersCLUE3DBarrel", hT)
        ev.getByLabel("ticlTracksterBarrelProperties", hP)
        ev.getByLabel("ticlBarrelGraphEdgeScore", hS)
        G, N, E = job[key]
        scores = hS.product().const_view()
        got = np.array([scores[e].egmLinkingProbability() for e in range(scores.metadata().size())], np.float32)
        expected = python_scores(model, node_features(event_columns(hT.product(), hP.product().const_view())),
                                 G, N, E)
        n_events += 1
        ok = len(got) == len(expected) and np.array_equal(got == -1, expected == -1)
        worst = float(np.abs(got - expected).max()) if ok and len(got) else 0.0
        ok = ok and worst < 1e-5
        n_good += ok
        n_big = sum(1 for g in G if g[1] > MAX_NODES)
        print(f"event {key}: {len(G)} graphs ({n_big} above {MAX_NODES} nodes), {len(got)} scores "
              f"({int((got == -1).sum())} = -1), {len(expected)} expected, max |diff| {worst:.2e}"
              f"{'' if ok else '  DIFFERENT'}")
    print(f"{n_good} of {n_events} events with the same scores")
    sys.exit(0 if n_good == n_events and n_events > 0 else 1)


if __name__ == "__main__":
    main()
