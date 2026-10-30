"""Compare the barrel superclusters of real events (ticlBarrelGraphSuperclusteringTask) with the Python mirror.

Inputs:
  - a file with the legacy barrel tracksters (ticlTrackstersCLUE3DBarrel), the edge scores (ticlBarrelGraphEdgeScore),
    the legacy supercluster tracksters and their links (ticlTracksterLinksSuperclusteringBarrelGraph) and the
    SuperClusters (ticlEGammaSuperClusterProducerBarrel), as written by the barrel SoA test job;
  - the log of the same job, with the graphs printed by TracksterGraphValidator (dump = True): the scores are in the
    order of its edge rows.

For each event:
  - the groups of resolve_groups() (compare_graph_superclusters.py) = linkedTracksterIdToInputTracksterId;
  - each supercluster trackster has the layer clusters of its tracksters, and their raw energy;
  - the SuperClusters are the supercluster tracksters with raw_pt >= superclusterEtThreshold, with the same members.

Usage: python3 compare_superclusters_events.py barrelSoA.root job.log [direct|transitive]
"""
import sys

import numpy as np
from DataFormats.FWLite import Events, Handle

from compare_graph_builder import F, SEED_ET, eta_phi_et
from compare_graph_events import read_job_blocks
from compare_graph_superclusters import LINKING_THRESHOLD, resolve_groups

SUPERCLUSTER_ET_THRESHOLD = 4.0  # superclusterEtThreshold of EGammaSuperclusterProducer


def main():
    job = read_job_blocks(sys.argv[2])
    transitive = len(sys.argv) > 3 and sys.argv[3] == "transitive"
    hT = Handle("std::vector<ticl::io_v1::Trackster>")
    hS = Handle("ticl::TracksterGraphEdgeScoresHostCollection")
    hSC = Handle("std::vector<ticl::io_v1::Trackster>")
    hL = Handle("std::vector<std::vector<unsigned int>>")
    hEG = Handle("std::vector<reco::SuperCluster>")
    n_events = n_good = n_groups = n_multi = n_superclusters = 0
    for ev in Events(sys.argv[1]):
        aux = ev.eventAuxiliary()
        key = (aux.run(), aux.luminosityBlock(), aux.event())
        if key not in job:
            print(f"event {key}: no graphs in the log")
            continue
        ev.getByLabel("ticlTrackstersCLUE3DBarrel", hT)
        ev.getByLabel("ticlBarrelGraphEdgeScore", hS)
        ev.getByLabel("ticlTracksterLinksSuperclusteringBarrelGraph", hSC)
        ev.getByLabel("ticlTracksterLinksSuperclusteringBarrelGraph", "linkedTracksterIdToInputTracksterId", hL)
        ev.getByLabel("ticlEGammaSuperClusterProducerBarrel", hEG)
        tracksters, superclusters = hT.product(), hSC.product()
        links = [list(link) for link in hL.product()]
        scores = hS.product().const_view()
        G, N, E = job[key]

        n = tracksters.size()
        col = lambda f: np.array([f(tracksters[i]) for i in range(n)], F)  # noqa: E731
        _, _, et, _ = eta_phi_et(col(lambda t: t.raw_energy()), col(lambda t: t.barycenter().x()),
                                 col(lambda t: t.barycenter().y()), col(lambda t: t.barycenter().z()))
        p = [F(scores[e].egmLinkingProbability()) for e in range(scores.metadata().size())]
        expected = resolve_groups(et, [t for t, _ in N], [(a, b) for a, b, *_ in E], p, SEED_ET, LINKING_THRESHOLD,
                                  transitive)

        problems = []
        if links != expected:
            problems.append(f"links differ: {sum(1 for a, b in zip(links, expected) if a != b)} of "
                            f"{max(len(links), len(expected))} groups")
        if superclusters.size() != len(links):
            problems.append(f"{superclusters.size()} supercluster tracksters for {len(links)} groups")
        for g, members in enumerate(links[:superclusters.size()]):
            sc = superclusters[g]
            lcs = sorted(lc for t in members for lc in tracksters[t].vertices())
            if sorted(sc.vertices()) != lcs:
                problems.append(f"group {g}: layer clusters differ")
            energy = sum(tracksters[t].raw_energy() for t in members)
            if abs(sc.raw_energy() - energy) > 1e-4 * max(1.0, energy):
                problems.append(f"group {g}: raw energy {sc.raw_energy()} vs {energy}")
        passing = [g for g in range(superclusters.size()) if superclusters[g].raw_pt() >= SUPERCLUSTER_ET_THRESHOLD]
        if hEG.product().size() != len(passing):
            problems.append(f"{hEG.product().size()} SuperClusters, {len(passing)} supercluster tracksters above "
                            f"{SUPERCLUSTER_ET_THRESHOLD} GeV")
        else:
            for s, g in zip(hEG.product(), passing):
                if s.clustersSize() != len(links[g]):
                    problems.append(f"group {g}: {s.clustersSize()} clusters in the SuperCluster, {len(links[g])} "
                                    "tracksters")

        n_events += 1
        n_good += not problems
        n_groups += len(links)
        n_multi += sum(1 for link in links if len(link) > 1)
        n_superclusters += hEG.product().size()
        print(f"event {key}: {len(G)} graphs, {len(links)} superclusters "
              f"({sum(1 for link in links if len(link) > 1)} with more than one trackster), "
              f"{hEG.product().size()} SuperClusters" + ("" if not problems else "  " + "; ".join(problems[:5])))
    print(f"{n_good} of {n_events} events with the same superclusters "
          f"({n_groups} superclusters, {n_multi} with more than one trackster, {n_superclusters} SuperClusters)")
    sys.exit(0 if n_good == n_events and n_events > 0 else 1)


if __name__ == "__main__":
    main()
