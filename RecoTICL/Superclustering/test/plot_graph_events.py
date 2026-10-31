"""Draw the trackster graphs of real events in the eta-phi plane, one page per event.

Inputs: the output file and the log of the barrel SoA test job, as for compare_graph_events.py (the graphs come
from the TracksterGraphValidator dump, eta, phi, et and selected are computed from the legacy tracksters).

Each page has up to 3 panels. The last 2 need the edge scores and the superclusters in the file.
  1. Input graphs:
     - the tracksters without an ECAL layer cluster (not selected): grey crosses;
     - the selected tracksters that are not in a graph: grey dots;
     - the seeds (filled) and the followers (hollow) in the colour of their graph; a follower in more than one graph
       has a black ring;
     - the edges of each graph, and the window of each seed (mirrored for negative eta, split at phi = +-pi).
  2. Edge scores: each link with the colour and the width of egmLinkingProbability. Grey dashed: not evaluated (-1).
  3. Superclusters: the tracksters of each group, and the accepted links in the group that have a seed at one end.
     Superclusters below the SuperCluster threshold are pale.
The marker area grows with et.

Usage: python3 plot_graph_events.py barrelSoA.root job.log [-o graphs.pdf] [--max-events N]
"""
import argparse
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from compare_graph_builder import SEED_ET, windows_for_eta

LINKING_THRESHOLD = 0.5  # linkingThreshold of TICLBarrelGraphSuperclusterProducer
SUPERCLUSTER_ET_THRESHOLD = 4.0  # superclusterEtThreshold of EGammaSuperclusterProducer


def marker(label, shape="o", face="0.45", edge="k", size=6):
    return Line2D([], [], linestyle="", marker=shape, markerfacecolor=face, markeredgecolor=edge, markersize=size,
                  label=label)


def finish_axes(ax, title, handles):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(title, fontsize=9)
    ax.legend(handles=handles, fontsize=7, loc="upper right")


def edge_segments(eta, phi, a, b):
    """the eta-phi segments of the link (a, b), split at +-pi"""
    return [[(eta[a] + t0 * (eta[b] - eta[a]), p0), (eta[a] + t1 * (eta[b] - eta[a]), p1)]
            for t0, t1, p0, p1 in phi_segments(phi[a], phi[b])]


def phi_segments(phi_a, phi_b):
    """the segment from phi_a to phi_b along the shorter way, split at +-pi: [(t0, t1, phi0, phi1)]"""
    d = phi_b - phi_a
    if d > np.pi:
        d -= 2 * np.pi
    elif d < -np.pi:
        d += 2 * np.pi
    end = phi_a + d
    if -np.pi <= end <= np.pi:
        return [(0.0, 1.0, phi_a, end)]
    edge = np.pi if end > np.pi else -np.pi
    t = (edge - phi_a) / d
    return [(0.0, t, phi_a, edge), (t, 1.0, -edge, end - 2 * edge)]


def draw_window(ax, eta, phi, colour):
    lo, hi, dphi = (float(v[0]) for v in windows_for_eta(np.array([eta], np.float32)))
    eta0, eta1 = (eta + lo, eta + hi) if eta >= 0 else (eta - hi, eta - lo)
    # the phi range [phi - dphi, phi + dphi], split at +-pi
    for shift in (-2 * np.pi, 0.0, 2 * np.pi):
        p0, p1 = max(phi - dphi + shift, -np.pi), min(phi + dphi + shift, np.pi)
        if p1 > p0:
            ax.add_patch(Rectangle((eta0, p0), eta1 - eta0, p1 - p0, facecolor=colour, edgecolor=colour, alpha=0.08,
                                   linestyle="--", linewidth=0.8))


def draw_event(ax, key, eta, phi, et, selected, graphs):
    """graphs: {g: (nodes set, sorted directed edges)}"""
    colours = plt.get_cmap("tab10").colors
    size = 12 + 40 * np.sqrt(np.clip(et, 0, None))
    membership = defaultdict(list)
    for g, (nodes, _) in graphs.items():
        for t in nodes:
            membership[t].append(g)

    unselected = ~selected
    alone = selected & np.array([t not in membership for t in range(len(eta))], bool)
    ax.scatter(eta[unselected], phi[unselected], s=size[unselected], marker="x", c="0.6", linewidths=0.8)
    ax.scatter(eta[alone], phi[alone], s=size[alone], c="0.6")

    for g, (nodes, edges) in sorted(graphs.items()):
        colour = colours[g % len(colours)]
        for a, b in edges:
            if a < b:  # each link once
                for t0, t1, p0, p1 in phi_segments(phi[a], phi[b]):
                    ax.plot([eta[a] + t0 * (eta[b] - eta[a]), eta[a] + t1 * (eta[b] - eta[a])], [p0, p1],
                            color=colour, linewidth=0.8, alpha=0.7)
        seeds = sorted(t for t in nodes if et[t] >= SEED_ET)
        followers = sorted(t for t in nodes if et[t] < SEED_ET)
        for s in seeds:
            draw_window(ax, float(eta[s]), float(phi[s]), colour)
        ax.scatter(eta[seeds], phi[seeds], s=size[seeds], color=colour, edgecolors="k", linewidths=0.6, zorder=3)
        ax.scatter(eta[followers], phi[followers], s=size[followers], facecolors="none", edgecolors=colour,
                   linewidths=1.2, zorder=3)
        shared = [t for t in followers if len(membership[t]) > 1]
        ax.scatter(eta[shared], phi[shared], s=size[shared] * 2.2, facecolors="none", edgecolors="k",
                   linewidths=0.8, zorder=3)
        for t in nodes:
            ax.annotate(str(t), (eta[t], phi[t]), xytext=(3, 3), textcoords="offset points", fontsize=6)

    finish_axes(ax, f"Input graphs. Run {key[0]}, lumi {key[1]}, event {key[2]}: {len(eta)} tracksters, "
                f"{int(selected.sum())} selected, {len(graphs)} graphs, {sum(len(n) for n, _ in graphs.values())} nodes",
                [marker("seed"), marker("follower", face="none", edge="0.45"),
                 marker("follower in more graphs (black ring)", face="none", size=9),
                 marker("selected, not in a graph", face="0.6", edge="0.6"),
                 marker("no ECAL cluster", shape="x", edge="0.6")])


def draw_scores(ax, eta, phi, et, graphs, scores):
    """scores: [(a, b, probability)] of the edge rows; each link once"""
    size = 12 + 40 * np.sqrt(np.clip(et, 0, None))
    links = {}
    for a, b, p in scores:
        links.setdefault((min(a, b), max(a, b)), p)
    done = sorted(((a, b, p) for (a, b), p in links.items() if p >= 0), key=lambda x: x[2])  # the high scores on top
    missing = [(a, b) for (a, b), p in links.items() if p < 0]
    segments, values = [], []
    for a, b, p in done:
        pieces = edge_segments(eta, phi, a, b)
        segments += pieces
        values += [p] * len(pieces)
    values = np.array(values, float)
    lines = LineCollection(segments, cmap="viridis", norm=Normalize(0, 1), linewidths=0.4 + 2.2 * values, zorder=2)
    lines.set_array(values)
    ax.add_collection(lines)
    ax.add_collection(LineCollection([s for a, b in missing for s in edge_segments(eta, phi, a, b)], colors="0.6",
                                     linestyles="--", linewidths=0.5, zorder=1))
    nodes = sorted({t for n, _ in graphs.values() for t in n})
    seeds = [t for t in nodes if et[t] >= SEED_ET]
    followers = [t for t in nodes if et[t] < SEED_ET]
    ax.scatter(eta[seeds], phi[seeds], s=size[seeds], c="0.45", edgecolors="k", linewidths=0.5, zorder=3)
    ax.scatter(eta[followers], phi[followers], s=size[followers], facecolors="w", edgecolors="0.35", linewidths=0.8,
               zorder=3)
    bar = ax.figure.colorbar(lines, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("egmLinkingProbability", fontsize=8)
    bar.ax.axhline(LINKING_THRESHOLD, color="r", linewidth=1.2)
    bar.ax.tick_params(labelsize=7)
    n_pass = sum(p > LINKING_THRESHOLD for _, _, p in done)
    finish_axes(ax, f"Edge scores: {len(links)} links, {n_pass} with p > {LINKING_THRESHOLD}, "
                f"{len(missing)} not evaluated",
                [marker("seed"), marker("follower", face="w", edge="0.35"),
                 Line2D([], [], color="0.6", linestyle="--", linewidth=0.8, label="not evaluated (graph > 50 nodes)"),
                 Line2D([], [], color="r", linewidth=1.2, label=f"threshold on the colour bar: {LINKING_THRESHOLD}")])


def draw_superclusters(ax, eta, phi, et, graphs, scores, groups, raw_pt):
    """groups: for each supercluster, its trackster indices with the seed first; raw_pt: its raw pt"""
    colours = plt.get_cmap("tab10").colors
    size = 12 + 40 * np.sqrt(np.clip(et, 0, None))
    nodes = sorted({t for n, _ in graphs.values() for t in n})
    taken = {t for group in groups for t in group}
    free = [t for t in nodes if t not in taken]
    ax.scatter(eta[free], phi[free], s=size[free], c="0.85", zorder=1)
    accepted = {(min(a, b), max(a, b)) for a, b, p in scores if p > LINKING_THRESHOLD}
    n_passing = 0
    for g, group in enumerate(groups):
        passing = raw_pt[g] >= SUPERCLUSTER_ET_THRESHOLD
        n_passing += passing
        colour, alpha = colours[g % len(colours)], 1.0 if passing else 0.3
        members = set(group)
        links = [(a, b) for a, b in accepted if a in members and b in members and max(et[a], et[b]) >= SEED_ET]
        ax.add_collection(LineCollection([s for a, b in links for s in edge_segments(eta, phi, a, b)], colors=[colour],
                                         linewidths=1.3, alpha=alpha, zorder=2))
        seed, others = group[0], group[1:]
        other_seeds = [t for t in others if et[t] >= SEED_ET]
        followers = [t for t in others if et[t] < SEED_ET]
        ax.scatter(eta[[seed]], phi[[seed]], s=size[[seed]] * 1.8, marker="*", color=colour, edgecolors="k",
                   linewidths=0.6, alpha=alpha, zorder=4)
        ax.scatter(eta[other_seeds], phi[other_seeds], s=size[other_seeds], color=colour, edgecolors="k",
                   linewidths=0.5, alpha=alpha, zorder=3)
        ax.scatter(eta[followers], phi[followers], s=size[followers], facecolors="none", edgecolors=colour,
                   linewidths=1.2, alpha=alpha, zorder=3)
        if passing:
            ax.annotate(f"{raw_pt[g]:.0f}", (eta[seed], phi[seed]), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, fontweight="bold")
    n_multi = sum(len(group) > 1 for group in groups)
    finish_axes(ax, f"Superclusters: {len(groups)} groups, {n_multi} with more than one trackster, {n_passing} with raw "
                f"pt >= {SUPERCLUSTER_ET_THRESHOLD:.0f} GeV (number at the seed: raw pt, GeV)",
                [marker("group seed", shape="*"), marker("other seed"), marker("follower in the group", face="none"),
                 marker("graph node, in no group", face="0.85", edge="0.85"),
                 Line2D([], [], color="0.45", linewidth=1.3, label=f"accepted link (p > {LINKING_THRESHOLD})")])


def main():
    from DataFormats.FWLite import Events, Handle
    from compare_graph_events import inputs_from_tracksters, read_job_blocks, read_job_graphs

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file")
    parser.add_argument("log")
    parser.add_argument("-o", "--output", default="graphs.pdf")
    parser.add_argument("--max-events", type=int, default=-1)
    args = parser.parse_args()

    job = read_job_graphs(args.log)
    blocks = read_job_blocks(args.log)
    hT = Handle("std::vector<ticl::io_v1::Trackster>")
    hP = Handle("ticl::TracksterBarrelPropertiesHostCollection")
    hS = Handle("ticl::TracksterGraphEdgeScoresHostCollection")
    hSC = Handle("std::vector<ticl::io_v1::Trackster>")
    hL = Handle("std::vector<std::vector<unsigned int>>")
    n = 0
    with PdfPages(args.output) as pdf:
        for ev in Events(args.file):
            if n == args.max_events:
                break
            aux = ev.eventAuxiliary()
            key = (aux.run(), aux.luminosityBlock(), aux.event())
            if key not in job:
                continue
            ev.getByLabel("ticlTrackstersCLUE3DBarrel", hT)
            ev.getByLabel("ticlTracksterBarrelProperties", hP)
            eta, phi, et, selected = inputs_from_tracksters(hT.product(), hP.product().const_view())
            ev.getByLabel("ticlBarrelGraphEdgeScore", hS)
            ev.getByLabel("ticlTracksterLinksSuperclusteringBarrelGraph", hSC)
            ev.getByLabel("ticlTracksterLinksSuperclusteringBarrelGraph", "linkedTracksterIdToInputTracksterId", hL)
            full = hS.isValid() and hSC.isValid() and hL.isValid()
            fig, axes = plt.subplots(1, 3 if full else 1, figsize=(27, 7.5) if full else (9, 6), squeeze=False,
                                     layout="constrained")
            draw_event(axes[0, 0], key, eta, phi, et, selected, job[key])
            if full:
                view = hS.product().const_view()
                scores = [(row[0], row[1], float(view[e].egmLinkingProbability()))
                          for e, row in enumerate(blocks[key][2])]
                groups = [list(link) for link in hL.product()]
                raw_pt = [hSC.product()[g].raw_pt() for g in range(len(groups))]
                draw_scores(axes[0, 1], eta, phi, et, job[key], scores)
                draw_superclusters(axes[0, 2], eta, phi, et, job[key], scores, groups, raw_pt)
            pdf.savefig(fig)
            plt.close(fig)
            n += 1
    print(f"{n} events drawn in {args.output}")


if __name__ == "__main__":
    main()
