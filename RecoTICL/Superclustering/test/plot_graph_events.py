"""Draw the trackster graphs of real events in the eta-phi plane, one page per event.

Inputs: the output file and the log of the barrel SoA test job, as for compare_graph_events.py (the graphs come
from the TracksterGraphValidator dump, eta, phi, et and selected are computed from the legacy tracksters).

Each page shows:
  - the tracksters without an ECAL layer cluster (not selected): grey crosses;
  - the selected tracksters that are not in a graph: grey dots;
  - the seeds (filled) and the followers (hollow) in the colour of their graph; a follower in more than one graph
    has a black ring;
  - the edges of each graph, and the window of each seed (mirrored for negative eta, split at phi = +-pi).
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
from matplotlib.patches import Rectangle

from compare_graph_builder import SEED_ET, windows_for_eta


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
    ax.scatter(eta[unselected], phi[unselected], s=size[unselected], marker="x", c="0.6", linewidths=0.8,
               label="no ECAL cluster")
    ax.scatter(eta[alone], phi[alone], s=size[alone], c="0.6", label="selected, not in a graph")

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
        ax.scatter(eta[seeds], phi[seeds], s=size[seeds], color=colour, edgecolors="k", linewidths=0.6, zorder=3,
                   label=f"graph {g}: {len(seeds)} seeds, {len(followers)} followers")
        ax.scatter(eta[followers], phi[followers], s=size[followers], facecolors="none", edgecolors=colour,
                   linewidths=1.2, zorder=3)
        shared = [t for t in followers if len(membership[t]) > 1]
        ax.scatter(eta[shared], phi[shared], s=size[shared] * 2.2, facecolors="none", edgecolors="k",
                   linewidths=0.8, zorder=3)
        for t in nodes:
            ax.annotate(str(t), (eta[t], phi[t]), xytext=(3, 3), textcoords="offset points", fontsize=6)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(f"run {key[0]}, lumi {key[1]}, event {key[2]}: {len(eta)} tracksters, {int(selected.sum())} "
                 f"selected, {len(graphs)} graphs", fontsize=9)
    ax.legend(fontsize=6, loc="upper right", ncol=2)


def main():
    from DataFormats.FWLite import Events, Handle
    from compare_graph_events import inputs_from_tracksters, read_job_graphs

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file")
    parser.add_argument("log")
    parser.add_argument("-o", "--output", default="graphs.pdf")
    parser.add_argument("--max-events", type=int, default=-1)
    args = parser.parse_args()

    job = read_job_graphs(args.log)
    hT = Handle("std::vector<ticl::io_v1::Trackster>")
    hP = Handle("ticl::TracksterBarrelPropertiesHostCollection")
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
            fig, ax = plt.subplots(figsize=(9, 6))
            draw_event(ax, key, eta, phi, et, selected, job[key])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            n += 1
    print(f"{n} events drawn in {args.output}")


if __name__ == "__main__":
    main()
