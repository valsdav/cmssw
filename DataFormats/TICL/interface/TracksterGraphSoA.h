#ifndef DataFormats_TICL_interface_TracksterGraphSoA_h
#define DataFormats_TICL_interface_TracksterGraphSoA_h

// The trackster graphs of one event, in three blocks.
//
// A graph = one connected component of seeds (seed-seed links) plus all the followers that sit in the window of at
// least one of its seeds. A follower can be in the window of seeds of different components, so the same trackster
// can be a node of several graphs.
//
//   graphs : one row per graph: where its nodes and edges start, and how many. Graph g is the component whose
//            smallest seed index is the g-th smallest.
//   nodes  : one row per (graph, trackster) node, grouped by graph and, inside a graph, in ascending trackster index.
//            The local index of a node in its graph is row - nodeOffset(graph).
//   edges  : one row per directed edge, grouped by graph and sorted by (localA, localB). Every link is stored in both
//            directions.

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  GENERATE_SOA_LAYOUT(TracksterGraphSoAGraphs,
                      SOA_COLUMN(int32_t, nodeOffset),
                      SOA_COLUMN(int32_t, nNodes),
                      SOA_COLUMN(int32_t, edgeOffset),
                      SOA_COLUMN(int32_t, nEdges))

  GENERATE_SOA_LAYOUT(TracksterGraphSoANodes,
                      SOA_COLUMN(int32_t, tracksterIdx),  // row in the TracksterSoA
                      SOA_COLUMN(int32_t, graphIdx))

  GENERATE_SOA_LAYOUT(TracksterGraphSoAEdges,
                      SOA_COLUMN(int32_t, tracksterA),  // row in the TracksterSoA
                      SOA_COLUMN(int32_t, tracksterB),
                      SOA_COLUMN(int32_t, graphIdx),
                      SOA_COLUMN(int32_t, localA),  // local index of tracksterA in the graph
                      SOA_COLUMN(int32_t, localB))

  GENERATE_SOA_BLOCKS(TracksterGraphSoALayout,
                      SOA_BLOCK(graphs, TracksterGraphSoAGraphs),
                      SOA_BLOCK(nodes, TracksterGraphSoANodes),
                      SOA_BLOCK(edges, TracksterGraphSoAEdges))

  using TracksterGraphSoA = TracksterGraphSoALayout<>;
  using TracksterGraphSoAView = TracksterGraphSoA::View;
  using TracksterGraphSoAConstView = TracksterGraphSoA::ConstView;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphSoA_h
