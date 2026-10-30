#ifndef DataFormats_TICL_interface_TracksterGraphEdgeScoresSoA_h
#define DataFormats_TICL_interface_TracksterGraphEdgeScoresSoA_h

// The scores of the directed edges of TracksterGraphSoA, one row per edge, in the same row order as its edges block:
// row e belongs to edges[e].tracksterA -> edges[e].tracksterB. A user of both products must check that the sizes are
// equal.
//   egmLinkingProbability : sigmoid(logit) of the graph edge model (EGM linking), or -1 if the model did not evaluate
//                           the edge (graph with more nodes than the model accepts)

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  GENERATE_SOA_LAYOUT(TracksterGraphEdgeScoresLayout, SOA_COLUMN(float, egmLinkingProbability))

  using TracksterGraphEdgeScoresSoA = TracksterGraphEdgeScoresLayout<>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphEdgeScoresSoA_h
