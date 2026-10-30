#ifndef DataFormats_TICL_interface_TracksterGraphEdgeScoreSoA_h
#define DataFormats_TICL_interface_TracksterGraphEdgeScoreSoA_h

// One score per directed edge of TracksterGraphSoA, in the same row order as its edges block: row e is the score of
// edges[e].tracksterA -> edges[e].tracksterB. The score is sigmoid(logit) of the graph edge model, or -1 if the model
// did not evaluate the edge (graph with more nodes than the model accepts).

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  GENERATE_SOA_LAYOUT(TracksterGraphEdgeScoreLayout, SOA_COLUMN(float, score))

  using TracksterGraphEdgeScoreSoA = TracksterGraphEdgeScoreLayout<>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphEdgeScoreSoA_h
