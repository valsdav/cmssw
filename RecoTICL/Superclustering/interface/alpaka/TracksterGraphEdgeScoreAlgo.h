#ifndef RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeScoreAlgo_h
#define RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeScoreAlgo_h

// The kernels around the graph edge model:
//
//   TracksterSoA + barrel properties + TracksterGraphSoA
//        |  memset 0 + launchFillModelInputs
//        v
//   GraphModelInputSoA   (one row per graph)
//        |  runGraphModel (TracksterGraphEdgeModel.h)
//        v
//   GraphModelOutputSoA  (one row per graph)
//        |  launchGatherEdgeScores
//        v
//   TracksterGraphEdgeScoreSoA (one row per edge, same order as the edges block)

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoreSoA.h"
#include "DataFormats/TICL/interface/TracksterGraphSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphEdgeModelSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel {

  // Fills the model inputs, one row per graph. The input buffer must be zeroed before (memset):
  //   nodeFeatures : the features of the nodes (fillNodeFeatures), 0 for the padding rows
  //   adjacency    : 1 on the edges, 1 on the diagonal of the real nodes
  // A graph with more than kMaxNodes nodes gets one dummy node (adjacency(0, 0) = 1), so that the model runs on a
  // valid input; launchGatherEdgeScores gives its edges -1.
  void launchFillModelInputs(Queue& queue,
                             ::ticl::TracksterSoAConstView tracksters,
                             ::ticl::TracksterBarrelPropertiesSoAConstView properties,
                             ::ticl::TracksterGraphSoAConstView graph,
                             ::ticl::graphmodel::GraphModelInputSoA::View inputs);

  // Valid dummy inputs (one real node per graph), to warm up the model. The input buffer must be zeroed before.
  void launchFillWarmupInputs(Queue& queue, ::ticl::graphmodel::GraphModelInputSoA::View inputs);

  // score[e] = sigmoid(pairLogits[graph](localA, localB)) for every edge e (the logit if applySigmoid is false);
  // -1 for the edges of the graphs with more than kMaxNodes nodes.
  void launchGatherEdgeScores(Queue& queue,
                              ::ticl::TracksterGraphSoAConstView graph,
                              ::ticl::graphmodel::GraphModelOutputSoA::ConstView outputs,
                              ::ticl::TracksterGraphEdgeScoreSoA::View scores,
                              bool applySigmoid);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel

#endif  // RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeScoreAlgo_h
