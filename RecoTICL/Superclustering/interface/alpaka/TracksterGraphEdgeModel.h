#ifndef RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeModel_h
#define RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeModel_h

// Runs the graph edge model on the model input SoA, as RecoTracker/FinalTrackSelectors/plugins/alpaka/
// PixelTrackTorchHighPuritySelector.cc does: the SoA columns are wrapped as tensors (no copy), and the model output
// is written into the output SoA.
//
//   inputs  : "node_features" [B, kMaxNodes, kNodeFeatures], "adjacency" [B, kMaxNodes, kMaxNodes]
//   outputs : "pair_logits"   [B, kMaxNodes, kMaxNodes]
//
// B = graphsPerBatch (0 = all the graphs of the event in one call). Header only, so that the package library does
// not link PyTorch.

#include <deque>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphEdgeModelSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel {

  // The tensors of one inference batch. They must live until the end of the event, as in the reference module.
  struct BatchIO {
    cms::torch::alpakatools::TensorCollection<Queue> inputs;
    cms::torch::alpakatools::TensorCollection<Queue> outputs;
  };

  inline void runGraphModel(Queue& queue,
                            torch::AlpakaModel& model,
                            ::ticl::graphmodel::GraphModelInputSoA::View inputs,
                            ::ticl::graphmodel::GraphModelOutputSoA::View outputs,
                            int graphsPerBatch,
                            std::deque<BatchIO>& batches) {
    using ::ticl::graphmodel::GraphModelInputSoA;
    using ::ticl::graphmodel::GraphModelOutputSoA;

    int const nGraphs = inputs.metadata().size();
    if (nGraphs == 0)
      return;
    int const batchSize = graphsPerBatch > 0 ? graphsPerBatch : nGraphs;
    int const nBatches = (nGraphs + batchSize - 1) / batchSize;

    auto inputRecords = inputs.records();
    auto outputRecords = outputs.records();
    for (int b = 0; b < nBatches; ++b) {
      batches.emplace_back(BatchIO{cms::torch::alpakatools::TensorCollection<Queue>(batchSize, nGraphs),
                                   cms::torch::alpakatools::TensorCollection<Queue>(batchSize, nGraphs)});
      auto& batch = batches.back();
      // the order must be the one of forward(node_features, adjacency) of the exported model
      batch.inputs.add<GraphModelInputSoA>("node_features", b, inputRecords.nodeFeatures());
      batch.inputs.add<GraphModelInputSoA>("adjacency", b, inputRecords.adjacency());
      batch.outputs.add<GraphModelOutputSoA>("pair_logits", b, outputRecords.pairLogits());
      model.forward(queue, batch.inputs, batch.outputs);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel

#endif  // RecoTICL_Superclustering_interface_alpaka_TracksterGraphEdgeModel_h
