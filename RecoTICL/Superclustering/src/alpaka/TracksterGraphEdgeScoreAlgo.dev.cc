// Memory note: in the model SoAs, coefficient k of the matrix of graph g is at k * columnStride + g. The kernels that
// write the matrices use g as the fastest index, so consecutive threads write consecutive addresses.

#include <alpaka/alpaka.hpp>
#include <xtd/xtd.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphEdgeFeatures.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphEdgeScoreAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel {

  using namespace cms::alpakatools;
  using ::ticl::graphmodel::GraphModelInputSoA;
  using ::ticl::graphmodel::GraphModelOutputSoA;
  using ::ticl::graphmodel::kMaxNodes;
  using ::ticl::graphmodel::kNodeFeatures;

  namespace {

    constexpr int32_t kThreads = 256;

    // One thread per (graph, node slot), graph fastest: the features and the diagonal of the real nodes.
    struct KernelFillNodes {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterSoAConstView tracksters,
                                    ::ticl::TracksterBarrelPropertiesSoAConstView properties,
                                    ::ticl::TracksterGraphSoAConstView graph,
                                    GraphModelInputSoA::View inputs) const {
        auto const graphs = graph.graphs();
        auto const nodes = graph.nodes();
        int32_t const nGraphs = graphs.metadata().size();
        for (int32_t idx : uniform_elements(acc, nGraphs * kMaxNodes)) {
          int32_t const g = idx % nGraphs;
          int32_t const i = idx / nGraphs;  // local node index = row of the matrices
          int32_t const nNodes = graphs[g].nNodes();
          if (nNodes > kMaxNodes) {
            if (i == 0)
              inputs[g].adjacency()(0, 0) = 1.f;  // one dummy node, so that the model sees a valid graph
            continue;
          }
          if (i >= nNodes)
            continue;  // padding: 0 from the memset
          float f[kNodeFeatures];
          ::ticl::graphmodel::fillNodeFeatures(
              tracksters, properties, nodes[graphs[g].nodeOffset() + i].tracksterIdx(), f);
          auto features = inputs[g].nodeFeatures();
          for (int32_t k = 0; k < kNodeFeatures; ++k)
            features(i, k) = f[k];
          inputs[g].adjacency()(i, i) = 1.f;
        }
      }
    };

    // One thread per edge: adjacency(localA, localB) = 1.
    struct KernelFillEdges {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterGraphSoAConstView graph,
                                    GraphModelInputSoA::View inputs) const {
        auto const graphs = graph.graphs();
        auto const edges = graph.edges();
        for (int32_t e : uniform_elements(acc, edges.metadata().size())) {
          int32_t const g = edges[e].graphIdx();
          if (graphs[g].nNodes() <= kMaxNodes)
            inputs[g].adjacency()(edges[e].localA(), edges[e].localB()) = 1.f;
        }
      }
    };

    // Graphs with links, so that the JIT profiles the edge head on the path and the shapes of real events. The features
    // stay 0 from the memset: the values do not change the operations.
    struct KernelFillWarmup {
      static constexpr int32_t kNodes = 4;
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, GraphModelInputSoA::View inputs) const {
        for (int32_t g : uniform_elements(acc, inputs.metadata().size())) {
          auto adjacency = inputs[g].adjacency();
          for (int32_t i = 0; i < kNodes; ++i) {
            adjacency(i, i) = 1.f;
            if (i > 0)
              adjacency(i, i - 1) = adjacency(i - 1, i) = 1.f;
          }
        }
      }
    };

    // One thread per edge: the pair logit of (localA, localB).
    struct KernelGatherEdgeScores {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterGraphSoAConstView graph,
                                    GraphModelOutputSoA::ConstView outputs,
                                    ::ticl::TracksterGraphEdgeScoresSoA::View scores,
                                    bool applySigmoid) const {
        auto const graphs = graph.graphs();
        auto const edges = graph.edges();
        for (int32_t e : uniform_elements(acc, edges.metadata().size())) {
          int32_t const g = edges[e].graphIdx();
          if (graphs[g].nNodes() > kMaxNodes) {
            scores[e].egmLinkingProbability() = -1.f;
            continue;
          }
          float const logit = outputs[g].pairLogits()(edges[e].localA(), edges[e].localB());
          scores[e].egmLinkingProbability() = applySigmoid ? 1.f / (1.f + xtd::exp(-logit)) : logit;
        }
      }
    };

    auto workDiv(int32_t n) { return make_workdiv<Acc1D>(divide_up_by(n, kThreads), kThreads); }

  }  // namespace

  void launchFillModelInputs(Queue& queue,
                             ::ticl::TracksterSoAConstView tracksters,
                             ::ticl::TracksterBarrelPropertiesSoAConstView properties,
                             ::ticl::TracksterGraphSoAConstView graph,
                             GraphModelInputSoA::View inputs) {
    int32_t const nGraphs = graph.graphs().metadata().size();
    int32_t const nEdges = graph.edges().metadata().size();
    if (nGraphs == 0)
      return;
    alpaka::exec<Acc1D>(
        queue, workDiv(nGraphs * kMaxNodes), KernelFillNodes{}, tracksters, properties, graph, inputs);
    if (nEdges > 0)
      alpaka::exec<Acc1D>(queue, workDiv(nEdges), KernelFillEdges{}, graph, inputs);
  }

  void launchFillWarmupInputs(Queue& queue, GraphModelInputSoA::View inputs) {
    int32_t const n = inputs.metadata().size();
    if (n > 0)
      alpaka::exec<Acc1D>(queue, workDiv(n), KernelFillWarmup{}, inputs);
  }

  void launchGatherEdgeScores(Queue& queue,
                              ::ticl::TracksterGraphSoAConstView graph,
                              GraphModelOutputSoA::ConstView outputs,
                              ::ticl::TracksterGraphEdgeScoresSoA::View scores,
                              bool applySigmoid) {
    int32_t const nEdges = graph.edges().metadata().size();
    if (nEdges > 0)
      alpaka::exec<Acc1D>(queue, workDiv(nEdges), KernelGatherEdgeScores{}, graph, outputs, scores, applySigmoid);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::graphmodel
