#ifndef RecoTICL_Superclustering_interface_alpaka_TracksterGroupAlgo_h
#define RecoTICL_Superclustering_interface_alpaka_TracksterGroupAlgo_h

// Groups the tracksters of the graphs into superclusters, with the EGM linking probabilities of the graph edges, in two
// phases:
//
//   count() : the seeds (graph nodes with et >= seedEtThreshold) in descending et, then the greedy resolution: each
//             seed that no earlier group took starts a group and takes its neighbours with a probability above the
//             threshold that no group took yet (transitive: also the neighbours of the seeds that it takes). Copies
//             the totals to the host asynchronously.                                    -> called in acquire()
//   fill()  : once the totals are on the host and the outputs are allocated, writes the groups (the trackster indices,
//             the seed first) and the layer cluster -> group map, from which TracksterSoAProducer makes one trackster
//             per group.                                                                 -> called in produce()
//
// The scratch buffers live between the two calls, so one TracksterGroupAlgo serves one event at a time (stream
// module).

#include <optional>

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/TICL/interface/LayerClusterToTracksterSoA.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoresSoA.h"
#include "DataFormats/TICL/interface/TracksterGraphSoA.h"
#include "DataFormats/TICL/interface/TracksterGroupSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphParams.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterGroupAlgo {
  public:
    explicit TracksterGroupAlgo(::ticl::graph::GroupParams const& params) : params_(params) {}

    // The number of scores must be the number of edges.
    void count(Queue& queue,
               ::ticl::TracksterSoAConstView tracksters,
               ::ticl::TracksterGraphSoAConstView graph,
               ::ticl::TracksterGraphEdgeScoresSoA::ConstView scores);

    // Valid once the queue has been synchronised after count().
    ::ticl::graph::GroupCounters const& counters() const { return *hostCounters_->data(); }

    // groups: constructed with (nGroups, nMembers) of counters(); layerClusters: one row per layer cluster of the
    // tracksters (the indices of their vertices).
    void fill(Queue& queue,
              ::ticl::TracksterSoAConstView tracksters,
              ::ticl::TracksterGroupSoAView groups,
              ::ticl::LayerClusterToTracksterSoA::View layerClusters);

  private:
    ::ticl::graph::GroupParams const params_;
    int32_t n_ = 0;

    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> ints_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> et_;
    std::optional<cms::alpakatools::device_buffer<Device, ::ticl::graph::GroupCounters>> deviceCounters_;
    std::optional<cms::alpakatools::host_buffer<::ticl::graph::GroupCounters>> hostCounters_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_interface_alpaka_TracksterGroupAlgo_h
