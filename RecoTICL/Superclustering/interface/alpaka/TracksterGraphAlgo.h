#ifndef RecoTICL_Superclustering_interface_alpaka_TracksterGraphAlgo_h
#define RecoTICL_Superclustering_interface_alpaka_TracksterGraphAlgo_h

// Builds the trackster graphs on the device, in two phases:
//
//   count() : seeds, seed-seed links, connected components (graph numbering), the node and edge counts and offsets
//             of each graph; copies the 3 totals to the host asynchronously.     -> called in acquire()
//   fill()  : once the totals are on the host and the output is allocated with the exact sizes, writes the graphs,
//             the nodes and the edges.                                            -> called in produce()
//
// The scratch buffers live between the two calls, so one TracksterGraphAlgo serves one event at a time (stream
// module).

#include <optional>

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphParams.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterGraphAlgo {
  public:
    explicit TracksterGraphAlgo(::ticl::graph::Params const& params) : params_(params) {}

    // Computes the inputs of the barrel tracksters: eta and phi of the barycenter, et = rawEnergy / cosh(eta), and
    // selected = the trackster has an ECAL layer cluster. The returned arrays live until the next call.
    ::ticl::graph::Inputs prepareInputs(Queue& queue,
                                        ::ticl::TracksterSoAConstView tracksters,
                                        ::ticl::TracksterBarrelPropertiesSoAConstView properties);

    // The inputs must stay valid until fill() is called.
    void count(Queue& queue, ::ticl::graph::Inputs const& inputs);

    // Valid once the queue has been synchronised after count().
    ::ticl::graph::Counters const& counters() const { return *hostCounters_->data(); }

    void fill(Queue& queue, ::ticl::TracksterGraphSoAView graphs);

  private:
    ::ticl::graph::Params const params_;
    ::ticl::graph::Inputs inputs_{};

    std::optional<cms::alpakatools::device_buffer<Device, float[]>> inputFloats_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> inputInts_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> ints_;
    std::optional<cms::alpakatools::device_buffer<Device, float[]>> floats_;
    std::optional<cms::alpakatools::device_buffer<Device, int32_t[]>> bins_;
    std::optional<cms::alpakatools::device_buffer<Device, ::ticl::graph::Counters>> deviceCounters_;
    std::optional<cms::alpakatools::host_buffer<::ticl::graph::Counters>> hostCounters_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_interface_alpaka_TracksterGraphAlgo_h
