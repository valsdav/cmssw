// Standalone check of TracksterGraphAlgo: reads events from stdin, prints the graphs, nodes and edges. Driven by
// compare_graph_builder.py, which runs the Python graph builder on the same events and compares graph by graph.
//
// stdin: seedEtThreshold
//        nDetaLow c0 ... ; nDetaHigh c0 ... ; nDphiMax c0 ...   (one line each)
//        nEvents, then for each event: n, and n lines "eta phi et selected"

#include <cstdio>
#include <iostream>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TICL/interface/TracksterGraphHostCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphAlgo.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  void readCoefficients(float* c, int32_t& n) {
    std::cin >> n;
    for (int32_t k = 0; k < n; ++k)
      std::cin >> c[k];
  }
}  // namespace

int main() {
  ::ticl::graph::Params p{};
  std::cin >> p.seedEtThreshold;
  readCoefficients(p.detaLow, p.nDetaLow);
  readCoefficients(p.detaHigh, p.nDetaHigh);
  readCoefficients(p.dphiMax, p.nDphiMax);
  int nEvents;
  std::cin >> nEvents;

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No device available for this backend\n";
    return 1;
  }
  Queue queue(devices[0]);
  TracksterGraphAlgo algo(p);

  for (int ev = 0; ev < nEvents; ++ev) {
    int32_t n;
    std::cin >> n;
    std::vector<float> eta(n), phi(n), et(n);
    std::vector<int32_t> selected(n);
    for (int32_t i = 0; i < n; ++i)
      std::cin >> eta[i] >> phi[i] >> et[i] >> selected[i];

    // the inputs on the device
    auto deviceFloats = cms::alpakatools::make_device_buffer<float[]>(queue, 3 * n);
    auto deviceInts = cms::alpakatools::make_device_buffer<int32_t[]>(queue, n);
    ::ticl::graph::Inputs inputs{n, deviceFloats.data(), deviceFloats.data() + n, deviceFloats.data() + 2 * n,
                                 deviceInts.data()};
    if (n > 0) {
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(queue, deviceFloats.data(), n),
                     cms::alpakatools::make_host_view(eta.data(), n));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(queue, deviceFloats.data() + n, n),
                     cms::alpakatools::make_host_view(phi.data(), n));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(queue, deviceFloats.data() + 2 * n, n),
                     cms::alpakatools::make_host_view(et.data(), n));
      alpaka::memcpy(queue, deviceInts, cms::alpakatools::make_host_view(selected.data(), n));
    }

    algo.count(queue, inputs);
    alpaka::wait(queue);
    auto const c = algo.counters();
    ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterGraphDeviceCollection graphs(queue, c.nGraphs, c.nNodes, c.nEdges);
    algo.fill(queue, graphs.view());
    ::ticl::TracksterGraphHostCollection host(queue, c.nGraphs, c.nNodes, c.nEdges);
    alpaka::memcpy(queue, host.buffer(), graphs.const_buffer());
    alpaka::wait(queue);

    auto const view = host.const_view();
    std::printf("EVENT %d %d %d\n", c.nGraphs, c.nNodes, c.nEdges);
    for (int32_t g = 0; g < c.nGraphs; ++g) {
      auto const graph = view.graphs()[g];
      std::printf("G %d %d %d %d\n", graph.nodeOffset(), graph.nNodes(), graph.edgeOffset(), graph.nEdges());
    }
    for (int32_t k = 0; k < c.nNodes; ++k)
      std::printf("N %d %d\n", view.nodes()[k].tracksterIdx(), view.nodes()[k].graphIdx());
    for (int32_t k = 0; k < c.nEdges; ++k) {
      auto const edge = view.edges()[k];
      std::printf("E %d %d %d %d %d\n",
                  edge.tracksterA(),
                  edge.tracksterB(),
                  edge.graphIdx(),
                  edge.localA(),
                  edge.localB());
    }
  }
  return 0;
}
