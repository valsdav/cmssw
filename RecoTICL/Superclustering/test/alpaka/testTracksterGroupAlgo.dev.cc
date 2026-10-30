// Standalone check of TracksterGroupAlgo (the superclustering on the graphs): reads events from stdin, prints the
// groups and the layer cluster -> group map. Driven by compare_graph_superclusters.py, which runs a Python mirror of
// the resolution on the same events.
//
// usage: testTracksterGroupAlgoSerialSync seedEtThreshold linkingThreshold direct|transitive
// stdin: nEvents, then for each event:
//        n nLayerClusters, and n lines "rawEnergy x y z nVertices vertex..."
//        nGraphs nNodes nEdges, then the G, N and E rows of the graph dump format without the letter,
//        then nEdges probabilities
// stdout, per event: "EVENT nGroups nMembers", one line "G member..." per group (the seed first),
//        "MAP nTracksters nVertices" and one line with the group of each layer cluster

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TICL/interface/LayerClusterToTracksterHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoresHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGroupHostCollection.h"
#include "DataFormats/TICL/interface/TracksterHostCollection.h"
#include "DataFormats/TICL/interface/alpaka/LayerClusterToTracksterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphEdgeScoresDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGroupDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGroupAlgo.h"

namespace A = ALPAKA_ACCELERATOR_NAMESPACE;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " seedEtThreshold linkingThreshold direct|transitive\n";
    return 1;
  }
  ::ticl::graph::GroupParams const params{std::stof(argv[1]), std::stof(argv[2]), std::string(argv[3]) == "transitive"};

  auto const& devices = cms::alpakatools::devices<A::Platform>();
  if (devices.empty()) {
    std::cerr << "No device available for this backend\n";
    return 1;
  }
  A::Queue queue(devices[0]);
  A::TracksterGroupAlgo algo(params);

  int nEvents;
  std::cin >> nEvents;
  for (int ev = 0; ev < nEvents; ++ev) {
    // ---- tracksters: energy, barycenter and layer clusters ----
    int32_t n, nLayerClusters;
    std::cin >> n >> nLayerClusters;
    std::vector<float> rawEnergy(n), x(n), y(n), z(n);
    std::vector<std::vector<uint32_t>> vertices(n);
    uint32_t nVertices = 0;
    for (int32_t i = 0; i < n; ++i) {
      int32_t size;
      std::cin >> rawEnergy[i] >> x[i] >> y[i] >> z[i] >> size;
      vertices[i].resize(size);
      for (auto& v : vertices[i])
        std::cin >> v;
      nVertices += size;
    }
    ::ticl::TracksterHostCollection tracksters(queue, n, n, n, n, n, n, nVertices);
    auto tv = tracksters.view();
    uint32_t offset = 0;
    tv.vertices().offsets()[0].keys_offsets() = 0;
    for (int32_t i = 0; i < n; ++i) {
      tv.energy()[i].rawEnergy() = rawEnergy[i];
      tv.barycenter()[i].x() = x[i];
      tv.barycenter()[i].y() = y[i];
      tv.barycenter()[i].z() = z[i];
      for (auto v : vertices[i])
        tv.vertices().content()[offset++].values() = v;
      tv.vertices().offsets()[i + 1].keys_offsets() = offset;
    }

    // ---- graphs and probabilities ----
    int32_t nGraphs, nNodes, nEdges;
    std::cin >> nGraphs >> nNodes >> nEdges;
    ::ticl::TracksterGraphHostCollection graph(queue, nGraphs, nNodes, nEdges);
    auto gv = graph.view();
    for (int32_t g = 0; g < nGraphs; ++g)
      std::cin >> gv.graphs()[g].nodeOffset() >> gv.graphs()[g].nNodes() >> gv.graphs()[g].edgeOffset() >>
          gv.graphs()[g].nEdges();
    for (int32_t k = 0; k < nNodes; ++k)
      std::cin >> gv.nodes()[k].tracksterIdx() >> gv.nodes()[k].graphIdx();
    for (int32_t e = 0; e < nEdges; ++e)
      std::cin >> gv.edges()[e].tracksterA() >> gv.edges()[e].tracksterB() >> gv.edges()[e].graphIdx() >>
          gv.edges()[e].localA() >> gv.edges()[e].localB();
    ::ticl::TracksterGraphEdgeScoresHostCollection scores(queue, nEdges);
    for (int32_t e = 0; e < nEdges; ++e)
      std::cin >> scores.view()[e].egmLinkingProbability();

    // ---- to the device (a no-op copy on the serial backend), then the producer's acquire() and produce() ----
    A::ticl::TracksterDeviceCollection dTracksters(queue, n, n, n, n, n, n, nVertices);
    A::ticl::TracksterGraphDeviceCollection dGraph(queue, nGraphs, nNodes, nEdges);
    A::ticl::TracksterGraphEdgeScoresDeviceCollection dScores(queue, nEdges);
    alpaka::memcpy(queue, dTracksters.buffer(), tracksters.const_buffer());
    alpaka::memcpy(queue, dGraph.buffer(), graph.const_buffer());
    alpaka::memcpy(queue, dScores.buffer(), scores.const_buffer());

    algo.count(queue, dTracksters.const_view(), dGraph.const_view(), dScores.const_view());
    alpaka::wait(queue);
    auto const c = algo.counters();
    A::ticl::TracksterGroupDeviceCollection dGroups(queue, c.nGroups, c.nMembers);
    A::ticl::LayerClusterToTracksterDeviceCollection dMap(queue, nLayerClusters);
    algo.fill(queue, dTracksters.const_view(), dGroups.view(), dMap.view());

    ::ticl::TracksterGroupHostCollection groups(queue, c.nGroups, c.nMembers);
    ::ticl::LayerClusterToTracksterHostCollection map(queue, nLayerClusters);
    alpaka::memcpy(queue, groups.buffer(), dGroups.const_buffer());
    alpaka::memcpy(queue, map.buffer(), dMap.const_buffer());
    alpaka::wait(queue);

    auto const groupsView = groups.const_view();
    std::printf("EVENT %d %d\n", c.nGroups, c.nMembers);
    for (int32_t g = 0; g < c.nGroups; ++g) {
      std::printf("G");
      for (auto t : groupsView[g])
        std::printf(" %u", t);
      std::printf("\n");
    }
    auto const mapView = map.const_view();
    std::printf("MAP %u %u\n", mapView.numberOfTrackstersScalar(), mapView.numberOfVerticesScalar());
    for (int32_t i = 0; i < nLayerClusters; ++i)
      std::printf("%s%d", i ? " " : "", mapView[i].tracksterIndex());
    std::printf("\n");
  }
  return 0;
}
