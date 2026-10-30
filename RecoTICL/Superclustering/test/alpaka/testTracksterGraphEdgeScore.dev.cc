// Standalone check of the edge score chain of TICLBarrelGraphEdgeScoreProducer (no cmsRun):
//   tracksters + barrel properties + graphs -> launchFillModelInputs -> runGraphModel -> launchGatherEdgeScores
// Driven by compare_edge_scores.py, which computes the same scores with the TorchScript model in Python.
//
// usage: testTracksterGraphEdgeScoreSerialSync model.pt [graphsPerBatch]
// stdin: nEvents, then for each event:
//        n, and n lines "rawEnergy rawEmEnergy x y z nEcal nHcal ecalX ecalY ecalZ ecalEta ecalPhi
//                        hcalX hcalY hcalZ hcalEta hcalPhi"
//        nGraphs nNodes nEdges, then the G, N and E rows of the graph dump format without the letter
// stdout: one line "EVENT score0 score1 ..." per event, one score per edge

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <cstdio>
#include <deque>
#include <iostream>
#include <string>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoreHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphHostCollection.h"
#include "DataFormats/TICL/interface/TracksterHostCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphEdgeScoreDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphEdgeModel.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphEdgeScoreAlgo.h"

namespace A = ALPAKA_ACCELERATOR_NAMESPACE;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " model.pt [graphsPerBatch]\n";
    return 1;
  }
  int const graphsPerBatch = argc > 2 ? std::stoi(argv[2]) : 0;

  auto const& devices = cms::alpakatools::devices<A::Platform>();
  if (devices.empty()) {
    std::cerr << "No device available for this backend\n";
    return 1;
  }
  A::Queue queue(devices[0]);
  A::torch::AlpakaModel model(argv[1]);

  int nEvents;
  std::cin >> nEvents;
  for (int ev = 0; ev < nEvents; ++ev) {
    int32_t n;
    std::cin >> n;
    ::ticl::TracksterHostCollection tracksters(queue, n, n, n, n, n, n, 0);
    ::ticl::TracksterBarrelPropertiesHostCollection properties(queue, n);
    auto tv = tracksters.view();
    auto pv = properties.view();
    for (int32_t i = 0; i < n; ++i) {
      auto p = pv[i];
      std::cin >> tv.energy()[i].rawEnergy() >> tv.energy()[i].rawEmEnergy() >> tv.barycenter()[i].x() >> tv.barycenter()[i].y() >>
          tv.barycenter()[i].z() >> p.nEcalClusters() >> p.nHcalClusters() >> p.ecalX() >>
          p.ecalY() >> p.ecalZ() >> p.ecalEta() >> p.ecalPhi() >> p.hcalBarycenterX() >>
          p.hcalBarycenterY() >> p.hcalBarycenterZ() >> p.hcalBarycenterEta() >> p.hcalBarycenterPhi();
    }

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

    // to the device (a no-op copy on the serial backend)
    A::ticl::TracksterDeviceCollection dTracksters(queue, n, n, n, n, n, n, 0);
    A::ticl::TracksterBarrelPropertiesDeviceCollection dProperties(queue, n);
    A::ticl::TracksterGraphDeviceCollection dGraph(queue, nGraphs, nNodes, nEdges);
    alpaka::memcpy(queue, dTracksters.buffer(), tracksters.const_buffer());
    alpaka::memcpy(queue, dProperties.buffer(), properties.const_buffer());
    alpaka::memcpy(queue, dGraph.buffer(), graph.const_buffer());

    // the produce() of TICLBarrelGraphEdgeScoreProducer
    A::ticl::TracksterGraphEdgeScoreDeviceCollection scores(queue, nEdges);
    std::deque<A::graphmodel::BatchIO> batches;
    if (nEdges > 0) {
      A::PortableCollection<::ticl::graphmodel::GraphModelInputSoA> inputs(queue, nGraphs);
      A::PortableCollection<::ticl::graphmodel::GraphModelOutputSoA> outputs(queue, nGraphs);
      auto inputBuffer = inputs.buffer();
      alpaka::memset(queue, inputBuffer, 0x00);
      A::graphmodel::launchFillModelInputs(
          queue, dTracksters.const_view(), dProperties.const_view(), dGraph.const_view(), inputs.view());
      A::graphmodel::runGraphModel(queue, model, inputs.view(), outputs.view(), graphsPerBatch, batches);
      A::graphmodel::launchGatherEdgeScores(queue, dGraph.const_view(), outputs.const_view(), scores.view(), true);
      alpaka::wait(queue);
    }
    ::ticl::TracksterGraphEdgeScoreHostCollection hostScores(queue, nEdges);
    alpaka::memcpy(queue, hostScores.buffer(), scores.const_buffer());
    alpaka::wait(queue);

    std::printf("EVENT");
    for (int32_t e = 0; e < nEdges; ++e)
      std::printf(" %.9g", hostScores.view()[e].score());
    std::printf("\n");
  }
  return 0;
}
