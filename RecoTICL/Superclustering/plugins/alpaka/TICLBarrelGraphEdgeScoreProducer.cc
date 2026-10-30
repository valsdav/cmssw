// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <deque>
#include <utility>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphEdgeScoresDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphEdgeModelSoA.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphEdgeModel.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphEdgeScoreAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Scores every edge of the trackster graphs with the graph edge model (TorchScript, PyTorchAlpaka), on the device:
  //   1. model inputs, one row per graph: node features and adjacency (TracksterGraphEdgeScoreAlgo)
  //   2. inference, no copy of the inputs (TracksterGraphEdgeModel.h)
  //   3. one score per edge, in the order of the edges block
  // The model is written by ticl-egm-linking/scripts/export_cmssw.py. The process needs the PyTorchService.
  class TICLBarrelGraphEdgeScoreProducer : public stream::FixedQueueEDProducer<> {
  public:
    using ModelInputCollection = PortableCollection<::ticl::graphmodel::GraphModelInputSoA>;
    using ModelOutputCollection = PortableCollection<::ticl::graphmodel::GraphModelOutputSoA>;

    explicit TICLBarrelGraphEdgeScoreProducer(edm::ParameterSet const& config)
        : FixedQueueEDProducer(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          propertiesToken_{consumes(config.getParameter<edm::InputTag>("barrelProperties"))},
          graphsToken_{consumes(config.getParameter<edm::InputTag>("graphs"))},
          scoresToken_{produces()},
          model_{config.getParameter<edm::FileInPath>("model").fullPath()},
          graphsPerBatch_{config.getParameter<int>("graphsPerBatch")},
          applySigmoid_{config.getParameter<bool>("applySigmoid")} {
      // a batch must start on an aligned row: 32 floats = 128 bytes
      if (graphsPerBatch_ < 0 or graphsPerBatch_ % 32 != 0)
        throw cms::Exception("Configuration") << "TICLBarrelGraphEdgeScoreProducer: graphsPerBatch must be 0 (one "
                                              << "batch) or a positive multiple of 32, not " << graphsPerBatch_;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void beginStream(edm::StreamID, Queue queue) override;
    void produce(device::Event& iEvent, device::EventSetup const&) override;

    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    device::EDGetToken<ticl::TracksterBarrelPropertiesDeviceCollection> const propertiesToken_;
    device::EDGetToken<ticl::TracksterGraphDeviceCollection> const graphsToken_;
    device::EDPutToken<ticl::TracksterGraphEdgeScoresDeviceCollection> const scoresToken_;
    torch::AlpakaModel model_;
    int const graphsPerBatch_;
    bool const applySigmoid_;
    static constexpr int kWarmupIterations = 3;
  };

  void TICLBarrelGraphEdgeScoreProducer::beginStream(edm::StreamID, Queue queue) {
    // moves the model to the device and freezes it, on valid dummy graphs
    int const n = graphsPerBatch_ > 0 ? graphsPerBatch_ : 32;
    ModelInputCollection inputs(queue, n);
    ModelOutputCollection outputs(queue, n);
    auto inputBuffer = inputs.buffer();  // memset needs an lvalue buffer on the GPU backends
    alpaka::memset(queue, inputBuffer, 0x00);
    graphmodel::launchFillWarmupInputs(queue, inputs.view());
    for (int it = 0; it < kWarmupIterations; ++it) {
      std::deque<graphmodel::BatchIO> batches;
      graphmodel::runGraphModel(queue, model_, inputs.view(), outputs.view(), 0, batches);
    }
  }

  void TICLBarrelGraphEdgeScoreProducer::produce(device::Event& iEvent, device::EventSetup const&) {
    auto& queue = iEvent.queue();
    auto const tracksters = iEvent.get(trackstersToken_).const_view();
    auto const properties = iEvent.get(propertiesToken_).const_view();
    auto const graph = iEvent.get(graphsToken_).const_view();
    if (properties.metadata().size() != tracksters.energy().metadata().size()) {
      throw cms::Exception("LogicError") << "TICLBarrelGraphEdgeScoreProducer: "
                                         << tracksters.energy().metadata().size() << " tracksters but "
                                         << properties.metadata().size() << " barrel properties";
    }

    // the sizes of the graph blocks are known on the host: no synchronisation
    int32_t const nGraphs = graph.graphs().metadata().size();
    int32_t const nEdges = graph.edges().metadata().size();
    ticl::TracksterGraphEdgeScoresDeviceCollection scores(queue, nEdges);
    if (nEdges == 0) {
      iEvent.emplace(scoresToken_, std::move(scores));
      return;
    }

    ModelInputCollection inputs(queue, nGraphs);
    ModelOutputCollection outputs(queue, nGraphs);
    auto inputBuffer = inputs.buffer();  // memset needs an lvalue buffer on the GPU backends
    alpaka::memset(queue, inputBuffer, 0x00);
    graphmodel::launchFillModelInputs(queue, tracksters, properties, graph, inputs.view());

    std::deque<graphmodel::BatchIO> batches;
    graphmodel::runGraphModel(queue, model_, inputs.view(), outputs.view(), graphsPerBatch_, batches);

    graphmodel::launchGatherEdgeScores(queue, graph, outputs.const_view(), scores.view(), applySigmoid_);
    iEvent.emplace(scoresToken_, std::move(scores));
  }

  void TICLBarrelGraphEdgeScoreProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersSoACLUE3DBarrel"));
    desc.add<edm::InputTag>("barrelProperties", edm::InputTag("ticlTracksterBarrelProperties"));
    desc.add<edm::InputTag>("graphs", edm::InputTag("ticlTracksterGraph"));
    // no default: a FileInPath default must exist when the descriptions are filled
    desc.add<edm::FileInPath>("model")->setComment("TorchScript file from ticl-egm-linking/scripts/export_cmssw.py");
    desc.add<int>("graphsPerBatch", 0)->setComment("0 = all the graphs in one call, else a multiple of 32");
    desc.add<bool>("applySigmoid", true)->setComment("store sigmoid(logit) instead of the logit");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TICLBarrelGraphEdgeScoreProducer);
