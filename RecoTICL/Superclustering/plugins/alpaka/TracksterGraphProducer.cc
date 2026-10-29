// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <utility>
#include <vector>

#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the trackster graphs (the graph construction used in the training of the graph model). The sizes of the
  // output depend on the data, so the graphs are counted on the device in acquire(), and the output is allocated with
  // the exact sizes and filled in produce().
  class TracksterGraphProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit TracksterGraphProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          propertiesToken_{consumes(config.getParameter<edm::InputTag>("barrelProperties"))},
          graphsToken_{produces()},
          algo_{makeParams(config)} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void acquire(device::Event const& iEvent, device::EventSetup const&) override {
      auto const tracksters = iEvent.get(trackstersToken_).const_view();
      auto const properties = iEvent.get(propertiesToken_).const_view();
      if (properties.metadata().size() != tracksters.energy().metadata().size()) {
        throw cms::Exception("LogicError") << "TracksterGraphProducer: " << tracksters.energy().metadata().size()
                                           << " tracksters but " << properties.metadata().size()
                                           << " barrel properties";
      }
      algo_.count(iEvent.queue(), algo_.prepareInputs(iEvent.queue(), tracksters, properties));
    }

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      auto const& c = algo_.counters();  // on the host: the queue was synchronised after acquire()
      ticl::TracksterGraphDeviceCollection graphs(iEvent.queue(), c.nGraphs, c.nNodes, c.nEdges);
      algo_.fill(iEvent.queue(), graphs.view());
      iEvent.emplace(graphsToken_, std::move(graphs));
    }

  private:
    static ::ticl::graph::Params makeParams(edm::ParameterSet const& config);

    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    device::EDGetToken<ticl::TracksterBarrelPropertiesDeviceCollection> const propertiesToken_;
    device::EDPutToken<ticl::TracksterGraphDeviceCollection> const graphsToken_;
    TracksterGraphAlgo algo_;
  };

  ::ticl::graph::Params TracksterGraphProducer::makeParams(edm::ParameterSet const& config) {
    using P = ::ticl::graph::Params;
    auto const edges = config.getParameter<std::vector<double>>("windowAbsEtaEdges");
    auto const lo = config.getParameter<std::vector<double>>("windowDEtaLow");
    auto const hi = config.getParameter<std::vector<double>>("windowDEtaHigh");
    auto const dphi = config.getParameter<std::vector<double>>("windowDPhiMax");

    int32_t const nBins = static_cast<int32_t>(lo.size());
    if (nBins < 1 || nBins > P::kMaxEtaBins || hi.size() != lo.size() || dphi.size() != lo.size() ||
        edges.size() != lo.size() + 1)
      throw cms::Exception("Configuration")
          << "TracksterGraphProducer: need 1.." << P::kMaxEtaBins
          << " eta bins, windowAbsEtaEdges with nBins+1 entries and windowDEtaLow/High/DPhiMax with nBins entries";

    P p{};
    p.seedEtThreshold = config.getParameter<double>("seedEtThreshold");
    p.nEtaBins = nBins;
    for (int32_t b = 0; b <= nBins; ++b)
      p.absEtaEdges[b] = edges[b];
    for (int32_t b = 0; b < nBins; ++b) {
      p.detaLow[b] = lo[b];
      p.detaHigh[b] = hi[b];
      p.dphiMax[b] = dphi[b];
    }
    return p;
  }

  void TracksterGraphProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersSoACLUE3DBarrel"));
    desc.add<edm::InputTag>("barrelProperties", edm::InputTag("ticlTracksterBarrelProperties"))
        ->setComment("only the tracksters with an ECAL layer cluster are nodes");
    // TODO: the values used in the training (SEED_ET_THRESHOLD and windows_for_eta)
    desc.add<double>("seedEtThreshold", 1.0)->setComment("seed: et >= seedEtThreshold");
    desc.add<std::vector<double>>("windowAbsEtaEdges", {0., 1.5})->setComment("|eta_seed| bin edges (nBins+1)");
    desc.add<std::vector<double>>("windowDEtaLow", {-0.1})->setComment("lowest eta - eta_seed, per bin");
    desc.add<std::vector<double>>("windowDEtaHigh", {0.1})->setComment("highest eta - eta_seed, per bin");
    desc.add<std::vector<double>>("windowDPhiMax", {0.1})->setComment("largest |phi - phi_seed|, per bin");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterGraphProducer);
