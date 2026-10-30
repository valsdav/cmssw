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
    P p{};
    p.seedEtThreshold = config.getParameter<double>("seedEtThreshold");
    // copies the polynomial coefficients of one window parameter
    auto coefficients = [&config](char const* name, float* c, int32_t& n) {
      auto const values = config.getParameter<std::vector<double>>(name);
      if (values.empty() or values.size() > P::kMaxCoefficients)
        throw cms::Exception("Configuration")
            << "TracksterGraphProducer: " << name << " needs 1.." << P::kMaxCoefficients << " coefficients";
      n = values.size();
      for (int32_t k = 0; k < n; ++k)
        c[k] = values[k];
    };
    coefficients("windowDEtaLow", p.detaLow, p.nDetaLow);
    coefficients("windowDEtaHigh", p.detaHigh, p.nDetaHigh);
    coefficients("windowDPhiMax", p.dphiMax, p.nDphiMax);
    return p;
  }

  void TracksterGraphProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersSoACLUE3DBarrel"));
    desc.add<edm::InputTag>("barrelProperties", edm::InputTag("ticlTracksterBarrelProperties"))
        ->setComment("only the tracksters with an ECAL layer cluster are nodes");
    desc.add<double>("seedEtThreshold", 1.0)->setComment("seed: et >= seedEtThreshold [GeV]");
    // The windows are polynomials in |eta_seed|, coefficients from the constant term up. The defaults are the windows
    // of the training (ticl-egm-linking, configs/graph_builder/graph_builder_cff.py): deta_lo = -0.08,
    // deta_hi = 0.08 + 0.1 / 1.479^2 * |eta|^2, dphi = (|eta| / 6 + 1.2) / 2.
    desc.add<std::vector<double>>("windowDEtaLow", {-0.08})
        ->setComment("lowest (eta - eta_seed) * sign(eta_seed), as a polynomial in |eta_seed|");
    desc.add<std::vector<double>>("windowDEtaHigh", {0.08, 0., 0.1 / (1.479 * 1.479)})
        ->setComment("highest (eta - eta_seed) * sign(eta_seed), as a polynomial in |eta_seed|");
    desc.add<std::vector<double>>("windowDPhiMax", {0.6, 1. / 12.})
        ->setComment("largest |phi - phi_seed|, as a polynomial in |eta_seed|");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterGraphProducer);
