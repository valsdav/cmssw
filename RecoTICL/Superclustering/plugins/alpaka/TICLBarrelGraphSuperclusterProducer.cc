// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <string>
#include <utility>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/LayerClusterToTracksterDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGraphEdgeScoresDeviceCollection.h"
#include "DataFormats/TICL/interface/alpaka/TracksterGroupDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGroupAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The superclustering of the barrel tracksters on their graphs, with the EGM linking probabilities of the edges.
  // Products:
  //   - TracksterGroupSoA: for each supercluster, its trackster indices (the seed first);
  //   - LayerClusterToTracksterSoA: the supercluster of each layer cluster, from which TracksterSoAProducer makes one
  //     trackster per supercluster, in the same order.
  // The sizes depend on the data, so the superclusters are resolved on the device in acquire(), and the outputs are
  // allocated with the exact sizes and filled in produce().
  class TICLBarrelGraphSuperclusterProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit TICLBarrelGraphSuperclusterProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          graphsToken_{consumes(config.getParameter<edm::InputTag>("graphs"))},
          scoresToken_{consumes(config.getParameter<edm::InputTag>("edgeScores"))},
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          groupsToken_{produces()},
          layerClusterToTracksterToken_{produces()},
          algo_{makeParams(config)} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void acquire(device::Event const& iEvent, device::EventSetup const&) override {
      auto const graph = iEvent.get(graphsToken_).const_view();
      auto const scores = iEvent.get(scoresToken_).const_view();
      if (scores.metadata().size() != graph.edges().metadata().size()) {
        throw cms::Exception("LogicError") << "TICLBarrelGraphSuperclusterProducer: " << graph.edges().metadata().size()
                                           << " graph edges but " << scores.metadata().size() << " edge scores";
      }
      algo_.count(iEvent.queue(), iEvent.get(trackstersToken_).const_view(), graph, scores);
    }

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      auto const& c = algo_.counters();  // on the host: the queue was synchronised after acquire()
      auto& queue = iEvent.queue();
      ticl::TracksterGroupDeviceCollection groups(queue, c.nGroups, c.nMembers);
      ticl::LayerClusterToTracksterDeviceCollection layerClusterToTrackster(
          queue, iEvent.get(layerClustersToken_).const_view().position().metadata().size());
      algo_.fill(queue, iEvent.get(trackstersToken_).const_view(), groups.view(), layerClusterToTrackster.view());
      iEvent.emplace(groupsToken_, std::move(groups));
      iEvent.emplace(layerClusterToTracksterToken_, std::move(layerClusterToTrackster));
    }

  private:
    static ::ticl::graph::GroupParams makeParams(edm::ParameterSet const& config) {
      auto const mode = config.getParameter<std::string>("linkMode");
      if (mode != "direct" and mode != "transitive")
        throw cms::Exception("Configuration")
            << "TICLBarrelGraphSuperclusterProducer: linkMode must be \"direct\" or \"transitive\", not \"" << mode
            << "\"";
      return {static_cast<float>(config.getParameter<double>("seedEtThreshold")),
              static_cast<float>(config.getParameter<double>("linkingThreshold")),
              mode == "transitive"};
    }

    device::EDGetToken<ticl::TracksterDeviceCollection> const trackstersToken_;
    device::EDGetToken<ticl::TracksterGraphDeviceCollection> const graphsToken_;
    device::EDGetToken<ticl::TracksterGraphEdgeScoresDeviceCollection> const scoresToken_;
    device::EDGetToken<reco::CaloClusterDeviceCollection> const layerClustersToken_;
    device::EDPutToken<ticl::TracksterGroupDeviceCollection> const groupsToken_;
    device::EDPutToken<ticl::LayerClusterToTracksterDeviceCollection> const layerClusterToTracksterToken_;
    TracksterGroupAlgo algo_;
  };

  void TICLBarrelGraphSuperclusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersSoACLUE3DBarrel"));
    desc.add<edm::InputTag>("graphs", edm::InputTag("ticlTracksterGraph"));
    desc.add<edm::InputTag>("edgeScores", edm::InputTag("ticlBarrelGraphEdgeScore"));
    desc.add<edm::InputTag>("layerClusters", edm::InputTag("ticlLayerClustersSoA"))
        ->setComment("the layer clusters of the tracksters: the size of the layer cluster -> supercluster map");
    desc.add<double>("seedEtThreshold", 1.0)
        ->setComment("seed of a supercluster: a graph node with et >= seedEtThreshold [GeV]; not lower than the seed "
                     "threshold of TracksterGraphProducer");
    desc.add<double>("linkingThreshold", 0.5)->setComment("a link is taken if its egmLinkingProbability is above");
    desc.add<std::string>("linkMode", "direct")
        ->setComment("direct: only the seed of a supercluster takes its neighbours; transitive: also each seed that "
                     "it takes (never a follower)");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TICLBarrelGraphSuperclusterProducer);
