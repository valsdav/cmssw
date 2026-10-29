#include <cstdint>
#include <utility>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TracksterBarrelPropertiesAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterBarrelPropertiesProducer : public stream::EDProducer<> {
  public:
    explicit TracksterBarrelPropertiesProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("src"))},
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          outputToken_{produces()} {}

    void produce(device::Event& event, const device::EventSetup&) override {
      const auto tracksters = event.get(trackstersToken_).const_view();
      const auto layerClusters = event.get(layerClustersToken_).const_view();
      const uint32_t nTracksters = tracksters.energy().metadata().size();

      ticl::TracksterBarrelPropertiesDeviceCollection output(event.queue(), nTracksters);
      algo_.run(event.queue(), tracksters, layerClusters, output.view());
      event.emplace(outputToken_, std::move(output));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("ticlTrackstersSoACLUE3DBarrel"))
          ->setComment("Device SoA barrel tracksters.");
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("ticlLayerClustersSoA"))
          ->setComment("Merged device layer-cluster SoA, in the order indexed by src.vertices().");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<ticl::TracksterDeviceCollection> trackstersToken_;
    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDPutToken<ticl::TracksterBarrelPropertiesDeviceCollection> outputToken_;
    const TracksterBarrelPropertiesAlgoWrapper algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterBarrelPropertiesProducer);
