#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Copies the legacy layer clusters to the layer cluster SoA, in their legacy order, so that the indices of the legacy
  // tracksters stay valid. reco::CaloCluster has no layer: as on the CPU, it is the layer of the first hit.
  class LayerClusterToSoAConverter : public stream::EDProducer<> {
  public:
    LayerClusterToSoAConverter(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          layerClustersTimeToken_{consumes(config.getParameter<edm::InputTag>("layerClustersTime"))},
          ticlGeomToken_{consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(
              edm::ESInputTag("", config.getParameter<std::string>("geometryLabel")))},
          ticlGeomLookupToken_{consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>(
              edm::ESInputTag("", config.getParameter<std::string>("geometryLabel")))},
          ticlGeomLayersToken_{
              consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>(edm::ESInputTag("", ""))},
          layerClustersSoAToken_{produces()} {}

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      const auto& layerClusters = iEvent.get(layerClustersToken_);
      const auto& layerClustersTime = iEvent.get(layerClustersTimeToken_);

      ticlgeom::Tools geometry;
      geometry.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));

      const uint32_t n = layerClusters.size();
      ::reco::CaloClusterHostCollection output(iEvent.queue(), n, n, n, n);
      auto view = output.view();
      for (uint32_t i = 0; i < n; ++i) {
        const auto& lc = layerClusters[i];
        auto position = view.position()[i];
        position.x() = lc.x();
        position.y() = lc.y();
        position.z() = lc.z();
        position.layer() = lc.hitsAndFractions().empty()
                               ? -1
                               : static_cast<int>(geometry.getLayerWithOffset(lc.hitsAndFractions()[0].first));
        position.cells() = lc.hitsAndFractions().size();
        auto energy = view.energy()[i];
        energy.energy() = lc.energy();
        energy.correctedEnergy() = lc.correctedEnergy();
        energy.correctedEnergyUncertainty() = lc.correctedEnergyUncertainty();
        auto indexes = view.indexes()[i];
        indexes.caloID() = lc.caloID();
        indexes.algoID() = lc.algo();
        indexes.seedID() = lc.seed();
        indexes.flags() = lc.flags();
        const auto& [time, timeError] = layerClustersTime.get(i);
        view.timing()[i].time() = time;
        view.timing()[i].timeError() = timeError;
      }

      if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
        iEvent.emplace(layerClustersSoAToken_, std::move(output));
      } else {
        reco::CaloClusterDeviceCollection deviceOutput(iEvent.queue(), n, n, n, n);
        alpaka::memcpy(iEvent.queue(), deviceOutput.buffer(), output.const_buffer());
        iEvent.emplace(layerClustersSoAToken_, std::move(deviceOutput));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
      desc.add<edm::InputTag>("layerClustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
      desc.add<std::string>("geometryLabel", "withBarrel")
          ->setComment("Label of the TICL geometry, which must contain the detectors of the layer clusters");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<std::vector<::reco::CaloCluster>> layerClustersToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> layerClustersTimeToken_;
    const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const device::EDPutToken<reco::CaloClusterDeviceCollection> layerClustersSoAToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(LayerClusterToSoAConverter);
