// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

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
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Copies the legacy barrel tracksters and the legacy merged layer clusters to the SoA, so that the SoA barrel
  // modules can run on the output of the CPU barrel pattern recognition. All the merged layer clusters are copied in
  // their legacy order, so that the trackster vertices keep their indices.
  class BarrelTrackstersToSoAProducer : public stream::EDProducer<> {
  public:
    BarrelTrackstersToSoAProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("tracksters"))},
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClusters"))},
          layerClustersTimeToken_{consumes(config.getParameter<edm::InputTag>("layerClustersTime"))},
          ticlGeomToken_{
              consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(edm::ESInputTag("", "withBarrel"))},
          ticlGeomLookupToken_{consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>(
              edm::ESInputTag("", "withBarrel"))},
          ticlGeomLayersToken_{
              consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>(edm::ESInputTag("", ""))},
          layerClustersSoAToken_{produces()},
          trackstersSoAToken_{produces()} {}

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      const auto& tracksters = iEvent.get(trackstersToken_);
      const auto& layerClusters = iEvent.get(layerClustersToken_);
      const auto& layerClustersTime = iEvent.get(layerClustersTimeToken_);

      ticlgeom::Tools geometry;
      geometry.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));

      // The layer clusters. The layer is not stored in reco::CaloCluster: as on the CPU, it is the layer of the first
      // hit (ECAL barrel 0, HCAL barrel its depth)
      const uint32_t nLayerClusters = layerClusters.size();
      ::reco::CaloClusterHostCollection layerClustersSoA(
          iEvent.queue(), nLayerClusters, nLayerClusters, nLayerClusters, nLayerClusters);
      auto lcView = layerClustersSoA.view();
      for (uint32_t i = 0; i < nLayerClusters; ++i) {
        const auto& lc = layerClusters[i];
        auto position = lcView.position()[i];
        position.x() = lc.x();
        position.y() = lc.y();
        position.z() = lc.z();
        position.layer() =
            lc.hitsAndFractions().empty() ? -1 : static_cast<int>(geometry.getLayerWithOffset(lc.hitsAndFractions()[0].first));
        position.cells() = lc.hitsAndFractions().size();
        auto energy = lcView.energy()[i];
        energy.energy() = lc.energy();
        energy.correctedEnergy() = lc.correctedEnergy();
        energy.correctedEnergyUncertainty() = lc.correctedEnergyUncertainty();
        auto indexes = lcView.indexes()[i];
        indexes.caloID() = lc.caloID();
        indexes.algoID() = lc.algo();
        indexes.seedID() = lc.seed();
        indexes.flags() = lc.flags();
        const auto& [time, timeError] = layerClustersTime.get(i);
        lcView.timing()[i].time() = time;
        lcView.timing()[i].timeError() = timeError;
      }

      // The tracksters
      const uint32_t nTracksters = tracksters.size();
      uint32_t nVertices = 0;
      for (const auto& trackster : tracksters) {
        nVertices += trackster.vertices().size();
      }
      ::ticl::TracksterHostCollection trackstersSoA(
          iEvent.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nVertices);
      auto tView = trackstersSoA.view();
      auto vertices = tView.vertices();
      uint32_t vertex = 0;
      vertices.offsets()[0].keys_offsets() = 0;
      for (uint32_t t = 0; t < nTracksters; ++t) {
        const auto& trackster = tracksters[t];
        // TracksterSoA has no vertex multiplicity: the barrel CLUE3D does not share layer clusters, so it is always 1
        for (const auto lc : trackster.vertices()) {
          if (lc >= nLayerClusters) {
            throw cms::Exception("LogicError") << "BarrelTrackstersToSoAProducer: trackster " << t << " uses layer cluster "
                                               << lc << " of " << nLayerClusters
                                               << ". The tracksters must index the layerClusters collection.";
          }
          vertices.content()[vertex++].values() = lc;
        }
        vertices.offsets()[t + 1].keys_offsets() = vertex;

        auto energy = tView.energy()[t];
        energy.rawEnergy() = trackster.raw_energy();
        energy.rawEmEnergy() = trackster.raw_em_energy();
        energy.regressedEnergy() = trackster.regressed_energy();

        auto barycenter = tView.barycenter()[t];
        barycenter.x() = trackster.barycenter().x();
        barycenter.y() = trackster.barycenter().y();
        barycenter.z() = trackster.barycenter().z();

        tView.timing()[t].time() = trackster.time();
        tView.timing()[t].timeError() = trackster.timeError();

        // The legacy trackster keeps the PCA in decreasing eigenvalue order, as the SoA does
        auto pca = tView.pca()[t];
        for (int k = 0; k < 3; ++k) {
          pca.eigenvalues()(k) = trackster.eigenvalues()[k];
          const auto& axis = trackster.eigenvectors(k);
          pca.eigenvectors().col(k) = Eigen::Vector3f(axis.x(), axis.y(), axis.z());
          pca.sigmas()(k) = trackster.sigmas()[k];
          pca.sigmasPCA()(k) = trackster.sigmasPCA()[k];
        }

        for (int k = 0; k < ::ticl::TracksterIdProbabilities::RowsAtCompileTime; ++k) {
          tView.id()[t].idProbabilities()(k) = trackster.id_probabilities(k);
        }
      }

      if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
        iEvent.emplace(layerClustersSoAToken_, std::move(layerClustersSoA));
        iEvent.emplace(trackstersSoAToken_, std::move(trackstersSoA));
      } else {
        reco::CaloClusterDeviceCollection layerClustersDevice(
            iEvent.queue(), nLayerClusters, nLayerClusters, nLayerClusters, nLayerClusters);
        alpaka::memcpy(iEvent.queue(), layerClustersDevice.buffer(), layerClustersSoA.const_buffer());
        ticl::TracksterDeviceCollection trackstersDevice(
            iEvent.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nVertices);
        alpaka::memcpy(iEvent.queue(), trackstersDevice.buffer(), trackstersSoA.const_buffer());
        iEvent.emplace(layerClustersSoAToken_, std::move(layerClustersDevice));
        iEvent.emplace(trackstersSoAToken_, std::move(trackstersDevice));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DBarrel"));
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"))
          ->setComment("The legacy layer clusters that the tracksters index");
      desc.add<edm::InputTag>("layerClustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<std::vector<::ticl::Trackster>> trackstersToken_;
    const edm::EDGetTokenT<std::vector<::reco::CaloCluster>> layerClustersToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> layerClustersTimeToken_;
    const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const device::EDPutToken<reco::CaloClusterDeviceCollection> layerClustersSoAToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> trackstersSoAToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelTrackstersToSoAProducer);
