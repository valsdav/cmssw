// TracksterHostCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/TracksterHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGroupHostCollection.h"

#include <algorithm>
#include <vector>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Converts the tracksters built on the device to ticl::Trackster, in the same order, and masks their layer clusters
// in a copy of the original layer cluster mask, as TrackstersProducer does. The layer clusters of the tracksters are
// indices in the merged layer clusters, which index the same layer clusters as hgcalMergeLayerClusters.
// If the tracksters are groups of other tracksters (for example the superclusters, one trackster per group), the
// optional groups input gives the indices of the grouped tracksters: they are written as
// linkedTracksterIdToInputTracksterId, as TracksterLinksProducer does, for EGammaSuperclusterProducer.
class TracksterSoAToLegacyConverter : public edm::global::EDProducer<> {
public:
  TracksterSoAToLegacyConverter(const edm::ParameterSet& config)
      : tracksterSoAToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        originalMaskToken_(consumes(config.getParameter<edm::InputTag>("originalMask"))),
        trackstersToken_(produces<std::vector<ticl::Trackster>>()),
        maskToken_(produces<std::vector<float>>()) {
    const auto groups = config.getParameter<edm::InputTag>("groups");
    if (not groups.label().empty()) {
      groupsToken_ = consumes(groups);
      linksToken_ = produces<std::vector<std::vector<unsigned int>>>("linkedTracksterIdToInputTracksterId");
    }
  }

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    const auto tracksterSoA = iEvent.get(tracksterSoAToken_).const_view();
    const auto vertices = tracksterSoA.vertices();
    std::vector<float> mask = iEvent.get(originalMaskToken_);

    std::vector<ticl::Trackster> tracksters(vertices.keys());
    for (unsigned int i = 0; i < tracksters.size(); ++i) {
      auto& trackster = tracksters[i];

      // the device fills the layer clusters of a trackster in no particular order
      const auto layerClusters = vertices[i];
      trackster.vertices().assign(layerClusters.begin(), layerClusters.end());
      std::ranges::sort(trackster.vertices());
      trackster.vertex_multiplicity().assign(layerClusters.size(), 1.f);
      for (const auto layerCluster : trackster.vertices()) {
        if (layerCluster >= mask.size()) {
          throw edm::Exception(edm::errors::LogicError)
              << "TracksterSoAToLegacyConverter: a trackster uses layer cluster " << layerCluster << " of "
              << mask.size() << ". The layer clusters must be merged on the device in the order of the legacy ones.";
        }
        mask[layerCluster] = 0.f;
      }

      const auto energy = tracksterSoA.energy()[i];
      trackster.setRawEnergy(energy.rawEnergy());
      trackster.setRawEmEnergy(energy.rawEmEnergy());
      trackster.setRegressedEnergy(energy.regressedEnergy());
      const auto barycenter = tracksterSoA.barycenter()[i];
      trackster.setBarycenter(ticl::Trackster::Vector(barycenter.x(), barycenter.y(), barycenter.z()));
      trackster.calculateRawPt();
      trackster.calculateRawEmPt();
      trackster.setTimeAndError(tracksterSoA.timing()[i].time(), tracksterSoA.timing()[i].timeError());

      // fillPCAVariables takes the squares of the spreads, and keeps the order and orientation of the device
      const auto pca = tracksterSoA.pca()[i];
      trackster.fillPCAVariables(pca.eigenvalues(),
                                 pca.eigenvectors(),
                                 pca.sigmas().cwiseAbs2(),
                                 pca.sigmasPCA().cwiseAbs2(),
                                 3,
                                 ticl::Trackster::PCAOrdering::descending);
      ticl::TracksterIdProbabilities idProbabilities = tracksterSoA.id()[i].idProbabilities();
      trackster.setProbabilities(idProbabilities.data());
      // The iteration index keeps its default value, which TrackstersProducer also gives the iterations it does not name
    }

    if (not groupsToken_.isUninitialized()) {
      const auto groups = iEvent.get(groupsToken_).const_view();
      if (static_cast<std::size_t>(groups.keys()) != tracksters.size()) {
        throw edm::Exception(edm::errors::LogicError)
            << "TracksterSoAToLegacyConverter: " << groups.keys() << " groups for " << tracksters.size()
            << " tracksters. There must be one trackster per group.";
      }
      std::vector<std::vector<unsigned int>> links(groups.keys());
      for (unsigned int g = 0; g < links.size(); ++g) {
        const auto members = groups[g];
        links[g].assign(members.begin(), members.end());
      }
      iEvent.emplace(linksToken_, std::move(links));
    }

    iEvent.emplace(trackstersToken_, std::move(tracksters));
    iEvent.emplace(maskToken_, std::move(mask));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("ticlSoATracksters"));
    desc.add<edm::InputTag>("originalMask", edm::InputTag("hgcalMergeLayerClusters", "InitialLayerClustersMask"))
        ->setComment("The mask of the layer clusters before the tracksters: theirs are set to 0 in the output mask.");
    desc.add<edm::InputTag>("groups", edm::InputTag())
        ->setComment("Optional: the trackster groups of which the tracksters are made, one trackster per group. If set, "
                     "their indices are written as linkedTracksterIdToInputTracksterId.");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<ticl::TracksterHostCollection> tracksterSoAToken_;
  const edm::EDGetTokenT<std::vector<float>> originalMaskToken_;
  const edm::EDPutTokenT<std::vector<ticl::Trackster>> trackstersToken_;
  const edm::EDPutTokenT<std::vector<float>> maskToken_;
  edm::EDGetTokenT<ticl::TracksterGroupHostCollection> groupsToken_;
  edm::EDPutTokenT<std::vector<std::vector<unsigned int>>> linksToken_;
};

DEFINE_FWK_MODULE(TracksterSoAToLegacyConverter);
