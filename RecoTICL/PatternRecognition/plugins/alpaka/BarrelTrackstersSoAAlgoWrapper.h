#ifndef RecoTICL_PatternRecognition_plugins_alpaka_BarrelTrackstersSoAAlgoWrapper_h
#define RecoTICL_PatternRecognition_plugins_alpaka_BarrelTrackstersSoAAlgoWrapper_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BarrelTrackstersSoAAlgoWrapper {
  public:
    // Fills the barrel tracksters from the clustering of the layer clusters: the layer clusters of each trackster, and
    // its energy, barycenter, time and principal components, as ticl::assignPCAtoTracksters does for the barrel with
    // energy weighting, the PCA cleaning and the local time, except for the spreads. The ECAL layer clusters (layer 0)
    // count to the electromagnetic energy. The sums over the layer clusters are atomic, so on a GPU their order, and
    // the results at the rounding level, vary from run to run.
    void run(Queue& queue,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
             ticl::TracksterDeviceCollection::View tracksters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_PatternRecognition_plugins_alpaka_BarrelTrackstersSoAAlgoWrapper_h
