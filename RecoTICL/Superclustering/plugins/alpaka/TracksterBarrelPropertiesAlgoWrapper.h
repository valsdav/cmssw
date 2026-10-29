#ifndef RecoTICL_Superclustering_plugins_alpaka_TracksterBarrelPropertiesAlgoWrapper_h
#define RecoTICL_Superclustering_plugins_alpaka_TracksterBarrelPropertiesAlgoWrapper_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers.
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TracksterBarrelPropertiesAlgoWrapper {
  public:
    void run(Queue& queue,
             ticl::TracksterDeviceCollection::ConstView tracksters,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             ticl::TracksterBarrelPropertiesDeviceCollection::View properties) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_plugins_alpaka_TracksterBarrelPropertiesAlgoWrapper_h
