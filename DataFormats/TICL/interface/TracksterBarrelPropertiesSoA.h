#ifndef DataFormats_TICL_interface_TracksterBarrelPropertiesSoA_h
#define DataFormats_TICL_interface_TracksterBarrelPropertiesSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  // The ECAL and HCAL properties of the barrel tracksters, one row per trackster
  GENERATE_SOA_LAYOUT(TracksterBarrelPropertiesSoALayout,
                      SOA_COLUMN(float, ecalX),
                      SOA_COLUMN(float, ecalY),
                      SOA_COLUMN(float, ecalZ),
                      SOA_COLUMN(float, ecalEta),
                      SOA_COLUMN(float, ecalPhi),
                      SOA_COLUMN(float, hcalBarycenterX),
                      SOA_COLUMN(float, hcalBarycenterY),
                      SOA_COLUMN(float, hcalBarycenterZ),
                      SOA_COLUMN(float, hcalBarycenterEta),
                      SOA_COLUMN(float, hcalBarycenterPhi),
                      SOA_COLUMN(uint32_t, nEcalClusters),
                      SOA_COLUMN(uint32_t, nHcalClusters))

  using TracksterBarrelPropertiesSoA = TracksterBarrelPropertiesSoALayout<>;
  using TracksterBarrelPropertiesSoAView = TracksterBarrelPropertiesSoA::View;
  using TracksterBarrelPropertiesSoAConstView = TracksterBarrelPropertiesSoA::ConstView;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterBarrelPropertiesSoA_h
