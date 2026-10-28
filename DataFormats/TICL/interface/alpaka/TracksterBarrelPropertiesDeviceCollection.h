#ifndef DataFormats_TICL_interface_alpaka_TracksterBarrelPropertiesDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterBarrelPropertiesDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterBarrelPropertiesSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterBarrelPropertiesDeviceCollection = PortableCollection<::ticl::TracksterBarrelPropertiesSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterBarrelPropertiesDeviceCollection_h
