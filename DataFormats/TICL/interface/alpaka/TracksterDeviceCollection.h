#ifndef DataFormats_TICL_interface_alpaka_TracksterDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterDeviceCollection_h

#include "DataFormats/TICL/interface/TracksterSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterDeviceCollection = PortableCollection<::ticl::TracksterSoA>;
  using TracksterBarrelPropertiesDeviceCollection = PortableCollection<::ticl::TracksterBarrelPropertiesSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterDeviceCollection_h
