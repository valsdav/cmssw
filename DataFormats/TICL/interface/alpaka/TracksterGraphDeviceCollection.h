#ifndef DataFormats_TICL_interface_alpaka_TracksterGraphDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterGraphDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterGraphDeviceCollection = PortableCollection<::ticl::TracksterGraphSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterGraphDeviceCollection_h
