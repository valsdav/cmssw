#ifndef DataFormats_TICL_interface_alpaka_TracksterGroupDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterGroupDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterGroupSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  // constructed with (queue, nGroups, nMembers)
  using TracksterGroupDeviceCollection = PortableCollection<::ticl::TracksterGroupSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterGroupDeviceCollection_h
