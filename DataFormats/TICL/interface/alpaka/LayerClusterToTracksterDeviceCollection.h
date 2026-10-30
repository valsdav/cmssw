#ifndef DataFormats_TICL_interface_alpaka_LayerClusterToTracksterDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_LayerClusterToTracksterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/LayerClusterToTracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using LayerClusterToTracksterDeviceCollection = PortableCollection<::ticl::LayerClusterToTracksterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_LayerClusterToTracksterDeviceCollection_h
