#ifndef DataFormats_TICL_interface_LayerClusterToTracksterHostCollection_h
#define DataFormats_TICL_interface_LayerClusterToTracksterHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/LayerClusterToTracksterSoA.h"

namespace ticl {

  using LayerClusterToTracksterHostCollection = PortableHostCollection<LayerClusterToTracksterSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_LayerClusterToTracksterHostCollection_h
