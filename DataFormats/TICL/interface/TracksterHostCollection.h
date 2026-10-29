#ifndef DataFormats_TICL_interface_TracksterHostCollection_h
#define DataFormats_TICL_interface_TracksterHostCollection_h

#include "DataFormats/TICL/interface/TracksterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ticl {

  using TracksterHostCollection = PortableHostCollection<TracksterSoA>;
  using TracksterBarrelPropertiesHostCollection = PortableHostCollection<TracksterBarrelPropertiesSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterHostCollection_h
