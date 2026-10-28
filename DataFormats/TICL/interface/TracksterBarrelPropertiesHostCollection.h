#ifndef DataFormats_TICL_interface_TracksterBarrelPropertiesHostCollection_h
#define DataFormats_TICL_interface_TracksterBarrelPropertiesHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/TracksterBarrelPropertiesSoA.h"

namespace ticl {

  using TracksterBarrelPropertiesHostCollection = PortableHostCollection<TracksterBarrelPropertiesSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterBarrelPropertiesHostCollection_h
