#ifndef DataFormats_TICL_interface_TracksterGroupHostCollection_h
#define DataFormats_TICL_interface_TracksterGroupHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGroupSoA.h"

namespace ticl {

  // constructed with (queue, nGroups, nMembers)
  using TracksterGroupHostCollection = PortableHostCollection<TracksterGroupSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGroupHostCollection_h
