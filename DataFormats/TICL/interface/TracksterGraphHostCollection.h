#ifndef DataFormats_TICL_interface_TracksterGraphHostCollection_h
#define DataFormats_TICL_interface_TracksterGraphHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphSoA.h"

namespace ticl {

  using TracksterGraphHostCollection = PortableHostCollection<TracksterGraphSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphHostCollection_h
