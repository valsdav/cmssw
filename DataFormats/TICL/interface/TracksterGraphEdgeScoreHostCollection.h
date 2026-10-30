#ifndef DataFormats_TICL_interface_TracksterGraphEdgeScoreHostCollection_h
#define DataFormats_TICL_interface_TracksterGraphEdgeScoreHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoreSoA.h"

namespace ticl {

  using TracksterGraphEdgeScoreHostCollection = PortableHostCollection<TracksterGraphEdgeScoreSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphEdgeScoreHostCollection_h
