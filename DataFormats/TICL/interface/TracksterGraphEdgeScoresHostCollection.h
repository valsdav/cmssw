#ifndef DataFormats_TICL_interface_TracksterGraphEdgeScoresHostCollection_h
#define DataFormats_TICL_interface_TracksterGraphEdgeScoresHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoresSoA.h"

namespace ticl {

  using TracksterGraphEdgeScoresHostCollection = PortableHostCollection<TracksterGraphEdgeScoresSoA>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGraphEdgeScoresHostCollection_h
