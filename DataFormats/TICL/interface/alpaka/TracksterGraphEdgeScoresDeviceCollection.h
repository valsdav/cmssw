#ifndef DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoresDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoresDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoresSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterGraphEdgeScoresDeviceCollection = PortableCollection<::ticl::TracksterGraphEdgeScoresSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoresDeviceCollection_h
