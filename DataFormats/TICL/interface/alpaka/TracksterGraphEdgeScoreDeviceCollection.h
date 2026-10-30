#ifndef DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoreDeviceCollection_h
#define DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoreDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoreSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterGraphEdgeScoreDeviceCollection = PortableCollection<::ticl::TracksterGraphEdgeScoreSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_TICL_interface_alpaka_TracksterGraphEdgeScoreDeviceCollection_h
