#ifndef DataFormats_TICL_interface_TracksterGroupSoA_h
#define DataFormats_TICL_interface_TracksterGroupSoA_h

// Groups of tracksters, for example the superclusters: for each group, the indices of its tracksters. Key g is
// group g; its values are rows of the TracksterSoA that the groups were made from, the seed first, then the others in
// ascending index. A TracksterSoA made from the groups (one trackster per group, with the layer clusters of its
// members) has the same row order as the keys.

#include <cstdint>

#include "DataFormats/TICL/interface/AssociationMap.h"

namespace ticl {

  using TracksterGroupSoA = AssociationMap<uint32_t, uint32_t>;
  using TracksterGroupSoAView = TracksterGroupSoA::View;
  using TracksterGroupSoAConstView = TracksterGroupSoA::ConstView;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_TracksterGroupSoA_h
