#ifndef DataFormats_TICL_interface_LayerClusterToTracksterSoA_h
#define DataFormats_TICL_interface_LayerClusterToTracksterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  // Clustering of the layer clusters into tracksters: the trackster of each layer cluster (-1 for none), the number of
  // tracksters and the number of layer clusters in them. TracksterSoAProducer builds the TracksterSoA from it.
  GENERATE_SOA_LAYOUT(LayerClusterToTracksterLayout,
                      SOA_COLUMN(int, tracksterIndex),
                      SOA_SCALAR(unsigned int, numberOfTrackstersScalar),
                      SOA_SCALAR(unsigned int, numberOfVerticesScalar))

  using LayerClusterToTracksterSoA = LayerClusterToTracksterLayout<>;

}  // namespace ticl

#endif  // DataFormats_TICL_interface_LayerClusterToTracksterSoA_h
