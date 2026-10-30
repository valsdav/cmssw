#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TICL/interface/LayerClusterToTracksterHostCollection.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoresHostCollection.h"
#include "DataFormats/TICL/interface/TracksterHostCollection.h"

// TracksterHostCollection uses SoA blocks, which the PortableHostCollection read rules do not support yet
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::TracksterBarrelPropertiesHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::TracksterGraphEdgeScoresHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::LayerClusterToTracksterHostCollection);
