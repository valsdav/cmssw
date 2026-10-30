#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TICL/interface/TracksterGraphEdgeScoreHostCollection.h"
#include "DataFormats/TICL/interface/TracksterHostCollection.h"

// TracksterHostCollection uses SoA blocks, which the PortableHostCollection read rules do not support yet
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::TracksterBarrelPropertiesHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(ticl::TracksterGraphEdgeScoreHostCollection);
