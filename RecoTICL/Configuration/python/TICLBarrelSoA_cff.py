import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustering.layerClusterToSoAConverter_cfi import layerClusterToSoAConverter as _layerClusterToSoAConverter
from RecoTICL.PatternRecognition.tracksterSoAProducer_cfi import tracksterSoAProducer as _tracksterSoAProducer
from RecoTICL.PatternRecognition.tracksterToSoAConverter_cfi import tracksterToSoAConverter as _tracksterToSoAConverter
from RecoTICL.Superclustering.tracksterBarrelPropertiesProducer_cfi import tracksterBarrelPropertiesProducer as _tracksterBarrelPropertiesProducer

# The legacy CLUE3D barrel tracksters and the legacy merged layer clusters are copied to the SoA, and the ECAL and
# HCAL properties of the barrel tracksters are computed on the device
ticlTrackstersSoACLUE3DBarrel = _tracksterToSoAConverter.clone(
    src = 'ticlTrackstersCLUE3DBarrel'
)

ticlLayerClustersSoA = _layerClusterToSoAConverter.clone(
    layerClusters = 'hgcalMergeLayerClusters',
    layerClustersTime = 'hgcalMergeLayerClusters:timeLayerCluster',
    geometryLabel = 'withBarrel'
)

ticlTracksterBarrelProperties = _tracksterBarrelPropertiesProducer.clone(
    src = 'ticlTrackstersSoACLUE3DBarrel',
    layerClusters = 'ticlLayerClustersSoA'
)

ticlBarrelSoAFromLegacyTask = cms.Task(ticlTrackstersSoACLUE3DBarrel
    ,ticlLayerClustersSoA
    ,ticlTracksterBarrelProperties)

# The barrel tracksters built on the device, from the barrel layer clusters on the device and their clustering. These
# are not produced yet: set the two inputs to them, and the inputs of ticlTracksterBarrelProperties to this module and
# to the layer clusters.
ticlSoATrackstersBarrel = _tracksterSoAProducer.clone(
    hgcalLayerClustersSoA = 'barrelSoALayerClusters',
    hgcalLayerClustersTrackstersSoA = 'ticlCLUEsteringBarrel',
    isBarrel = True
)
