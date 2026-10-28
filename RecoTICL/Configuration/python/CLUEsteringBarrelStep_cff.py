import FWCore.ParameterSet.Config as cms

from RecoTICL.PatternRecognition.barrelSoATrackstersProducer_cfi import barrelSoATrackstersProducer as _barrelSoATrackstersProducer
from RecoTICL.PatternRecognition.barrelTrackstersToSoAProducer_cfi import barrelTrackstersToSoAProducer as _barrelTrackstersToSoAProducer
from RecoTICL.Superclustering.tracksterBarrelPropertiesProducer_cfi import tracksterBarrelPropertiesProducer as _tracksterBarrelPropertiesProducer

# The legacy barrel tracksters of CLUE3D and the legacy merged layer clusters are copied to the SoA, and the ECAL and
# HCAL properties of the barrel tracksters are computed on the device.
ticlBarrelTrackstersToSoA = _barrelTrackstersToSoAProducer.clone(
    tracksters = 'ticlTrackstersCLUE3DBarrel',
    layerClusters = 'hgcalMergeLayerClusters',
    layerClustersTime = 'hgcalMergeLayerClusters:timeLayerCluster'
)

ticlTracksterBarrelProperties = _tracksterBarrelPropertiesProducer.clone(
    src = 'ticlBarrelTrackstersToSoA',
    layerClusters = 'ticlBarrelTrackstersToSoA'
)

ticlBarrelSoAFromLegacyTask = cms.Task(ticlBarrelTrackstersToSoA
    ,ticlTracksterBarrelProperties)

# The barrel tracksters built on the device from the barrel layer clusters on the device and their clustering.
# These are not produced yet: set the two labels below to them, and the src and layerClusters of
# ticlTracksterBarrelProperties to ticlSoATrackstersBarrel and to the layer clusters.
_barrelLayerClustersSoA = 'barrelSoALayerClusters'
_barrelLayerClustersTrackstersSoA = 'ticlCLUEsteringBarrel'

ticlSoATrackstersBarrel = _barrelSoATrackstersProducer.clone(
    layerClustersSoA = _barrelLayerClustersSoA,
    layerClustersTrackstersSoA = _barrelLayerClustersTrackstersSoA
)

ticlCLUEsteringBarrelStepTask = cms.Task(ticlSoATrackstersBarrel)
