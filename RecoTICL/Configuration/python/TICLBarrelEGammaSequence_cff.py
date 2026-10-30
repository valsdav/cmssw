import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustering.layerClusterToSoAConverter_cfi import layerClusterToSoAConverter as _layerClusterToSoAConverter
from RecoTICL.PatternRecognition.tracksterSoAProducer_cfi import tracksterSoAProducer as _tracksterSoAProducer
from RecoTICL.PatternRecognition.tracksterToSoAConverter_cfi import tracksterToSoAConverter as _tracksterToSoAConverter
from RecoTICL.Superclustering.tracksterBarrelPropertiesProducer_cfi import tracksterBarrelPropertiesProducer as _tracksterBarrelPropertiesProducer
from RecoTICL.Superclustering.tracksterGraphProducer_cfi import tracksterGraphProducer as _tracksterGraphProducer
from RecoTICL.Superclustering.ticlBarrelGraphEdgeScoreProducer_cfi import ticlBarrelGraphEdgeScoreProducer as _ticlBarrelGraphEdgeScoreProducer
from RecoTICL.Superclustering.ticlBarrelGraphSuperclusterProducer_cfi import ticlBarrelGraphSuperclusterProducer as _ticlBarrelGraphSuperclusterProducer
from RecoTICL.PatternRecognition.tracksterSoAToLegacyConverter_cfi import tracksterSoAToLegacyConverter as _tracksterSoAToLegacyConverter
from RecoTICL.Superclustering.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer as _ticlEGammaSuperClusterProducer

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

# The graphs of the barrel tracksters with an ECAL layer cluster, for the graph model
# The seed threshold and the windows are the defaults of TracksterGraphProducer (see its fillDescriptions)
ticlTracksterGraph = _tracksterGraphProducer.clone(
    tracksters = 'ticlTrackstersSoACLUE3DBarrel',
    barrelProperties = 'ticlTracksterBarrelProperties'
)

ticlBarrelSoAFromLegacyTask = cms.Task(ticlTrackstersSoACLUE3DBarrel
    ,ticlLayerClustersSoA
    ,ticlTracksterBarrelProperties
    ,ticlTracksterGraph)

# One score per graph edge, from the graph edge model. In its own task: it needs the model file and the
# PyTorchService (process.add_(cms.Service("PyTorchService"))). The model is exported with
# ticl-egm-linking/scripts/export_cmssw.py; there is no trained model in the release yet.
ticlBarrelGraphEdgeScore = _ticlBarrelGraphEdgeScoreProducer.clone(
    tracksters = 'ticlTrackstersSoACLUE3DBarrel',
    barrelProperties = 'ticlTracksterBarrelProperties',
    graphs = 'ticlTracksterGraph',
    model = cms.FileInPath('RecoTICL/Superclustering/data/TICLBarrelGraphEdgeModel.pt')
)

ticlBarrelGraphEdgeScoreTask = cms.Task(ticlBarrelGraphEdgeScore)

# The superclustering on the graphs, with the edge scores (it needs ticlBarrelGraphEdgeScoreTask):
# - the superclusters as groups of barrel tracksters, and the supercluster of each layer cluster (on the device);
# - one trackster per supercluster, with the layer clusters of its tracksters and the energy, barycenter and PCA
#   computed again from them (on the device);
# - the legacy supercluster tracksters and linkedTracksterIdToInputTracksterId, as TracksterLinksProducer writes them
#   for the HGCAL superclustering, and the legacy SuperClusters.
ticlBarrelGraphSupercluster = _ticlBarrelGraphSuperclusterProducer.clone(
    tracksters = 'ticlTrackstersSoACLUE3DBarrel',
    graphs = 'ticlTracksterGraph',
    edgeScores = 'ticlBarrelGraphEdgeScore',
    layerClusters = 'ticlLayerClustersSoA'
)

ticlTrackstersSoASuperclustersBarrel = _tracksterSoAProducer.clone(
    layerClusters = 'ticlLayerClustersSoA',
    layerClusterToTrackster = 'ticlBarrelGraphSupercluster',
    isBarrel = True
)

ticlTracksterLinksSuperclusteringBarrelGraph = _tracksterSoAToLegacyConverter.clone(
    src = 'ticlTrackstersSoASuperclustersBarrel',
    groups = 'ticlBarrelGraphSupercluster'
)

# The supercluster regression model is for HGCAL
ticlEGammaSuperClusterProducerBarrel = _ticlEGammaSuperClusterProducer.clone(
    ticlSuperClusters = 'ticlTracksterLinksSuperclusteringBarrelGraph',
    ticlTrackstersEM = 'ticlTrackstersCLUE3DBarrel',
    layerClusters = 'hgcalMergeLayerClusters',
    isBarrel = True,
    enableRegression = False
)

ticlBarrelGraphSuperclusteringTask = cms.Task(ticlBarrelGraphSupercluster
    ,ticlTrackstersSoASuperclustersBarrel
    ,ticlTracksterLinksSuperclusteringBarrelGraph
    ,ticlEGammaSuperClusterProducerBarrel)

# The barrel tracksters built on the device, from the barrel layer clusters on the device and their clustering. These
# are not produced yet: set the two inputs to them, and the inputs of ticlTracksterBarrelProperties to this module and
# to the layer clusters.
ticlSoATrackstersBarrel = _tracksterSoAProducer.clone(
    layerClusters = 'barrelSoALayerClusters',
    layerClusterToTrackster = 'ticlCLUEsteringBarrel',
    isBarrel = True
)
