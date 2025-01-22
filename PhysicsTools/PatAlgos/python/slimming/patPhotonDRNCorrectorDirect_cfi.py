from RecoEgamma.EgammaTools.patPhotonDRNCorrectionProducerDirect_cfi import patPhotonDRNCorrectionProducerDirect

import FWCore.ParameterSet.Config as cms 

patPhotonsDRNDirect = patPhotonDRNCorrectionProducerDirect.clone(
                            particleSource = 'selectedPatPhotons',
                            rhoName = 'fixedGridRhoFastjetAll',
                            modelPath ='RecoEgamma/EgammaPhotonProducers/data/models/photonObjectCombined/1/model.pt',
                    )