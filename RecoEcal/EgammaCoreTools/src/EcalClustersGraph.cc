#include "RecoEcal/EgammaCoreTools/interface/EcalClustersGraph.h"
#include <algorithm>
#include "TVector2.h"


using namespace std;

EcalClustersGraph::EcalClustersGraph(CalibratedClusterPtrVector clusters, int nSeeds):
        _clusters(clusters), _nSeeds(nSeeds){
    _nCls = _clusters.size();
    _inWindows = ublas::matrix<int> (_nSeeds, _nCls);
    _scoreMatrix = ublas::matrix<int> (_nSeeds, _nCls);

    //test
    std::cout << "ClustersGraph created. Nseeds "<< _nSeeds << " nClusters "<< _nCls << endl;
}

void EcalClustersGraph::initWindows(double etaWidth, double phiWidth){

    for (int is=0; is < _nSeeds; is++){
        double seed_eta = _clusters.at(is)->eta();
        double seed_phi = _clusters.at(is)->phi();
        _inWindows(is,is) = 1;

        for (int icl=is+1 ; icl < _nCls; icl++){
            double cl_eta = _clusters.at(icl)->eta();
            double cl_phi = _clusters.at(icl)->phi();
            double dphi = std::abs(TVector2::Phi_mpi_pi(seed_phi - cl_phi));  
            
            if( std::abs(cl_eta - seed_eta) <= etaWidth && dphi <= phiWidth){
                _inWindows(is, icl) = 1;
                //Save also symmetric part of the adj matrix
                if (icl < _nSeeds)  _inWindows(icl,is) = 1;
            }else{
                 _inWindows(is, icl) = 0;
                 if (icl < _nSeeds)  _inWindows(icl,is) = 0;
            }
        }
    }

    //test
    for (int is=0; is < _nSeeds; is++){
        for (int ic=0; ic < _nCls; ic++){
            cout << _inWindows(is,ic) << " ";
        }
        cout << endl;
    }
    

}