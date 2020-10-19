/**
   \file
   Tools for manipulating ECAL Clusters as graphs

   \author Davide Valsecchi, Badder Marzocchi
   \date 05 October 2020
*/

#ifndef _EcalClustersGraph__
#define _EcalClustersGraph__

#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

class EcalClustersGraph {

  typedef std::shared_ptr<PFECALSuperClusterAlgo::CalibratedPFCluster> CalibratedClusterPtr;
  typedef std::vector<PFECALSuperClusterAlgo::CalibratedClusterPtr> CalibratedClusterPtrVector;

private:

   CalibratedClusterPtrVector _clusters;
   int _nSeeds;
   int _nCls;

   // Adjacency matrix defining which clusters are inside the seeds windows.
   // row: seeds (Et ordered), column: clusters (Et ordered)
   ublas::matrix<int> _inWindows;
   // Adjacency matrix defining how much each cluster is linked to the seed
   // row: seeds (Et ordered), column: clusters (Et ordered)
   ublas::matrix<double> _scoreMatrix;


public:

   EcalClustersGraph(CalibratedClusterPtrVector clusters, int nSeeds);

   void initWindows(double etaWidth, double phiWidth);

};

#endif