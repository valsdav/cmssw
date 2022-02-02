#include "RecoEcal/EgammaCoreTools/interface/GraphMap.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

GraphMap::GraphMap(uint nNodes, const std::vector<uint> & categories)
  : nNodes_(nNodes)
{
  for(const uint & c: categories){
    nodesCount_[c] = 0;
    nodesCategory_[c] = {};
  }
  // One entry for node in the edges_ list
  edgesIn_.resize(nNodes);
  edgesOut_.resize(nNodes);
}


void GraphMap::addNode(const uint & index, const uint& category){
  nodesCategory_[category].push_back(index); // --> Not sure this works
  nodesCount_[category] += 1;
}

void GraphMap::addNodes(const std::vector<uint> & indeces, const std::vector<uint> & categories){
  if (indeces.size() != categories.size()){
    throw cms::Exception("Adding nodes to GraphMap with wrong size");
  }
  for(size_t i=0; i<indeces.size(); i++){
    addNode(indeces.at(i), categories.at(i));
  }
}

void GraphMap::addEdge(const uint & i, const uint & j){
  // The first index is the starting node of the outcoming edge. 
  edgesOut_.at(i).push_back(j);
  edgesIn_.at(j).push_back(i);
  adjMatrix_[{i,j}] = 1.; 
}


