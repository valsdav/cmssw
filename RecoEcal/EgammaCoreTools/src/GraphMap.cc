#include "RecoEcal/EgammaCoreTools/interface/GraphMap.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

GraphMap::GraphMap(uint nNodes, const std::vector<uint> &categories) : nNodes_(nNodes) {
  // One entry for node in the edges_ list
  edgesIn_.resize(nNodes);
  edgesOut_.resize(nNodes);
}

void GraphMap::addNode(const uint &index, const uint &category) {
  nodesCategories_[category].push_back(index);
  nodesCount_[category] += 1;
}

void GraphMap::addNodes(const std::vector<uint> &indeces, const std::vector<uint> &categories) {
  if (indeces.size() != categories.size()) {
    throw cms::Exception("Adding nodes to GraphMap with wrong size");
  }
  for (size_t i = 0; i < indeces.size(); i++) {
    addNode(indeces.at(i), categories.at(i));
  }
}

void GraphMap::addEdge(const uint &i, const uint &j) {
  // The first index is the starting node of the outcoming edge.
  edgesOut_.at(i).push_back(j);
  edgesIn_.at(j).push_back(i);
  adjMatrix_[{i, j}] = 1.;
}

void GraphMap::setAdjMatrix(const uint &i, const uint &j, const float &score) {
  adjMatrix_[{i, j}] = score;
};

void GraphMap::setAdjMatrixSym(const uint &i, const uint &j, const float &score) {
  adjMatrix_[{i, j}] = score;
  adjMatrix_[{j, i}] = score;
};

//Getters
const std::vector<uint> & GraphMap::getOutEdges(const uint &i) const {
  return edgesOut_.at(i);
};

const std::vector<uint> & GraphMap::getInEdges(const uint &i) const {
  return edgesIn_.at(i);
};

uint GraphMap::getAdjMatrix(const uint &i, const uint &j) const {
  return adjMatrix_.at({i, j});
};

std::vector<float> GraphMap::getAdjMatrixRow(const uint &i) const {
  std::vector<float> out;
  for (const auto &j : getOutEdges(i)) {
    out.push_back(adjMatrix_.at({i, j}));
  }
  return out;
};

std::vector<float> GraphMap::getAdjMatrixCol(const uint &j) const {
  std::vector<float> out;
  for (const auto &i : getInEdges(j)) {
    out.push_back(adjMatrix_.at({i, j}));
  }
  return out;
};


//--------------------------------------------------------------

std::vector<std::pair<uint, std::vector<uint>>> GraphMap::collectNodes(GraphMap::CollectionStrategy strategy, float threshold){
  std::vector<std::pair<uint, std::vector<uint>>> out;

    if (strategy == GraphMap::CollectionStrategy::A){
      const auto & superNodes = nodesCategories_[1];
      // superNodes are already included in order
      for(const auto & s : superNodes){
        std::vector<uint> collectedNodes;
        // Check if the seed if still available
        if (adjMatrix_[{s,s}] < threshold) continue;
        // Loop on the out-coming edges 
        for (const auto & out : edgesOut_[s]){
          // Check the threshold for association
          if (adjMatrix_[{s, out}] >= threshold){
            // Save the node
            collectedNodes.push_back(out);
            // Remove all incoming edges to the selected node
            // So that it cannot be taken from other SuperNodes
            for (const auto & out_in : edgesIn_[out] ){
              // Since there is a self loop on the superNode
              // this will select also other superNodes pointing to the
              // current superNode.
              // Also the self-loop on the correct superNodes is removed. 
              // The edge is removed in the adjacency matrix score
              adjMatrix_[{out_in, out}] = 0.;
            }
            // Remove also any self-loop so that if the out node is superNode,
            //it cannot be used in the next iteration
            adjMatrix_[{out,out}] = 0.;
          }
        }
        out.push_back({s, collectedNodes});
      }
    }
    return out;

}

