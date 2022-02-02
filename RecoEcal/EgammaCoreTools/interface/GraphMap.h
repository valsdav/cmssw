#ifndef RecoEcal_EgammaCoreTools_GraphMap_h
#define RecoEcal_EgammaCoreTools_GraphMap_h
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <array>
#include <map>
#include <algorithm>

/*
 * Class handling a sparse graph of clusters
 *
 */

namespace reco {
    class GraphMap {

    public:
      GraphMap(uint nNodes, const std::vector<uint> & categories);
      ~GraphMap(){};

      void addNode(const uint & index, const uint & category); 
      void addNodes(const std::vector<uint> & indices, const std::vector<uint> & categories);
      void addEdge(const uint &i, const uint &j);
      void setAdjMatrix(const uint &i, const uint &j, const float& score){
        adjMatrix_[{i,j}] = score;
      };
      void setAdjMatrixSym(const uint &i, const uint &j, const float& score){
        adjMatrix_[{i,j}] = score;
        adjMatrix_[{j,i}] = score;
      };
      

      //Getters
      const std::vector<uint> & getOutEdges(const uint & i) const{
        return edgesOut_.at(i);
      };
      const std::vector<uint> & getInEdges(const uint & i) const{
        return edgesIn_.at(i);
      };

      uint getAdjMatrix(const uint &i, const uint &j) const {
        return adjMatrix_.at({i,j});
      };

      std::vector<float> getAdjMatrixRow(const uint &i) const{
        std::vector<float> out;
        for (const auto & j : getOutEdges(i)){
          out.push_back(adjMatrix_.at({i, j}));
        }
        return out;
      };

      std::vector<float> getAdjMatrixCol(const uint &j) const{
        std::vector<float> out;
        for (const auto & i : getInEdges(j)){
          out.push_back(adjMatrix_.at({i, j}));
        }
        return out;
      };      
      
      
    private:
      uint nNodes_;
      // Map with list of indices of nodes for each category
      std::map<uint, std::vector<uint>> nodesCategory_;
      // Count of nodes for each category
      std::map<uint, uint> nodesCount_;
      // Incoming edges, one list for each node (no distinction between type)
      std::vector<std::vector<uint>> edgesIn_;
      // Outcoming edges, one list for each node
      std::vector<std::vector<uint>> edgesOut_;
      // Adjacency matrix (i,j) --> score
      // Rows are interpreted as OUT edges
      // Columns are interpreted as IN edges
      std::map<std::pair<uint,uint>, float> adjMatrix_;  
         
    };
    
}

#endif
