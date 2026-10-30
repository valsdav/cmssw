#ifndef RecoTICL_Superclustering_interface_TracksterGraphEdgeModelSoA_h
#define RecoTICL_Superclustering_interface_TracksterGraphEdgeModelSoA_h

// Input and output of the graph edge model, as SoAs that PyTorchAlpaka wraps as tensors without a copy.
//
// ONE ROW PER GRAPH. A fixed-size Eigen column of shape [R, C] becomes a tensor [rows, R, C] in TensorCollection:
//
//   GraphModelInputSoA
//     nodeFeatures  [kMaxNodes, kNodeFeatures]  ->  "node_features" [graphs, kMaxNodes, kNodeFeatures]
//     adjacency     [kMaxNodes, kMaxNodes]      ->  "adjacency"     [graphs, kMaxNodes, kMaxNodes]
//   GraphModelOutputSoA
//     pairLogits    [kMaxNodes, kMaxNodes]      <-  "pair_logits"   [graphs, kMaxNodes, kMaxNodes]
//
// Contract with the exported model (ticl-egm-linking/scripts/export_cmssw.py):
//   - local node i of graph g is row i (local index = row in the nodes block - nodeOffset of the graph);
//   - nodeFeatures are the raw features of fillNodeFeatures(); the model applies the training normalisation;
//   - padding rows: features 0, adjacency row and column 0;
//   - adjacency(i, j) = 1 if there is an edge i -> j; adjacency(i, i) = 1 for a real node, 0 for padding (the
//     diagonal is the padding mask, so the model needs no third input);
//   - pairLogits(i, j) = logit of the edge i -> j, 0 where there is no edge (the model evaluates only the edges).
//
// These are module buffers, not event products: no dictionaries.

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl::graphmodel {

  // ponytail: graphs with more than kMaxNodes nodes are not evaluated (their edges get score -1); split the graphs
  // in a small and a large batch when the real graph sizes need it.
  constexpr int kMaxNodes = 50;
  constexpr int kNodeFeatures = 10;

  using NodeFeatureMatrix = Eigen::Matrix<float, kMaxNodes, kNodeFeatures>;
  using NodePairMatrix = Eigen::Matrix<float, kMaxNodes, kMaxNodes>;

  GENERATE_SOA_LAYOUT(GraphModelInputLayout,
                      SOA_EIGEN_COLUMN(NodeFeatureMatrix, nodeFeatures),
                      SOA_EIGEN_COLUMN(NodePairMatrix, adjacency))
  using GraphModelInputSoA = GraphModelInputLayout<>;

  GENERATE_SOA_LAYOUT(GraphModelOutputLayout, SOA_EIGEN_COLUMN(NodePairMatrix, pairLogits))
  using GraphModelOutputSoA = GraphModelOutputLayout<>;

}  // namespace ticl::graphmodel

#endif  // RecoTICL_Superclustering_interface_TracksterGraphEdgeModelSoA_h
