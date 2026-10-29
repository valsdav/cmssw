// Checks the layout of the trackster graphs on real events, and optionally prints them in the format of
// testTracksterGraphAlgo, so that they can be compared offline with the Python graph builder. The graphs cannot be
// read back from a file (the read rules do not support SoA blocks yet), so they are checked in the job.

#include <set>
#include <utility>

#include "DataFormats/TICL/interface/TracksterGraphHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

class TracksterGraphValidator : public edm::global::EDAnalyzer<> {
public:
  explicit TracksterGraphValidator(edm::ParameterSet const& config)
      : graphsToken_{consumes(config.getParameter<edm::InputTag>("graphs"))}, dump_{config.getParameter<bool>("dump")} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("graphs", edm::InputTag("ticlTracksterGraph"));
    desc.add<bool>("dump", false)->setComment("print all the graphs, nodes and edges");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
    auto const view = event.get(graphsToken_).const_view();
    int32_t const nGraphs = view.graphs().metadata().size();
    int32_t const nNodes = view.nodes().metadata().size();
    int32_t const nEdges = view.edges().metadata().size();
    auto check = [&event](bool ok, char const* what, int32_t g) {
      if (not ok)
        throw cms::Exception("TracksterGraphValidator")
            << "event " << event.id() << ", graph " << g << ": " << what;
    };

    int32_t nodeOffset = 0, edgeOffset = 0;
    for (int32_t g = 0; g < nGraphs; ++g) {
      auto const graph = view.graphs()[g];
      check(graph.nodeOffset() == nodeOffset and graph.edgeOffset() == edgeOffset, "offsets not contiguous", g);
      check(graph.nNodes() > 0, "graph without nodes", g);
      std::set<std::pair<int32_t, int32_t>> links;
      for (int32_t k = 0; k < graph.nNodes(); ++k) {
        auto const node = view.nodes()[nodeOffset + k];
        check(node.graphIdx() == g, "node of another graph", g);
        check(k == 0 or view.nodes()[nodeOffset + k - 1].tracksterIdx() < node.tracksterIdx(),
              "nodes not in ascending trackster index",
              g);
      }
      for (int32_t k = 0; k < graph.nEdges(); ++k) {
        auto const edge = view.edges()[edgeOffset + k];
        check(edge.graphIdx() == g, "edge of another graph", g);
        check(edge.localA() >= 0 and edge.localA() < graph.nNodes() and edge.localB() >= 0 and
                  edge.localB() < graph.nNodes() and edge.localA() != edge.localB(),
              "local index out of range",
              g);
        check(view.nodes()[nodeOffset + edge.localA()].tracksterIdx() == edge.tracksterA() and
                  view.nodes()[nodeOffset + edge.localB()].tracksterIdx() == edge.tracksterB(),
              "local index and trackster do not agree",
              g);
        if (k > 0) {
          auto const previous = view.edges()[edgeOffset + k - 1];
          check(std::pair(previous.localA(), previous.localB()) < std::pair(edge.localA(), edge.localB()),
                "edges not sorted by (localA, localB)",
                g);
        }
        links.emplace(edge.localA(), edge.localB());
      }
      for (auto const& [a, b] : links)
        check(links.count({b, a}) == 1, "edge without its reverse", g);
      nodeOffset += graph.nNodes();
      edgeOffset += graph.nEdges();
    }
    check(nodeOffset == nNodes and edgeOffset == nEdges, "the graphs do not cover all the nodes and edges", -1);

    // one message per event, so that the lines of different events do not mix in a multi-threaded job
    edm::LogPrint out("TracksterGraphValidator");
    out << "event " << event.id().run() << ":" << event.id().luminosityBlock() << ":" << event.id().event() << ": "
        << nGraphs << " graphs, " << nNodes << " nodes, " << nEdges << " edges";
    if (dump_) {
      out << "\nEVENT " << nGraphs << " " << nNodes << " " << nEdges << "\n";
      for (int32_t g = 0; g < nGraphs; ++g) {
        auto const graph = view.graphs()[g];
        out << "G " << graph.nodeOffset() << " " << graph.nNodes() << " " << graph.edgeOffset() << " "
            << graph.nEdges() << "\n";
      }
      for (int32_t k = 0; k < nNodes; ++k)
        out << "N " << view.nodes()[k].tracksterIdx() << " " << view.nodes()[k].graphIdx() << "\n";
      for (int32_t k = 0; k < nEdges; ++k) {
        auto const edge = view.edges()[k];
        out << "E " << edge.tracksterA() << " " << edge.tracksterB() << " " << edge.graphIdx() << " "
            << edge.localA() << " " << edge.localB() << "\n";
      }
    }
  }

private:
  edm::EDGetTokenT<ticl::TracksterGraphHostCollection> const graphsToken_;
  bool const dump_;
};

DEFINE_FWK_MODULE(TracksterGraphValidator);
