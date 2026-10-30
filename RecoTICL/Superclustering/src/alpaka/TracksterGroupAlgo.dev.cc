// The superclustering on the trackster graphs.
//
// count():
//   KernelInit          et of each trackster, no owner
//   KernelSeeds         the graph nodes with et >= seedEtThreshold, compacted in any order
//   KernelRankSeeds     each seed counts the seeds before it (larger et; equal et: smaller index) -> sorted order
//   KernelEdgeRanges    the out-edges of a seed are one contiguous range of the edges block of its graph (rows sorted
//                       by (localA, localB)): the range boundaries
//   KernelResolve       one thread: the greedy resolution in descending seed et
// fill():
//   KernelFillGroups            one thread per group: offsets and members (seed first, then ascending index)
//   KernelFillLayerClusterMap   one thread per trackster in a group: its layer clusters -> the group
//
// The results do not depend on the order of the threads.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGroupAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using ::ticl::graph::GroupCounters;
  using ::ticl::graph::GroupParams;

  namespace {

    constexpr int32_t kThreads = 256;

    // Number of int32 scratch arrays of size nTracksters (the group offsets have nTracksters + 1 entries)
    constexpr int32_t kNInts = 8;

    // Raw pointers into the scratch buffers. A trackster is a member of at most one group, so the arrays per group and
    // per member have at most nTracksters entries.
    struct Scratch {
      float* et;
      int32_t* canSeed;      // 1 for the seeds
      int32_t* seeds;        // the seeds, compacted in any order
      int32_t* sorted;       // the seeds in descending et
      int32_t* edgeBegin;    // per seed: its out-edges are the rows [edgeBegin, edgeEnd) of the edges block
      int32_t* edgeEnd;
      int32_t* owner;        // per trackster: its group, -1 for none
      int32_t* members;      // the members of the groups, group after group, the seed first, in the order taken
      int32_t* groupOffset;  // per group, + 1: where its members start in members
      GroupCounters* counters;
    };

    Scratch makeScratch(int32_t* ints, float* et, GroupCounters* counters, int32_t n) {
      Scratch s;
      s.et = et;
      s.canSeed = ints;
      s.seeds = ints + n;
      s.sorted = ints + 2 * n;
      s.edgeBegin = ints + 3 * n;
      s.edgeEnd = ints + 4 * n;
      s.owner = ints + 5 * n;
      s.members = ints + 6 * n;
      s.groupOffset = ints + 7 * n;  // n + 1 entries: the last array
      s.counters = counters;
      return s;
    }

    auto workDiv(int32_t n) { return make_workdiv<Acc1D>(divide_up_by(n, kThreads), kThreads); }

    struct KernelInit {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::ticl::TracksterSoAConstView tracksters, int32_t n, Scratch s) const {
        for (int32_t t : uniform_elements(acc, n)) {
          auto const barycenter = tracksters.barycenter()[t];
          s.et[t] = ::ticl::graph::barycenterKinematics(
                        barycenter.x(), barycenter.y(), barycenter.z(), tracksters.energy()[t].rawEnergy())
                        .et;
          s.owner[t] = -1;
        }
      }
    };

    // A seed of the graph building is a node of exactly one graph; canSeed makes each seed count once also otherwise.
    struct KernelSeeds {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterGraphSoAConstView graph,
                                    float seedEtThreshold,
                                    Scratch s) const {
        auto const nodes = graph.nodes();
        for (int32_t k : uniform_elements(acc, nodes.metadata().size())) {
          int32_t const t = nodes[k].tracksterIdx();
          if (s.et[t] >= seedEtThreshold and alpaka::atomicExch(acc, &s.canSeed[t], 1) == 0)
            s.seeds[alpaka::atomicAdd(acc, &s.counters->nSeeds, 1)] = t;
        }
      }
    };

    struct KernelRankSeeds {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, int32_t n, Scratch s) const {
        int32_t const nSeeds = s.counters->nSeeds;
        // ponytail: O(nSeeds^2) comparisons, in parallel; a sort if the number of seeds grows a lot
        for (int32_t i : uniform_elements(acc, n)) {
          if (i >= nSeeds)
            continue;
          int32_t const t = s.seeds[i];
          float const et = s.et[t];
          int32_t rank = 0;
          for (int32_t j = 0; j < nSeeds; ++j) {
            int32_t const u = s.seeds[j];
            rank += (s.et[u] > et or (s.et[u] == et and u < t)) ? 1 : 0;
          }
          s.sorted[rank] = t;
        }
      }
    };

    struct KernelEdgeRanges {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, ::ticl::TracksterGraphSoAConstView graph, Scratch s) const {
        auto const edges = graph.edges();
        int32_t const nEdges = edges.metadata().size();
        for (int32_t e : uniform_elements(acc, nEdges)) {
          int32_t const a = edges[e].tracksterA();
          if (s.canSeed[a] == 0)
            continue;
          // the rows of one source node are contiguous inside its graph
          auto sameSource = [&](int32_t other) {
            return edges[other].graphIdx() == edges[e].graphIdx() and edges[other].localA() == edges[e].localA();
          };
          if (e == 0 or not sameSource(e - 1))
            s.edgeBegin[a] = e;
          if (e == nEdges - 1 or not sameSource(e + 1))
            s.edgeEnd[a] = e + 1;
        }
      }
    };

    // ponytail: the greedy resolution runs in one thread, as each decision depends on the earlier ones; one thread per
    // graph with atomic claims of the shared followers if the timing needs it
    struct KernelResolve {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterGraphSoAConstView graph,
                                    ::ticl::TracksterGraphEdgeScoresSoA::ConstView scores,
                                    GroupParams params,
                                    Scratch s) const {
        if (not once_per_grid(acc))
          return;
        auto const edges = graph.edges();
        int32_t const nSeeds = s.counters->nSeeds;
        int32_t nGroups = 0;
        int32_t nMembers = 0;
        for (int32_t k = 0; k < nSeeds; ++k) {
          int32_t const seed = s.sorted[k];
          if (s.owner[seed] >= 0)
            continue;
          int32_t const g = nGroups++;
          int32_t const first = nMembers;
          s.groupOffset[g] = first;
          s.owner[seed] = g;
          s.members[nMembers++] = seed;
          // direct: only the group seed takes; transitive: also each seed that the group takes
          int32_t const last = params.transitive ? -1 : first + 1;
          for (int32_t c = first; c < nMembers and c != last; ++c) {
            int32_t const m = s.members[c];
            if (s.canSeed[m] == 0)
              continue;
            for (int32_t e = s.edgeBegin[m]; e < s.edgeEnd[m]; ++e) {
              int32_t const b = edges[e].tracksterB();
              if (s.owner[b] < 0 and scores[e].egmLinkingProbability() > params.linkingThreshold) {
                s.owner[b] = g;
                s.members[nMembers++] = b;
              }
            }
          }
        }
        s.groupOffset[nGroups] = nMembers;
        s.counters->nGroups = nGroups;
        s.counters->nMembers = nMembers;
      }
    };

    struct KernelFillGroups {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, int32_t nGroups, Scratch s, ::ticl::TracksterGroupSoAView groups) const {
        uint32_t* content = groups.content().values().data();
        for (int32_t g : uniform_elements(acc, nGroups)) {
          int32_t const begin = s.groupOffset[g];
          int32_t const end = s.groupOffset[g + 1];
          if (g == 0)
            groups.offsets()[0].keys_offsets() = 0;
          groups.offsets()[g + 1].keys_offsets() = end;
          // the seed first, then the other members in ascending index (insertion sort: the groups are small)
          content[begin] = s.members[begin];
          for (int32_t i = begin + 1; i < end; ++i) {
            uint32_t const value = s.members[i];
            int32_t j = i;
            for (; j > begin + 1 and content[j - 1] > value; --j)
              content[j] = content[j - 1];
            content[j] = value;
          }
        }
      }
    };

    struct KernelFillLayerClusterMap {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterSoAConstView tracksters,
                                    int32_t n,
                                    int32_t nGroups,
                                    Scratch s,
                                    ::ticl::LayerClusterToTracksterSoA::View layerClusters) const {
        if (once_per_grid(acc))
          layerClusters.numberOfTrackstersScalar() = nGroups;
        auto const vertices = tracksters.vertices();
        int32_t const nLayerClusters = layerClusters.metadata().size();
        for (int32_t t : uniform_elements(acc, n)) {
          int32_t const g = s.owner[t];
          if (g < 0)
            continue;
          // a barrel layer cluster is in only one trackster, so no two threads write the same row
          for (uint32_t const layerCluster : vertices[t]) {
            if (static_cast<int32_t>(layerCluster) < nLayerClusters)
              layerClusters[layerCluster].tracksterIndex() = g;
          }
          alpaka::atomicAdd(
              acc, &layerClusters.numberOfVerticesScalar(), static_cast<unsigned int>(vertices.count(t)));
        }
      }
    };

  }  // namespace

  void TracksterGroupAlgo::count(Queue& queue,
                                 ::ticl::TracksterSoAConstView tracksters,
                                 ::ticl::TracksterGraphSoAConstView graph,
                                 ::ticl::TracksterGraphEdgeScoresSoA::ConstView scores) {
    n_ = tracksters.energy().metadata().size();
    hostCounters_.emplace(make_host_buffer<GroupCounters>(queue));
    if (n_ == 0 or graph.nodes().metadata().size() == 0) {
      *hostCounters_->data() = GroupCounters{0, 0, 0};
      return;
    }

    ints_.emplace(make_device_buffer<int32_t[]>(queue, kNInts * n_ + 1));
    et_.emplace(make_device_buffer<float[]>(queue, n_));
    deviceCounters_.emplace(make_device_buffer<GroupCounters>(queue));
    alpaka::memset(queue, *ints_, 0x00);
    alpaka::memset(queue, *deviceCounters_, 0x00);
    Scratch const s = makeScratch(ints_->data(), et_->data(), deviceCounters_->data(), n_);

    alpaka::exec<Acc1D>(queue, workDiv(n_), KernelInit{}, tracksters, n_, s);
    alpaka::exec<Acc1D>(
        queue, workDiv(graph.nodes().metadata().size()), KernelSeeds{}, graph, params_.seedEtThreshold, s);
    alpaka::exec<Acc1D>(queue, workDiv(n_), KernelRankSeeds{}, n_, s);
    if (graph.edges().metadata().size() > 0)
      alpaka::exec<Acc1D>(queue, workDiv(graph.edges().metadata().size()), KernelEdgeRanges{}, graph, s);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, 1), KernelResolve{}, graph, scores, params_, s);

    alpaka::memcpy(queue, *hostCounters_, *deviceCounters_);
  }

  void TracksterGroupAlgo::fill(Queue& queue,
                                ::ticl::TracksterSoAConstView tracksters,
                                ::ticl::TracksterGroupSoAView groups,
                                ::ticl::LayerClusterToTracksterSoA::View layerClusters) {
    int32_t const nGroups = counters().nGroups;
    // no group: the offsets of the groups start at 0, and no layer cluster is in a group
    auto firstOffset = make_device_view(queue, groups.offsets().keys_offsets().data(), 1);
    alpaka::memset(queue, firstOffset, 0x00);
    int32_t const nLayerClusters = layerClusters.metadata().size();
    if (nLayerClusters > 0) {
      auto index = make_device_view(queue, layerClusters.tracksterIndex().data(), nLayerClusters);
      alpaka::memset(queue, index, 0xff);  // -1
    }
    auto nVertices = make_device_view(queue, &layerClusters.numberOfVerticesScalar(), 1);
    alpaka::memset(queue, nVertices, 0x00);
    auto nTracksters = make_device_view(queue, &layerClusters.numberOfTrackstersScalar(), 1);
    alpaka::memset(queue, nTracksters, 0x00);
    if (nGroups == 0)
      return;

    Scratch const s = makeScratch(ints_->data(), et_->data(), deviceCounters_->data(), n_);
    alpaka::exec<Acc1D>(queue, workDiv(nGroups), KernelFillGroups{}, nGroups, s, groups);
    alpaka::exec<Acc1D>(
        queue, workDiv(n_), KernelFillLayerClusterMap{}, tracksters, n_, nGroups, s, layerClusters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
