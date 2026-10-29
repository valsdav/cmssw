// The trackster graph building used in the training of the graph model.
//
// training (Python)                        | here
// -----------------------------------------+--------------------------------------------------------------
// seed_mask = et >= seed_et_threshold      | KernelInit           (only for selected tracksters)
// lo, hi, dphi_max = windows_for_eta(...)  | KernelInit           (per seed)
// a_sees_b | b_sees_a -> connected         | KernelLinkSeeds      (lock-free union-find)
// _connected_components labels             | KernelNumberGraphs   (root = smallest seed, graphs numbered by root)
// in_win & ~seed_mask -> followers         | KernelCountGraphs / KernelFillNodesEdges
// g_nodes (sets), g_edges (both directions)|   one block per graph: nodes in ascending trackster index,
//                                          |   edges by (localA, localB)
//
// All kernels work in trackster-index space (arrays of size nTracksters), so no stream compaction is needed and the
// result is deterministic.

#include <algorithm>
#include <bit>

#include <alpaka/alpaka.hpp>
#include <xtd/xtd.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Superclustering/interface/alpaka/TracksterGraphAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using ::ticl::graph::Counters;
  using ::ticl::graph::Inputs;
  using ::ticl::graph::Params;

  namespace {

    constexpr int32_t kMaxBlockThreads = 1024;  // size of the shared scan buffer
    constexpr int32_t kSingleBlockThreads = 512;
    constexpr int32_t kGraphBlockThreads = 256;
    constexpr int32_t kElementwiseThreads = 256;
    constexpr int32_t kMaxGraphBlocks = 1024;

    // Number of int32 and float scratch arrays of size nTracksters
    constexpr int32_t kNInts = 13;
    constexpr int32_t kNFloats = 3;

    // eta bins of the selected tracksters, to find the candidates of a seed window without a loop over all the
    // tracksters. The tracksters outside the range go to the first or the last bin.
    // ponytail: fixed bins; make them configurable if the windows or the eta range change a lot
    constexpr float kEtaBinMin = -3.f;
    constexpr float kEtaBinWidth = 0.05f;
    constexpr int32_t kNEtaBins = 120;
    // binCount, binOffset (kNEtaBins + 1), binCursor, reachBits
    constexpr int32_t kNBinInts = 3 * kNEtaBins + 2;

    // Raw pointers into the scratch buffers (all the arrays have nTracksters entries; the arrays per graph use only
    // the first nGraphs <= nTracksters entries).
    struct Scratch {
      // the selected tracksters grouped by eta bin: bin b is binContent[binOffset[b] .. binOffset[b + 1])
      int32_t* binContent;  // nTracksters entries
      int32_t* binCount;    // per bin
      int32_t* binOffset;   // per bin, + 1
      int32_t* binCursor;   // per bin
      int32_t* reachBits;   // the largest eta reach of a seed window, max(-lo, hi), as the bits of a float >= 0
      int32_t* isSeed;      // selected and et >= threshold
      int32_t* isFollower;  // selected and not a seed
      int32_t* parent;      // union-find over the seeds
      int32_t* graphOfRoot;
      int32_t* seedGraph;   // seed -> graph, -1 for the other tracksters
      int32_t* seedCount;   // per graph
      int32_t* seedOffset;  // per graph
      int32_t* seeds;       // seeds grouped by graph, in ascending trackster index
      int32_t* nodeCount;   // per graph
      int32_t* nodeOffset;  // per graph
      int32_t* edgeCount;   // per graph
      int32_t* edgeOffset;  // per graph
      float* lo;
      float* hi;
      float* dphiMax;
      Counters* counters;
    };

    // ---------------------------------------------------------------------------------------------------------------
    // small helpers
    // ---------------------------------------------------------------------------------------------------------------

    ALPAKA_FN_ACC inline int32_t blockThreads(Acc1D const& acc) {
      return static_cast<int32_t>(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
    }
    ALPAKA_FN_ACC inline int32_t threadInBlock(Acc1D const& acc) {
      return static_cast<int32_t>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    }

    // Exclusive scan of one value per thread over the block (Hillis-Steele). All the threads of the block must call
    // it. Returns the exclusive prefix, and sets total to the sum over the block.
    ALPAKA_FN_ACC int32_t blockExclusiveScan(Acc1D const& acc, int32_t value, int32_t* buf, int32_t& total) {
      int32_t const nt = blockThreads(acc);
      int32_t const tid = threadInBlock(acc);
      buf[tid] = value;
      alpaka::syncBlockThreads(acc);
      for (int32_t off = 1; off < nt; off <<= 1) {
        int32_t const x = tid >= off ? buf[tid - off] : 0;
        alpaka::syncBlockThreads(acc);
        buf[tid] += x;
        alpaka::syncBlockThreads(acc);
      }
      total = buf[nt - 1];
      int32_t const inclusive = buf[tid];
      alpaka::syncBlockThreads(acc);  // buf can be used again right after
      return inclusive - value;
    }

    // out[i] = sum(in[0..i)), for i < m. Single block. Returns the total.
    ALPAKA_FN_ACC int32_t blockExclusiveScanArray(
        Acc1D const& acc, int32_t const* in, int32_t* out, int32_t m, int32_t* buf) {
      int32_t const nt = blockThreads(acc);
      int32_t const tid = threadInBlock(acc);
      int32_t carry = 0;
      for (int32_t base = 0; base < m; base += nt) {
        int32_t const i = base + tid;
        int32_t total;
        int32_t const ex = blockExclusiveScan(acc, i < m ? in[i] : 0, buf, total);
        if (i < m)
          out[i] = carry + ex;
        carry += total;
      }
      return carry;
    }

    // Union-find with lock-free hooking: the larger root is always hooked under the smaller one, so the final root of
    // a component is its smallest seed. parent[] is read while other threads (possibly in other blocks) hook roots
    // with atomicCas. The reads go through a volatile pointer, so that they always fetch the current value from
    // memory: a non-coherent cached copy (e.g. the GPU L1) could keep returning a stale "x is a root" and make
    // unite() retry forever.
    ALPAKA_FN_ACC inline int32_t findRoot(int32_t const* parent, int32_t x) {
      int32_t const volatile* p = parent;
      while (true) {
        int32_t const next = p[x];
        if (next == x)
          return x;
        x = next;
      }
    }

    ALPAKA_FN_ACC inline void unite(Acc1D const& acc, int32_t* parent, int32_t a, int32_t b) {
      while (true) {
        a = findRoot(parent, a);
        b = findRoot(parent, b);
        if (a == b)
          return;
        if (a < b) {
          int32_t const tmp = a;
          a = b;
          b = tmp;
        }
        // Hook a (the larger root) under b, but only if a is still a root: atomicCas writes b into parent[a] only if
        // parent[a] == a, and returns the value that it found. If another thread hooked a in the meantime, nothing is
        // written and the loop retries from the new roots. hierarchy::Blocks: the atomic is coherent across all the
        // blocks of the grid.
        if (alpaka::atomicCas(acc, &parent[a], a, b, alpaka::hierarchy::Blocks{}) == a)
          return;
      }
    }

    ALPAKA_FN_ACC inline int32_t etaBin(float eta) {
      int32_t const b = static_cast<int32_t>((eta - kEtaBinMin) / kEtaBinWidth);
      return b < 0 ? 0 : (b >= kNEtaBins ? kNEtaBins - 1 : b);
    }

    // The rows of binContent with the selected tracksters that can be in the window of a seed at eta, or that can
    // see it from their own window: all the bins within the largest reach, plus one bin on each side as a margin for
    // the rounding.
    struct Candidates {
      int32_t begin;
      int32_t end;
    };
    ALPAKA_FN_ACC inline Candidates candidates(Scratch const& s, float eta) {
      float const reach = std::bit_cast<float>(*s.reachBits);
      int32_t const first = etaBin(eta - reach);
      int32_t const last = etaBin(eta + reach);
      return {s.binOffset[first > 0 ? first - 1 : 0], s.binOffset[last < kNEtaBins - 1 ? last + 2 : kNEtaBins]};
    }

    // Python: deta = (eta[t] - eta[seed]) * sign(eta[seed]); lo < deta < hi and |dphi| < dphi_max
    ALPAKA_FN_ACC inline bool inWindow(Scratch const& s, Inputs const& in, int32_t seed, int32_t t) {
      // mirrored for negative eta: positive deta points to larger |eta|
      float const deta = in.eta[seed] < 0.f ? in.eta[seed] - in.eta[t] : in.eta[t] - in.eta[seed];
      float const dphi = ::ticl::graph::absDeltaPhi(in.phi[t], in.phi[seed]);
      return deta > s.lo[seed] && deta < s.hi[seed] && dphi < s.dphiMax[seed];
    }

    // Python: connected = a_sees_b | b_sees_a
    ALPAKA_FN_ACC inline bool seedsLinked(Scratch const& s, Inputs const& in, int32_t a, int32_t b) {
      return inWindow(s, in, a, b) || inWindow(s, in, b, a);
    }

    // Is u a neighbour of the seed t (of graph g)? A linked seed, or a follower in the window of t.
    ALPAKA_FN_ACC inline bool isNeighbourOfSeed(Scratch const& s, Inputs const& in, int32_t g, int32_t t, int32_t u) {
      if (u == t)
        return false;
      if (s.isSeed[u])
        return s.seedGraph[u] == g && seedsLinked(s, in, t, u);
      return s.isFollower[u] && inWindow(s, in, t, u);
    }

    struct NodeInfo {
      bool member;
      int32_t degree;  // number of directed edges that leave this node in graph g
    };

    // Membership and degree of the trackster t in the graph g. A seed counts its neighbours only among the candidates
    // of its eta bins.
    // ponytail: still called for every (graph, trackster) pair; build neighbour lists per seed if the timing with
    // pileup needs it.
    ALPAKA_FN_ACC NodeInfo nodeInfo(Scratch const& s, Inputs const& in, int32_t g, int32_t t) {
      if (s.isSeed[t]) {
        if (s.seedGraph[t] != g)
          return {false, 0};
        int32_t degree = 0;
        auto const [begin, end] = candidates(s, in.eta[t]);
        for (int32_t k = begin; k < end; ++k)
          degree += isNeighbourOfSeed(s, in, g, t, s.binContent[k]) ? 1 : 0;
        return {true, degree};
      }
      if (!s.isFollower[t])
        return {false, 0};
      int32_t hits = 0;  // the seeds of g that see t: each one gives seed->t and t->seed
      int32_t const first = s.seedOffset[g];
      int32_t const last = first + s.seedCount[g];
      for (int32_t k = first; k < last; ++k)
        hits += inWindow(s, in, s.seeds[k], t) ? 1 : 0;
      return {hits > 0, hits};
    }

    // Local index of the trackster t in its graph: binary search in the (ascending) node rows.
    template <typename TNodes>
    ALPAKA_FN_ACC inline int32_t localIndexOf(TNodes const& nodes, int32_t nodeOffset, int32_t nNodes, int32_t t) {
      int32_t lo = 0, hi = nNodes;
      while (lo < hi) {
        int32_t const mid = (lo + hi) / 2;
        if (nodes[nodeOffset + mid].tracksterIdx() < t)
          lo = mid + 1;
        else
          hi = mid;
      }
      return lo;
    }

    template <typename TEdges>
    ALPAKA_FN_ACC inline void writeEdge(
        TEdges& edges, int32_t row, int32_t g, int32_t tA, int32_t localA, int32_t tB, int32_t localB) {
      edges[row].tracksterA() = tA;
      edges[row].tracksterB() = tB;
      edges[row].graphIdx() = g;
      edges[row].localA() = localA;
      edges[row].localB() = localB;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // kernel: inputs from the trackster SoAs
    // ---------------------------------------------------------------------------------------------------------------

    struct KernelInputs {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::ticl::TracksterSoAConstView tracksters,
                                    ::ticl::TracksterBarrelPropertiesSoAConstView properties,
                                    int32_t n,
                                    float* eta,
                                    float* phi,
                                    float* et,
                                    int32_t* selected) const {
        for (int32_t i : uniform_elements(acc, n)) {
          auto const barycenter = tracksters.barycenter()[i];
          float const x = barycenter.x();
          float const y = barycenter.y();
          float const z = barycenter.z();
          float const rho = xtd::sqrt(x * x + y * y);
          // eta = asinh(z / rho), as Trackster::barycenter().eta(); et = rawEnergy / cosh(eta) = rawEnergy * rho / r
          float const r = xtd::sqrt(rho * rho + z * z);
          eta[i] = rho > 0.f ? (z >= 0.f ? xtd::log((z + r) / rho) : -xtd::log((r - z) / rho)) : 0.f;
          phi[i] = xtd::atan2(y, x);
          et[i] = r > 0.f ? tracksters.energy()[i].rawEnergy() * rho / r : 0.f;
          selected[i] = properties[i].nEcalClusters() >= 1;
        }
      }
    };

    // ---------------------------------------------------------------------------------------------------------------
    // kernels: counting phase
    // ---------------------------------------------------------------------------------------------------------------

    // seeds, followers, seed windows, union-find initialisation
    struct KernelInit {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Inputs in, Params params, Scratch s) const {
        for (int32_t i : uniform_elements(acc, in.n)) {
          bool const sel = in.selected[i] != 0;
          bool const seed = sel && in.et[i] >= params.seedEtThreshold;
          s.isSeed[i] = seed;
          s.isFollower[i] = sel && !seed;
          s.parent[i] = i;
          float lo = 0.f, hi = 0.f, dphiMax = 0.f;
          if (seed) {
            ::ticl::graph::windowsForEta(params, in.eta[i], lo, hi, dphiMax);
            // the bits of a float >= 0 are ordered as its value, so an integer atomicMax gives the largest reach
            float const reach = std::max(std::max(-lo, hi), 0.f);
            alpaka::atomicMax(acc, s.reachBits, std::bit_cast<int32_t>(reach), alpaka::hierarchy::Blocks{});
          }
          s.lo[i] = lo;
          s.hi[i] = hi;
          s.dphiMax[i] = dphiMax;
          if (sel)
            alpaka::atomicAdd(acc, &s.binCount[etaBin(in.eta[i])], 1, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Single block: the offsets of the eta bins
    struct KernelBinOffsets {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Scratch s) const {
        auto& buf = alpaka::declareSharedVar<int32_t[kMaxBlockThreads], __COUNTER__>(acc);
        int32_t const total = blockExclusiveScanArray(acc, s.binCount, s.binOffset, kNEtaBins, buf);
        if (threadInBlock(acc) == 0)
          s.binOffset[kNEtaBins] = total;
      }
    };

    // the selected tracksters grouped by eta bin (in no particular order inside a bin)
    struct KernelBinFill {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Inputs in, Scratch s) const {
        for (int32_t i : uniform_elements(acc, in.n)) {
          if (in.selected[i] == 0)
            continue;
          int32_t const b = etaBin(in.eta[i]);
          s.binContent[s.binOffset[b] + alpaka::atomicAdd(acc, &s.binCursor[b], 1, alpaka::hierarchy::Blocks{})] = i;
        }
      }
    };

    // seed-seed links -> union-find; each pair (i, j > i) is tested once, among the candidates of i
    struct KernelLinkSeeds {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Inputs in, Scratch s) const {
        for (int32_t i : uniform_elements(acc, in.n)) {
          if (!s.isSeed[i])
            continue;
          auto const [begin, end] = candidates(s, in.eta[i]);
          for (int32_t k = begin; k < end; ++k) {
            int32_t const j = s.binContent[k];
            if (j > i && s.isSeed[j] && seedsLinked(s, in, i, j))
              unite(acc, s.parent, i, j);
          }
        }
      }
    };

    // Single block: graph numbering, seed -> graph, seeds grouped by graph.
    struct KernelNumberGraphs {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, int32_t n, Scratch s) const {
        auto& buf = alpaka::declareSharedVar<int32_t[kMaxBlockThreads], __COUNTER__>(acc);
        int32_t const nt = blockThreads(acc);
        int32_t const tid = threadInBlock(acc);

        // a) graph index = rank of the root among the roots, in ascending trackster index. The root is the smallest
        //    seed of the component, so this is the order in which the Python DFS (start = 0, 1, 2, ...) labels the
        //    components.
        int32_t carry = 0;
        for (int32_t base = 0; base < n; base += nt) {
          int32_t const t = base + tid;
          bool const root = t < n && s.isSeed[t] && s.parent[t] == t;
          int32_t total;
          int32_t const ex = blockExclusiveScan(acc, root ? 1 : 0, buf, total);
          if (root)
            s.graphOfRoot[t] = carry + ex;
          carry += total;
        }
        int32_t const nGraphs = carry;
        if (tid == 0)
          s.counters->nGraphs = nGraphs;
        alpaka::syncBlockThreads(acc);

        // b) graph of every seed, number of seeds per graph
        for (int32_t t = tid; t < n; t += nt) {
          if (s.isSeed[t]) {
            int32_t const g = s.graphOfRoot[findRoot(s.parent, t)];
            s.seedGraph[t] = g;
            alpaka::atomicAdd(acc, &s.seedCount[g], 1, alpaka::hierarchy::Blocks{});
          } else {
            s.seedGraph[t] = -1;
          }
        }
        alpaka::syncBlockThreads(acc);

        // c) offsets of the seed lists of the graphs
        blockExclusiveScanArray(acc, s.seedCount, s.seedOffset, nGraphs, buf);
        alpaka::syncBlockThreads(acc);

        // d) seeds grouped by graph, in ascending trackster index inside a graph
        for (int32_t t = tid; t < n; t += nt) {
          if (!s.isSeed[t])
            continue;
          int32_t const g = s.seedGraph[t];
          int32_t rank = 0;
          for (int32_t u = 0; u < t; ++u)
            rank += (s.isSeed[u] && s.seedGraph[u] == g) ? 1 : 0;
          s.seeds[s.seedOffset[g] + rank] = t;
        }
      }
    };

    // One block per graph: number of nodes and of directed edges.
    struct KernelCountGraphs {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Inputs in, Scratch s) const {
        auto& nodeCnt = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
        auto& edgeCnt = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
        int32_t const nt = blockThreads(acc);
        int32_t const tid = threadInBlock(acc);
        int32_t const nGraphs = s.counters->nGraphs;

        for (int32_t g : independent_groups(acc, nGraphs)) {
          if (tid == 0) {
            nodeCnt = 0;
            edgeCnt = 0;
          }
          alpaka::syncBlockThreads(acc);
          for (int32_t t = tid; t < in.n; t += nt) {
            NodeInfo const info = nodeInfo(s, in, g, t);
            if (info.member) {
              alpaka::atomicAdd(acc, &nodeCnt, 1, alpaka::hierarchy::Threads{});
              alpaka::atomicAdd(acc, &edgeCnt, info.degree, alpaka::hierarchy::Threads{});
            }
          }
          alpaka::syncBlockThreads(acc);
          if (tid == 0) {
            s.nodeCount[g] = nodeCnt;
            s.edgeCount[g] = edgeCnt;
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    };

    // Single block: node and edge offsets of the graphs, and the totals.
    struct KernelOffsets {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, Scratch s) const {
        auto& buf = alpaka::declareSharedVar<int32_t[kMaxBlockThreads], __COUNTER__>(acc);
        int32_t const nGraphs = s.counters->nGraphs;
        int32_t const nNodes = blockExclusiveScanArray(acc, s.nodeCount, s.nodeOffset, nGraphs, buf);
        int32_t const nEdges = blockExclusiveScanArray(acc, s.edgeCount, s.edgeOffset, nGraphs, buf);
        if (threadInBlock(acc) == 0) {
          s.counters->nNodes = nNodes;
          s.counters->nEdges = nEdges;
        }
      }
    };

    // ---------------------------------------------------------------------------------------------------------------
    // kernels: fill phase
    // ---------------------------------------------------------------------------------------------------------------

    struct KernelFillGraphs {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    int32_t nGraphs,
                                    Scratch s,
                                    ::ticl::TracksterGraphSoAView view) const {
        auto graphs = view.graphs();
        for (int32_t g : uniform_elements(acc, nGraphs)) {
          graphs[g].nodeOffset() = s.nodeOffset[g];
          graphs[g].nNodes() = s.nodeCount[g];
          graphs[g].edgeOffset() = s.edgeOffset[g];
          graphs[g].nEdges() = s.edgeCount[g];
        }
      }
    };

    // One block per graph: the node rows (in ascending trackster index), then the edge rows, ordered by
    // (localA, localB).
    struct KernelFillNodesEdges {
      ALPAKA_FN_ACC void operator()(
          Acc1D const& acc, Inputs in, int32_t nGraphs, Scratch s, ::ticl::TracksterGraphSoAView view) const {
        auto& buf = alpaka::declareSharedVar<int32_t[kMaxBlockThreads], __COUNTER__>(acc);
        int32_t const nt = blockThreads(acc);
        int32_t const tid = threadInBlock(acc);
        auto nodes = view.nodes();
        auto edges = view.edges();
        int32_t const n = in.n;

        for (int32_t g : independent_groups(acc, nGraphs)) {
          int32_t const nodeOffset = s.nodeOffset[g];
          int32_t const nNodes = s.nodeCount[g];
          int32_t const edgeOffset = s.edgeOffset[g];
          int32_t const seedFirst = s.seedOffset[g];
          int32_t const seedLast = seedFirst + s.seedCount[g];

          // pass A: node rows; local index = rank among the members of g
          int32_t carry = 0;
          for (int32_t base = 0; base < n; base += nt) {
            int32_t const t = base + tid;
            bool const member = t < n && nodeInfo(s, in, g, t).member;
            int32_t total;
            int32_t const ex = blockExclusiveScan(acc, member ? 1 : 0, buf, total);
            if (member) {
              nodes[nodeOffset + carry + ex].tracksterIdx() = t;
              nodes[nodeOffset + carry + ex].graphIdx() = g;
            }
            carry += total;
          }
          alpaka::syncBlockThreads(acc);  // the node rows of g are complete

          // pass B: edges; the edges of a node start after those of all the smaller nodes
          carry = 0;
          for (int32_t base = 0; base < n; base += nt) {
            int32_t const t = base + tid;
            NodeInfo const info = t < n ? nodeInfo(s, in, g, t) : NodeInfo{false, 0};
            int32_t total;
            int32_t const ex = blockExclusiveScan(acc, info.member ? info.degree : 0, buf, total);
            if (info.member) {
              int32_t row = edgeOffset + carry + ex;
              int32_t const lt = localIndexOf(nodes, nodeOffset, nNodes, t);
              if (s.isSeed[t]) {
                // neighbours in ascending trackster index: linked seeds and followers
                for (int32_t u = 0; u < n; ++u)
                  if (isNeighbourOfSeed(s, in, g, t, u))
                    writeEdge(edges, row++, g, t, lt, u, localIndexOf(nodes, nodeOffset, nNodes, u));
              } else {
                // follower: the seeds of g that see it (the seed list is ascending)
                for (int32_t k = seedFirst; k < seedLast; ++k) {
                  int32_t const u = s.seeds[k];
                  if (inWindow(s, in, u, t))
                    writeEdge(edges, row++, g, t, lt, u, localIndexOf(nodes, nodeOffset, nNodes, u));
                }
              }
            }
            carry += total;
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    };

    Scratch makeScratch(int32_t* ints, float* floats, int32_t* bins, Counters* counters, int32_t n) {
      Scratch s;
      int32_t** intArrays[kNInts] = {&s.binContent,
                                     &s.isSeed,
                                     &s.isFollower,
                                     &s.parent,
                                     &s.graphOfRoot,
                                     &s.seedGraph,
                                     &s.seedCount,
                                     &s.seedOffset,
                                     &s.seeds,
                                     &s.nodeCount,
                                     &s.nodeOffset,
                                     &s.edgeCount,
                                     &s.edgeOffset};
      for (int32_t k = 0; k < kNInts; ++k)
        *intArrays[k] = ints + static_cast<size_t>(k) * n;
      s.lo = floats;
      s.hi = floats + n;
      s.dphiMax = floats + 2 * static_cast<size_t>(n);
      s.binCount = bins;
      s.binOffset = bins + kNEtaBins;
      s.binCursor = bins + 2 * kNEtaBins + 1;
      s.reachBits = bins + 3 * kNEtaBins + 1;
      s.counters = counters;
      return s;
    }

  }  // namespace

  // -----------------------------------------------------------------------------------------------------------------
  // host side
  // -----------------------------------------------------------------------------------------------------------------

  Inputs TracksterGraphAlgo::prepareInputs(Queue& queue,
                                           ::ticl::TracksterSoAConstView tracksters,
                                           ::ticl::TracksterBarrelPropertiesSoAConstView properties) {
    int32_t const n = tracksters.energy().metadata().size();
    if (n == 0)
      return Inputs{0, nullptr, nullptr, nullptr, nullptr};
    inputFloats_.emplace(make_device_buffer<float[]>(queue, 3 * n));
    inputInts_.emplace(make_device_buffer<int32_t[]>(queue, n));
    float* eta = inputFloats_->data();
    float* phi = eta + n;
    float* et = phi + n;
    int32_t* selected = inputInts_->data();
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kElementwiseThreads), kElementwiseThreads),
                        KernelInputs{},
                        tracksters,
                        properties,
                        n,
                        eta,
                        phi,
                        et,
                        selected);
    return Inputs{n, eta, phi, et, selected};
  }

  void TracksterGraphAlgo::count(Queue& queue, Inputs const& inputs) {
    int32_t const n = inputs.n;
    inputs_ = inputs;

    hostCounters_.emplace(make_host_buffer<Counters>(queue));
    if (n == 0) {
      *hostCounters_->data() = Counters{0, 0, 0};
      return;
    }

    ints_.emplace(make_device_buffer<int32_t[]>(queue, kNInts * n));
    floats_.emplace(make_device_buffer<float[]>(queue, kNFloats * n));
    bins_.emplace(make_device_buffer<int32_t[]>(queue, kNBinInts));
    deviceCounters_.emplace(make_device_buffer<Counters>(queue));
    alpaka::memset(queue, *ints_, 0x00);
    alpaka::memset(queue, *bins_, 0x00);
    alpaka::memset(queue, *deviceCounters_, 0x00);

    Scratch const s = makeScratch(ints_->data(), floats_->data(), bins_->data(), deviceCounters_->data(), n);

    auto const elementwise = make_workdiv<Acc1D>(divide_up_by(n, kElementwiseThreads), kElementwiseThreads);
    auto const singleBlock = make_workdiv<Acc1D>(1, kSingleBlockThreads);
    auto const perGraph = make_workdiv<Acc1D>(std::min(n, kMaxGraphBlocks), kGraphBlockThreads);

    alpaka::exec<Acc1D>(queue, elementwise, KernelInit{}, inputs, params_, s);
    alpaka::exec<Acc1D>(queue, singleBlock, KernelBinOffsets{}, s);
    alpaka::exec<Acc1D>(queue, elementwise, KernelBinFill{}, inputs, s);
    alpaka::exec<Acc1D>(queue, elementwise, KernelLinkSeeds{}, inputs, s);
    alpaka::exec<Acc1D>(queue, singleBlock, KernelNumberGraphs{}, n, s);
    alpaka::exec<Acc1D>(queue, perGraph, KernelCountGraphs{}, inputs, s);
    alpaka::exec<Acc1D>(queue, singleBlock, KernelOffsets{}, s);

    alpaka::memcpy(queue, *hostCounters_, *deviceCounters_);
  }

  void TracksterGraphAlgo::fill(Queue& queue, ::ticl::TracksterGraphSoAView graphs) {
    int32_t const nGraphs = counters().nGraphs;
    if (nGraphs == 0)
      return;
    int32_t const n = inputs_.n;
    Scratch const s = makeScratch(ints_->data(), floats_->data(), bins_->data(), deviceCounters_->data(), n);

    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nGraphs, kElementwiseThreads), kElementwiseThreads),
                        KernelFillGraphs{},
                        nGraphs,
                        s,
                        graphs);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(std::min(nGraphs, kMaxGraphBlocks), kGraphBlockThreads),
                        KernelFillNodesEdges{},
                        inputs_,
                        nGraphs,
                        s,
                        graphs);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
