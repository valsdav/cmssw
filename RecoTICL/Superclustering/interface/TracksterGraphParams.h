#ifndef RecoTICL_Superclustering_interface_TracksterGraphParams_h
#define RecoTICL_Superclustering_interface_TracksterGraphParams_h

// The configuration, the inputs and the totals of the trackster graph building (TracksterGraphAlgo) and of the
// superclustering on the graphs (TracksterGroupAlgo).
// Backend independent, usable on the host and on the device.

#include <cstdint>

#include <xtd/xtd.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ticl::graph {

  struct Params {
    static constexpr int32_t kMaxCoefficients = 4;

    float seedEtThreshold;  // a selected trackster with et >= seedEtThreshold is a seed

    // Seed windows, as polynomials in |eta_seed|: value = sum_k c[k] * |eta_seed|^k. A trackster t is in the window
    // of the seed s if
    //   detaLow < (eta[t] - eta[s]) * sign(eta[s]) < detaHigh  and  |phi[t] - phi[s]| < dphiMax
    // The window is mirrored for negative eta, so detaHigh is always on the side of larger |eta|.
    int32_t nDetaLow;
    int32_t nDetaHigh;
    int32_t nDphiMax;
    float detaLow[kMaxCoefficients];
    float detaHigh[kMaxCoefficients];
    float dphiMax[kMaxCoefficients];
  };

  // The quantities of the tracksters that the graph building uses, as device arrays of n entries.
  struct Inputs {
    int32_t n;
    float const* eta;
    float const* phi;
    float const* et;
    int32_t const* selected;  // 0: the trackster is not a node of any graph
  };

  // The totals of the counting phase, copied to the host to size the output.
  struct Counters {
    int32_t nGraphs;
    int32_t nNodes;
    int32_t nEdges;
  };

  // The configuration and the totals of the superclustering on the graphs (TracksterGroupAlgo)
  struct GroupParams {
    // A graph node with et >= seedEtThreshold can be a group seed. It must not be lower than the seed threshold of the
    // graph building: then each such node is a seed of exactly one graph.
    float seedEtThreshold;
    float linkingThreshold;  // a link is taken if its probability is > linkingThreshold
    // false: only the group seed takes its neighbours; true: also each seed that the group takes (never a follower)
    bool transitive;
  };

  struct GroupCounters {
    int32_t nSeeds;
    int32_t nGroups;
    int32_t nMembers;
  };

  // eta, phi and transverse energy of a trackster, from its barycenter
  struct BarycenterKinematics {
    float eta;
    float phi;
    float et;
    float rho;  // transverse distance of the barycenter from the beam line
  };

  // eta = asinh(z / rho), as Trackster::barycenter().eta(), and et = rawEnergy / cosh(eta) = rawEnergy * rho / r, as
  // the CPU raw_pt. The two forms of eta never subtract two nearly equal numbers. The graph building, the graph edge
  // model and the superclustering use this function, so that they agree.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE BarycenterKinematics barycenterKinematics(float x, float y, float z, float rawEnergy) {
    float const rho = xtd::sqrt(x * x + y * y);
    float const r = xtd::sqrt(rho * rho + z * z);
    return {rho > 0.f ? (z >= 0.f ? xtd::log((z + r) / rho) : -xtd::log((r - z) / rho)) : 0.f,
            xtd::atan2(y, x),
            r > 0.f ? rawEnergy * rho / r : 0.f,
            rho};
  }

  // sum_k c[k] * x^k, with the Horner method
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float polynomial(float const* c, int32_t n, float x) {
    float value = c[n - 1];
    for (int32_t k = n - 2; k >= 0; --k)
      value = value * x + c[k];
    return value;
  }

  // The window of a seed at seedEta
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void windowsForEta(
      Params const& p, float seedEta, float& lo, float& hi, float& dphiMax) {
    float const a = seedEta < 0.f ? -seedEta : seedEta;
    lo = polynomial(p.detaLow, p.nDetaLow, a);
    hi = polynomial(p.detaHigh, p.nDetaHigh, a);
    dphiMax = polynomial(p.dphiMax, p.nDphiMax, a);
  }

  // |a - b|, with the difference wrapped to (-pi, pi]
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float absDeltaPhi(float a, float b) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float k2Pi = 2.f * kPi;
    float d = a - b;
    while (d > kPi)
      d -= k2Pi;
    while (d <= -kPi)
      d += k2Pi;
    return d < 0.f ? -d : d;
  }

}  // namespace ticl::graph

#endif  // RecoTICL_Superclustering_interface_TracksterGraphParams_h
