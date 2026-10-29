#ifndef RecoTICL_Superclustering_interface_TracksterGraphParams_h
#define RecoTICL_Superclustering_interface_TracksterGraphParams_h

// The configuration, the inputs and the totals of the trackster graph building (TracksterGraphAlgo).
// Backend independent, usable on the host and on the device.

#include <cstdint>

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
