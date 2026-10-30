#ifndef RecoTICL_Superclustering_interface_TracksterGraphEdgeFeatures_h
#define RecoTICL_Superclustering_interface_TracksterGraphEdgeFeatures_h

// The node features of the graph edge model: the ONLY C++ place that knows them. They must be the features of the
// training, in the same order: FEATURE_NAMES in ticl-egm-linking/src/build_dataset/node_features.py. They are raw
// (physical units): the exported model applies the normalisation of the training.
// The Python mirror is node_features() in RecoTICL/Superclustering/test/compare_edge_scores.py.

#include <cstdint>

#include <xtd/xtd.h>

#include "DataFormats/TICL/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphEdgeModelSoA.h"
#include "RecoTICL/Superclustering/interface/TracksterGraphParams.h"

namespace ticl::graphmodel {

  // value of the ECAL-HCAL features of a trackster without HCAL clusters (ECAL_ONLY_PLACEHOLDER)
  constexpr float kEcalOnlyPlaceholder = -1.f;

  ALPAKA_FN_HOST_ACC inline void fillNodeFeatures(TracksterSoAConstView const& tracksters,
                                                  TracksterBarrelPropertiesSoAConstView const& properties,
                                                  int32_t t,
                                                  float (&f)[kNodeFeatures]) {
    constexpr float kPi = 3.14159265358979323846f;
    auto const barycenter = tracksters.barycenter()[t];
    auto const p = properties[t];
    // A barrel trackster has one ECAL layer cluster: the EM energy is its energy, the rest is the HCAL energy
    float const rawEnergy = tracksters.energy()[t].rawEnergy();
    float const ecalEnergy = tracksters.energy()[t].rawEmEnergy();
    float const hcalEnergy = rawEnergy - ecalEnergy;
    auto const k = ::ticl::graph::barycenterKinematics(barycenter.x(), barycenter.y(), barycenter.z(), rawEnergy);

    f[0] = ecalEnergy;                                                         // ecal_energy
    f[1] = k.et;                                                               // et = rawEnergy / cosh(eta)
    f[2] = k.eta;                                                              // eta
    f[3] = k.rho > 0.f ? barycenter.y() / k.rho : 0.f;                         // sin_phi
    f[4] = k.rho > 0.f ? barycenter.x() / k.rho : 1.f;                         // cos_phi
    f[5] = p.nHcalClusters();                                                  // n_hcal
    f[6] = hcalEnergy;                                                         // hcal_energy
    // deta_ecal_hcal, dphi_ecal_hcal, dz_ecal_hcal: HCAL barycenter - ECAL cluster. The training gives the
    // placeholder without HCAL clusters, and also if their energy sum is 0 (NaN barycenter).
    if (p.nHcalClusters() > 0 and hcalEnergy > 0.f) {
      float dphi = p.hcalBarycenterPhi() - p.ecalPhi();
      if (dphi >= kPi)
        dphi -= 2.f * kPi;
      else if (dphi < -kPi)
        dphi += 2.f * kPi;
      f[7] = p.hcalBarycenterEta() - p.ecalEta();
      f[8] = dphi;
      f[9] = xtd::fabs(p.ecalZ() - p.hcalBarycenterZ());
    } else {
      f[7] = f[8] = f[9] = kEcalOnlyPlaceholder;
    }
  }

}  // namespace ticl::graphmodel

#endif  // RecoTICL_Superclustering_interface_TracksterGraphEdgeFeatures_h
