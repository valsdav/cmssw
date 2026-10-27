#include <alpaka/alpaka.hpp>

#include <cstdint>

#include <xtd/xtd.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TracksterBarrelPropertiesAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    struct KernelBarrelProperties {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    ticl::TracksterDeviceCollection::ConstView tracksters,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    ticl::TracksterBarrelPropertiesDeviceCollection::View properties) const {
        const auto vertices = tracksters.vertices();

        for (auto t : uniform_elements(acc, properties.metadata().size())) {
          auto row = properties[t];
          row.ecalX() = 0.f;
          row.ecalY() = 0.f;
          row.ecalZ() = 0.f;
          row.ecalEta() = 0.f;
          row.ecalPhi() = 0.f;
          row.hcalBarycenterX() = 0.f;
          row.hcalBarycenterY() = 0.f;
          row.hcalBarycenterZ() = 0.f;
          row.hcalBarycenterEta() = 0.f;
          row.hcalBarycenterPhi() = 0.f;

          uint32_t nEcalClusters = 0;
          uint32_t nHcalClusters = 0;
          uint32_t ecalCluster = 0;
          float hcalEnergy = 0.f;
          float sumEX = 0.f, sumEY = 0.f, sumEZ = 0.f;

          const uint32_t first = vertices.offsets()[t].keys_offsets();
          const uint32_t last = vertices.offsets()[t + 1].keys_offsets();
          for (uint32_t vertex = first; vertex < last; ++vertex) {
            const uint32_t clusterIndex = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[clusterIndex];
            if (position.layer() == 0) {
              ++nEcalClusters;
              ecalCluster = clusterIndex;
            } else if (position.layer() > 0) {
              ++nHcalClusters;
              const float energy = layerClusters.energy()[clusterIndex].energy();
              hcalEnergy += energy;
              sumEX += energy * position.x();
              sumEY += energy * position.y();
              sumEZ += energy * position.z();
            }
          }
          row.nEcalClusters() = nEcalClusters;
          row.nHcalClusters() = nHcalClusters;

          // A barrel trackster has one ECAL layer cluster; keep invalid values if this is not true.
          if (nEcalClusters == 1) {
            const auto position = layerClusters.position()[ecalCluster];
            const float x = position.x();
            const float y = position.y();
            const float z = position.z();
            row.ecalX() = x;
            row.ecalY() = y;
            row.ecalZ() = z;
            const float rho = xtd::sqrt(x * x + y * y);
            if (rho > 0.f) {
              const float theta = xtd::atan2(rho, z);
              row.ecalEta() = -xtd::log(xtd::tan(theta * 0.5f));
              row.ecalPhi() = xtd::atan2(y, x);
            }
          }

          // Use the stored HCAL layer-cluster energies, without clipping negative energies.
          if (hcalEnergy > 0.f) {
            const float x = sumEX / hcalEnergy;
            const float y = sumEY / hcalEnergy;
            const float z = sumEZ / hcalEnergy;
            row.hcalBarycenterX() = x;
            row.hcalBarycenterY() = y;
            row.hcalBarycenterZ() = z;
            const float rho = xtd::sqrt(x * x + y * y);
            if (rho > 0.f) {
              const float theta = xtd::atan2(rho, z);
              row.hcalBarycenterEta() = -xtd::log(xtd::tan(theta * 0.5f));
              row.hcalBarycenterPhi() = xtd::atan2(y, x);
            }
          }
        }
      }
    };

  }  // namespace

  void TracksterBarrelPropertiesAlgoWrapper::run(
      Queue& queue,
      ticl::TracksterDeviceCollection::ConstView tracksters,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      ticl::TracksterBarrelPropertiesDeviceCollection::View properties) const {
    const uint32_t nTracksters = properties.metadata().size();
    if (nTracksters == 0) {
      return;
    }

    constexpr uint32_t blockSize = 256;
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(nTracksters, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue, workDiv, KernelBarrelProperties{}, tracksters, layerClusters, properties);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
