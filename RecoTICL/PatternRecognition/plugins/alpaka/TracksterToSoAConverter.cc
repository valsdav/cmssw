// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/TICL/interface/alpaka/TracksterDeviceCollection.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TICL/interface/TracksterHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    Eigen::Vector3f toEigen(::ticl::Trackster::Vector const& vector) { return {vector.x(), vector.y(), vector.z()}; }

  }  // namespace

  // Copies the legacy tracksters to the trackster SoA. The SoA has no vertex multiplicity, edges or track indices, so
  // these are not copied. Ported from TrackstersToSoAProducer by E. Coradin.
  class TracksterToSoAConverter : public stream::EDProducer<> {
  public:
    explicit TracksterToSoAConverter(edm::ParameterSet const& config)
        : EDProducer<>(config),
          trackstersToken_{consumes(config.getParameter<edm::InputTag>("src"))},
          trackstersSoAToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription description;
      description.add<edm::InputTag>("src", edm::InputTag("ticlTrackstersCLUE3DHigh"));
      descriptions.addWithDefaultLabel(description);
    }

    void produce(device::Event& event, device::EventSetup const&) override {
      const auto& input = event.get(trackstersToken_);

      uint32_t nVertices = 0;
      for (const auto& trackster : input) {
        if (trackster.vertices().size() != trackster.vertex_multiplicity().size()) {
          throw cms::Exception("TracksterToSoAConverter")
              << "A Trackster has " << trackster.vertices().size() << " vertices but "
              << trackster.vertex_multiplicity().size() << " multiplicities.";
        }
        nVertices += trackster.vertices().size();
      }

      const uint32_t nTracksters = input.size();
      ::ticl::TracksterHostCollection output(
          event.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nVertices);
      auto view = output.view();
      auto vertices = view.vertices();

      uint32_t vertexOffset = 0;
      vertices.offsets()[0].keys_offsets() = 0;
      for (uint32_t i = 0; i < nTracksters; ++i) {
        const auto& source = input[i];

        auto energy = view.energy()[i];
        energy.rawEnergy() = source.raw_energy();
        energy.rawEmEnergy() = source.raw_em_energy();
        energy.regressedEnergy() = source.regressed_energy();

        auto barycenter = view.barycenter()[i];
        barycenter.x() = source.barycenter().x();
        barycenter.y() = source.barycenter().y();
        barycenter.z() = source.barycenter().z();

        view.timing()[i].time() = source.time();
        view.timing()[i].timeError() = source.timeError();

        // The legacy trackster keeps the PCA in decreasing eigenvalue order, as the SoA does
        auto pca = view.pca()[i];
        for (int k = 0; k < 3; ++k) {
          pca.eigenvalues()(k) = source.eigenvalues()[k];
          pca.eigenvectors().col(k) = toEigen(source.eigenvectors(k));
          pca.sigmas()(k) = source.sigmas()[k];
          pca.sigmasPCA()(k) = source.sigmasPCA()[k];
        }

        for (int k = 0; k < ::ticl::TracksterIdProbabilities::RowsAtCompileTime; ++k) {
          view.id()[i].idProbabilities()(k) = source.id_probabilities(k);
        }

        for (auto vertex : source.vertices()) {
          vertices.content()[vertexOffset++].values() = vertex;
        }
        vertices.offsets()[i + 1].keys_offsets() = vertexOffset;
      }

      if constexpr (std::is_same_v<Device, alpaka_common::DevHost>) {
        event.emplace(trackstersSoAToken_, std::move(output));
      } else {
        ticl::TracksterDeviceCollection deviceOutput(
            event.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nVertices);
        alpaka::memcpy(event.queue(), deviceOutput.buffer(), output.const_buffer());
        event.emplace(trackstersSoAToken_, std::move(deviceOutput));
      }
    }

  private:
    const edm::EDGetTokenT<std::vector<::ticl::Trackster>> trackstersToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> trackstersSoAToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TracksterToSoAConverter);
