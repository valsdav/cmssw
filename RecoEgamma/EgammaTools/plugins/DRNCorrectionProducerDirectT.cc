#include <torch/script.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include <cmath>


/*
 * DRNCorrectionProducerDirectT
 *
 * Producer template generate a ValueMap of corrected energies and resolutions
 * ValueMap contains std::pair<float, float> of corrected energy, resolution 
 *
 * Author: Simon Rothman (MIT)
 * Written 2022
 *
 */

namespace {
  float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

  float logcorrection(float x) {
    static float ln2 = std::log(2);
    return ln2 * 2 * (sigmoid(x) - 0.5);
  }

  //correction factor is transformed by sigmoid and "logratioflip target"
  float correction(float x) { return std::exp(-logcorrection(x)); }

  inline float rescale(float x, float min, float range) { return (x - min) / range; }

  //resolution is transformed by softplus function
  float resolution(float x) { return std::log(1 + std::exp(x)); }

  const float RHO_MIN = 0.0f;
  const float RHO_RANGE = 13.0f;

  const float HOE_MIN = 0.0f;
  const float HOE_RANGE = 0.05f;

  const float XY_MIN = -150.0f;
  const float XY_RANGE = 300.0f;

  const float Z_MIN = -330.0f;
  const float Z_RANGE = 660.0f;

  const float NOISE_MIN = 0.9f;
  const float NOISE_RANGE = 3.0f;

  const float ECAL_MIN = 0.0f;
  const float ECAL_RANGE = 250.0f;

  const float ES_MIN = 0.0f;
  const float ES_RANGE = 0.1f;

}  // namespace

template <typename T>
class DRNCorrectionProducerDirectT : public edm::stream::EDProducer<> {
public:
  explicit DRNCorrectionProducerDirectT(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  std::string modelPath_;
  //edm::EDGetTokenT<edm::View<T>> particleToken_;
  edm::EDGetTokenT<edm::View<T>> particleToken_;

  edm::EDGetTokenT<double> rhoToken_;

  edm::EDGetTokenT<EcalRecHitCollection> EBRecHitsToken_, EERecHitsToken_, ESRecHitsToken_;

  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

  size_t nPart_, nValidPart_;

  bool isEB(const T& part);
  bool isEE(const T& part);
  bool skip(const T& part);
};

template <typename T>
DRNCorrectionProducerDirectT<T>::DRNCorrectionProducerDirectT(const edm::ParameterSet& iConfig)
    : modelPath_(iConfig.getParameter<std::string>("modelPath")),
      particleToken_(consumes(iConfig.getParameter<edm::InputTag>("particleSource"))),
      rhoToken_(consumes(iConfig.getParameter<edm::InputTag>("rhoName"))),
      EBRecHitsToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEcalRecHitsEB"))),
      EERecHitsToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEcalRecHitsEE"))),
      ESRecHitsToken_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEcalRecHitsES"))),
      pedToken_(esConsumes()),
      geomToken_(esConsumes()) {
  produces<edm::ValueMap<std::pair<float, float>>>();
}

template <typename T>
bool DRNCorrectionProducerDirectT<T>::isEB(const T& part) {
  return part.superCluster()->seed()->hitsAndFractions().at(0).first.subdetId() == EcalBarrel;
}

template <typename T>
bool DRNCorrectionProducerDirectT<T>::isEE(const T& part) {
  return part.superCluster()->seed()->hitsAndFractions().at(0).first.subdetId() == EcalEndcap;
}

template <typename T>
bool DRNCorrectionProducerDirectT<T>::skip(const T& part) {
  /*
   * Separated out from acquire() and produce() to ensure that skipping check is identical in both
   * N.B. in MiniAOD there are sometimes particles with no RecHits
   * We can not apply our regression to these, so we skip them
   */
  return (!isEB(part) && !isEE(part)) || part.superCluster()->hitsAndFractions().empty();
}

torch::Device device(torch::kCPU);
torch::jit::script::Module module_;

template <typename T>
void DRNCorrectionProducerDirectT<T>::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {

  try {
    module_ = torch::jit::load(edm::FileInPath(modelPath_).fullPath());
    module_.to(device);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }
  auto method = module_.get_method("forward");
  auto schema = method.function().getSchema();
  std::vector<torch::jit::IValue> inputs;
  /*
   * Get products from event and event setup
   */
  const auto& particles_ = iEvent.getHandle(particleToken_);
  float rho = iEvent.get(rhoToken_);

  const auto& ped = &iSetup.getData(pedToken_);
  const auto& geo = &iSetup.getData(geomToken_);

  const CaloSubdetectorGeometry* ecalEBGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
  const CaloSubdetectorGeometry* ecalEEGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap));
  const CaloSubdetectorGeometry* ecalESGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalPreshower));

  const auto& recHitsEB = iEvent.get(EBRecHitsToken_);
  const auto& recHitsEE = iEvent.get(EERecHitsToken_);
  const auto& recHitsES = iEvent.get(ESRecHitsToken_);

  nPart_ = particles_->size();

  //if (nPart_ == 0) {
  //  client_->setBatchSize(0);
  //  return;
  //} else {
  //  client_->setBatchSize(1);
  //}

  /*
   * Determine how many particles, how many RecHits there are in each subdetector
   */
  unsigned nHitsECAL = 0, nHitsES = 0;
  nValidPart_ = 0;
  for (auto& part : *particles_) {
    const reco::SuperClusterRef& sc = part.superCluster();

    if (skip(part))
      continue;

    nHitsECAL += sc->hitsAndFractions().size();

    for (auto iES = sc->preshowerClustersBegin(); iES != sc->preshowerClustersEnd(); ++iES) {
      nHitsES += (*iES)->hitsAndFractions().size();
    }

    ++nValidPart_;
  }

  /*
   * Allocate DRN inputs ({SB} is one of ECAL, ES):
   * x{SB}: (x, y, z, energy, [noise]) continuous-valued inputs per RecHit
   * f{SB}: (flagVal) integer denoting RecHit flag values
   * gainECAL: (gain) integer in (0, 1, 2) denoting gain value
   * gx: (rho, H/E) additional high-level features.
   * batch{SB}: graph models require explicitely passing the particle index for each vertex
   */

  
  auto dataxECAL = torch::empty({1, nHitsECAL, 5}, torch::kFloat32);
  auto datafECAL = torch::empty({1}, torch::kInt64);
  auto dataGainECAL = torch::empty({1}, torch::kInt64);
  auto dataGx = torch::empty({1, 2}, torch::kFloat32);
  auto dataBatchECAL = torch::empty({1, nHitsECAL}, torch::kInt64);
  auto dataxES = torch::empty({1, nHitsES, 4}, torch::kFloat32);
  auto datafES = torch::empty({1}, torch::kInt64);
  auto dataBatchES = torch::empty({1, nHitsES}, torch::kInt64);

  // accessors
  auto inputxECAL = dataxECAL.accessor<float, 2>();
  auto inputfECAL = datafECAL.accessor<int64_t, 1>();
  auto inputGainECAL = dataGainECAL.accessor<int64_t, 1>();
  auto inputxES = dataxES.accessor<float, 2>();
  auto inputfES = datafES.accessor<int64_t, 1>();
  auto inputGx = dataGx.accessor<float, 2>();
  auto inputBatchECAL = dataBatchECAL.accessor<int64_t, 1>();
  auto inputBatchES = dataBatchES.accessor<int64_t, 1>();  

  /*
  auto& inputxECAL = iInput.at("xECAL__0");
  inputxECAL.setShape(0, nHitsECAL);
  auto dataxECAL = inputxECAL.allocate<float>();
  auto& vdataxECAL = (*dataxECAL)[0];

  auto& inputfECAL = iInput.at("fECAL__1");
  inputfECAL.setShape(0, nHitsECAL);
  auto datafECAL = inputfECAL.allocate<int64_t>();
  auto& vdatafECAL = (*datafECAL)[0];

  auto& inputGainECAL = iInput.at("gain__2");
  inputGainECAL.setShape(0, nHitsECAL);
  auto dataGainECAL = inputGainECAL.allocate<int64_t>();
  auto& vdataGainECAL = (*dataGainECAL)[0];

  auto& inputGx = iInput.at("graph_x__5");
  inputGx.setShape(0, nValidPart_);
  auto dataGx = inputGx.allocate<float>();
  auto& vdataGx = (*dataGx)[0];

  auto& inputBatchECAL = iInput.at("xECAL_batch__6");
  inputBatchECAL.setShape(0, nHitsECAL);
  auto dataBatchECAL = inputBatchECAL.allocate<int64_t>();
  auto& vdataBatchECAL = (*dataBatchECAL)[0];

  auto& inputxES = iInput.at("xES__3");
  inputxES.setShape(0, nHitsES);
  auto dataxES = inputxES.allocate<float>();
  auto& vdataxES = (*dataxES)[0];

  auto& inputfES = iInput.at("fES__4");
  inputfES.setShape(0, nHitsES);
  auto datafES = inputfES.allocate<int64_t>();
  auto& vdatafES = (*datafES)[0];

  auto& inputBatchES = iInput.at("xES_batch__7");
  inputBatchES.setShape(0, nHitsES);
  auto dataBatchES = inputBatchES.allocate<int64_t>();
  auto& vdataBatchES = (*dataBatchES)[0];
  */

  /*
   * Fill input tensors by iterating over particles...
   */
  int64_t partNum = 0;
  CaloCellGeometryMayOwnPtr geom;
  for (auto& part : *particles_) {
    const reco::SuperClusterRef& sc = part.superCluster();

    if (skip(part))
      continue;

    std::vector<std::pair<DetId, float>> hitsAndFractions = sc->hitsAndFractions();
    EcalRecHitCollection::const_iterator hit;

    //iterate over ECAL hits...
    int64_t hitNum = 0;
    for (const auto& detitr : hitsAndFractions) {
      DetId id = detitr.first.rawId();
      if (isEB(part)) {
        geom = ecalEBGeom->getGeometry(id);
        hit = recHitsEB.find(detitr.first);
      } else {
        geom = ecalEEGeom->getGeometry(id);
        hit = recHitsEE.find(detitr.first);
      }

      //fill xECAL

      
      auto pos = geom->getPosition();
      inputxECAL[hitNum][0] = rescale(pos.x(), XY_MIN, XY_RANGE);
      inputxECAL[hitNum][1] = rescale(pos.y(), XY_MIN, XY_RANGE);
      inputxECAL[hitNum][2] = rescale(pos.z(), Z_MIN, Z_RANGE);
      inputxECAL[hitNum][3] = rescale(hit->energy() * detitr.second, ECAL_MIN, ECAL_RANGE);
      inputxECAL[hitNum][4] = rescale(ped->find(detitr.first)->rms(1), NOISE_MIN, NOISE_RANGE);

      //vdataxECAL.push_back(rescale(pos.x(), XY_MIN, XY_RANGE));
      //vdataxECAL.push_back(rescale(pos.y(), XY_MIN, XY_RANGE));
      //vdataxECAL.push_back(rescale(pos.z(), Z_MIN, Z_RANGE));
      //vdataxECAL.push_back(rescale(hit->energy() * detitr.second, ECAL_MIN, ECAL_RANGE));
      //vdataxECAL.push_back(rescale(ped->find(detitr.first)->rms(1), NOISE_MIN, NOISE_RANGE));

      //fill fECAL
      int64_t flagVal = 0;
      if (hit->checkFlag(EcalRecHit::kGood))
        flagVal += 1;
      if (hit->checkFlag(EcalRecHit::kOutOfTime))
        flagVal += 2;
      if (hit->checkFlag(EcalRecHit::kPoorCalib))
        flagVal += 4;

      inputfECAL[0] = flagVal;
      //vdatafECAL.push_back(flagVal);

      //fill gain
      int64_t gainVal = 0;
      if (hit->checkFlag(EcalRecHit::kHasSwitchToGain6))
        gainVal = 1;
      else if (hit->checkFlag(EcalRecHit::kHasSwitchToGain1))
        gainVal = 0;
      else
        gainVal = 2;

      inputGainECAL[hitNum] = gainVal;
      //vdataGainECAL.push_back(gainVal);

      //fill batch number
      inputBatchECAL[hitNum] = partNum;
      ++hitNum;
      //vdataBatchECAL.push_back(partNum);
    }  //end iterate over ECAL hits

    //iterate over ES clusters...
    int64_t hitNumES = 0;
    for (auto iES = sc->preshowerClustersBegin(); iES != sc->preshowerClustersEnd(); ++iES) {
      for (const auto& ESitr : (*iES)->hitsAndFractions()) {  //iterate over ES hits
        hit = recHitsES.find(ESitr.first);
        geom = ecalESGeom->getGeometry(ESitr.first);
        auto& pos = geom->getPosition();

        //fill xES
        inputxES[hitNumES][0] = rescale(pos.x(), XY_MIN, XY_RANGE);
	      inputxES[hitNumES][1] = rescale(pos.y(), XY_MIN, XY_RANGE);
	      inputxES[hitNumES][2] = rescale(pos.z(), Z_MIN, Z_RANGE);
	      inputxES[hitNumES][3] = rescale(hit->energy(), ES_MIN, ES_RANGE);
        //vdataxES.push_back(rescale(pos.x(), XY_MIN, XY_RANGE));
        //vdataxES.push_back(rescale(pos.y(), XY_MIN, XY_RANGE));
        //vdataxES.push_back(rescale(pos.z(), Z_MIN, Z_RANGE));
        //vdataxES.push_back(rescale(hit->energy(), ES_MIN, ES_RANGE));

        //fill fES
        int64_t flagVal = 0;
        if (hit->checkFlag(EcalRecHit::kESGood))
          flagVal += 1;

        inputfES[hitNumES] = flagVal;
        //vdatafES.push_back(flagVal);

        //fill batchES
        inputBatchES[hitNumES] = partNum;
        ++hitNumES;
        //vdataBatchES.push_back(partNum);
      }  //end iterate over ES hits
    }  //end iterate over ES clusters

    //fill gx
    inputGx[partNum][0] = rescale(rho, RHO_MIN, RHO_RANGE);
    inputGx[partNum][1] = rescale(part.hadronicOverEm(), HOE_MIN, HOE_RANGE);
    //vdataGx.push_back(rescale(rho, RHO_MIN, RHO_RANGE));
    //vdataGx.push_back(rescale(part.hadronicOverEm(), HOE_MIN, HOE_RANGE));

    //increment particle number
    ++partNum;
  }  // end iterate over particles

  /*
   * Convert input tensors to server data format
   */

  inputs.push_back(dataxECAL);
  inputs.push_back(datafECAL);
  inputs.push_back(dataGainECAL);
  inputs.push_back(dataBatchECAL);
  inputs.push_back(dataGx);
  inputs.push_back(dataxES);
  inputs.push_back(datafES);
  inputs.push_back(dataBatchES);

  /*
  inputxECAL.toServer(dataxECAL);
  inputfECAL.toServer(datafECAL);
  inputGainECAL.toServer(dataGainECAL);
  inputBatchECAL.toServer(dataBatchECAL);

  inputGx.toServer(dataGx);

  inputxES.toServer(dataxES);
  inputfES.toServer(datafES);
  inputBatchES.toServer(dataBatchES);
  */
  //const auto& particles_ = iEvent.getHandle(particleToken_);

  std::vector<std::pair<float, float>> corrections;
  corrections.reserve(nPart_);

  //if there are no particles, the fromServer() call will fail
  //but we can just put() an empty valueMap
  if (nPart_) {
    const auto& out_raw = module_.forward(inputs).toTensor();
    auto out = out_raw.accessor<float, 2>();
    //const auto& out = iOutput.at("combined_output__0").fromServer<float>();

    unsigned i = 0;
    float mu, sigma, Epred, sigmaPred, rawE;
    for (unsigned iPart = 0; iPart < nPart_; ++iPart) {
      const auto& part = particles_->at(iPart);
      if (!skip(part)) {
        mu = correction(out[0][0 + 11 * i]);
        sigma = resolution(out[0][6 + 11 * i]);
        ++i;

        rawE = part.superCluster()->rawEnergy();
        Epred = mu * rawE;
        sigmaPred = sigma * rawE;
        corrections.emplace_back(Epred, sigmaPred);
      } else {
        corrections.emplace_back(-1.0f, -1.0f);
      }
    }
  }

  //fill
  auto out = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*out);
  filler.insert(particles_, corrections.begin(), corrections.end());
  filler.fill();

  iEvent.put(std::move(out));
}

template <typename T>
void DRNCorrectionProducerDirectT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //TritonClient::fillPSetDescription(desc);
  desc.add<std::string>("modelPath");
  desc.add<edm::InputTag>("particleSource");
  desc.add<edm::InputTag>("rhoName");
  desc.add<edm::InputTag>("reducedEcalRecHitsEB", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("reducedEcalRecHitsEE", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("reducedEcalRecHitsES", edm::InputTag("reducedEcalRecHitsES"));
  descriptions.addWithDefaultLabel(desc);
}

//reco:: template instances are supported
//uncomment the lines below to enable them

//using GsfElectronDRNCorrectionProducer = DRNCorrectionProducerT<reco::GsfElectron>;
//using GedPhotonDRNCorrectionProducer = DRNCorrectionProducerT<reco::Photon>;
//DEFINE_FWK_MODULE(GedPhotonDRNCorrectionProducer);
//DEFINE_FWK_MODULE(GsfElectronDRNCorrectionProducer);

using PatElectronDRNCorrectionProducerDirect = DRNCorrectionProducerDirectT<pat::Electron>;
using PatPhotonDRNCorrectionProducerDirect = DRNCorrectionProducerDirectT<pat::Photon>;

DEFINE_FWK_MODULE(PatPhotonDRNCorrectionProducerDirect);
DEFINE_FWK_MODULE(PatElectronDRNCorrectionProducerDirect);