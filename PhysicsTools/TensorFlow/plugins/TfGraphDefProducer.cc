// -*- C++ -*-
//
// Package:    PhysicsTools/TensorFlow
// Class:      TFGraphDefProducer
//
/**\class TFGraphDefProducer
 Description: Produces TfGraphRecord into the event containing a tensorflow GraphDef object that can be used for running inference on a pretrained network
*/
//
// Original Author:  Joona Havukainen
//         Created:  Fri, 24 Jul 2020 08:04:00 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"

// class declaration

class TfGraphDefProducer : public edm::ESProducer {
public:
  TfGraphDefProducer(const edm::ParameterSet&);
  using ReturnType = std::unique_ptr<TfGraphDefWrapper>;

  ReturnType produce(const TfGraphRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const std::string filename_;
  const tensorflow::Backend backend_;
  // ----------member data ---------------------------
};

TfGraphDefProducer::TfGraphDefProducer(const edm::ParameterSet& iConfig)
    : filename_(iConfig.getParameter<edm::FileInPath>("FileName").fullPath()),
      backend_(tensorflow::Options::getBackendFromString(iConfig.getParameter<std::string>("backend"))) {
  auto componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
}

// ------------ method called to produce the data  ------------
TfGraphDefProducer::ReturnType TfGraphDefProducer::produce(const TfGraphRecord& iRecord) {
  auto* graph = tensorflow::loadGraphDef(filename_);
  tensorflow::Options options(backend_);
  return std::make_unique<TfGraphDefWrapper>(tensorflow::createSession(graph, options), graph);
}

void TfGraphDefProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "tfGraphDef");
  desc.add<edm::FileInPath>("FileName", edm::FileInPath());
  desc.add<std::string>("backend", "cpu");
  descriptions.add("tfGraphDefProducer", desc);
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(TfGraphDefProducer);
