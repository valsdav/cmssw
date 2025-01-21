#include <torch/script.h>
#include "testBase.h"
#include <iostream>
#include <cppunit/extensions/HelperMacros.h>
#include <memory>
#include <vector>

void tabs(size_t num) {
  for (size_t i = 0; i < num; i++) {
    std::cout << "\t";
  }
}

void print_modules(const torch::jit::script::Module& module, size_t level = 0) {
  std::cout << module.dump_to_str(true, false,false) << " (\n";
  for (const auto& child : module.children()) {
    tabs(level + 1);
    print_modules(child, level + 1);
  }
  tabs(level);
  std::cout << ")\n";
}

class testSimpleDNN : public testBasePyTorch {
  CPPUNIT_TEST_SUITE(testSimpleDNN);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  std::string pyScript() const override;
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSimpleDNN);

std::string testSimpleDNN::pyScript() const { return ""; }

void testSimpleDNN::test() {
  
  std::string model_path = cmsswPath("src/PhysicsTools/RecoEcal-EgammaClusterProducers/models/MustacheEB/1/model.pt");
  torch::Device device(torch::kCPU);
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_path);
    module.to(device);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
  }
  std::cout << "Model loaded\n";

  // // Print info about the Pytorch jit model
  // std::cout << "Model info:\n";
  // std::cout << "Model parameters: " << module.named_parameters().size() << '\n';
  // std::cout << "Model buffers: " << module.named_buffers().size() << '\n';
  // std::cout << "Model attributes: " << module.named_attributes().size() << '\n';
  // // print the jitted graph of the model
  // std::cout << "Model graph:\n";
  // print_modules(module); 
  // Access the method schema for 'forward'
  auto method = module.get_method("forward");
  auto schema = method.function().getSchema();
  
  // Print the schema
  std::cout << "Method Schema: " << schema << std::endl;
  
  /*
  - [Npart, Nrechit, 5 ] floats, rechit features
- [Npart, Nrechit, 1] integer, rechit flag
- [Npart, Nrechit, 1] integer, rechit gain
- [Npart] index over batching for different particles
- [Npart, 2] floats, global features
- [Npart, Nrechit, 5 ] floats, rechit features ES
- [Npart, Nrechit, 1] integer, rechit flag ES
- [Npart] index over batching for different particles ES
  */
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;

  auto rechit_features = torch::rand({30, 4}, device);
  //auto rechit_flag = torch::ones({3, 10, 1}, device);
  //auto rechit_gain = torch::ones({3, 10, 1}, device);
  auto particle_index = torch::ones({30}, torch::TensorOptions().dtype(torch::kLong).device(device));
  
  auto global_features = torch::rand({2}, device);
  //auto rechit_features_es = torch::ones({3, 10, 5}, device);
  //auto rechit_flag_es = torch::ones({3, 10, 1}, device);
  //auto particle_index_es = torch::ones({3}, device);
  // Increase the index for each item
  for (int i = 0; i < 30; i++) {
    int j = i % 10;
    particle_index[j] = i;
    //particle_index_es[i] = i;
  }

  inputs.push_back(rechit_features);
  //inputs.push_back(rechit_flag);
  //inputs.push_back(rechit_gain);
  inputs.push_back(particle_index);
  inputs.push_back(global_features);
  //inputs.push_back(rechit_features_es);
  //inputs.push_back(rechit_flag_es);
  //inputs.push_back(particle_index_es);
  

  // // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "output: " << output << '\n';
  // CPPUNIT_ASSERT(output.item<float_t>() == 110.);
  // std::cout << "ok\n";
}
