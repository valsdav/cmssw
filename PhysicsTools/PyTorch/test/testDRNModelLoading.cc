#include <torch/script.h>
#include "testBase.h"
#include <iostream>
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
  
  std::string model_path = "model.pt";
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

  // Print info about the Pytorch jit model
  std::cout << "Model info:\n";
  std::cout << "Model parameters: " << module.named_parameters().size() << '\n';
  std::cout << "Model buffers: " << module.named_buffers().size() << '\n';
  std::cout << "Model attributes: " << module.named_attributes().size() << '\n';
  // print the jitted graph of the model
  std::cout << "Model graph:\n";
  print_modules(module); 
  
  // // Create a vector of inputs.
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones(10, device));

  // // Execute the model and turn its output into a tensor.
  // at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << "output: " << output << '\n';
  // CPPUNIT_ASSERT(output.item<float_t>() == 110.);
  // std::cout << "ok\n";
}
