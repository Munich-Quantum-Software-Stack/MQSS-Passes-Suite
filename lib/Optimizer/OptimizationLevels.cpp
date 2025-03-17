#include "Optimizer/Pipelines.hpp"
#include "Passes/Transforms.hpp"

using namespace mlir;

void mqss::opt::O1(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void mqss::opt::O2(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // more passes to be added here
}

void mqss::opt::O3(PassManager &pm) {
  // more passes to be added here
  pm.addNestedPass<func::FuncOp>(createDoubleCnotCancellationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
