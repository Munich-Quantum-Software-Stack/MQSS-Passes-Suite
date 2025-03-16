#include "Optimizer/Pipelines.h"

using namespace mlir;

void cudaq::opt::o1(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void cudaq::opt::o2(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void cudaq::opt::o3 : (PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}
