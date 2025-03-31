#include "Support/CodeGen/Quake.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// Base class extending PassWrapper with a common method
template <typename DerivedT>
class BaseMQSSPass
    : public PassWrapper<DerivedT, OperationPass<mlir::ModuleOp>> {
public:
  virtual void operationsOnQuantumKernel(
      func::FuncOp kernel) = 0; // this has to be re-written by each pass
private:
  std::tuple<SmallVector<Operation *, 16>, mlir::WalkResult>
  getQuakeKernels(mlir::ModuleOp module) {
    SmallVector<Operation *, 16> kernels;
    auto walkResult = module.walk([&kernels](Operation *op) {
      // Check if it is a quantum kernel
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp->hasAttr(cudaq::entryPointAttrName)) {
          kernels.push_back(funcOp);
          return WalkResult::advance();
        }
        for (auto arg : funcOp.getArguments())
          if (isa<quake::RefType, quake::VeqType>(arg.getType())) {
            kernels.push_back(funcOp);
            return WalkResult::advance();
          }
        // Skip functions which are not quantum kernels
        return WalkResult::skip();
      }
      // Check if it is controlled quake.apply
      if (auto applyOp = dyn_cast<quake::ApplyOp>(op))
        if (!applyOp.getControls().empty())
          return WalkResult::interrupt();

      return WalkResult::advance();
    });
    return std::make_tuple(kernels, walkResult);
  }

  void runOnOperation() override {
    auto module = this->getOperation();
    auto [kernels, walkResult] = getQuakeKernels(module);
    if (walkResult.wasInterrupted()) {
      module.emitError("Basis conversion doesn't work with `quake.apply`");
      this->signalPassFailure();
      return;
    }
    if (kernels.empty())
      return;
    // Process kernels in parallel
    parallelForEach(module.getContext(), kernels, [this](Operation *kernel) {
      if (auto funcOp = dyn_cast<func::FuncOp>(kernel)) {
        static_cast<DerivedT *>(this)->operationsOnQuantumKernel(funcOp);
      }
    });
  }
};
