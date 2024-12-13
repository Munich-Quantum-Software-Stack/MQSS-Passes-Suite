#ifndef PASSES_H
#define PASSES_H

#pragma once

#include <stdexcept>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "qdmi.h"
#include "sc/heuristic/HeuristicMapper.hpp"

#define CUDAQ_PREFIX_FUNCTION "__nvqpp__mlirgen__"

namespace mqss::opt{

std::unique_ptr<mlir::Pass> createCustomExamplePass();
std::unique_ptr<mlir::Pass> createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream);

std::unique_ptr<mlir::Pass> createQuakeQMapPass(Architecture &architecture, const Configuration &settings);

} // end namespace
#endif // PASSES_H
