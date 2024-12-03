#ifndef PASSES_H
#define PASSES_H

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace mqss::opt{

std::unique_ptr<mlir::Pass> createCustomExamplePass();
std::unique_ptr<mlir::Pass> createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream);
std::unique_ptr<mlir::Pass> createQuakeQMapPass(llvm::raw_string_ostream &ostream);

} // end namespace
#endif // PASSES_H
