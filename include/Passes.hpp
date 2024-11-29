#ifndef PASSES_H
#define PASSES_H

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

std::unique_ptr<mlir::Pass> createCustomExamplePass();
std::unique_ptr<mlir::Pass> createPrintQuakeGatesPass(llvm::raw_string_ostream &ostream);

#endif // PASSES_H
