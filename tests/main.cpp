#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <iostream>
#include <dlfcn.h>

#include "../src/QirPassManager.h"
#include "../src/headers/QirGrouping.h"
#include "../src/headers/QirBarrierBeforeFinalMeasurements.h"
#include "../src/headers/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.h"
#include "../src/headers/QirCXCancellation.h"
#include "../src/headers/QirRemoveNonEntrypointFunctions.h"

using namespace llvm;

int main() {
    ModuleAnalysisManager MAM;
    QirPassManager        QPM;
    LLVMContext           Context;
    SMDiagnostic          error;

    std::unique_ptr<Module> module = parseIRFile("../benchmarks/bell_state.ll", error, Context);
    if(!module) {
        std::cerr << "Error reading .ll file." << std::endl;
        error.print("apply_pass", errs());
        return 1;
    }

    QPM.append(new QirRemoveNonEntrypointFunctionsPass());
    QPM.append(new QirGroupingPass());
    QPM.append(new QirBarrierBeforeFinalMeasurementsPass());
    QPM.append(new QirCXCancellationPass());
    QPM.append(new QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass());

    QPM.run(module.get(), MAM);

    module->print(outs(), nullptr);

    for(PassModule* pass : QPM.getPasses())
        delete pass;

    return 0;
}

