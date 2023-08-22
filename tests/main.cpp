// QIR Pass Manager
#include "../src/QirPassManager.h"
#include <iostream>

using namespace llvm;

int main() {
    /*PassBuilder             PB;
    LoopAnalysisManager     LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager    CGAM;*/
    ModuleAnalysisManager   MAM; 

    // Set up the analysis managers
    /*PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);*/

    QirPassManager QPM;

    LLVMContext Context;

    // Step 1: Read the LLVM IR from .ll file
    SMDiagnostic error;
    std::unique_ptr<Module> module = parseIRFile("../benchmarks/bell_state.ll", error, Context);
    if (!module) {
        std::cerr << "Error reading .ll file." << std::endl;
        error.print("apply_pass", errs());
        return 1;
    }

    QPM.append("./src/passes/libQirRemoveNonEntrypointFunctionsPass.so");
    QPM.append("./src/passes/libQirGroupingPass.so");
    QPM.append("./src/passes/libQirBarrierBeforeFinalMeasurementsPass.so");
    QPM.append("./src/passes/libQirCXCancellationPass.so");
    QPM.append("./src/passes/libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so");

    QPM.run(module.get(), MAM);

    module->print(outs(), nullptr);

    return 0;
}

