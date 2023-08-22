// QIR Pass Manager
#include "../src/QirPassManager.h"
#include <iostream>

using namespace llvm;

int main() {
	ModuleAnalysisManager MAM;
    QirPassManager        QPM;
    LLVMContext           Context;
    SMDiagnostic          error;

   	// Read QIR 
    std::unique_ptr<Module> module = parseIRFile("../benchmarks/bell_state.ll", error, Context);
    if (!module) {
        std::cerr << "Error reading .ll file." << std::endl;
        return 1;
    }

	// Append passes
    QPM.append("./src/passes/libQirRemoveNonEntrypointFunctionsPass.so");
    QPM.append("./src/passes/libQirGroupingPass.so");
    QPM.append("./src/passes/libQirBarrierBeforeFinalMeasurementsPass.so");
    QPM.append("./src/passes/libQirCXCancellationPass.so");
    QPM.append("./src/passes/libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so");

	// Run the passes
    QPM.run(module.get(), MAM);

	// Print the result
    module->print(outs(), nullptr);

    return 0;
}

