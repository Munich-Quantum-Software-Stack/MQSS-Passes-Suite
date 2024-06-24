/**
 * @file QirBarrierBeforeFinalMeasurements.cpp
 * @brief Implementation of the 'QirBarrierBeforeFinalMeasurementsPass' class.
 * <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/tree/Plugins/src/passes?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://qiskit.org/documentation/stubs/qiskit.transpiler.passes.BarrierBeforeFinalMeasurements.html
 */

#include <QirBarrierBeforeFinalMeasurements.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 *
 * @param module The module of the submitted QIR.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirBarrierBeforeFinalMeasurementsPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/)
{
    std::vector<Instruction *> mz_instructions;
    bool barrier_found = false;
    for (auto &function : module)
    {
        for (auto &block : function)
        {
            for (auto &instruction : block)
            {
                CallInst *call_instruction = dyn_cast<CallInst>(&instruction);

                if (call_instruction)
                {
                    Function *mz_function =
                        call_instruction->getCalledFunction();

                    if (mz_function == nullptr)
                        continue;

                    std::string call_name =
                        static_cast<std::string>(mz_function->getName());

                    if (call_name == "__quantum__qis__barrier__body")
                        goto exit_loops;

                    if (call_name == "__quantum__qis__mz__body")
                    {
                        mz_instructions.push_back(&instruction);
                        goto exit_loops;
                    }
                }
            }
        }

    exit_loops:

        if (mz_instructions.empty())
            return PreservedAnalyses::none();

        LLVMContext &Ctx = function.getContext();

        FunctionType *function_type =
            FunctionType::get(Type::getVoidTy(Ctx), // return void
                              false);               // no variable arguments

        Function *barrier_function;
        if (barrier_found)
            barrier_function =
                module.getFunction("__quantum__qis__barrier__body");
        else
        {
            barrier_function = Function::Create(
                function_type,
                function.getLinkage(), // Function::ExternalWeakLinkage,
                "__quantum__qis__barrier__body", module);
        }

        // while(!mz_instructions.empty()){
        Instruction *mz_instruction = mz_instructions.back();

        CallInst::Create(function_type,
                         barrier_function, // new function
                         "",               // no name required
                         mz_instruction);  // insert before

        mz_instructions.pop_back();
        //}
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the
 * 'QirBarrierBeforeFinalMeasurementsPass' as a 'PassModule'.
 * @return QirBarrierBeforeFinalMeasurementsPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirBarrierBeforeFinalMeasurementsPass();
}
