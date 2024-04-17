/**
 * @file QirDeferMeasurement.cpp
 * @brief Implementation of the 'QirDeferMeasurementPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirDeferMeasurement.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/DeferMeasurementPass/DeferMeasurementPass.cpp
 */

#include <QirDeferMeasurement.hpp>

using namespace llvm;

/**
 * @var QirDeferMeasurementPass::RECORD_INSTR_END
 * @brief Used within the 'QirDeferMeasurementPass' to define the record prefix.
 */
std::string const QirDeferMeasurementPass::RECORD_INSTR_END = "_record_output";

/**
 * @brief Constructor for QirDeferMeasurementPass.
 *
 * This constructor initializes the QirDeferMeasurementPass object.
 */
QirDeferMeasurementPass::QirDeferMeasurementPass()
{
    readout_names_.insert("__quantum__qis__m__body");
    readout_names_.insert("__quantum__qis__mz__body");
    readout_names_.insert("__quantum__qis__reset__body");
    readout_names_.insert("__quantum__qis__read_result__body");
}

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirDeferMeasurementPass::run(Module &module,
                                               ModuleAnalysisManager &MAM,
                                               QDMI_Device dev)
{
    for (auto &function : module)
    {
        for (auto &block : function)
        {
            // Identifying record functions
            std::vector<Instruction *> records;
            for (auto &instr : block)
            {
                auto call = dyn_cast<CallBase>(&instr);
                if (call != nullptr)
                {
                    auto f = call->getCalledFunction();
                    if (f != nullptr)
                    {
                        auto name = static_cast<std::string>(f->getName());
                        bool is_readout =
                            (name.size() >= RECORD_INSTR_END.size() &&
                             name.substr(name.size() - RECORD_INSTR_END.size(),
                                         RECORD_INSTR_END.size()) ==
                                 RECORD_INSTR_END);

                        if (is_readout ||
                            readout_names_.find(name) != readout_names_.end())
                        {
                            records.push_back(&instr);
                        }
                    }
                }
            }

            // Moving function calls
            if (!records.empty())
            {
                IRBuilder<> builder(function.getContext());
                builder.SetInsertPoint(block.getTerminator());

                for (auto instr : records)
                {
                    auto new_instr = instr->clone();
                    new_instr->takeName(instr);
                    builder.Insert(new_instr);
                    instr->replaceAllUsesWith(new_instr);

                    if (!instr->use_empty())
                    {
                        //errs() << "   [Pass]..............Error: unexpected "
                        //          "uses of "
                        //          "instruction "
                        //          "while moving records to the bottom of the "
                        //          "block\n";
                        return PreservedAnalyses::none();
                    }

                    instr->eraseFromParent();
                }
            }
        }
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirDeferMeasurementPass' as a
 * 'PassModule'.
 * @return QirDeferMeasurementPass
 */
extern "C" PassModule *loadQirPass() { return new QirDeferMeasurementPass(); }
