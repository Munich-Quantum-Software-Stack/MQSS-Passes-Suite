/**
 * @file QirAnnotateUnsupportedGates.cpp
 * @brief Implementation of the 'QirAnnotateUnsupportedGatesPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirAnnotateUnsupportedGates.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass inserts an "unsupported" attribute to the appropriate gates
 * after querying the target platform using QDMI.
 */

#include <QirAnnotateUnsupportedGates.hpp>

using namespace llvm;

/**
 * @var QirAnnotateUnsupportedGatesPass::QIS_START
 * @brief Used within the 'QirAnnotateUnsupportedGatesPass' to define the
 * quantum prefix.
 */
std::string const QirAnnotateUnsupportedGatesPass::QIS_START = "__quantum"
                                                               "__qis_";

/**
 * @brief Applies this pass to the QIR's LLVM module.
 *
 * @param module The module of the submitted QIR.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirAnnotateUnsupportedGatesPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/, QDMI_Device dev)
{
    bool changed = false;

    // Fetch the supported gate set using qdmi
    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();
    auto targetArchitecture = qirMetadata.targetPlatform;

    int err = 0, num_gates = 0;
    // TODO Handle err
    QDMI_Gate gates;
    err = QDMI_query_all_gates(dev, &gates);
    err = QDMI_query_gateset_num(dev, &num_gates);

    std::vector<std::string> supported_gate_set;
    for (int i = 0; i < num_gates; i++)
        supported_gate_set.push_back(gates[i].name);

    // Adding  as requested
    for (auto &function : module)
    {
        auto original_gate = static_cast<std::string>(function.getName());

        bool is_quantum =
            (original_gate.size() >= QIS_START.size() &&
             original_gate.substr(0, QIS_START.size()) == QIS_START);

        // We only want to annotate quantum gates
        if (!is_quantum)
            continue;

        // Insert attribute to each unsupported gate
        auto it = std::find(supported_gate_set.begin(),
                            supported_gate_set.end(), original_gate);
        if (it == supported_gate_set.end())
        {
            errs() << "   [Pass]................Transpiling: "
                   << original_gate << '\n';
            function.addFnAttr("unsupported");
            changed = true;
        }
    }

    free(gates);

    if (changed)
        return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}

/**
 * @brief External function for loading the 'QirAnnotateUnsupportedGatesPass' as
 * a 'SpecificPassModule'.
 * @return QirAnnotateUnsupportedGatesPass
 */
extern "C" SpecificPassModule *loadQirPass()
{
    return new QirAnnotateUnsupportedGatesPass();
}
