/**
 * @file QirAllocationAnalysis.cpp
 * @brief Implementation of the 'QirAllocationAnalysisPass' analysis pass. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirAllocationAnalysis.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/StaticResourceComponent/AllocationAnalysisPass.cpp
 */

#include <QirAllocationAnalysis.hpp>

using namespace llvm;

// Checks for Qubit and Result pointer types.
bool QirAllocationAnalysisPass::extractResourceId(Value *value,
                                                  uint64_t &return_value,
                                                  ResourceType &type) const
{
    auto *instruction_ptr = dyn_cast<IntToPtrInst>(value);
    auto *operator_ptr =
        dyn_cast<ConcreteOperator<Operator, Instruction::IntToPtr>>(value);
    auto *nullptr_cast = dyn_cast<ConstantPointerNull>(value);

    if (instruction_ptr || operator_ptr || nullptr_cast)
    {
        if (!value->getType()->isPointerTy())
            return false;

        Type *element_type = value->getType()->getPointerElementType();

        if (!element_type->isStructTy())
            return false;

        type = ResourceType::NotResource;
        auto type_name =
            static_cast<std::string>(element_type->getStructName());

        if (type_name == "Qubit")
            type = ResourceType::QubitResource;

        if (type_name == "Result")
            type = ResourceType::ResultResource;

        if (type == ResourceType::NotResource)
            return false;

        bool is_constant_int = nullptr_cast != nullptr;
        uint64_t n = 0;
        auto user = dyn_cast<User>(value);

        // In case there exists a user, it must have exactly one argument
        // which should be an integer. In case of deferred integers, the mapping
        // will not work
        if (user && user->getNumOperands() == 1)
        {
            auto cst = dyn_cast<ConstantInt>(user->getOperand(0));

            if (cst)
            {
                is_constant_int = true;
                n = cst->getValue().getZExtValue();
            }
        }

        if (is_constant_int)
        {
            return_value = n;
            return true;
        }
    }

    return false;
}

/**
 * @brief Applies an analysis pass to the 'function' function.
 * @param function The function.
 * @param FAM The function analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirAllocationAnalysisPass::run(
    Function &function, FunctionAnalysisManager & /*FAM*/)
{
    AllocationAnalysis result;

    std::unordered_set<uint64_t> qubits_used{};
    std::unordered_set<uint64_t> results_used{};

    for (auto &block : function)
    {
        for (auto &instr : block)
        {
            for (uint64_t i = 0; i < instr.getNumOperands(); ++i)
            {
                auto op = instr.getOperand(static_cast<uint32_t>(i));
                ResourceType type = ResourceType::NotResource;
                uint64_t n = 0;

                // Checking if it is a qubit resource reference and extracting
                // the corresponding id and type. Otherwise, we skip to the next
                // operand.
                if (!extractResourceId(op, n, type))
                    continue;

                ResourceAccessLocation value{op, type, n, &instr, i};
                result.access_map[op] = value;
                result.resource_access.push_back(value);

                switch (type)
                {
                case ResourceType::QubitResource:
                    qubits_used.insert(n);
                    result.largest_qubit_index =
                        result.largest_qubit_index < n
                            ? n
                            : result.largest_qubit_index;

                    break;
                case ResourceType::ResultResource:
                    results_used.insert(n);
                    result.largest_result_index =
                        result.largest_result_index < n
                            ? n
                            : result.largest_result_index;
                    break;
                case ResourceType::NotResource:
                    break;
                }
            }
        }
    }

    result.usage_qubit_counts = qubits_used.size();
    result.usage_result_counts = results_used.size();

    AnalysisResult = result;

    return PreservedAnalyses::none();
}
