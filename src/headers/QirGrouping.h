#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>

#include <unordered_set>

#include "PassModule.h"

namespace llvm {

class QirGroupingPass : public PassModule { //PassInfoMixin<QirGroupingPass> {
    public:
        enum class ResourceType
        {
            UNDEFINED,
            QUBIT,
            RESULT
        };

        struct ResourceAnalysis{
            bool         is_const{false};
            uint64_t     id{0};
            ResourceType type{ResourceType::UNDEFINED};
        }; 

        enum
        {
            PureClassical              = 0,
            SourceQuantum              = 1,
            DestQuantum                = 2,
            PureQuantum                = SourceQuantum | DestQuantum,
            TransferClassicalToQuantum = DestQuantum,
            TransferQuantumToClassical = SourceQuantum,
            InvalidMixedLocation       = -1
        };

        void prepareSourceSeparation(Module *module, BasicBlock *block);
        void nextQuantumCycle(Module *module, BasicBlock* block);
        void expandBasedOnSource(Module *module, BasicBlock *block);
        void expandBasedOnDest(Module *module, BasicBlock* block, bool move_quatum, std::string const& name);
        bool isQuantumRegister(Type const *type);		
        int64_t classifyInstruction(Instruction const *instr);
        PreservedAnalyses run(Module *module, ModuleAnalysisManager &/*mam*/);
    private:
        void deleteInstructions();

        ResourceAnalysis operandAnalysis(Value* val) const;

        BasicBlock* pre_quantum_block{nullptr};
        BasicBlock* quantum_block{nullptr};
        BasicBlock* post_quantum_block{nullptr};

        std::shared_ptr<IRBuilder<>> pre_quantum_builder{};
        std::shared_ptr<IRBuilder<>> quantum_builder{};
        std::shared_ptr<IRBuilder<>> post_quantum_builder{};

        std::vector<BasicBlock*> quantum_blocks{};
        std::vector<BasicBlock*> classical_blocks{};

	    std::unordered_set<std::string> quantum_register_types = {
            "Qubit", 
            "Result"};
        std::unordered_set<std::string> irreversible_operations = {
            "__quantum__qis__reset__body",
            "__quantum__qis__mz__body"};
	    std::string qir_runtime_prefix = "__quantum__rt__";

        std::vector<Instruction*> to_delete;
};

}


