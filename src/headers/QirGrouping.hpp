#pragma once

#include "PassModule.h"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

struct GroupAnalysis
{
    std::vector<BasicBlock*> qc_cc_blocks{};
    std::vector<BasicBlock*> qc_mc_cc_blocks{};
    GroupAnalysis() : qc_cc_blocks(), qc_mc_cc_blocks() {}
};

class QirGroupingPass : public PassModule {
public:
    using Result = GroupAnalysis;

    static std::string const QIS_START;
    static std::string const READ_INSTR_START;

	enum class ResourceType {
	    UNDEFINED,
	    QUBIT,
	    RESULT
    };

    struct ResourceAnalysis {
        bool         is_const{false};
        uint64_t     id{0};
        ResourceType type{ResourceType::UNDEFINED};
    }; 

	enum {
	    PureClassical              = 0,
	    SourceQuantum              = 1,
	    DestQuantum                = 2,
	    PureQuantum                = SourceQuantum | DestQuantum,
	    TransferClassicalToQuantum = DestQuantum,
	    TransferQuantumToClassical = SourceQuantum,
	    InvalidMixedLocation       = -1
    };

    void prepareSourceSeparation(Module &module, BasicBlock *block);
    void nextQuantumCycle(Module &module, BasicBlock* block);
    void expandBasedOnSource(Module &module, BasicBlock *block);
	void expandBasedOnDest(Module &module, BasicBlock* block, bool move_quatum, std::string const& name);
	bool isQuantumRegister(Type const *type);		
	int64_t classifyInstruction(Instruction const *instr);
	PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam);
	void runBlockAnalysis(Module &module);
    Result runGroupingAnalysis(Module &module);
private:
    void deleteInstructions();

    ResourceAnalysis operandAnalysis(Value* val) const;

    BasicBlock* post_classical_block_{nullptr};
    BasicBlock* quantum_block_{nullptr};
    BasicBlock* pre_classical_block_{nullptr};

    std::shared_ptr<IRBuilder<>> pre_classical_builder_{};
    std::shared_ptr<IRBuilder<>> quantum_builder_{};
    std::shared_ptr<IRBuilder<>> post_classical_builder_{};

    std::vector<BasicBlock*> quantum_blocks_{};
    std::vector<BasicBlock*> classical_blocks_{};

	std::unordered_set<BasicBlock*> visited_blocks_; // TODO Do we need this?

    std::unordered_set<BasicBlock*> contains_quantum_circuit_{};
    std::unordered_set<BasicBlock*> contains_quantum_measurement_{};
    std::unordered_set<BasicBlock*> pure_quantum_instructions_{};
    std::unordered_set<BasicBlock*> pure_quantum_measurement_{};

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

