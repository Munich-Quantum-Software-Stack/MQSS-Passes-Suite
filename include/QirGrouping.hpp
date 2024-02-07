/**
 * @file QirGrouping.hpp
 * @brief Declaration of the 'QirGroupingPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @struct GroupAnalysis
 * @brief TODO
 */
struct GroupAnalysis
{
    std::vector<BasicBlock *> qc_cc_blocks{};              /**< TODO */
    std::vector<BasicBlock *> qc_mc_cc_blocks{};           /**< TODO */
    GroupAnalysis() : qc_cc_blocks(), qc_mc_cc_blocks() {} /**< TODO */
};

/**
 * @class QirGroupingPass
 * @brief This pass groups the instructions into purely-quantum and
 * purely-classical blocks.
 */
class QirGroupingPass : public PassModule
{
  public:
    using Result = GroupAnalysis;

    static std::string const QIS_START;        /**< TODO */
    static std::string const READ_INSTR_START; /**< TODO */

    /**
     * @enum ResourceType
     * @brief The `ResourceType` enum class defines the resource
     * types Qubit and Result.
     */
    enum class ResourceType
    {
        UNDEFINED, /**< An undefined resource. */
        QUBIT,     /**< A Qubit resource. */
        RESULT     /**< A Result resource. */
    };

    /**
     * @struct ResourceAnalysis
     * @brief TODO
     */
    struct ResourceAnalysis
    {
        bool is_const{false};                       /**< TODO */
        uint64_t id{0};                             /**< TODO */
        ResourceType type{ResourceType::UNDEFINED}; /**< TODO */
    };

    /**
     * @brief This anonymous enum defines various resource types and mixed
     * locations.
     */
    enum
    {
        PureClassical = 0, /**< A pure classical resource. */
        SourceQuantum = 1, /**< A source quantum resource. */
        DestQuantum = 2,   /**< A destination quantum resource. */
        PureQuantum =
            SourceQuantum | DestQuantum, /**< A pure quantum resource. */
        TransferClassicalToQuantum =
            DestQuantum, /**< Representes the transfer of classical to quantum
                            resources. */
        TransferQuantumToClassical =
            SourceQuantum, /**< Represents the transfer of quantum to classical
                              resources. */
        InvalidMixedLocation = -1 /**< Represents an invalid mixed location. */
    };

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @param block TODO
     */
    void prepareSourceSeparation(Module &module, BasicBlock *block);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @param block TODO
     */
    void nextQuantumCycle(Module &module, BasicBlock *block);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @param block TODO
     */
    void expandBasedOnSource(Module &module, BasicBlock *block);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @param block TODO
     * @param move_quatum TODO
     * @param name TODO
     */
    void expandBasedOnDest(Module &module, BasicBlock *block, bool move_quatum,
                           std::string const &name);

    /**
     * @brief TODO
     * @param type TODO
     * @return bool
     */
    bool isQuantumRegister(Type const *type);

    /**
     * @brief TODO
     * @param instr TODO
     * @return int64_t
     */
    int64_t classifyInstruction(Instruction const *instr);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     */
    void runBlockAnalysis(Module &module);

    /**
     * @brief TODO
     * @param module The module of the submitted QIR.
     * @return Result
     */
    Result runGroupingAnalysis(Module &module);

  private:
    void deleteInstructions();

    ResourceAnalysis operandAnalysis(Value *val) const;

    BasicBlock *post_classical_block_{nullptr};
    BasicBlock *quantum_block_{nullptr};
    BasicBlock *pre_classical_block_{nullptr};

    std::shared_ptr<IRBuilder<>> pre_classical_builder_{};
    std::shared_ptr<IRBuilder<>> quantum_builder_{};
    std::shared_ptr<IRBuilder<>> post_classical_builder_{};

    std::vector<BasicBlock *> quantum_blocks_{};
    std::vector<BasicBlock *> classical_blocks_{};

    std::unordered_set<BasicBlock *> visited_blocks_; // TODO Do we need this?

    std::unordered_set<BasicBlock *> contains_quantum_circuit_{};
    std::unordered_set<BasicBlock *> contains_quantum_measurement_{};
    std::unordered_set<BasicBlock *> pure_quantum_instructions_{};
    std::unordered_set<BasicBlock *> pure_quantum_measurement_{};

    std::unordered_set<std::string> quantum_register_types = {"Qubit",
                                                              "Result"};

    std::string qir_runtime_prefix = "__quantum__rt__";

    std::vector<Instruction *> to_delete;
};

} // namespace llvm
