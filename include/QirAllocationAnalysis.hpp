/**
 * @file QirAllocationAnalysis.hpp
 * @brief Declaration of the 'QirAllocationAnalysisPass' class.
 */

#pragma once

#include "llvm.hpp"
#include <qdmi.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm
{

/**
 * @struct AllocationAnalysis
 * @brief TODO
 */
struct AllocationAnalysis
{
    /**
     * @enum ResourceType
     * @brief TODO
     */
    enum ResourceType
    {
        NotResource,   /**< Not a resource. */
        QubitResource, /**< A Qubit resource. */
        ResultResource /**< A Result resource. */
    };

    /**
     * @struct ResourceAccessLocation
     * @brief TODO
     */
    struct ResourceAccessLocation
    {
        Value *operand{nullptr};                      /**< TODO. */
        ResourceType type{ResourceType::NotResource}; /**< TODO. */
        uint64_t index{static_cast<uint64_t>(-1)};    /**< TODO. */
        Instruction *used_by{nullptr};                /**< TODO. */
        uint64_t operand_id{0};                       /**< TODO. */
    };

    using ResourceValueToId =
        std::unordered_map<Value *, ResourceAccessLocation>;
    using ResourceAccessList = std::vector<ResourceAccessLocation>;

    uint64_t largest_qubit_index{0};  /**< TODO. */
    uint64_t largest_result_index{0}; /**< TODO. */
    uint64_t usage_qubit_counts{0};   /**< TODO. */
    uint64_t usage_result_counts{0};  /**< TODO. */

    ResourceValueToId access_map{};       /**< TODO. */
    ResourceAccessList resource_access{}; /**< TODO. */
};

/**
 * @class QirAllocationAnalysisPass
 * @brief TODO
 */
class QirAllocationAnalysisPass
    : public AnalysisInfoMixin<QirAllocationAnalysisPass>
{
  public:
    using Result = AllocationAnalysis;
    using BlockSet = std::unordered_set<BasicBlock *>;
    using ResourceType = AllocationAnalysis::ResourceType;
    using ResourceAccessLocation = AllocationAnalysis::ResourceAccessLocation;

    Result AnalysisResult;

    /**
     * @brief Applies this pass to the function 'function'.
     *
     * @param function The function.
     * @param FAM The function analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Function &function, FunctionAnalysisManager &FAM,
                          QDMI_Device dev);

  private:
    bool extractResourceId(Value *value, uint64_t &return_value,
                           ResourceType &type) const;
};

} // namespace llvm
