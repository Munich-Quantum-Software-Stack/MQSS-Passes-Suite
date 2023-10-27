/**
 * @file QirPassRunner.hpp
 * @brief Declaration of the 'QirPassRunner' class.
 */

#ifndef QIR_MODULE_PASS_MANAGER_H
#define QIR_MODULE_PASS_MANAGER_H

#include "headers/PassModule.hpp"

#include <dlfcn.h>
#include <unordered_map>
#include <string>

using namespace llvm;

/**
 * @enum MetadataType
 * @brief Enumerated type for appending information to the metadata
 */
enum MetadataType {
    SUPPORTED_GATE,     /**< A supported gate. */
    REVERSIBLE_GATE,    /**< A reversible gate. */
    IRREVERSIBLE_GATE,  /**< An irreversile gate. */
    AVAILABLE_PLATFORM, /**< An available platfirm. */
    UNKNOWN             /**< Unknown value. */
};

/**
 * @struct QirMetadata
 * @brief This struct holds all required metadata shared amongst the
 * QIR Pass Runner, the QIR Selector Runner, and each pass.
 */
struct QirMetadata {
    // All required metadata shall be declared here
    std::vector<std::string> reversibleGates;                           /**< Vector of reversible gates. */
    std::vector<std::string> irreversibleGates;                         /**< Vector of irreversible gates. */
    std::vector<std::string> supportedGates;                            /**< Vector of supported gates. */
    std::vector<std::string> availablePlatforms;                        /**< Vectors of available platforms. */
    std::unordered_map<std::string, std::string> injectedAnnotations;   /**< Map of injected annotations. */
    bool shouldRemoveCallAttributes;                                    /**< Boolean value for controlling the removal of call attributes. */

    /**
     * @brief Adds entries to multiple vectors of the metadata. Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QirMetadata &qirMetadata = QPR.getMetadata();
     *     qirMetadata.append(REVERSIBLE_GATE, "__quantum__qis__x__body");
     *     QPR.setMetadata(qirMetadata);
     *
     * @param key It indicates the right metadata vector  according to 'MetadataType'
     * @param value The value to push into the vector.
     */
	void append(const int key, const std::string &value) {
        switch(key) {
            case SUPPORTED_GATE:
                supportedGates.push_back(value);
                break;
            case REVERSIBLE_GATE:
                reversibleGates.push_back(value);
                break;
            case IRREVERSIBLE_GATE:
                irreversibleGates.push_back(value);
                break;
            case AVAILABLE_PLATFORM:
                availablePlatforms.push_back(value);
                break;
            default:
                errs() << "Warning: Unknown metadata type: " << key <<  "\n";
        }
    }

    /**
     * @brief The 'injecteAnnotation' inserts in a map a pair of call instructions.
     * This map is currently used to keep track of those LLVM functions with 
     * a 'replaceWith' attribute. Use example:
     * 
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QirMetadata &qirMetadata = QPR.getMetadata();
     *     auto key   = static_cast<std::string>(function1->getName());
     *     auto value = static_cast<std::string>(function2->getName());
     *     qirMetadata.injectAnnotation(key, value);
     *     QPR.setMetadata(qirMetadata);
     *
     * @param key The key of the map entry.
     * @param value The value of the map entry.
     */
    void injectAnnotation(const std::string &key, const std::string &value) {
        injectedAnnotations[key] = value;
    }

    /**
     * @brief Function for removing all attributes associated to LLVM functions.
     * Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QirMetadata &qirMetadata = QPR.getMetadata();
     *     qirMetadata.setRemoveCallAttributes(true);
     *     QPR.setMetadata(qirMetadata);
     *
     * @param value Boolean value for controlling the removal of call attributes.
     */
    void setRemoveCallAttributes(const bool value) {
        shouldRemoveCallAttributes = value;
    }
};


/**
 * @class QirPassRunner
 * @brief The 'QirPassRunner' class is a derived class of the 'PassModule'
 * abstract class. This derived class keeps track of metadata shared
 * among its daemon, the passes, and the selector runner. This
 * class also manages the passes and their invokation.
 *
 * This class is effectively a Pass Manager. It contains a 
 * list of names of shared libraries (.so) implementing 
 * different kinds of passes. This class also contains a 
 * functions for invoking the pre-compiled passes.
 */
class QirPassRunner {
public:
    /**
     * @brief Returns a reference to a 'QirPassRuner' object. This
     * object is created as a 'static' variable inside the
     * 'getInstance' function such that 'getInstance' has a
     * single instance shared across all calls to this
     * function. Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *
     * @return QirPassRunner
     */
    static QirPassRunner &getInstance();

    /**
     * @brief Fills the private vector 'passes_' with the names of the 
     * passes compiled as shared objects (.so) Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QPR.append("libNamePass.so");
     *
     * @param pass Name of the pass compiled as a shared object
     */
    void append(std::string pass);

    /**
     * @brief Invokes each of the passes listed in the private vector
     * 'passes_'. Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QPR.append("libName1Pass.so");
     *     QPR.append("libName2Pass.so");
     *     QPR.append("libName3Pass.so");
     *     QPR.run(*module, MAM);
     * 
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     */
    void run(Module &module, ModuleAnalysisManager &MAM);

    /**
     * @brief Returns the private metadata of 'QirPassRunner', that is, 'qirMetadata_',
     * as a 'QirMetadata' object. Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QirMetadata &qirMetadata = QPR.getMetadata();
     *
     * @return QirMetadata
     */
    QirMetadata &getMetadata();

    /**
     * @brief Saves 'metadata' as the private metadata ('qirMetadata_') of 
     * the 'QirPassRunner' class. Use example:
     *
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QirMetadata &qirMetadata = QPR.getMetadata();
     *     qirMetadata.append(REVERSIBLE_GATE, "__quantum__qis__x__body");
     *     QPR.setMetadata(qirMetadata);
     *
     * @param metadata The medatada of type 'QirMetadata' attached to the 
     * module.
     */
    void setMetadata(const QirMetadata &metadata);

    /**
     * @brief Empties all structures within the metadata. Use example:
     * 
     *     QirPassRunner &QPR = QirPassRunner::getInstance();
     *     QPR.clearMetadata();
     */
    void clearMetadata();

private:
    // Private constructor of the 'QirPassRunner' class
    QirPassRunner();

    // List of passes to be applied
    std::vector<std::string> passes_;

    // Metadata reachable by all passes and the Pass Runner daemon
	QirMetadata qirMetadata_;
};

#endif // QIR_MODULE_PASS_MANAGER_H

