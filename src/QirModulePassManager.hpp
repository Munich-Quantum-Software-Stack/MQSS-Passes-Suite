#ifndef QIR_MODULE_PASS_MANAGER_H
#define QIR_MODULE_PASS_MANAGER_H

#include "headers/PassModule.hpp"

#include <dlfcn.h>

using namespace llvm;

enum MetadataType {
    SUITABLE_PASS,
    REVERSIBLE_GATE,
    INJECTED_ANNOTATION,
    SHOULD_REMOVE_CALL_ATTRIBUTES,
    UNKNOWN
};

struct QirMetadata {
    std::vector<std::string> reversibleGates;
    std::vector<std::string> suitablePasses;
    std::vector<std::string> injectedAnnotations;

    bool shouldRemoveCallAttributes;

	void append(const int key, const std::string &value) {
        switch(key) {
            case SUITABLE_PASS:
                suitablePasses.push_back(value);
                break;
            case REVERSIBLE_GATE:
                reversibleGates.push_back(value);
                break;
            case INJECTED_ANNOTATION:
                injectedAnnotations.push_back(value);
                break;
            default:
                errs() << "Warning: Unknown metadata type: " << key <<  "\n";
        }
    }

    void setRemoveCallAttributes(const bool &value) {
        shouldRemoveCallAttributes = value;
    }
};

class QirModulePassManager {
public:
    static QirModulePassManager &getInstance();

    std::vector<std::string> getPasses() const {
        return passes_;
    }
    
    void append(std::string pass);
    void run(Module &module, ModuleAnalysisManager &MAM);
    void setMetadata(const QirMetadata &metadata);
    void clearMetadata();

    QirMetadata &getMetadata();

private:
    QirModulePassManager();
    std::vector<std::string> passes_;
	QirMetadata qirMetadata_;
};

#endif // QIR_MODULE_PASS_MANAGER_H

