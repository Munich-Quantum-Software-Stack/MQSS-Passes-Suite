#ifndef QIR_MODULE_PASS_MANAGER_H
#define QIR_MODULE_PASS_MANAGER_H

#include "headers/PassModule.hpp"

#include <dlfcn.h>
#include <unordered_map>
#include <string>

using namespace llvm;

enum MetadataType {
    SUPPORTED_GATE,
    REVERSIBLE_GATE,
    IRREVERSIBLE_GATE,
    AVAILABLE_PLATFORM,
    UNKNOWN
};

struct QirMetadata {
    std::vector<std::string> reversibleGates;
    std::vector<std::string> irreversibleGates;
    std::vector<std::string> supportedGates;
    std::vector<std::string> availablePlatforms;
    std::unordered_map<std::string, std::string> injectedAnnotations;

    bool shouldRemoveCallAttributes;

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

    void injectAnnotation(const std::string &key, const std::string &value) {
        injectedAnnotations[key] = value;
    }

    void setRemoveCallAttributes(const bool value) {
        shouldRemoveCallAttributes = value;
    }
};

class QirPassRunner {
public:
    static QirPassRunner &getInstance();

    std::vector<std::string> getPasses() const {
        return passes_;
    }
    
    void append(std::string pass);
    void run(Module &module, ModuleAnalysisManager &MAM);
    void setMetadata(const QirMetadata &metadata);
    void clearMetadata();

    QirMetadata &getMetadata();

private:
    QirPassRunner();
    std::vector<std::string> passes_;
	QirMetadata qirMetadata_;
};

#endif // QIR_MODULE_PASS_MANAGER_H

