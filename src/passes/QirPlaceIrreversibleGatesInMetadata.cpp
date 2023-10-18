#include "../headers/QirPlaceIrreversibleGatesInMetadata.hpp"

using namespace llvm;

std::string const QirPlaceIrreversibleGatesInMetadataPass::QIS_START = "__quantum"
                                                                       "__qis_";

PreservedAnalyses QirPlaceIrreversibleGatesInMetadataPass::run(Module &module, ModuleAnalysisManager &MAM) {
    QirPassRunner &QPR = QirPassRunner::getInstance();
	QirMetadata &qirMetadata = QPR.getMetadata();

    for (auto &function : module) {
        auto name = static_cast<std::string>(function.getName());
        bool is_quantum = (name.size() >= QIS_START.size() &&
                           name.substr(0, QIS_START.size()) == QIS_START);

        if (is_quantum) {
			auto name = static_cast<std::string>(function.getName());
            if (!function.hasFnAttribute("irreversible")) {
                qirMetadata.append(REVERSIBLE_GATE, name);
				errs() << "              Reversible gate found: " << name << '\n';
			}
            else {
                qirMetadata.append(IRREVERSIBLE_GATE, name);
				errs() << "              Irreversible gate found: " << name << '\n';
			}
        }
    }
 
    QPR.setMetadata(qirMetadata);

    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirPass() {
    return new QirPlaceIrreversibleGatesInMetadataPass();
}
