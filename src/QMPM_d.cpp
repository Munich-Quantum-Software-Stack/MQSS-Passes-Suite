// QIR Pass Manager
#include "QirModulePassManager.hpp"

#include <iostream>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>

const int PORT = 8081;
int qpmSocket = -1;

const std::string QIS_START = "__quantum__qis_";

void handleClient(int clientSocket) {
    // Receive generic QIR
    ssize_t qirMessageSizeNetwork;
    recv(clientSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);

    char* genericQir = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(clientSocket, genericQir, qirMessageSize, 0);
    genericQir[qirBytesRead] = '\0';

	// Parse generic QIR into an LLVM module
    LLVMContext  Context;
    SMDiagnostic error;
    
	auto memoryBuffer = MemoryBuffer::getMemBuffer(genericQir, "QIR (LRZ)", false);
	
    MemoryBufferRef QIRRef = *memoryBuffer;
    std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
    if (!module) {
        std::cout << "Warning: There was an error parsing the generic QIR" << std::endl;
        return;
    }
   
    std::cout << "Generic QIR received from a client" << std::endl;
    
    // Append the desired metadata to the module
    // These metadata will be attached to the IR of this module
    Metadata* metadata = ConstantAsMetadata::get(ConstantInt::get(Context, APInt(1, true)));
    module->addModuleFlag(Module::Warning, "lrz_supports_qir", metadata);
    module->setSourceFileName("");

    Metadata* metadataSupport = module->getModuleFlag("lrz_supports_qir");
    if (metadataSupport)
        if (ConstantAsMetadata* boolMetadata = dyn_cast<ConstantAsMetadata>(metadataSupport))
            if (ConstantInt* boolConstant = dyn_cast<ConstantInt>(boolMetadata->getValue()))
                errs() << "\tModule-level Metadata: \"lrz_supports_qir\" = " << (boolConstant->isOne() ? "true" : "false") << '\n';

    // Receive the list of passes 
	std::vector<std::string> passes;
    while (true) {
        ssize_t passMessageSizeNetwork;
        recv(clientSocket, &passMessageSizeNetwork, sizeof(passMessageSizeNetwork), 0);
        ssize_t passMessageSize = ntohl(passMessageSizeNetwork);

        char* passBuffer = new char[passMessageSize];
        ssize_t passBytesRead = recv(clientSocket, passBuffer, passMessageSize, 0);

        if (passBytesRead > 0) {
            passBuffer[passBytesRead] = '\0';

            if (strcmp(passBuffer, "EOT") == 0) {
                delete[] passBuffer;
                break;
            }

            passes.push_back(passBuffer);
            delete[] passBuffer;
        }
    }

    if (passes.empty()) {
		std::cout << "Warning: A client did not send any pass to the QMPM" << std::endl;
		close(clientSocket);
        std::cout << "Client disconnected";
		return;
	}

    // Append the desired metadata to each gate
    // These metadata will be attached to the module's IR
	Function *function = module->getFunction("__quantum__qis__rxryrx__body");
	
	if (!function) {
        Type        *doubleType   = Type::getDoubleTy(Context);
		StructType  *qubitType    = StructType::getTypeByName(Context, "Qubit");
		PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

		FunctionType *funcType = FunctionType::get(
			Type::getVoidTy(Context), 
			{
                doubleType,
                doubleType,
                doubleType,
				qubitPtrType
			}, 
			false
		);

		function = Function::Create(
			funcType, 
			Function::ExternalLinkage,
			"__quantum__qis__rxryrx__body",
			module.get()
		);

		BasicBlock *entryBlock = BasicBlock::Create(Context, "entry", function);
		IRBuilder<> builder(entryBlock);

		Function *qis_rx_body = module->getFunction("__quantum__qis__rx__body");
		Function *qis_ry_body = module->getFunction("__quantum__qis__ry__body");
		
		if (!qis_rx_body) {
			FunctionType *funcTypeRx = FunctionType::get(
				Type::getVoidTy(Context),
				{
                    doubleType,
                    qubitPtrType
                },
				false
			);

			qis_rx_body = Function::Create(
				funcTypeRx,
				Function::ExternalLinkage,
				"__quantum__qis__rx__body",
				module.get()
			);
		}

		if (!qis_ry_body) {
			FunctionType *funcTypeRy = FunctionType::get(
				Type::getVoidTy(Context),
				{
					doubleType,
					qubitPtrType
				}, 
				false
			);

			qis_ry_body = Function::Create(
				funcTypeRy,
				Function::ExternalLinkage,
				"__quantum__qis__ry__body",
				module.get()
			);
		}

        Value *a = function->getArg(0);
        Value *b = function->getArg(1);
        Value *c = function->getArg(2);

        Value *q = /*builder.CreateLoad(*/function->getArg(3)/*)*/;

        Value *sum_ab         = builder.CreateFAdd(a, b);
        Value *sum_bc         = builder.CreateFAdd(b, c);
        Value *prod_bc        = builder.CreateFMul(b, c);
        Value *sum_ab_prod_bc = builder.CreateFAdd(sum_ab, prod_bc);

		builder.CreateCall(
			qis_rx_body, 
			{sum_ab, q}
		);
		builder.CreateCall(
			qis_ry_body, 
			{sum_bc, q}
		);
		builder.CreateCall(
			qis_rx_body, 
			{sum_ab_prod_bc, q}
		);

		builder.CreateRetVoid();
	}

    // Append the desired metadata to the module's context
    // These metadata will NOT be attached to the module's IR
    QirMetadata &qirMetadata = QirModulePassManager::getInstance().getMetadata();

    for (auto &function : module->getFunctionList()) {
		auto name = static_cast<std::string>(function.getName());
		bool is_quantum = (name.size() >= QIS_START.size() &&
						   name.substr(0, QIS_START.size()) == QIS_START);

        if (is_quantum && !function.hasFnAttribute("irreversible"))
			qirMetadata.append(REVERSIBLE_GATE, static_cast<std::string>(function.getName()));
    }

    /*Function *functionToBeReplaced = module->getFunction("__quantum__qis__U3__body");
    if (functionToBeReplaced)
        functionToBeReplaced->addFnAttr("replaceWith", "__quantum__qis__rxryrx__body");*/
    Function *functionKey   = module->getFunction("__quantum__qis__U3__body");
    Function *functionValue = module->getFunction("__quantum__qis__rxryrx__body");
    auto key   = static_cast<std::string>(functionKey->getName());
    auto value = static_cast<std::string>(functionValue->getName());
    qirMetadata.injectAnnotation(key, value);

    QirModulePassManager::getInstance().setMetadata(qirMetadata);

    // Append all received passes
    QirModulePassManager &QMPM = QirModulePassManager::getInstance();
    ModuleAnalysisManager MAM;
    
    std::reverse(passes.begin(), passes.end());
    while (!passes.empty()) {
        auto pass = passes.back();
        QMPM.append("./src/passes/" + pass);
        passes.pop_back();
    }

	// Run QIR passes
	QMPM.run(*module, MAM);

    // Print the result
    //module->print(outs(), nullptr);
 
    std::string str;
    raw_string_ostream OS(str);
    OS << *module;
    OS.flush();
    const char* qir = str.data();
    send(clientSocket, qir, strlen(qir), 0);
    std::cout << "Adapted QIR sent to client" << std::endl;

    QMPM.clearMetadata();
    delete[] genericQir;
	close(clientSocket);

	std::cout << "Client disconnected";
}

void signalHandler(int signum) {
	close(qpmSocket);
	exit(0);
}

int main(void) {
	signal(SIGTERM, signalHandler);

    qpmSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qpmSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Enable the SO_REUSEADDR option
    int optval = 1;
    setsockopt(qpmSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(qpmSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding" << std::endl;
        close(qpmSocket);
        return 1;
    }

    if (listen(qpmSocket, 5) == -1) {
        std::cerr << "Error listening" << std::endl;
        close(qpmSocket);
        return 1;
    }

    std::cout << "QMPM listening on port " << PORT << std::endl;

    while (true) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(qpmSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

        if (clientSocket == -1) {
            std::cerr << "Error accepting connection" << std::endl;
            continue;
        }

        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        std::cout << std::endl;
        for (int i = 0; i < w.ws_col; i++)
            std::cout << '-';
        std::cout << "\nClient connected" << std::endl;

        std::thread clientThread(handleClient, clientSocket);
        clientThread.detach();
    }

    close(qpmSocket);
	std::cerr << "QMPM stopped" << std::endl;

    return 0;
}

