// QIR Pass Runner
#include "QirPassRunner.hpp"

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
int qprSocket = -1;

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
		std::cout << "Warning: A client did not send any pass to the pass runner" << std::endl;
		close(clientSocket);
        std::cout << "Client disconnected";
		return;
	}

    // Attach metadata to the IR
    Metadata* metadata = ConstantAsMetadata::get(ConstantInt::get(Context, APInt(1, true)));
    module->addModuleFlag(Module::Warning, "lrz_supports_qir", metadata);
    module->setSourceFileName("");

    Metadata* metadataSupport = module->getModuleFlag("lrz_supports_qir");
    if (metadataSupport)
        if (ConstantAsMetadata* boolMetadata = dyn_cast<ConstantAsMetadata>(metadataSupport))
            if (ConstantInt* boolConstant = dyn_cast<ConstantInt>(boolMetadata->getValue()))
                errs() << "\tFlag inserted: \"lrz_supports_qir\" = " << (boolConstant->isOne() ? "true" : "false") << '\n';

    // Append all received passes
    QirPassRunner &QPR = QirPassRunner::getInstance();
    ModuleAnalysisManager MAM;
    
    std::reverse(passes.begin(), passes.end());
    while (!passes.empty()) {
        auto pass = passes.back();
        QPR.append("./src/passes/" + pass);
        passes.pop_back();
    }

	// Run QIR passes
	QPR.run(*module, MAM);

    // Print the result
    //module->print(outs(), nullptr);
 
    std::string str;
    raw_string_ostream OS(str);
    OS << *module;
    OS.flush();
    const char* qir = str.data();
    send(clientSocket, qir, strlen(qir), 0);
    std::cout << "Adapted QIR sent to client" << std::endl;

    QPR.clearMetadata();
    delete[] genericQir;
	close(clientSocket);

	std::cout << "Client disconnected";
}

void signalHandler(int signum) {
	close(qprSocket);
	exit(0);
}

int main(void) {
	signal(SIGTERM, signalHandler);

    qprSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qprSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Enable the SO_REUSEADDR option
    int optval = 1;
    setsockopt(qprSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(qprSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding" << std::endl;
        close(qprSocket);
        return 1;
    }

    if (listen(qprSocket, 5) == -1) {
        std::cerr << "Error listening" << std::endl;
        close(qprSocket);
        return 1;
    }

    std::cout << "Pass Runner listening on port " << PORT << std::endl;

    while (true) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(qprSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

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

    close(qprSocket);
	std::cerr << "Pass runner stopped" << std::endl;

    return 0;
}

