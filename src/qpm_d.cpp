// QIR Pass Manager
#include "../src/QirPassManager.h"
#include <iostream>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <string>

using namespace llvm;

const int PORT = 8081;
const int BUFFER_SIZE = 65536;

int serverSocket = -1;

void handleClient(int clientSocket) {
    char genericQir[BUFFER_SIZE];
    ssize_t bytesRead;

    while ((bytesRead = recv(clientSocket, genericQir, BUFFER_SIZE, 0)) > 0) {
        genericQir[bytesRead] = '\0';
        std::cout << "Generic QIR received" << std::endl;

		ModuleAnalysisManager MAM;
		QirPassManager        QPM;
		LLVMContext           Context;
		SMDiagnostic          error;

		// Read generic QIR
        auto memoryBuffer = MemoryBuffer::getMemBuffer(genericQir, "QIR Buffer", false);
        MemoryBufferRef QIRRef = *memoryBuffer;

		std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
		if(!module) {
			std::cerr << "Error parsing Generic QIR" << std::endl;
			exit(1);
		}

		std::vector<std::string> passes {
			"./src/passes/libQirRemoveNonEntrypointFunctionsPass.so",
			"./src/passes/libQirGroupingPass.so",
			"./src/passes/libQirBarrierBeforeFinalMeasurementsPass.so",
			"./src/passes/libQirCXCancellationPass.so",
			"./src/passes/libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so"
		};
		
		// Append passes
		for(std::string pass : passes)
			QPM.append(pass);

		// Run the passes
		QPM.run(module.get(), MAM);

		// Print the result
		//module->print(outs(), nullptr);
		
		std::string str;
		raw_string_ostream OS(str);
		OS << *module;
		OS.flush();
		const char* qir = str.data();
		send(clientSocket, qir, strlen(qir), 0);
		std::cout << "Adapted QIR sent" << std::endl;
	}

	close(clientSocket);
	std::cout << "Client disconnected" << std::endl;
}

void signalHandler(int signum) {
	close(serverSocket);
	exit(0);
}

int main(void) {
	signal(SIGTERM, signalHandler);

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding" << std::endl;
        close(serverSocket);
        return 1;
    }

    if (listen(serverSocket, 5) == -1) {
        std::cerr << "Error listening" << std::endl;
        close(serverSocket);
        return 1;
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    while (true) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

        if (clientSocket == -1) {
            std::cerr << "Error accepting connection" << std::endl;
            continue;
        }

        std::cout << "Client connected" << std::endl;

        std::thread clientThread(handleClient, clientSocket);
        clientThread.detach();
    }

    close(serverSocket);
	std::cerr << "Server stopped" << std::endl;

    return 0;
}

