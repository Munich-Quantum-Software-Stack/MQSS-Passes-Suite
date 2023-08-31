// QIR Pass Manager
#include "QirModulePassManager.hpp"

#include <iostream>
#include <cstring>
#include <csignal>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>

const int PORT = 8081;

int qpmSocket = -1;

void handleClient(int clientSocket) {
    ssize_t qirMessageSizeNetwork;
    recv(clientSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);

    char* genericQir = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(clientSocket, genericQir, qirMessageSize, 0);
    genericQir[qirBytesRead] = '\0';

    std::cout << "Generic QIR received from a client" << std::endl;

    LLVMContext  Context;
    SMDiagnostic error;
    
	// Parse generic QIR into a module
	auto memoryBuffer = MemoryBuffer::getMemBuffer(genericQir, "QIR (LRZ)", false);
	
    MemoryBufferRef QIRRef = *memoryBuffer;
    std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
    if (!module) {
        std::cout << "Warning: There was an error parsing the generic QIR" << std::endl;
        return;
    }

    module->setSourceFileName("");
 
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
    	std::cout << "Client disconnected" << std::endl;
		return;
	}
	
    // Append all received passes
    QirModulePassManager    QMPM;
    ModuleAnalysisManager   MAM;
    
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

    delete[] genericQir;
	close(clientSocket);

	std::cout << "Client disconnected" << std::endl;
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

        std::cout << "\nClient connected" << std::endl;

        std::thread clientThread(handleClient, clientSocket);
        clientThread.detach();
    }

    close(qpmSocket);
	std::cerr << "QMPM stopped" << std::endl;

    return 0;
}

