#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

const char* SERVER_IP   = "127.0.0.1";
const int   PORT        = 8081;
const int   BUFFER_SIZE = 65536; // TODO

int main(void) {
    // Create socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Connect to the QPM
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &serverAddr.sin_addr);

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error connecting to the QPM" << std::endl;
        close(clientSocket);
        return 1;
    }

    // Open the QIR file
    const char* filename = "../benchmarks/bell_state.ll";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file content into a buffer
    char* genericQir = new char[fileSize];
    file.read(genericQir, fileSize);
    file.close();

    // Send generic QIR to the QPM
    ssize_t fileSizeNetwork = htonl(fileSize);
    std::cout << "Sending generic QIR" << std::endl << std::endl;
	ssize_t bytesSent = send(clientSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0);
    if (bytesSent == -1) {
        std::cerr << "Error: Failed to send size of generic QIR to the QPM" << std::endl;
		exit(1);
	}
    bytesSent = send(clientSocket, genericQir, fileSize, 0);
    if (bytesSent == -1) {
        std::cerr << "Error: Failed to send generic QIR to the QPM" << std::endl;
        exit(1);
    }
    delete[] genericQir;

    // Append the desired passes
    std::vector<std::string> passes {
        "libQirRedundantGatesCancellationPass.so",
        "libQirFunctionReplacementPass.so",
        "libQirReplaceConstantBranchesPass.so",
        "libQirGroupingPass.so", // TODO: Does __quantum__rt__initialize belong to post-quantum?
		"libQirRemoveNonEntrypointFunctionsPass.so",
        "libQirBarrierBeforeFinalMeasurementsPass.so",
        "libQirDeferMeasurementPass.so",
        "libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so",
        "libQirQubitRemapPass.so",
    };

    // Send each of the passes to the QPM
    while (!passes.empty()) {
        auto pass = passes.back();
        const char* libPass = pass.c_str();
		ssize_t passSizeNetwork = htonl(strlen(libPass));
		
        std::cout << "Sending pass " << pass << std::endl;
		
		if (send(clientSocket, &passSizeNetwork, sizeof(passSizeNetwork), 0)  < 0) {
			std::cout << "Warning: Failed to send size of the following pass to the QPM: \n" 
					  << pass << std::endl;
			continue;
		}
		if (send(clientSocket, libPass, strlen(libPass), 0) < 0) {
			std::cout << "Warning: Failed to send the followig pass to the QPM: \n"
					  << pass << std::endl;
			continue;
		}
        passes.pop_back();
	}
    std::cout << std::endl;

    // Send End Of Transmission (EOF)
    const char* eot = "EOT";
    ssize_t eotSizeNetwork = htonl(strlen(eot));
	if (send(clientSocket, &eotSizeNetwork, sizeof(eotSizeNetwork), 0) < 0) {
		std::cout << "Warning: Failed to send size of the EOT to the QPM" << std::endl;
		close(clientSocket);
		return 1;
	}
    if (send(clientSocket, eot, strlen(eot), 0) < 0) {
        std::cerr << "Error: failed to send end of transmission to the QPM" << std::endl;
		close(clientSocket);
        return 1;
    }

    // Receive response from QPM
    char adapted_qir[BUFFER_SIZE];
    ssize_t bytesRead = recv(clientSocket, adapted_qir, BUFFER_SIZE, 0);
    if (bytesRead > 0) {
        adapted_qir[bytesRead] = '\0';
        std::cout << "Received adapted QIR:\n\n" << adapted_qir << std::endl;
    }

    // Close connection with the QPM
    close(clientSocket);
    return 0;
}

