/**
 * @file selector_all.cpp
 * @brief Implementation of a dummy selector.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

/**
 * @var SERVER_IP
 * @brief IP to connect to the QIR Pass Runner
 */
const char* SERVER_IP   = "127.0.0.1";

/**
 * @var PORT
 * @brief The port number from which the selector will listen for
 * incomming connections.
 */
const int   PORT        = 8081;

/**
 * @var BUFFER_SIZE
 * @brief Size of the buffer that will store the binary blob of a
 * QIR 
 * @todo This variable should not be a fixed value.
 */
const int   BUFFER_SIZE = 65536;

/**
 * @brief The main entry point of the program.
 *
 * The Pass Selector.
 *
 * @return int
 */
int main(void) {
    // Create socket
    int passrunnerSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (passrunnerSocket == -1) {
        std::cerr << "[Selector] Error creating socket" << std::endl;
        return 1;
    }

    // Connect to qpassrunner_d
    sockaddr_in passrunnerAddr;
    passrunnerAddr.sin_family = AF_INET;
    passrunnerAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &passrunnerAddr.sin_addr);

    if (connect(passrunnerSocket, (struct sockaddr*)&passrunnerAddr, sizeof(passrunnerAddr)) == -1) {
        std::cerr << "[Selector] Error connecting to qpassrunner_d" << std::endl;
        close(passrunnerSocket);
        return 1;
    }

    //// Open the QIR file
    //const char* filename = ".qpassrunner_d
    //std::ifstream file(filename, std::ios::binary);
    //if (!file.is_open()) {
    //    std::cerr << "[Selector] Failed to open file: " << filename << std::endl;
    //    return 1;
    //}

    //// Get the file size
    //file.seekg(0, std::ios::end);
    //std::streampos fileSize = file.tellg();
    //file.seekg(0, std::ios::beg);

    //// Read the file content into a buffer
    //char* genericQir = new char[fileSize];
    //file.read(genericQir, fileSize);
    //file.close();

    //// Send generic QIR to qpassrunner_d
    //ssize_t fileSizeNetwork = htonl(fileSize);
    //std::cout << "[Selector] Sending generic QIR" << std::endl << std::endl;
	//ssize_t bytesSent = send(passrunnerSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0);
    //if (bytesSent == -1) {
    //    std::cerr << "[Selector] Error: Failed to send size of generic QIR module to qpassrunner_d" 
    //              << std::endl;
	//	exit(1);
	//}
    //bytesSent = send(passrunnerSocket, genericQir, fileSize, 0);
    //if (bytesSent == -1) {
    //    std::cerr << "[Selector] Error: Failed to send generic QIR to qpassrunner_d" << std::endl;
    //    exit(1);
    //}
    //delete[] genericQir;

    // Append the desired passes
    std::vector<std::string> passes {
        "libQirNormalizeArgAnglePass.so",
        "libQirXCnotXReductionPass.so",
        "libQirCommuteCnotRxPass.so",
        "libQirCommuteRxCnotPass.so",
        "libQirCommuteCnotXPass.so",
        "libQirCommuteXCnotPass.so",
        "libQirCommuteCnotZPass.so",
        "libQirCommuteZCnotPass.so",
        "libQirPlaceIrreversibleGatesInMetadataPass.so",
	    "libQirAnnotateUnsupportedGatesPass.so",
        "libQirU3ToRzRyRzDecompositionPass.so",
        "libQirRzToRxRyRxDecompositionPass.so",
        "libQirCNotToHCZHDecompositionPass.so",
	    "libQirFunctionAnnotatorPass.so",
        "libQirRedundantGatesCancellationPass.so",
        "libQirFunctionReplacementPass.so",
        "libQirReplaceConstantBranchesPass.so",
        "libQirGroupingPass.so", // TODO: Does __quantum__rt__initialize belong to post-quantum?
	    "libQirRemoveNonEntrypointFunctionsPass.so",
        "libQirDeferMeasurementPass.so",
        "libQirBarrierBeforeFinalMeasurementsPass.so",
        "libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so",
        "libQirQubitRemapPass.so",
        "libQirResourceAnnotationPass.so",
	    "libQirNullRotationCancellationPass.so",
	    "libQirMergeRotationsPass.so",
        "libQirDoubleCnotCancellationPass.so",
    };

    // Send each of the passes to qpassrunner_d
    while (!passes.empty()) {
        auto pass = passes.back();
        const char* libPass = pass.c_str();
		ssize_t passSizeNetwork = htonl(strlen(libPass));
		
        std::cout << "[Selector] Sending pass " << pass << std::endl;
		
		if (send(passrunnerSocket, &passSizeNetwork, sizeof(passSizeNetwork), 0)  < 0) {
			std::cout << "[Selector] Warning: Failed to send size of the following pass to qpassrunner_d: \n" 
					  << pass << std::endl;
			continue;
		}
		if (send(passrunnerSocket, libPass, strlen(libPass), 0) < 0) {
			std::cout << "[Selector] Warning: Failed to send the followig pass to qpassrunner_d: \n"
					  << pass << std::endl;
			continue;
		}
        passes.pop_back();
	}
    std::cout << std::endl;

    // Send End Of Transmission (EOF)
    const char* eot = "EOT";
    ssize_t eotSizeNetwork = htonl(strlen(eot));
	if (send(passrunnerSocket, &eotSizeNetwork, sizeof(eotSizeNetwork), 0) < 0) {
		std::cout << "[Selector] Warning: Failed to send size of the EOT to qpassrunner_d" << std::endl;
		close(passrunnerSocket);
		return 1;
	}
    if (send(passrunnerSocket, eot, strlen(eot), 0) < 0) {
        std::cerr << "[Selector] Error: failed to send end of transmission to qpassrunner_d" << std::endl;
		close(passrunnerSocket);
        return 1;
    }

    // Receive response from qpassrunner_d
    char adapted_qir[BUFFER_SIZE];
    ssize_t bytesRead = recv(passrunnerSocket, adapted_qir, BUFFER_SIZE, 0);
    if (bytesRead > 0) {
        adapted_qir[bytesRead] = '\0';
        std::cout << "[Selector] Received adapted QIR:\n\n" << adapted_qir << std::endl;
    }

    // Close connection with qpassrunner_d
    close(passrunnerSocket);
    return 0;
}

