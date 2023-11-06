/**
 * @file scheduler_round_robin.cpp 
 * @brief Implementation of a dummy scheduler.
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
#include <qdmi.hpp>

/**
 * @var SERVER_IP
 * @brief IP to connect to the QIR Selector Runner
 */
const char* SERVER_IP   = "127.0.0.1";

/**
 * @var PORT
 * @brief The port number from which the scheduler will listen for
 * incomming connections.
 */
const int   PORT        = 8082;

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
 * The Scheduler.
 *
 * @return int
 */
int main(void) {
    // Accept an incoming connection from qschedulerrunner_d
    sockaddr_in schedulerAddr;
    socklen_t schedulerAddrLen = sizeof(schedulerAddr);
    int schedulerSocket = accept(qsrSocket, (struct sockaddr*)&schedulerAddr, &schedulerAddrLen);

    if (schedulerSocket == -1) {
        std::cerr << "[Scheduler] Warning: Error accepting connection from qschedulerrunner_d"
                  << std::endl;
        return 1;
    }

    std::cout << "[Scheduler] qschedulerrunner_d connected" << std::endl;

    // Receive the generic QIR module
    ssize_t qirMessageSizeNetwork;
    recv(schedulerSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);
    char* receivedQir = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(schedulerSocket, receivedQir, qirMessageSize, 0);
    receivedQir[qirBytesRead] = '\0';    

    // Create socket to connect to qselectorrunner_d
    int selectorSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (selectorSocket == -1) {
        std::cerr << "[Scheduler] Error creating socket to connect to qselectorrunner_d" << std::endl;
        return 1;
    }

    // Connect to qselectorrunner_d 
    sockaddr_in selectorAddr;
    selectorAddr.sin_family = AF_INET;
    selectorAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &selectorAddr.sin_addr);

    if (connect(selectorSocket, (struct sockaddr*)&selectorAddr, sizeof(selectorAddr)) == -1) {
        std::cerr << "[Scheduler] Error connecting to qselectorrunner_d" << std::endl;
        close(selectorSocket);
        return 1;
    }

    // Send the generic QIR module to qselectorrunner_d
    std::cout << "[Scheduler] Sending generic QIR" << std::endl << std::endl;

    ssize_t fileSizeNetwork = htonl(qirMessageSize);
	ssize_t bytesSent       = send(selectorSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0);

    if (bytesSent == -1) {
        std::cerr << "[Scheduler] Error: Failed to send size of generic QIR module to qselectorrunner_d" 
                  << std::endl;
		exit(1);
	}
    bytesSent = send(selectorSocket, receivedQir, fileSize, 0);
    if (bytesSent == -1) {
        std::cerr << "[Scheduler] Error: Failed to send generic QIR to qselectorrunner_d" 
                  << std::endl;
        exit(1);
    }
    delete[] receivedQir;

    // Choose target architecture
    std::vector<std::string> platforms = qdmi_available_platforms();
    const char *targetArchitecture = platforms.back();

    // Send target architecture to qselectorrunner_d
    ssize_t targetArchSizeNetwork = htonl(strlen(targetArchitecture));
    
    std::cout << "[Scheduler] Sending target architecture " << targetArchitecture << std::endl;
    
    if (send(selectorSocket, &targetArchSizeNetwork, sizeof(targetArchSizeNetwork), 0)  < 0) {
        std::cout << "[Scheduler] Error: Failed to send size of target architecture ("
		          << targetArchitecture
                  << ") to qselectorrunner_d"
                  << std::endl;  
        
        close(selectorSocket);
        return 1;
    }
    if (send(selectorSocket, targetArchitecture, strlen(target), 0) < 0) {
        std::cout << "[Scheduler] Errir: Failed to send name of target architecture ("
		          << targetArchitecture
                  << ") to qselectorrunner_d"
                  << std::endl;
        
        close(selectorSocket);
        return 1;
    }

    // Send selector to qselectorrunner_d
    const char *selectorName   = "selectors/libselector_all.so";
    ssize_t    fileSizeNetwork = htonl(strlen(selectorName));

    std::cout << "[Scheduler] Sending selector " << selectorName << " to qselectorrunner_d" << std::endl;

    if (send(selectorSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0) < 0) {
        std::cerr << "[Scheduler] Error: Failed to send size of selector to qselectorrunner_d"
                  << std::endl;

        close(selectorSocket);
        return 1;
    }

    if (send(selectorSocket, selectorName, strlen(selectorName), 0) < 0) {
        std::cerr << "[Scheduler] Error: Failed to send the selector to qselectorrunner_d"
                  << std::endl;

        close(selectorSocket);
        return 1;
    }

    // Receive response from the QPR
    char adapted_qir[BUFFER_SIZE];
    ssize_t bytesRead = recv(selectorSocket, adapted_qir, BUFFER_SIZE, 0);
    if (bytesRead > 0) {
        adapted_qir[bytesRead] = '\0';
        std::cout << "[Scheduler] Received adapted QIR:\n\n" << adapted_qir << std::endl;
    }

    // Close connection with the QPR
    close(selectorSocket);
    return 0;
}

