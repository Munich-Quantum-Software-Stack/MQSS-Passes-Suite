/**
 * @file qselectorrunner_d.cpp
 * @brief Implementation of the QIR Selector Runner daemon.
 */

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
#include <dlfcn.h>

/**
 * @var PORT
 * @brief The port number from which the daemon will listen for
 * incomming connections.
 */
const int PORT      = 8080;

/**
 * @var qsrSocket
 * @brief Socket for transfering data from and to schedulers
 */
int       qsrSocket = -1;

//                                                                 ┌───────────────────────────────────────────────────────────────────────────┐
//                                                                 │   struct QirMetadata                                                      │
//                                                                 ├───────────────────────────────────────────────────────────────────────────┤
//                                                                 │ ─ void injectAnnotation(const std: string &key, const std: string &value) │
//                                                                 │ ─ void setRemoveCallAttributes(const bool value)                          │
//                                                                 │ ─ void append(const int key, const std::string &value)                    │
//                                                                 │ ─ std::vector<std:string> reversibleGates                                 │
//                                                                 │ ─ std::vector<std::string> irreversibleGates                              │
//                                                                 │ ─ std::vector<std::string> supportedGates                                 │
//                                                                 │ ─ std::vector<std::string> availablePlatforms                             │
//                                                                 │ ─ std::unordered_map<std::string, std::string> injectedAnnotations        │
//                                                                 │ ─ bool shouldRemoveCallAttributes                                         │
//                                                                 └─────────────────────────────────────┬─────────────────────────────────────┘
//                                                                                                       │
//                                                                                                       │
//                                                                                                       │
//                                                                                                       V
//                                                                 ┌───────────────────────────────────────────────────────────────────────────┐
//                                                                 │   class QirPassRunner                                                     │
//                                                                 ├───────────────────────────────────────────────────────────────────────────┤
//                                                                 │   public                                                                  │
//                                                                 ├───────────────────────────────────────────────────────────────────────────┤
//                                                                 │ ─ static QirPassRunner &getInstance()                                     │
//                                                                 │ ─ void append(std: string pass)                                           │
//                                                                 │ ─ void run(Module &module, ModuleAnalysisManager &MAM)                    │
//                                                                 │ ─ std: vector<std::string> getPasses)                                     │
//                                                                 │ ─ QirMetadata &getMetadata()                                              │
//                                                                 │ ─ void setMetadata(const QirMetadata &metadata)                           │
//                                                                 │ ─ void clearMetadata()                                                    │
//                                                                 ├───────────────────────────────────────────────────────────────────────────┤
//                                                                 │   private                                                                 │
//                                                                 ├───────────────────────────────────────────────────────────────────────────┤
//                                                                 │ ─ QirPassRunner()                                                         │
//                                                                 │ ─ std::vector<std::string> passes_                                        │
//                                                                 │ ─ QirMetadata qirMetadata_                                                │
//                                                                 └─────────────────────────────────────┬─────────────────────────────────────┘
//                                                                                                       │
//                                                                          ┌────────────────────────────┼────────────────//─────┐
//                                                                          │                            │                       │
//                                                                          V                            V                       │
// ┌───────────────────┐             ┌────────────────┐             ┌───────────────┐             ┌─────────────┐                │                                      ┌──────────────┐
// │ qselectorrunner_d ├─ ─invoke─ ─>│ libSelector.so │=====QIR====>│ qpassrunner_d ├─ ─invoke─ ─>│ libPass1.so │................│..........qdmi_supported_gate_set()..>│    Target    │
// │                   │             │                │==NamePass1=>│               │=====QIR====>│             │<===============│==================GateSet=============│ Architecture │
// │                   │             │                │     ...     │               │<====QIR=====│             │                │                                      │              │
// │                   │             │                │==NamePassN=>│               │             └─────────────┘                │                                      │              │
// │                   │             │                │             │               │                    .                       │                                      │              │
// │                   │             │                │             │               │                    .                       │                                      │              │
// │                   │             │                │             │               │                    .                       V                                      │              │
// │                   │             │                │             │               │                                     ┌─────────────┐                               │              │
// │                   │             │                │             │               ├─ ─ ─ ─ ─ ─ ─ ─ -invoke─ ─ ─ ─ ─ ─ ─>│ libPassN.so │...qdmi_supported_gate_set()..>│              │
// │                   │             │                │             │               │==================QIR===============>│             │<==========GateSet=============│              │
// │                   │             │                │<====QIR=====│               │<=================QIR================│             │                               │              │
// └───────────────────┘             └────────────────┘             └───────────────┘                                     └─────────────┘                               └──────────────┘
//          ^                                                               ^
//          │                                                               │
// +++++++++++++++++++++                                            +++++++++++++++++
// +                   +                                            +               +
// +     Selectors     +                                            +     Passes    +
// +                   +                                            +               +
// +++++++++++++++++++++                                            +++++++++++++++++

/**
 * @brief Function triggered whenever a scheduler connects to this daemon.
 * Its job is to receive the name of a selector and subsequently
 * invoke it. It is the invoked selector itself the one submitting
 * the QIR binary blob and a set of selected passes to the Pass 
 * Runner daemon.
 * @param schedulerSocket The socket to connect with a scheduler
 */
void handleScheduler(int schedulerSocket) {
    // Receive the generic QIR module
    ssize_t qirMessageSizeNetwork;
    recv(schedulerSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);
    char* receivedQir = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(schedulerSocket, receivedQir, qirMessageSize, 0);
    receivedQir[qirBytesRead] = '\0';

    std::cout << "[qselectorrunner_d] Generic QIR module (binary blob) received from a scheduler" 
              << std::endl;

    // Receive the target architecture
    recv(schedulerSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    qirMessageSize = ntohl(qirMessageSizeNetwork);
    char* receivedTargetArch = new char[qirMessageSize];
    qirBytesRead = recv(schedulerSocket, receivedTargetArch, qirMessageSize, 0);
    receivedTargetArch[qirBytesRead] = '\0';

    std::cout << "[qselectorrunner_d] Target architecture received from a scheduler: " 
              << receivedTargetArch
              << std::endl;

    // Receive name of the desired selector
    recv(schedulerSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    qirMessageSize = ntohl(qirMessageSizeNetwork);
    char* receivedSelector = new char[qirMessageSize];
    qirBytesRead = recv(schedulerSocket, receivedSelector, qirMessageSize, 0);
    receivedSelector[qirBytesRead] = '\0';

    std::cout << "[qselectorrunner_d] Selector received from a scheduler: " 
              << receivedSelector 
              << std::endl;

    // Load the selector as a shared library
    void* lib_handle = dlopen(receivedSelector, RTLD_LAZY);
 
    if (!lib_handle) {
        std::cerr << "[qselectorrunner_d] Error loading selector as a shared library: " 
                  << dlerror() 
                  << std::endl;

        close(schedulerSocket);
        return;
    }

    // TODO SEND receivedQir, receivedTargetArch, and receivedSelector to 

    

    // Dynamic loading and linking of the shared library
	// TODO DO NOT CALL MAIN BUT ANOTHER FUNCTION
    typedef int (*SelectorFunction)();
    SelectorFunction selector = reinterpret_cast<SelectorFunction>(dlsym(lib_handle, "main"));

    if (!selector) {
        std::cerr << "[qselectorrunner_d] Error finding function in shared library: " 
                  << dlerror() 
                  << std::endl;

        dlclose(lib_handle);
        return;
    }

    // Call the selector function
    int returnedValue = selector();

    // Inform the scheduler wheter the selector finished successfully
    const char *job_status = returnedValue == 0 ? "1" : "0";
    ssize_t    bytesSent   = send(schedulerSocket, job_status, strlen(job_status), 0);

    if (bytesSent == -1 || bytesSent < strlen(job_status)) {
        std::cerr << "[qselectorrunner_d] Error reporting success of job to the scheduler" 
                  << std::endl;
    }

    // Free memory
    delete[] receivedSelector;
    dlclose(lib_handle);

    // Disconnect from the scheduler
	close(schedulerSocket);
	std::cout << "[qselectorrunner_d] Client disconnected";
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting.
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
    std::cerr << "[qselectorrunner_d] Stoping" << std::endl;
	close(qsrSocket);
	exit(0);
}

/**
 * @brief The main entry point of the program.
 *
 * The QIR Selector Runner daemon.
 *
 * @return int
 */
int main(void) {
	// Go to function 'signalHandler' whenever the 'SIGTERM' (graceful
    // termination) signal is sent to this process
	signal(SIGTERM, signalHandler);

	// Create a socket for transfering data from and to schedulers
    qsrSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qsrSocket == -1) {
        std::cerr << "[qselectorrunner_d] Error creating own socket" << std::endl;
        return 1;
    }

	// Enable the 'SO_REUSEADDR' option to avoid blocking the IP and port
    // used by the daemon in case it is not terminated gracefully
    int optval = 1;
    setsockopt(qsrSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

	// Bind the socket
    sockaddr_in serverAddr;

    serverAddr.sin_family      = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port        = htons(PORT);

    if (bind(qsrSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "[qselectorrunner_d] Error binding" << std::endl;
        close(qsrSocket);
        return 1;
    }

	// Start listening for incomming schedulers
    // TODO HOW MANY PENDING CONNECTIONS SHALL BE ALLOW TO QUEUE?
    if (listen(qsrSocket, 5) == -1) {
        std::cerr << "[qselectorrunner_d] Error listening" << std::endl;
        close(qsrSocket);
        return 1;
    }

    std::cout << "[qselectorrunner_d] Listening on port " << PORT << std::endl;

	// Enter to an infinite loop
    while (true) {
		// Accept an incoming connection from a scheduler
        sockaddr_in schedulerAddr;
        socklen_t schedulerAddrLen = sizeof(schedulerAddr);
        int schedulerSocket = accept(qsrSocket, (struct sockaddr*)&schedulerAddr, &schedulerAddrLen);

        if (schedulerSocket == -1) {
            std::cerr << "[qselectorrunner_d] Error accepting connection from a scheduler" 
			          << std::endl;

            continue;
        }

        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        std::cout << std::endl;
        for (int i = 0; i < w.ws_col; i++)
            std::cout << '-';
        std::cout << "[qselectorrunner_d] Client connected" << std::endl;

		// Create a new thread that executes 'handleScheduler' to receive
        // a selector from the scheduler and then run it
        std::thread schedulerThread(handleScheduler, schedulerSocket);

		// Detach from this thread once done. The 'schedulerSocket' socket
        // was closed by the 'handleScheduler' function
        schedulerThread.detach();
    }

    return 1;
}

