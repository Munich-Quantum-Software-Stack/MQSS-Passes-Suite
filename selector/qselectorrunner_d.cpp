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
 * @brief Socket for transfering data from and to clients
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
 * @brief Function triggered whenever a client connects to this daemon.
 * Its job is to receive the name of a selector and subsequently
 * invoke it. It is the invoked selector itself the one submitting
 * the QIR binary blob and a set of selected passes to the Pass 
 * Runner daemon.
 * @param clientSocket The socket to connect with a client
 */
void handleClient(int clientSocket) {
    // Receive name of the desired selector
    ssize_t qirMessageSizeNetwork;
    recv(clientSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);

    char* receivedSelector = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(clientSocket, receivedSelector, qirMessageSize, 0);
    receivedSelector[qirBytesRead] = '\0';

    // Load the selector as a shared library
    void* lib_handle = dlopen(receivedSelector, RTLD_LAZY);
 
    if (!lib_handle) {
        std::cerr << "[Selector Runner] Error loading selector as a shared library: " 
                  << dlerror() 
                  << std::endl;

        close(clientSocket);
        return;
    }

    std::cout << "[Selector Runner] Selector received from a client: " << receivedSelector << std::endl;

    // Dynamic loading and linking of the shared library
	// TODO DO NOT CALL MAIN BUT ANOTHER FUNCTION
    typedef int (*SelectorFunction)();
    SelectorFunction selector = reinterpret_cast<SelectorFunction>(dlsym(lib_handle, "main"));

    if (!selector) {
        std::cerr << "[Selector Runner] Error finding function in shared library: " 
                  << dlerror() 
                  << std::endl;

        dlclose(lib_handle);
        return;
    }

    // Call the selector function
    int returnedValue = selector();

    // Inform the client wheter the selector finished successfully
    const char *job_status = returnedValue == 0 ? "1" : "0";
    ssize_t    bytesSent   = send(clientSocket, job_status, strlen(job_status), 0);

    if (bytesSent == -1 || bytesSent < strlen(job_status)) {
        std::cerr << "[Selector Runner] Error reporting success of job to the client" 
                  << std::endl;
    }

    // Free memory
    delete[] receivedSelector;
    dlclose(lib_handle);

    // Disconnect from the client
	close(clientSocket);
	std::cout << "[Selector Runner] Client disconnected";
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting.
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
    std::cerr << "[Selector Runner] Stoping" << std::endl;
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

	// Create a socket for transfering data from and to clients
    qsrSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qsrSocket == -1) {
        std::cerr << "[Selector Runner] Error creating own socket" << std::endl;
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
        std::cerr << "[Selector Runner] Error binding" << std::endl;
        close(qsrSocket);
        return 1;
    }

	// Start listening for incomming clients
    // TODO HOW MANY PENDING CONNECTIONS SHALL BE ALLOW TO QUEUE?
    if (listen(qsrSocket, 5) == -1) {
        std::cerr << "[Selector Runner] Error listening" << std::endl;
        close(qsrSocket);
        return 1;
    }

    std::cout << "[Selector Runner] Listening on port " << PORT << std::endl;

	// Enter to an infinite loop
    while (true) {
		// Accept an incoming connection from a client
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(qsrSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

        if (clientSocket == -1) {
            std::cerr << "[Selector Runner] Error accepting connection from a client" 
			          << std::endl;

            continue;
        }

        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        std::cout << std::endl;
        for (int i = 0; i < w.ws_col; i++)
            std::cout << '-';
        std::cout << "\n[Selector Runner] Client connected" << std::endl;

		// Create a new thread that executes 'handleClient' to receive
        // a selector from the client and then run it
        std::thread clientThread(handleClient, clientSocket);

		// Detach from this thread once done. The 'clientSocket' socket
        // was closed by the 'handleClient' function
        clientThread.detach();
    }

    return 1;
}

