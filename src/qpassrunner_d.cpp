/**
 * @file qpassrunner_d.cpp
 * @brief Implementation of the QIR Pass Runner daemon. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/qpassrunner_d.cpp?ref_type=heads">Source code.</a> 
 */

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
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>

/**
 * @var PORT
 * @brief The port number from which the daemon will listen for
 * incomming connections.
 */
const int         PORT      = 8081;

/**
 * @var qsrSocket
 * @brief Socket for transfering data from and to selectors
 */
int               qprSocket = -1;

/**
 * @var QIS_START
 * @brief Used to define the quantum prefix.
 */
const std::string QIS_START = "__quantum__qis_";

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
 * @brief Function triggered whenever a selector connects to this daemon.
 * Its job is to receive the QIR in binary blob and parse it into
 * an LLVM module. Then, it receives the names of all those passes
 * that should subsequently be applied to the QIR. Finally, it 
 * receives from the selector an End Of Transmission (EOT) message
 * to stop lisenting for more passes. All passes are appended to 
 * a 'QirPassRunner' instance which is also used to run the passes.
 *
 * @param selectorSocket The socket to connect with a selector
 */
void handleSelector(int selectorSocket) {
    // Receive generic QIR from the selector
    ssize_t qirMessageSizeNetwork;
    recv(selectorSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);

    char* genericQir = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(selectorSocket, genericQir, qirMessageSize, 0);
    genericQir[qirBytesRead] = '\0';

	// Parse generic QIR into an LLVM module
    LLVMContext  Context;
    SMDiagnostic error;
    
	auto memoryBuffer = MemoryBuffer::getMemBuffer(genericQir, "QIR (LRZ)", false);
	
    MemoryBufferRef QIRRef = *memoryBuffer;
    std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
    if (!module) {
        std::cout << "[qpassrunner_d] Warning: There was an error parsing the generic QIR" << std::endl;
        return;
    }
   
    std::cout << "[qpassrunner_d] Generic QIR received from a selector" << std::endl;
    
    // Receive the list of passes from the selector
	std::vector<std::string> passes;
    while (true) {
        ssize_t passMessageSizeNetwork;
        recv(selectorSocket, &passMessageSizeNetwork, sizeof(passMessageSizeNetwork), 0);
        ssize_t passMessageSize = ntohl(passMessageSizeNetwork);

        char* passBuffer = new char[passMessageSize];
        ssize_t passBytesRead = recv(selectorSocket, passBuffer, passMessageSize, 0);

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
		std::cout << "[qpassrunner_d] Warning: A selector did not send any pass to the pass runner" << std::endl;
		close(selectorSocket);
        std::cout << "[qpassrunner_d] Selector disconnected" << std::endl;
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
                errs() << "[qpassrunner_d] Flag inserted: \"lrz_supports_qir\" = " << (boolConstant->isOne() ? "true" : "false") << '\n';

    // Create an instance of the QirPassRunner and append to it all the received passes
    QirPassRunner &QPR = QirPassRunner::getInstance();
    ModuleAnalysisManager MAM;
    
    std::reverse(passes.begin(), passes.end());
    while (!passes.empty()) {
        auto pass = passes.back();
        QPR.append("/usr/local/bin/src/passes/" + pass);
        passes.pop_back();
    }

	// Run QIR passes
	QPR.run(*module, MAM);

    // Send the adapted QIR back to the selector
    std::string str;
    raw_string_ostream OS(str);
    OS << *module;
    OS.flush();
    const char* qir = str.data();
    send(selectorSocket, qir, strlen(qir), 0);
    std::cout << "[qpassrunner_d] Adapted QIR sent to selector" << std::endl;

    // Free memory
    QPR.clearMetadata();
    delete[] genericQir;

    // Disconnect from the selector
	close(selectorSocket);
	std::cout << "[qpassrunner_d] Selector disconnected" << std::endl;
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
    if (signum == SIGTERM) {
        std::cerr << "[qpassrunner_d] Stoping" << std::endl;
        close(qprSocket);
        exit(0);
    }
}

/**
 * @brief The main entry point of the program.
 *
 * The QIR Pass Runner daemon.
 *
 * @return int
 */
int main(int argc, char* argv[]) {
    std::string stream = "screen";
    if (argc != 2) {
        std::cerr << "[qpassrunner_d] Usage: qpassrunner_d [screen|log]" << std::endl;
        return 1;
    } else {
        stream = argv[1];
        if (stream != "screen" && stream != "log") {
            std::cerr << "[qpassrunner_d] Usage: qpassrunner_d [screen|log]" << std::endl;
            return 1;
        }
    }

    // Fork the process to create a daemon
    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "[qpassrunner_d] Failed to fork" << std::endl;
        return 1;
    }

    // If we are the parent process, exit
    const char *homeDirectory = getenv("HOME");
    if (!homeDirectory) {
        std::cerr << "[qpassrunner_d] Error getting the home directory" << std::endl;
        return 1;
    }
    
    std::string filePath = std::string(homeDirectory) + "/qpassrunner_d.log";
    
    if (pid > 0) {
        std::cout << "[qpassrunner_d] To stop this daemon type: kill -15 " << pid << std::endl;
        std::cout << "[qpassrunner_d] The log can be found in ~/passrunner_d.log" << std::endl;

        return 0;
    }

    // Create a socket for transfering data from and to selectors
    qprSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qprSocket == -1) {
        std::cerr << "[qpassrunner_d] Error creating own socket" << std::endl;
        return 1;
    }

    // Enable the 'SO_REUSEADDR' option to avoid blocking the IP and port 
    // used by the daemon in case it is not terminated gracefully
    int optval = 1;
    setsockopt(qprSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    // Bind the socket
    sockaddr_in qprAddr;

    qprAddr.sin_family      = AF_INET;
    qprAddr.sin_addr.s_addr = INADDR_ANY;
    qprAddr.sin_port        = htons(PORT);

    // Create a new session and become the session leader
    setsid();

    // Change the working directory to root to avoid locking the current directory
    chdir("/");

    signal(SIGTERM, signalHandler);  // Set up a signal handler for graceful termination

    if (bind(qprSocket, (struct sockaddr*)&qprAddr, sizeof(qprAddr)) != 0) {
        std::cerr << "[qpassrunner_d] Error binding" << std::endl;
        close(qprSocket);
        return 1;
    }

    // Start listening for incomming selectors
    // TODO HOW MANY PENDING CONNECTIONS SHALL BE ALLOW TO QUEUE?
    if (listen(qprSocket, 5) == -1) {
        std::cerr << "[qpassrunner_d] Error listening" << std::endl;
        close(qprSocket);
        return 1;
    }

    std::cout << "[qpassrunner_d] Listening on port " << PORT;

    // Set the output stream
    if (stream == "log") {
        int logFileDescriptor = -1;

        logFileDescriptor = open(filePath.c_str(), O_CREAT | O_RDWR | O_APPEND, S_IRUSR | S_IWUSR);

        if (logFileDescriptor == -1) {
            std::cerr << "[qpassrunner_d] Warning: Could not open the log file" << std::endl;
        }
        else {
            dup2(logFileDescriptor, STDOUT_FILENO);
            dup2(logFileDescriptor, STDERR_FILENO);
        }
    }

    // Enter to an infinite loop
    while (true) {
        // Accept an incoming connection from a selector
        sockaddr_in selectorAddr;
        socklen_t selectorAddrLen = sizeof(selectorAddr);
        int selectorSocket = accept(qprSocket, (struct sockaddr*)&selectorAddr, &selectorAddrLen);

        if (selectorSocket == -1) {
            std::cerr << "[qpassrunner_d] Error accepting connection from a selector" 
                      << std::endl;

            continue;
        }

        std::cout << "\n\n[qpassrunner_d] Selector connected" << std::endl;

        // Create a new thread that executes 'handleSelector' to receive
        // a QIR and a set of passes from the selector to subsequently
        // apply them to the QIR
        std::thread selectorThread(handleSelector, selectorSocket);

        // Detach from this thread once done. The 'selectorSocket' socket
        // was closed by the 'handleSelector' function
        selectorThread.detach();
    }

    return 1;
}

