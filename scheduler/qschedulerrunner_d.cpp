/**
 * @file qschedulerrunner_d.cpp
 * @brief Implementation of the QIR Scheduler Runner daemon.
 */

#include <iostream>
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
#include <dlfcn.h>
#include <mutex>

#include "connection_handling.hpp"

// Define a mutex for protecting shared resources
std::mutex sharedMutex;

/**
 * @var SERVER_IP
 * @brief IP to connect to the scheduler
 */
const char* SERVER_IP   = "127.0.0.1";

/**
 * @var PORT
 * @brief The port number from which the scheduler will listen for
 * incomming connections from this daemon.
 */
const int   PORT = 8082;

/**
 * @brief Function triggered whenever a client connects to this daemon.
 * Its job is to receive the name of a scheduler and subsequently
 * invoke it. It is the invoked scheduler itself the one querying the
 * available architectures via QDMI and pass the name of the target 
 * architecture to the Selector daemon.
 * @param conn TODO
 * @param ClientQueue TODO
 * @param DaemonQueue TODO
 * @param ReceiveChannel TODO
 * @param SendChannel TODO
 * @return const char*
 */
const char* handleClient(amqp_connection_state_t  conn,
                  char const              *ClientQueue,
                  int                      SendChannel,
                  const std::string       &receivedScheduler) {

        // Open the QIR file
        const char* filename = "/usr/local/bin/benchmarks/test.ll";
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[qschedulerrunner_d] Failed to open file: " << filename << std::endl;
            return NULL;
        }

        // Get the file size
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the file content into a buffer
        char* genericQir = new char[fileSize];
        file.read(genericQir, fileSize);
        file.close();

        // Parse generic QIR into an LLVM module
        LLVMContext  Context;
        SMDiagnostic error;

        auto memoryBuffer = MemoryBuffer::getMemBuffer(genericQir, "QIR (LRZ)", false);

        delete[] genericQir;

        MemoryBufferRef QIRRef = *memoryBuffer;
        std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
        if (!module) {
            std::cout << "[qschedulerrunner_d] Error: There was an error parsing the generic QIR" << std::endl;
            return NULL;
        }
 
		// Load the scheduler as a shared library
        std::string path = "/usr/local/bin/src/schedulers/" + receivedScheduler;

		void* lib_handle = dlopen(path.c_str(), RTLD_LAZY);
	 
		if (!lib_handle) {
			std::cerr << "[qschedulerrunner_d] Error loading scheduler as a shared library: " 
					  << dlerror() 
					  << std::endl;

			return NULL;
		}

		// Dynamic loading and linking of the shared library
		// TODO DO NOT CALL MAIN BUT ANOTHER FUNCTION
		typedef int (*SchedulerFunction)();
		SchedulerFunction scheduler = reinterpret_cast<SchedulerFunction>(dlsym(lib_handle, "main"));

		if (!scheduler) {
			std::cerr << "[qschedulerrunner_d] Error finding function in shared library: " 
					  << dlerror() 
					  << std::endl;

			dlclose(lib_handle);
			return NULL;
		}

        // Create a socket for transfering data to receivedScheduler
        // listening in port 'PORT'
        int receivedSchedulerSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (receivedSchedulerSocket == -1) {
            std::cerr << "[qschedulerrunner_d] Error creating socket" << std::endl;
            return 1;
        }

        // Connect to receivedScheduler 
        sockaddr_in clientAddr;

        clientAddr.sin_family = AF_INET;
        clientAddr.sin_port   = htons(PORT);

        inet_pton(AF_INET, SERVER_IP, &clientAddr.sin_addr);

        if (connect(receivedSchedulerSocket, (struct sockaddr*)&clientAddr, sizeof(clientAddr)) == -1) {
            std::cerr << "[qschedulerrunner_d] Error connecting to"
                      << receivedScheduler
                      << std::endl;

            close(receivedSchedulerSocket);
            return 1;
        }

        std::string str;
        raw_string_ostream OS(str);
        OS << *module;
        OS.flush();

        const char *qir             = str.data();
        size_t      qirSize         = str.size();
        ssize_t     fileSizeNetwork = htonl(qirSize);
        
        // Send generic QIR module to receivedScheduler 
        std::cout << "[qschedulerrunner_dr] Sending generic QIR module to "
                  << receivedScheduler
                  << std::endl 
                  << std::endl;

        {
            std::lock_guard<std::mutex> lock(sharedMutex);  // Lock the mutex to protect shared resources
        
            ssize_t bytesSent = send(receivedSchedulerSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0);

            if (bytesSent == -1) {
                std::cerr << "[qschedulerrunner_d] Error: Failed to send size of generic QIR module to "
                          << receivedScheduler
                          << std::endl;
                return NULL;;
            }

            bytesSent = send(receivedSchedulerSocket, qir, qirSize, 0);

            if (bytesSent == -1) {
                std::cerr << "[qschedulerrunner_d] Error: Failed to send generic QIR to "
                          << receivedScheduler
                          << std::endl;
                exit(1);
            }
        }

		// Call the scheduler function
		/*int returnedValue =*/ scheduler();

		// Send the target architecture to the client 
		const char *target_architecture = "<<Target Architecture>>";

        send_message(&conn,                         // conn
                      (char *)target_architecture,  // message
                      ClientQueue,                  // queue
                      SendChannel);                 // SendChannel

        return target_architecture;
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting.
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
	if (signum == SIGTERM) {
        std::cerr << "[qschedulerrunner_d] Stoping" << std::endl;
        exit(0);
    }
}

/**
 * @brief The main entry point of the program.
 *
 * The QIR Scheduler Runner daemon.
 *
 * @return int
 */
int main(int argc, char* argv[]) {
    std::string stream = "screen";
    if (argc != 2) {
        std::cerr << "[qschedulerrunner_d] Usage: qschedulerrunner_d [screen|log]" << std::endl;
        return 1;
    } else {
        stream = argv[1];
        if (stream != "screen" && stream != "log") {
            std::cerr << "[qschedulerrunner_d] Usage: qschedulerrunner_d [screen|log]" << std::endl;
            return 1;
        }
    }

    // Fork the process to create a daemon
    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "[qschedulerrunner_d] Failed to fork" << std::endl;
        return 1;
    }

    // If we are the parent process, exit
    const char *homeDirectory = getenv("HOME");
    if (!homeDirectory) {
        std::cerr << "[qschedulerrunner_d] Error getting the home directory" << std::endl;
        return 1;
    }

    std::string filePath = std::string(homeDirectory) + "/qschedulerrunner_d.log";

    if (pid > 0) {
        std::cout << "[qschedulerrunner_d] To stop this daemon type: kill -15 " << pid << std::endl;
        std::cout << "[qschedulerrunner_d] The log can be found in ~/qschedulerrunner_d.log" << std::endl;

        return 0;
    }

    // Create a new session and become the session leader
    setsid();

    // Change the working directory to root to avoid locking the current directory
    chdir("/");

    // Set up a signal handler for graceful termination
    signal(SIGTERM, signalHandler);

	const char *ClientQueue    = "client_queue";
    const char *DaemonQueue    = "daemon_queue";
    int         ReceiveChannel = 1;
    int         SendChannel    = 2;

    // Establish a connection to the RabbitMQ server
    amqp_connection_state_t conn;

    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket, SendChannel, ReceiveChannel);

	// Declare the client queue
    amqp_queue_declare(conn,
                       ReceiveChannel,
                       amqp_cstring_bytes(ClientQueue),
                       0,
                       1,
                       0,
                       0,
                       amqp_empty_table);

    // Declare the daemon queue
    amqp_queue_declare(conn,
                       ReceiveChannel,
                       amqp_cstring_bytes(DaemonQueue),
                       0,
                       1,
                       0,
                       0,
                       amqp_empty_table);

    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cout << "[qschedulerrunner_d] Error starting to consume messages" << std::endl;
        return 1;
    }

    // Set the output stream
    if (stream == "log") {
        int logFileDescriptor = -1;

        logFileDescriptor = open(filePath.c_str(), O_CREAT | O_RDWR | O_APPEND, S_IRUSR | S_IWUSR);

        if (logFileDescriptor == -1) {
            std::cerr << "[qschedulerrunner_d] Warning: Could not open the log file" << std::endl;
        }
        else {
            dup2(logFileDescriptor, STDOUT_FILENO);
            dup2(logFileDescriptor, STDERR_FILENO);
        }
    }

    std::cout << "[qschedulerrunner_d] Listening on queue " << DaemonQueue << std::endl << std::endl;

	// Enter to an infinite loop
    while (true) {
        // Receive name of the desired scheduler
        auto *receivedScheduler = receive_message(&conn,              // conn
                                                   DaemonQueue,       // queue
                                                   ReceiveChannel);   // SendChannel

        if (receivedScheduler) {
            std::cout << "[qschedulerrunner_d] Received a scheduler: " 
                      << receivedScheduler 
                      << std::endl;

            // Create a new thread that executes 'handleClient' to run
            // the received scheduler
            std::thread clientThread(handleClient, 
                                     conn,
                                     ClientQueue,
                                     SendChannel,
                                     std::string(receivedScheduler));

            delete[] receivedScheduler;

            // Detach from this thread once done
            clientThread.detach();
        }
    }

    // Close the connections
    close_connections(&conn, ReceiveChannel, SendChannel);

    return 1;
}

