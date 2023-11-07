/**
 * @file daemon_d.cpp
 * @brief TODO
 */

#include "scheduler_runner/SchedulerRunner.hpp"
#include "selector_runner/SelectorRunner.hpp"
#include "pass_runner/PassRunner.hpp"

#include "connection_handling.hpp"

//#include <cstdlib>
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
#include <chrono>

/**
 * @brief TODO
 * @param conn TODO
 * @param ClientQueue TODO
 * @param receivedQirModule TODO
 * @param receivedScheduler TODO
 * @param receivedSelector TODO
 */
void handleCircuit(amqp_connection_state_t  conn,
                   char const              *ClientQueue,
                   char const              *receivedQirModule,
                   char const              *receivedScheduler,
                   char const              *receivedSelector) {
                   //std::promise<void>      &p) {

    // Invoke the scheduler
    const char *homeDirectory = std::getenv("HOME");
    std::string scheduler = std::string(homeDirectory);
    scheduler.append("/bin/src/schedulers/");
    scheduler.append(receivedScheduler);

    std::string targetArchitecture = invokeScheduler(scheduler);

    // Invoke the selector
    std::vector<std::string> passes = invokeSelector(receivedSelector);

    // Parse generic QIR into an LLVM module
    LLVMContext  Context;
    SMDiagnostic error;
    
    auto memoryBuffer = MemoryBuffer::getMemBuffer(receivedQirModule, "QIR (LRZ)", false);
    MemoryBufferRef QIRRef = *memoryBuffer;
    std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
    if (!module) {
        std::cout << "[daemon_d] Warning: There was an error parsing the generic QIR" << std::endl;
        return;
    }

    // Invoke the passes
    invokePasses(module, passes);

    // Send the adapted QIR back to the client
    std::string str;
    raw_string_ostream OS(str);
    OS << *module;
    OS.flush();
    const char* qir = str.data();

    send_message(&conn,         // conn
                 (char *)qir,   // message
                 ClientQueue);  // queue

    std::cout << "[daemon_d] Adapted QIR sent to the client" 
              << std::endl;

    // This thread has compleated its task
    //p.set_value();
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
    if (signum == SIGTERM) {
        std::cerr << "[daemon_d] Stoping" << std::endl;
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
    if (argc != 2 && argc != 3) {
        std::cerr << "[daemon_d] Usage: daemon_d [screen|log PATH]" << std::endl;
        return 1;
    } else {
        stream = argv[1];
        if (stream != "screen" && stream != "log") {
            std::cerr << "[daemon_d] Usage: daemon_d [screen|log PATH]" << std::endl;
            return 1;
        }
    }

    // Fork the process to create a daemon
    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "[daemon_d] Failed to fork" << std::endl;
        return 1;
    }

    // If we are the parent process, exit
    const char *homeDirectory = getenv("HOME");
    if (!homeDirectory) {
        std::cerr << "[daemon_d] Error getting the home directory" 
                  << std::endl;
        return 1;
    }
    
    std::string filePath = std::string(argv[2]) + "/logs/daemon_d.log";
    
    if (pid > 0) {
        std::cout << "[daemon_d] To stop this daemon type: kill -15 " << pid << std::endl;
        std::cout << "[daemon_d] The log can be found in " << filePath  << std::endl;

        return 0;
    }

    // Create a new session and become the session leader
    setsid();

    // Change the working directory to root to avoid locking the current directory
    chdir("/");

    signal(SIGTERM, signalHandler);  // Set up a signal handler for graceful termination

    // Set the output stream
    if (stream == "log") {
        int logFileDescriptor = -1;

        logFileDescriptor = open(filePath.c_str(), O_CREAT | O_RDWR | O_APPEND, S_IRUSR | S_IWUSR);

        if (logFileDescriptor == -1) {
            std::cerr << "[daemon_d] Warning: Could not open the log file" << std::endl;
        }
        else {
            dup2(logFileDescriptor, STDOUT_FILENO);
            dup2(logFileDescriptor, STDERR_FILENO);
        }
    }

    // Start listening for incomming selectors
    const char *ClientQueue    = "client_queue";
    const char *DaemonQueue    = "daemon_queue";

    // Establish a connection to the RabbitMQ server
    amqp_connection_state_t conn;

    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket);

    // Declare the client queue
    amqp_queue_declare(conn,
                       1,
                       amqp_cstring_bytes(ClientQueue),
                       0,
                       1,
                       0,
                       0,
                       amqp_empty_table);

    // Declare the daemon queue
    amqp_queue_declare(conn,
                       1,
                       amqp_cstring_bytes(DaemonQueue),
                       0,
                       1,
                       0,
                       0,
                       amqp_empty_table);
    
    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cout << "[daemon_d] Error starting to consume messages" << std::endl;
        return 1;
    }

    std::cout << "[daemon_d] Listening on queue " 
              << DaemonQueue 
              << std::endl 
              << std::endl;

    while (true) {
        // Receive a QIR module as a binary blob
        auto *receivedQirModule = receive_message(&conn,              // conn
                                                   DaemonQueue);      // queue

        // Receive name of the desired scheduler
        auto *receivedScheduler = receive_message(&conn,              // conn
                                                   DaemonQueue);      // queue

        // Receive name of the desired selector
        auto *receivedSelector  = receive_message(&conn,              // conn
                                                   DaemonQueue);      // queue

        if (!(receivedQirModule && receivedScheduler && receivedSelector)) {
            std::cerr << "[daemon_d] Failed to receive the job"
                      << std::endl;
            continue;
        }

        std::cout << "[daemon_d] Received a QIR module: \n"
                  << receivedQirModule
                  << std::endl;

        std::cout << "[daemon_d] Received a scheduler: "
                  << receivedScheduler
                  << std::endl;

        std::cout << "[daemon_d] Received a selector: "
                  << receivedSelector
                  << std::endl;

        // Create a new thread that executes 'handleCircuit' to run
        // the received scheduler, and the received selector targeting
        // the received QIR
        //std::promise<void> threadPromise;
        //std::future<void>  threadFuture = threadPromise.get_future();

        std::thread clientThread(handleCircuit,
                                 conn,
                                 ClientQueue,
                                 receivedQirModule,
                                 receivedScheduler,
                                 receivedSelector);
                                 //std::ref(threadPromise));

        clientThread.join();

        delete[] receivedQirModule;
        delete[] receivedScheduler;
        delete[] receivedSelector;

        // Detach from this thread once done
        clientThread.detach();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Close the connections
    close_connections(&conn);

    return 1;
}

