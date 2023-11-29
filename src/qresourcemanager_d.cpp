/**
 * @file qresourcemanager_d.cpp
 * @brief TODO
 */

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "connection_handling.hpp"
#include "pass_runner/PassRunner.hpp"
#include "pass_runner/QirPassRunner.hpp"
#include "scheduler_runner/SchedulerRunner.hpp"
#include "selector_runner/SelectorRunner.hpp"

using json = nlohmann::json;

/**
 * @todo Comment this
 */
struct QuantumTask
{
    int task_id;
    int n_qbits;
    int n_shots;
    std::string circuit_file;
    std::string circuit_file_type;
    std::string result_destination;
    std::string preferred_qpu;
    std::string scheduled_qpu;
    int priority;
    int optimisation_level;
    bool no_modify;
    bool transpiler_flag;
    int result_type;
    std::string submit_time;
    std::string circuit_qiskit;
    std::string additional_information;
    std::string change_selector;
    std::string change_scheduler;
};

/**
 * @var conn
 * @brief TODO
 */
amqp_connection_state_t conn;

/**
 * @todo Comment this function
 */
QuantumTask JSONToQuantumTask(const char *QuantumTask_str)
{
    QuantumTask task;

    json QuantumTask_json = json::parse(QuantumTask_str);

    task.task_id = QuantumTask_json["task_id"];
    task.n_qbits = QuantumTask_json["n_qbits"];
    task.n_shots = QuantumTask_json["n_shots"];
    task.circuit_file = QuantumTask_json["circuit_file"];
    task.circuit_file_type = QuantumTask_json["circuit_file_type"];
    task.result_destination = QuantumTask_json["result_destination"];
    task.preferred_qpu = QuantumTask_json["preferred_qpu"];
    task.scheduled_qpu = QuantumTask_json["scheduled_qpu"];
    task.priority = QuantumTask_json["priority"];
    task.optimisation_level = QuantumTask_json["optimisation_level"];
    task.no_modify = QuantumTask_json["no_modify"];
    task.transpiler_flag = QuantumTask_json["transpiler_flag"];
    task.result_type = QuantumTask_json["result_type"];
    task.submit_time = QuantumTask_json["submit_time"];
    task.circuit_qiskit = QuantumTask_json["circuit_qiskit"];
    task.additional_information = QuantumTask_json["additional_information"];
    task.change_selector = QuantumTask_json["change_selector"];
    task.change_scheduler = QuantumTask_json["change_scheduler"];

    return task;
}

/**
 * @brief TODO
 * @param conn TODO
 * @param QDQueue TODO
 * @param receivedQirModule TODO
 * @param receivedScheduler TODO
 * @param receivedSelector TODO
 */
void handleQuantumDaemon(amqp_connection_state_t &conn, char const *QDQueue,
                         const QuantumTask &quantumTask)
{
    // Invoke the scheduler
    std::string scheduler;
    if (quantumTask.change_scheduler != "")
    {
        scheduler = quantumTask.change_scheduler;
        /*std::string scheduler;
        char buffer[PATH_MAX];

        ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
        if (len != -1)
        {
            buffer[len] = '\0';
            scheduler = std::string(buffer);
            size_t lastSlash = scheduler.find_last_of("/\\");
            scheduler = scheduler.substr(0, lastSlash) +
                        "/lib/scheduler_runner/schedulers/";
        }
        //scheduler.append(receivedScheduler.get());
        scheduler.append(quantumTask.change_scheduler);

        if (invokeScheduler(scheduler) > 0)*/
    }
    else
    {
        scheduler = "libscheduler_round_robin.so";
    }

    if (invokeScheduler(scheduler) > 0)
    {
        std::cout
            << "   [qresourcemanager_d]..Warning: There was an error obtaining "
               "the target architecture"
            << std::endl;
        return;
    }

    // Invoke the selector
    std::string selector;
    if (quantumTask.change_selector != "")
    {
        selector = quantumTask.change_selector;
        /*std::string selector;
        char selector_buffer[PATH_MAX];

        len = readlink("/proc/self/exe", selector_buffer,
                       sizeof(selector_buffer) - 1);
        if (len != -1)
        {
            selector_buffer[len] = '\0';
            selector = std::string(selector_buffer);
            size_t lastSlash = selector.find_last_of("/\\");
            selector =
                selector.substr(0, lastSlash) +
        "/lib/selector_runner/selectors/";
        }
        selector.append(receivedSelector.get());

        std::vector<std::string> passes = invokeSelector(selector.c_str());*/
    }
    else
    {
        selector = "libselector_all.so";
    }

    std::vector<std::string> passes = invokeSelector(selector);

    // Parse generic QIR into an LLVM module
    LLVMContext Context;
    SMDiagnostic error;

    auto memoryBuffer =
        // MemoryBuffer::getMemBuffer(receivedQirModule.get(), "QIR (LRZ)",
        // false);
        MemoryBuffer::getMemBuffer(quantumTask.circuit_qiskit, "QIR (LRZ)",
                                   false);
    MemoryBufferRef QIRRef = *memoryBuffer;
    std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
    if (!module)
    {
        std::cout << "   [qresourcemanager_d]..Warning: There was an error "
                     "parsing the "
                     "generic QIR"
                  << std::endl;
        return;
    }

    // Invoke the passes
    invokePasses(module, passes);

    // Fetch the target architecture from the metadata
    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();
    auto targetArchitecture = qirMetadata.targetPlatform;

    // Obtain handle of the target architecture
    std::shared_ptr<JobRunner> backend = qdmi_get_backend(targetArchitecture);

    if (!backend)
    {
        std::cout << "   [qresourcemanager_d]..Warning: Unavailable target "
                     "architecture: "
                  << targetArchitecture << std::endl;
        return;
    }

    // Submit the adapted QIR to the target platform
    int n_shots = quantumTask.n_shots;
    auto start = std::chrono::steady_clock::now();
    std::unordered_map<std::string, int> results =
        qdmi_launch_qir(backend, module, n_shots);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    // Get human-readable QIR from the adapted LLVM module
    std::string str;
    raw_string_ostream OS(str);
    OS << *module;
    OS.flush();
    const char *qir = str.data();

    // Create JSON string to send back to the Quantum Daemon
    json QuantumResult_json = {
        {"task_id", -1},
        {"results", results},
        {"destination", ""},
        {"execution_status", true},
        {"executed_qpu", targetArchitecture},
        {"executed_circuit", (char *)qir},
        {"additional_information", ""},
        {"execution_time", elapsed_seconds.count()},
    };

    std::string QuantumResult_str = QuantumResult_json.dump();

    // Send the results back to the Quantum Daemon
    send_message(&conn,                     // conn
                 QuantumResult_str.c_str(), // message
                 QDQueue);                  // queue

    std::cout
        << "   [qresourcemanager_d]..Adapted QIR sent to the Quantum Daemon"
        << std::endl;
}

/**
 * @brief Function for the graceful termination of this daemon closing
 * its own socket before exiting
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum)
{
    if (signum == SIGTERM)
    {
        std::cerr << "   [qresourcemanager_d]..Stoping" << std::endl;

        // Close the connections
        close_connections(&conn);

        exit(0);
    }
}

/**
 * @brief The main entry point of the program.
 *
 * The Quantum Resource Manager daemon.
 *
 * @return int
 */
int main(int argc, char *argv[])
{
    setbuf(stdout, NULL);

    if (argc != 2 && argc != 3)
    {
        std::cerr << "   [qresourcemanager_d]..aemon_d [screen|log PATH]"
                  << std::endl;
        return 1;
    }

    std::string stream;

    if (argc == 2)
    {
        stream = argv[1];
        if (stream != "screen")
        {
            std::cerr << "   [qresourcemanager_d]..aemon_d [screen|log PATH]"
                      << std::endl;
            return 1;
        }
    }

    if (argc == 3)
    {
        stream = argv[1];
        if (stream != "log")
        {
            std::cerr << "   [qresourcemanager_d]..aemon_d [screen|log PATH]"
                      << std::endl;
            return 1;
        }
    }

    // Fork the process to create a daemon
    pid_t pid = fork();

    if (pid < 0)
    {
        std::cerr << "   [qresourcemanager_d]..Failed to fork" << std::endl;
        return 1;
    }

    std::string filePath;

    if (stream == "log")
        filePath = std::string(argv[2]) + "/logs/qresourcemanager_d.log";

    if (pid > 0)
    {
        std::cout
            << "   [qresourcemanager_d]..To stop this daemon type: kill -15 "
            << pid << std::endl;
        if (stream == "log")
            std::cout << "   [qresourcemanager_d]..The log can be found in "
                      << filePath << std::endl;

        return 0;
    }

    // Create a new session and become the session leader
    setsid();

    // Change the working directory to root to avoid locking the current
    // directory
    chdir("/");

    // Set up a signal handler for graceful termination
    signal(SIGTERM, signalHandler);

    // Set the output stream
    if (stream == "log")
    {
        int logFileDescriptor = -1;

        logFileDescriptor = open(filePath.c_str(), O_CREAT | O_RDWR | O_APPEND,
                                 S_IRUSR | S_IWUSR);

        if (logFileDescriptor == -1)
        {
            std::cerr << "   [qresourcemanager_d]..Warning: Could not open the "
                         "log file"
                      << std::endl;
        }
        else
        {
            dup2(logFileDescriptor, STDOUT_FILENO);
            dup2(logFileDescriptor, STDERR_FILENO);
        }
    }

    // Establish a connection to the RabbitMQ server
    const char *QDQueue = "qd_queue";
    const char *QRMQueue = "qrm_queue";
    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket);

    // Declare the Quantum Daemon queue
    amqp_queue_declare(conn, 1, amqp_cstring_bytes(QDQueue), 0, 1, 0, 0,
                       amqp_empty_table);

    // Declare the Quantum Resource Manager queue
    amqp_queue_declare(conn, 1, amqp_cstring_bytes(QRMQueue), 0, 1, 0, 0,
                       amqp_empty_table);

    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cout
            << "   [qresourcemanager_d]..Error starting to consume messages"
            << std::endl;
        return 1;
    }

    std::cout << "   [qresourcemanager_d]..Listening on queue " << QRMQueue
              << std::endl;

    while (true)
    {
        // Receive a QuantumTask
        auto *task = receive_message(&conn,     // conn
                                     QRMQueue); // queue

        if (task)
        {
            QuantumTask quantumTask = JSONToQuantumTask(task);

            //// Receive a QIR module as a binary blob
            // auto *qirmodule = receive_message(&conn,     // conn
            //                                   QRMQueue); // queue

            // auto receivedQirModule =
            //     std::make_unique<char[]>(strlen(qirmodule) + 1);
            // strcpy(receivedQirModule.get(), qirmodule);

            //// Receive name of the desired scheduler
            // auto *scheduler = receive_message(&conn,     // conn
            //                                   QRMQueue); // queue

            // auto receivedScheduler =
            //     std::make_unique<char[]>(strlen(scheduler) + 1);
            // strcpy(receivedScheduler.get(), scheduler);

            //// Receive name of the desired selector
            // auto *selector = receive_message(&conn,     // conn
            //                                  QRMQueue); // queue

            // auto receivedSelector = std::make_unique<char[]>(strlen(selector)
            // + 1); strcpy(receivedSelector.get(), selector);

            std::cout << "   [qresourcemanager_d]..Received a QuantumTask"
                      << std::endl;

            // std::cout << "   [qresourcemanager_d]..Received a QIR module"
            //           << std::endl;

            // std::cout << "   [qresourcemanager_d]..Received a scheduler: "
            //           << receivedScheduler.get() << std::endl;

            // std::cout << "   [qresourcemanager_d]..Received a selector: "
            //           << receivedSelector.get() << std::endl;

            // Create a new thread that executes 'handleQuantumDaemon' to run
            // the received scheduler, and the received selector targeting
            // the received QIR
            std::thread QuantumDaemonThread(handleQuantumDaemon, std::ref(conn),
                                            QDQueue, quantumTask);
            // std::move(receivedQirModule),
            // std::move(receivedScheduler),
            // std::move(receivedSelector));

            // Detach from this thread once done
            QuantumDaemonThread.detach();
        }
        else
        {
            std::cout
                << "   [qresourcemanager_d]..Error: Failed to receive the task"
                << std::endl;
        }
    }

    return 1;
}
