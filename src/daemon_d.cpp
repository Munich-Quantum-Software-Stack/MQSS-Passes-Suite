/**
 * @file daemon_d.cpp
 * @brief TODO
 */

#include "pass_runner/PassRunner.hpp"
#include "pass_runner/QirPassRunner.hpp"
#include "scheduler_runner/SchedulerRunner.hpp"
#include "selector_runner/SelectorRunner.hpp"

#include "connection_handling.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <signal.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

/**
 * @var conn
 * @brief TODO
 */
amqp_connection_state_t conn;

/**
 * @brief TODO
 * @param conn TODO
 * @param ClientQueue TODO
 * @param receivedQirModule TODO
 * @param receivedScheduler TODO
 * @param receivedSelector TODO
 */
void handleCircuit(amqp_connection_state_t &conn, char const *ClientQueue,
                   std::unique_ptr<char[]> receivedQirModule,
                   std::unique_ptr<char[]> receivedScheduler,
                   std::unique_ptr<char[]> receivedSelector) {

  // Invoke the scheduler
  std::string scheduler;
  char buffer[PATH_MAX];

  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (len != -1) {
    buffer[len] = '\0';
    scheduler = std::string(buffer);
    size_t lastSlash = scheduler.find_last_of("/\\");
    scheduler =
        scheduler.substr(0, lastSlash) + "/lib/scheduler_runner/schedulers/";
  }
  scheduler.append(receivedScheduler.get());

  std::string targetArchitecture = invokeScheduler(scheduler);
  std::cout << "[daemon_d].........Target architecture: " << targetArchitecture
            << std::endl;

  // Invoke the selector
  std::string selector;
  char selector_buffer[PATH_MAX];

  len =
      readlink("/proc/self/exe", selector_buffer, sizeof(selector_buffer) - 1);
  if (len != -1) {
    selector_buffer[len] = '\0';
    selector = std::string(selector_buffer);
    size_t lastSlash = selector.find_last_of("/\\");
    selector =
        selector.substr(0, lastSlash) + "/lib/selector_runner/selectors/";
  }
  selector.append(receivedSelector.get());

  std::vector<std::string> passes = invokeSelector(selector.c_str());

  // Parse generic QIR into an LLVM module
  LLVMContext Context;
  SMDiagnostic error;

  auto memoryBuffer =
      MemoryBuffer::getMemBuffer(receivedQirModule.get(), "QIR (LRZ)", false);
  MemoryBufferRef QIRRef = *memoryBuffer;
  std::unique_ptr<Module> module = parseIR(QIRRef, error, Context);
  if (!module) {
    std::cout << "[daemon_d].........Warning: There was an error parsing the "
                 "generic QIR"
              << std::endl;
    return;
  }

  // Invoke the passes
  invokePasses(module, passes);

  // Send the adapted QIR back to the client
  std::string str;
  raw_string_ostream OS(str);
  OS << *module;
  OS.flush();
  const char *qir = str.data();

  send_message(&conn,        // conn
               (char *)qir,  // message
               ClientQueue); // queue

  std::cout << "[daemon_d].........Adapted QIR sent to the client" << std::endl;
}

/**
 * @brief Function for the graceful termination of the daemon closing
 * its own socket before exiting
 * @param signum Number of the interrupt signal
 */
void signalHandler(int signum) {
  if (signum == SIGTERM) {
    std::cerr << "[daemon_d].........Stoping" << std::endl;

    // Close the connections
    close_connections(&conn);

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
int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 3) {
    std::cerr << "[daemon_d] Usage: daemon_d [screen|log PATH]" << std::endl;
    return 1;
  }

  std::string stream;

  if (argc == 2) {
    stream = argv[1];
    if (stream != "screen") {
      std::cerr << "[daemon_d] Usage: daemon_d [screen|log PATH]" << std::endl;
      return 1;
    }
  }

  if (argc == 3) {
    stream = argv[1];
    if (stream != "log") {
      std::cerr << "[daemon_d] Usage: daemon_d [screen|log PATH]" << std::endl;
      return 1;
    }
  }

  // Fork the process to create a daemon
  pid_t pid = fork();

  if (pid < 0) {
    std::cerr << "[daemon_d].........Failed to fork" << std::endl;
    return 1;
  }

  std::string filePath;

  if (stream == "log")
    filePath = std::string(argv[2]) + "/logs/daemon_d.log";

  if (pid > 0) {
    std::cout << "[daemon_d].........To stop this daemon type: kill -15 " << pid
              << std::endl;
    if (stream == "log")
      std::cout << "[daemon_d].........The log can be found in " << filePath
                << std::endl;

    return 0;
  }

  // Create a new session and become the session leader
  setsid();

  // Change the working directory to root to avoid locking the current directory
  chdir("/");

  // Set up a signal handler for graceful termination
  signal(SIGTERM, signalHandler);

  // Set the output stream
  if (stream == "log") {
    int logFileDescriptor = -1;

    logFileDescriptor =
        open(filePath.c_str(), O_CREAT | O_RDWR | O_APPEND, S_IRUSR | S_IWUSR);

    if (logFileDescriptor == -1) {
      std::cerr << "[daemon_d].........Warning: Could not open the log file"
                << std::endl;
    } else {
      dup2(logFileDescriptor, STDOUT_FILENO);
      dup2(logFileDescriptor, STDERR_FILENO);
    }
  }

  // Establish a connection to the RabbitMQ server
  const char *ClientQueue = "client_queue";
  const char *DaemonQueue = "daemon_queue";
  amqp_socket_t *socket = NULL;

  rabbitmq_new_connection(&conn, &socket);

  // Declare the client queue
  amqp_queue_declare(conn, 1, amqp_cstring_bytes(ClientQueue), 0, 1, 0, 0,
                     amqp_empty_table);

  // Declare the daemon queue
  amqp_queue_declare(conn, 1, amqp_cstring_bytes(DaemonQueue), 0, 1, 0, 0,
                     amqp_empty_table);

  amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(conn);

  if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
    std::cout << "[daemon_d].........Error starting to consume messages"
              << std::endl;
    return 1;
  }

  std::cout << "[daemon_d].........Listening on queue " << DaemonQueue
            << std::endl;

  while (true) {
    // Receive a QIR module as a binary blob
    auto *qirmodule = receive_message(&conn,        // conn
                                      DaemonQueue); // queue

    auto receivedQirModule = std::make_unique<char[]>(strlen(qirmodule) + 1);
    strcpy(receivedQirModule.get(), qirmodule);

    // Receive name of the desired scheduler
    auto *scheduler = receive_message(&conn,        // conn
                                      DaemonQueue); // queue

    auto receivedScheduler = std::make_unique<char[]>(strlen(scheduler) + 1);
    strcpy(receivedScheduler.get(), scheduler);

    // Receive name of the desired selector
    auto *selector = receive_message(&conn,        // conn
                                     DaemonQueue); // queue

    auto receivedSelector = std::make_unique<char[]>(strlen(selector) + 1);
    strcpy(receivedSelector.get(), selector);

    if (!(receivedQirModule.get() && receivedScheduler.get() &&
          receivedSelector.get())) {

      std::cerr << "[daemon_d].........Failed to receive the job" << std::endl;

      if (receivedQirModule.get())
        delete[] receivedQirModule.get();

      if (receivedScheduler.get())
        delete[] receivedScheduler.get();

      if (receivedSelector.get())
        delete[] receivedSelector.get();

      continue;
    }

    std::cout << "[daemon_d].........Received a QIR module"
              //<< nameOfQir
              << std::endl;

    std::cout << "[daemon_d].........Received a scheduler: "
              << receivedScheduler.get() << std::endl;

    std::cout << "[daemon_d].........Received a selector: "
              << receivedSelector.get() << std::endl;

    // Create a new thread that executes 'handleCircuit' to run
    // the received scheduler, and the received selector targeting
    // the received QIR
    std::thread clientThread(handleCircuit, std::ref(conn), ClientQueue,
                             std::move(receivedQirModule),
                             std::move(receivedScheduler),
                             std::move(receivedSelector));

    delete[] receivedQirModule.get();
    delete[] receivedScheduler.get();
    delete[] receivedSelector.get();

    // Detach from this thread once done
    clientThread.detach();
  }

  return 1;
}
