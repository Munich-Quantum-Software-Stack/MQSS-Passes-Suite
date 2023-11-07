#include "../src/connection_handling.hpp"

#include <iostream>
#include <fstream>

int main() {
    const char *ClientQueue    = "client_queue";
    const char *DaemonQueue    = "daemon_queue";

    // Establish a connection to the RabbitMQ server
    amqp_connection_state_t conn;

    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket);

    // Send the generic QIR to the daemon
    std::cout << "[Client] Sending generic QIR to the daemon" 
              << std::endl;

    // Open the QIR file
    const char* filename = "../benchmarks/test.ll";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Selector] Failed to open file: " << filename << std::endl;
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

    send_message(&conn, genericQir, DaemonQueue);

    // Send the desired scheduler to the dameon
    const char *schedulerName = "libscheduler_round_robin.so";
    std::cout << "[Client] Sending scheduler " 
              << schedulerName 
              << " to the daemon"
              << std::endl;
    send_message(&conn, (char *)schedulerName, DaemonQueue);

    // Send the desired selector to the dameon
    const char *selectorName = "libselector_all.so";
    std::cout << "[Client] Sending selector " 
              << selectorName 
              << " to the daemon"
              << std::endl;
    send_message(&conn, (char *)selectorName, DaemonQueue);

    // Receive the response from the daemon
    const char *adaptedQir = receive_message(&conn, ClientQueue);

    if (adaptedQir) {
        std::cout << "[Client] Received adapted QIR: " 
                  << adaptedQir 
                  << std::endl;
        delete[] adaptedQir;
    } else {
        std::cout << "[Client] Error: Failed to receive the adapted QIR" << std::endl;
    }

    // Close the connections
    close_connections(&conn);

    return 0;
}

