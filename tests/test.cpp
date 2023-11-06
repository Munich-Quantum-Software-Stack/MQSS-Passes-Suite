#include "connection_handling.hpp"

#include <iostream>

int main() {
    const char *ClientQueue    = "client_queue";
    const char *DaemonQueue    = "daemon_queue";
    int         SendChannel    = 1;
    int         ReceiveChannel = 2;

    // Establish a connection to the RabbitMQ server
    amqp_connection_state_t conn /*SendConnection, ReceiveConnection*/;

    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket, SendChannel, ReceiveChannel);

    // Send a message to the daemon
    const char *schedulerName = "libscheduler_round_robin.so";
    send_message(&conn, (char *)schedulerName, DaemonQueue, SendChannel);

    std::cout << "[Client] Sending scheduler: " << schedulerName << std::endl;

    // Receive the response from the daemon
    const char *targetArchitecture = receive_message(&conn, ClientQueue, ReceiveChannel);

    if (targetArchitecture) {
        std::cout << "[Client] Received target architecture: " << targetArchitecture << std::endl;
        delete[] targetArchitecture;
    } else {
        std::cout << "[Client] Error: Failed to receive target architecture" << std::endl;
    }

    // Close the connections
    close_connections(&conn, ReceiveChannel, SendChannel);

    return 0;
}

