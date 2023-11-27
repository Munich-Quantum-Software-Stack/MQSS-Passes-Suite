#include "../src/connection_handling.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

std::unordered_map<std::string, int> JSONToMap(const char *jsonString)
{
    std::unordered_map<std::string, int> resultMap;

    try
    {
        json jsonObject = json::parse(jsonString);

        if (jsonObject.is_object())
        {
            for (auto &element : jsonObject.items())
            {
                std::string key = element.key();
                int value = element.value().get<int>();
                resultMap[key] = value;
            }
        }
        else
        {
            std::cout
                << "[Client]............Corrupt message received from the QRM."
                << std::endl;
        }
    }
    catch (const json::parse_error &e)
    {
        std::cout << "[Client]............JSON parsing error: " << e.what()
                  << std::endl;
    }

    return resultMap;
}

int main()
{
    setbuf(stdout, NULL);

    const char *ClientQueue = "client_queue";
    const char *DaemonQueue = "daemon_queue";

    // Establish a connection to the RabbitMQ server
    amqp_connection_state_t conn;

    amqp_socket_t *socket = NULL;

    rabbitmq_new_connection(&conn, &socket);

    // Send the generic QIR to the daemon
    std::cout << "[Client]............Sending generic QIR to the daemon"
              << std::endl;

    // Open the QIR file
    const char *filename = "../../benchmarks/test.ll";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[Client]............Failed to open file: " << filename
                  << std::endl;
        return 1;
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file content into a buffer
    char *genericQir = new char[fileSize];
    file.read(genericQir, fileSize);
    file.close();

    send_message(&conn, genericQir, DaemonQueue);

    // Send the desired scheduler to the dameon
    const char *schedulerName = "libscheduler_round_robin.so";
    std::cout << "[Client]............Sending scheduler " << schedulerName
              << " to the daemon" << std::endl;
    send_message(&conn, (char *)schedulerName, DaemonQueue);

    // Send the desired selector to the dameon
    const char *selectorName = "libselector_all.so";
    std::cout << "[Client]............Sending selector " << selectorName
              << " to the daemon" << std::endl;
    send_message(&conn, (char *)selectorName, DaemonQueue);

    // Receive the response from the daemon
    const char *results = receive_message(&conn, ClientQueue);

    if (results)
    {
        std::unordered_map<std::string, int> measurements = JSONToMap(results);

        std::cout << "[Client]............Received results: " << std::endl;
        for (const auto &measurement : measurements)
        {
            std::cout << "[Client]............|" << measurement.first
                      << "\u27E9 : " << measurement.second << std::endl;
        }
    }
    else
    {
        std::cout
            << "[Client]............Error: Failed to receive the measurements"
            << std::endl;
    }

    // Close the connections
    close_connections(&conn);

    return 0;
}
