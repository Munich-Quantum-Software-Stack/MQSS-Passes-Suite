#include "../src/connection_handling.hpp"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

struct QuantumResult
{
    int task_id;
    std::unordered_map<std::string, int> results;
    std::string destination;
    bool execution_status;
    std::string executed_qpu;
    std::string executed_circuit;
    std::string additional_information;
    double execution_time;
};

QuantumResult JSONToQuantumResult(const char *QuantumResult_str)
{
    QuantumResult result;

    json QuantumResult_json = json::parse(QuantumResult_str);

    result.task_id = QuantumResult_json["task_id"];
    result.results = QuantumResult_json["results"];
    result.destination = QuantumResult_json["destination"];
    result.execution_status = QuantumResult_json["execution_status"];
    result.executed_qpu = QuantumResult_json["executed_qpu"];
    result.executed_circuit = QuantumResult_json["executed_circuit"];
    result.additional_information =
        QuantumResult_json["additional_information"];
    result.execution_time = QuantumResult_json["execution_time"];

    return result;
}

int main(int argc, char *argv[])
{
    setbuf(stdout, NULL);

    // Establish a connection to the RabbitMQ server
    const char *QDQueue = "qd_queue";
    const char *QRMQueue = "qrm_queue";

    amqp_connection_state_t conn;
    amqp_socket_t *socket = NULL;
    rabbitmq_new_connection(&conn, &socket);

    // Open the QIR file
    const char *filename = "../../benchmarks/test.ll";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[Quantum Daemon]......Failed to open file with QIR: "
                  << filename << std::endl;
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

    // Create JSON string to send to the Quantum Resource Manager
    std::time_t currentTime = std::time(nullptr);
    const int bufferSize = 80;
    char buffer[bufferSize];
    std::strftime(buffer, bufferSize, "%Y-%m-%d %H:%M:%S",
                  std::localtime(&currentTime));
    std::string submit_time(buffer);
    json QuantumTask_json = {
        {"task_id", -1},
        {"n_qbits", 0},
        {"n_shots", 10000},
        {"circuit_file", ""},
        {"circuit_file_type", "QIR"},
        {"result_destination", ""},
        {"preferred_qpu", "Q20"},
        {"scheduled_qpu", ""},
        {"priority", 0},
        {"optimisation_level", 0},
        {"no_modify", false},
        {"transpiler_flag", true},
        {"result_type", 0},
        {"submit_time", submit_time},
        {"circuit_qiskit", genericQir},
        {"additional_information", ""},
        {"change_selector", "libselector_all.so"},
        {"change_scheduler", "libscheduler_round_robin.so"},
    };

    std::string QuantumTask_str = QuantumTask_json.dump();

    // Send the QuantumTask to the Quantum Resource Manager
    std::cout << "[Quantum Daemon]......Sending QuantumTask to the QRM"
              << std::endl;

    send_message(&conn, QuantumTask_str.c_str(), QRMQueue);

    // Receive the response from the daemon
    const char *results = receive_message(&conn, QDQueue);

    if (results)
    {
        QuantumResult quantumResult = JSONToQuantumResult(results);

        std::cout << "[Quantum Daemon]......Received QuantumResult"
                  << std::endl;
        std::cout << "                      L ...task_id: "
                  << quantumResult.task_id << std::endl;
        std::cout << "                      L ...destination: "
                  << quantumResult.destination << std::endl;
        std::cout << "                      L ...execution_status: "
                  << quantumResult.execution_status << std::endl;
        std::cout << "                      L ...executed_qpu: "
                  << quantumResult.executed_qpu << std::endl;
        std::cout << "                      L ...additional_information: "
                  << quantumResult.additional_information << std::endl;
        std::cout << "                      L ...execution_time: "
                  << quantumResult.execution_time << " s." << std::endl;
        std::cout << "                      L ...executed_circuit: "
                  << std::endl
                  << quantumResult.executed_circuit << std::endl;
    }
    else
    {
        std::cout
            << "[Quantum Daemon]......Error: Failed to receive the results"
            << std::endl;
    }

    // Close the connections
    close_connections(&conn);

    return 0;
}
