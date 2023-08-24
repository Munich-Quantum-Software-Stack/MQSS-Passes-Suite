#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

const char* SERVER_IP = "127.0.0.1";
const int PORT = 8081;
const int BUFFER_SIZE = 65536;

int main() {
    // Create socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Connect to server
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, SERVER_IP, &serverAddr.sin_addr);

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error connecting to server" << std::endl;
        close(clientSocket);
        return 1;
    }

    // Send data to server
    //const char* message = "../benchmarks/bell_state.ll";
	const char* filename = "../benchmarks/bell_state.ll";

    // Open the file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file content into a buffer
    char* generic_qir = new char[fileSize];
    file.read(generic_qir, fileSize);
    file.close();

	ssize_t bytesSent = send(clientSocket, generic_qir, fileSize, 0);
    if (bytesSent == -1) {
        std::cerr << "Error: Failed to send generic QIR over the socket" << std::endl;
		delete[] generic_qir;
		exit(1);
	}

	delete[] generic_qir;

    // Receive response from server
    char adapted_qir[BUFFER_SIZE];
    ssize_t bytesRead = recv(clientSocket, adapted_qir, BUFFER_SIZE, 0);
    if (bytesRead > 0) {
        adapted_qir[bytesRead] = '\0';
        std::cout << adapted_qir << std::endl;
    }

    close(clientSocket);
    return 0;
}

