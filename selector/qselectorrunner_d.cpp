// QIR Selector Runner
//#include "QirSelectorRunner.hpp"

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
#include <dlfcn.h>

const int PORT = 8080;
int qsrSocket = -1;

void handleClient(int clientSocket) {
    // Receive path to selector
    ssize_t qirMessageSizeNetwork;
    recv(clientSocket, &qirMessageSizeNetwork, sizeof(qirMessageSizeNetwork), 0);
    ssize_t qirMessageSize = ntohl(qirMessageSizeNetwork);

    char* receivedSelector = new char[qirMessageSize];
    ssize_t qirBytesRead = recv(clientSocket, receivedSelector, qirMessageSize, 0);
    receivedSelector[qirBytesRead] = '\0';

    // Load the shared object selector
    void* lib_handle = dlopen(receivedSelector, RTLD_LAZY);
 
    if (!lib_handle) {
        std::cerr << "Error loading selector as a shared library: " << dlerror() << std::endl;
        return;
    }

    std::cout << "Selector received from a client: " << receivedSelector << std::endl;

    // Get a function pointer to the selector function in the shared library
    typedef void (*SelectorFunction)();
    SelectorFunction selector = reinterpret_cast<SelectorFunction>(dlsym(lib_handle, "main"));

    if (!selector) {
        std::cerr << "Error finding function in shared library: " << dlerror() << std::endl;
        return;
    }

    // Call the selector function
    selector();

    // Close the library
    dlclose(lib_handle);

    delete[] receivedSelector;
	close(clientSocket);

	std::cout << "Client disconnected";
}

void signalHandler(int signum) {
	close(qsrSocket);
	exit(0);
}

int main(void) {
	signal(SIGTERM, signalHandler);

    qsrSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (qsrSocket == -1) {
        std::cerr << "Error creating socket" << std::endl;
        return 1;
    }

    // Enable the SO_REUSEADDR option
    int optval = 1;
    setsockopt(qsrSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(qsrSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Error binding" << std::endl;
        close(qsrSocket);
        return 1;
    }

    if (listen(qsrSocket, 5) == -1) {
        std::cerr << "Error listening" << std::endl;
        close(qsrSocket);
        return 1;
    }

    std::cout << "Selector Runner listening on port " << PORT << std::endl;

    while (true) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(qsrSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

        if (clientSocket == -1) {
            std::cerr << "Error accepting connection" << std::endl;
            continue;
        }

        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        std::cout << std::endl;
        for (int i = 0; i < w.ws_col; i++)
            std::cout << '-';
        std::cout << "\nClient connected" << std::endl;

        std::thread clientThread(handleClient, clientSocket);
        clientThread.detach();
    }

    close(qsrSocket);
	std::cerr << "Selector Runner stopped" << std::endl;

    return 0;
}

