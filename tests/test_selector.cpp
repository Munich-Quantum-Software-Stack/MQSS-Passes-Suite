// Client

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

const char* SERVER_IP   = "127.0.0.1";
const int   PORT        = 8080;
const int   BUFFER_SIZE = 65536; // TODO

/* Main function
 */
int main(void) {
	// Create a socket for transfering data to a the Selector Runner
    // daemon listening in port 'PORT'
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "[Client] Error creating socket" << std::endl;
        return 1;
    }

    // Connect to the Selector Runner daemon
    sockaddr_in clientAddr;

    clientAddr.sin_family = AF_INET;
    clientAddr.sin_port   = htons(PORT);

    inet_pton(AF_INET, SERVER_IP, &clientAddr.sin_addr);

    if (connect(clientSocket, (struct sockaddr*)&clientAddr, sizeof(clientAddr)) == -1) {
        std::cerr << "[Client] Error connecting to the selector" 
		          << std::endl;

        close(clientSocket);
        return 1;
    }

    // Send name of the selector to the Selector Runner  daemon
    const char *selectorName   = "selectors/libselector_all.so";
    ssize_t    fileSizeNetwork = htonl(strlen(selectorName));

    std::cout << "[Client] Sending selector: " << selectorName << std::endl;

    if (send(clientSocket, &fileSizeNetwork, sizeof(fileSizeNetwork), 0) < 0) {
        std::cerr << "[Client] Error: Failed to send size of selector to the Selector Runner" 
		          << std::endl;

		close(clientSocket);
		return 1;
	}

    if (send(clientSocket, selectorName, strlen(selectorName), 0) < 0) {
        std::cerr << "[Client] Error: Failed to send the selector to the Selector Runner" 
		          << std::endl;

		close(clientSocket);
        return 1;
    }

    // Receive response from the selector
    char    success[2];
    ssize_t bytesRead = recv(clientSocket, success, 2, 0);

    if (bytesRead > 0) {
        success[bytesRead] = '\0';
       
        if (strcmp(success, "1") == 0) {
            std::cout << "[Client] Selector Runner reported finishing successfully" 
			          << std::endl;

            // Close connection with the Selector Runner daemon
            close(clientSocket);
            return 0;
        }
    }

	// Report the unsuccessful task
    std::cout << "[Client] Error: The chosen selector did not finish successfully" 
	          << std::endl;

    // Close connection with the QSR
    close(clientSocket);

    return 1;
}

