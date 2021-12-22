/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "../server/connect.h"
#include "../server/client_service.h"
#include "../core/log.h"

#ifdef KAI2021_WINDOWS //hs.cho
KaiServer kai_server; 

KaiServer::KaiServer() {
    m_inService = false;
    m_contService = false;
}

KaiServer::~KaiServer() {
}

void KaiServer::openService() {
    if (m_inService) {
        logger.Print("Kai server is already in service.");
        return;
    }

    m_inService = true;
    m_contService = true;

    int rc = m_openService();

    if (rc != 0) {
        m_inService = false;
        m_contService = false;
    }

}

void KaiServer::closeService() {
    if (!m_inService) {
        logger.Print("Kai server is not in service.");
        return;
    }

    m_contService = false;
}

int KaiServer::m_openService() {
    WSADATA wsaData;
    int iResult;

    m_listenSocket = INVALID_SOCKET;

    struct addrinfo* result = NULL;
    struct addrinfo hints;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        logger.Print("WSAStartup failed with error: %d", iResult);
        return 1;
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    iResult = getaddrinfo(NULL, KAI_PORT, &hints, &result);
    if (iResult != 0) {
        logger.Print("getaddrinfo failed with error: %d", iResult);
        WSACleanup();
        return 1;
    }

    // Create a SOCKET for connecting to server
    m_listenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (m_listenSocket == INVALID_SOCKET) {
        logger.Print("socket failed with error: %ld", WSAGetLastError());
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Setup the TCP listening socket
    iResult = bind(m_listenSocket, result->ai_addr, (int)result->ai_addrlen);
    if (iResult == SOCKET_ERROR) {
        logger.Print("bind failed with error: %d", WSAGetLastError());
        freeaddrinfo(result);
        closesocket(m_listenSocket);
        WSACleanup();
        return 1;
    }

    freeaddrinfo(result);

    iResult = listen(m_listenSocket, SOMAXCONN);
    if (iResult == SOCKET_ERROR) {
        logger.Print("listen failed with error: %d", WSAGetLastError());
        closesocket(m_listenSocket);
        WSACleanup();
        return 1;
    }

    new std::thread(ms_acceptLoop, this);

    return 0;
}

void KaiServer::ms_acceptLoop(void* aux) {
    KaiServer* pInstance = (KaiServer*)aux;
    pInstance->m_acceptLoop();
}

void KaiServer::m_acceptLoop() {
    while (m_contService) {
        // Accept a client socket
        SOCKET clientSocket = INVALID_SOCKET;
        
        clientSocket = accept(m_listenSocket, NULL, NULL);

        if (clientSocket == INVALID_SOCKET) {
            logger.Print("accept failed with error: %d", WSAGetLastError());
            closesocket(m_listenSocket);
            WSACleanup();
            return;
        }

        ClientService* pService = new ClientService(clientSocket);

        pService->invokeClientService();
    }

    // No longer need server socket
    closesocket(m_listenSocket);
    WSACleanup();

    m_inService = false;
}
#endif