/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/

#include "../server/client_service.h"
#include "../core/log.h"
#ifdef KAI2021_WINDOWS
ClientService::ClientService(SOCKET clientSocket) {
	m_clientSocket = clientSocket;
}

ClientService::~ClientService() {
}

void ClientService::invokeClientService() {
	new std::thread(ms_startPassiveLoop, this);
}

void ClientService::ms_startPassiveLoop(void* aux) {
	ClientService* pInstance = (ClientService*)aux;
	pInstance->m_passiveLoop();
}

void ClientService::m_passiveLoop() {
    while (true) {
        int iResult;

        while (true) {
            char recvbuf[DEFAULT_BUFLEN];
            int recvbuflen = DEFAULT_BUFLEN;

            iResult = recv(m_clientSocket, recvbuf, recvbuflen, 0);
            if (iResult > 0) {
                logger.Print("Bytes received: %d", iResult);

                // Echo the buffer back to the sender
                int iSendResult = send(m_clientSocket, recvbuf, iResult, 0);
                if (iSendResult == SOCKET_ERROR) {
                    logger.Print("send failed with error: %d", WSAGetLastError());
                    closesocket(m_clientSocket);
                    WSACleanup();
                    break;
                }
                logger.Print("Bytes sent: %d", iSendResult);
            }
            else if (iResult == 0) {
                break;
            }
            else {
                logger.Print("recv failed with error: %d", WSAGetLastError());
                closesocket(m_clientSocket);
                WSACleanup();
                break;
            }

        }

        if (iResult < 0) break;

        // shutdown the connection since we're done
        iResult = shutdown(m_clientSocket, SD_SEND);
        if (iResult == SOCKET_ERROR) {
            logger.Print("shutdown failed with error: %d", WSAGetLastError());
            closesocket(m_clientSocket);
            WSACleanup();
            break;
        }
    }

    logger.Print("Connection closing to a client...");
    // cleanup
    closesocket(m_clientSocket);
    WSACleanup();
}
#endif

