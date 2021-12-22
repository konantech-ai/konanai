/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once
//hs.cho
#ifdef KAI2021_WINDOWS
#include "connect.h"

#define DEFAULT_BUFLEN 512

class ClientService {
public:
	ClientService(SOCKET clientSocket);
	virtual ~ClientService();

	void invokeClientService();

protected:
	SOCKET m_clientSocket;

	void m_activeLoop();
	void m_passiveLoop();

	static void ms_startActiveLoop(void* aux);
	static void ms_startPassiveLoop(void* aux);
};
#endif