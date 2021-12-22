/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once
//hs.cho
#ifdef KAI2021_WINDOWS //hs.cho
#include "../core/common.h"

#include<winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include<stdio.h>

#pragma comment(lib,"ws2_32.lib") //Winsock Library

#define KAI_PORT "7569"

class KaiServer {
public:
	KaiServer();
	virtual ~KaiServer();

	void openService();
	void closeService();

protected:
	bool m_inService;
	bool m_contService;

	SOCKET m_listenSocket;

	int m_openService();

	void m_acceptLoop();

	static void ms_acceptLoop(void* aux);
};

extern KaiServer kai_server;
#endif