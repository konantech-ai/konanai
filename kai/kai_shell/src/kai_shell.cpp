/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/

//hs.cho
//#include <boost/algorithm/string/join.hpp>

#include "common.h"

#include "kai_shell.h"
#include "utils/utils.h"

#include "missions/abalone.h"
#include "missions/flower.h"
#include "missions/office31.h"
#include "missions/pulsar.h"
#include "missions/steel.h"
#include "missions/water.h"
#include "missions/urban.h"
#include "missions/mnist_reader.h"
#include "missions/bert.h"
#include "missions/yolo3.h"

void report_shell_error(int errCode) {
	fprintf(stdout, "Kai_cuda error-%d\n", errCode);
	assert(0);
}

#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <iostream>
#include <cstring>
int main(int argc, char** argv) {
	Shell shell;
	shell.run();

	return 0;
}

Shell::Shell() {
	m_prompt = "kai> ";

	m_testLevel = Ken_test_level::core;
	m_next_mission = 0;

	KERR_CHK(KAI_OpenSession(&m_hSession));
	KERR_CHK(KAI_GetVersion(m_hSession, &m_sVersion));
	KERR_CHK(KAI_GetMissionNames(m_hSession, &m_run_mission_names));
	KERR_CHK(KAI_Session_set_callback(m_hSession, Ken_session_cb_event::print, this,reinterpret_cast<void*>( ms_cbExecOutput), NULL));
	KERR_CHK(KAI_OpenCudaOldVersion(m_hSession));

	m_mission_count = (int)m_run_mission_names.size();

	m_exec_mission_names.push_back("abalone");	// based on MnistReaderMission + refactoring
	m_exec_mission_names.push_back("flower");
	m_exec_mission_names.push_back("mnist_reader");
	m_exec_mission_names.push_back("office31");
	m_exec_mission_names.push_back("pulsar");
	m_exec_mission_names.push_back("steel");
	m_exec_mission_names.push_back("urban");
	m_exec_mission_names.push_back("water");
}

Shell::~Shell() {
	KERR_CHK(KAI_CloseSession(m_hSession));
}

void Shell::run() {
	fprintf(stdout, "**********************************************************************\n");
	fprintf(stdout, "                         Welcome to KAI Shell                         \n");
	fprintf(stdout, "**********************************************************************\n");
	fprintf(stdout, "\n");

	while (true) {
		char command[1024];

		fprintf(stdout, "%s", m_prompt.c_str());
		fflush(stdout);
		fgets(command, 1024, stdin);

		if (command[strlen(command) - 1] == '\n') command[strlen(command) - 1] = 0;
		if (!m_exec_command(command)) break;
	}
}

bool Shell::m_exec_command(KString cmd_line) {
	try {
		std::stringstream ss(cmd_line);
		
		KString token;
		KStrList tokens;

		while (ss >> token) tokens.push_back(token);

		if (tokens.size() == 0) return true;

		KString cmd = tokens[0];

		if (cmd == "quit") return false;
		else if (cmd == "list") m_list_missions();
		else if (cmd == "run") m_run_mission(tokens, true);
		else if (cmd == "exec" || cmd == "x") m_run_mission(tokens, false);
		else if (cmd == "help") m_help();
		else if (cmd == "cuda") m_set_cuda(tokens);
		else if (cmd == "image") m_set_image(tokens);
		else if (cmd == "test") m_set_test_level(tokens);
		else if (cmd == "set") m_set_select(tokens);
		else if (!m_status_select(cmd)) m_usage_error();
	}
	catch (KValueException ex) {
		printf("KValueException %d occured\n", ex.m_nErrCode);
	}
	catch (...) {
		printf("Unknown Exception On Command Execution\n");
	};

	return true;
}

void Shell::m_help() {
	fprintf(stdout, "You can use following commands in this shell.\n");
	fprintf(stdout, "    'list' -- will show you all prepared mission list.\n\n");

	fprintf(stdout, "    'run' -- will run the first mission or the next mission of recent executed one.\n");
	fprintf(stdout, "    'run <mission-1>' -- will run a single mission named <mission-1>.\n");
	fprintf(stdout, "    'run <mission-1> <mission-2> ... <mission-n>' -- will run the named missions.\n");
	fprintf(stdout, "    'run from <mission-1> to <mission-2>' -- will run the missions in the named range.\n");
	fprintf(stdout, "    'run all' -- will run all prepared missions.\n");
	fprintf(stdout, "        * <mission-i> can be either the name of mission or the number of mission.\n\n");

	fprintf(stdout, "    'exec:  similar to run, but will execute the mission not in the DLL but by calling the primitive API functions.\n\n");

	fprintf(stdout, "    'test [core | brief | detail | full]: select how deep test functions in executing the mission.\n\n");

	fprintf(stdout, "    'image' -- will show you the option setting for image output.\n");
	fprintf(stdout, "    'image off' -- will set the image output option not to show at all.\n");
	fprintf(stdout, "    'image screen [on]' -- will set to show the image output on the screen.\n");
	fprintf(stdout, "    'image screen off' -- will set not to show the image output on the screen.\n");
	fprintf(stdout, "    'image save <folder-name>' -- will set to save the image output into the named folder.\n");
	fprintf(stdout, "    'image save off' -- will set not to save the image output.\n\n");

	fprintf(stdout, "    'cuda' -- will show you the cuda option setting and available cuda device.\n");
	fprintf(stdout, "    'cuda on' -- will set the cuda option to use cuda.\n");
	fprintf(stdout, "    'cuda off' -- will set the cuda option not to use cuda functions.\n");
	fprintf(stdout, "    'cuda device <device-num>' -- will set the cuda functions to use the named device number.\n");
	fprintf(stdout, "    'cuda blocksize <size>' -- will set the cuda functions to use the named block size.\n\n");

	KStrList pl_type_names;
	KERR_CHK(KAI_GetPluginTypeNames(m_hSession, &pl_type_names));

	int iptype_cnt = (int)pl_type_names.size();

	for (int n = 0; n < iptype_cnt; n++) {
		KStrList pl_nom_names;
		KERR_CHK(KAI_GetPluginNomNames(m_hSession, pl_type_names[n], &pl_nom_names));
		KString noms = Utils::join(pl_nom_names, " | ");
		fprintf(stdout, "    'set %s [%s]\n", pl_type_names[n].c_str(), noms.c_str());
	}
	if (iptype_cnt > 0) fprintf(stdout, "\n");

	fprintf(stdout, "    'quit' -- will terminate this shell program.\n\n");
}

void Shell::m_list_missions() {
	fprintf(stdout, "You can run following missions by exec command in this shell.");

	for (int n = 0; n < (int)m_exec_mission_names.size(); n++) {
		if (n % 4 == 0) fprintf(stdout, "\n   ");
		fprintf(stdout, " %2d. %-24s", n + 1, m_exec_mission_names[n].c_str());
	}

	fprintf(stdout, "\n\n");

	fprintf(stdout, "You can run following missions by run command in this shell.");

	for (int n = 0; n < m_mission_count; n++) {
		if (n % 4 == 0) fprintf(stdout, "\n   ");
		fprintf(stdout, " %2d. %-24s", n + 1, m_run_mission_names[n].c_str());
	}

	fprintf(stdout, "\n\n");
}

void Shell::m_run_mission(KStrList tokens, KBool bInDLL) {
	if (tokens.size() == 1) {
		int nth = m_next_mission++;
		m_next_mission = m_next_mission % m_mission_count;
		m_exec_mission(m_run_mission_names[nth], bInDLL);
	}
	else if (tokens.size() == 2 && tokens[1] == "all") {
		if (bInDLL) {
			for (int nth = 0; nth < m_mission_count; nth++) {
				m_exec_mission(m_run_mission_names[nth], true);
			}
		}
		else {
			KInt time1 = time(NULL);
			for (int n = 0; n < (int)m_exec_mission_names.size(); n++) {
				m_exec_mission(m_exec_mission_names[n], false);
				printf("\n\n\n");
			}
			KInt time2 = time(NULL);
			printf("Total time to execute all missions: %lld secs\n", time2 - time1);
		}
	}
	else if (tokens.size() == 5 && tokens[1] == "from" && tokens[3] == "to") {
		int nFrom = m_get_mission_idx(tokens[2]);
		int nTo = m_get_mission_idx(tokens[4]) + 1;

		for (int nth = nFrom; nth < nTo; nth++) {
			m_exec_mission(m_run_mission_names[nth], bInDLL);
		}
	}
	else {
		for (int n = 1; n < tokens.size(); n++) {
			char* p;
			long converted = strtol(tokens[n].c_str(), &p, 10);
			if (*p) m_exec_mission(tokens[n], bInDLL);
			else m_exec_mission(m_run_mission_names[converted-1], bInDLL);
			//int nth = m_get_mission_idx(tokens[n], bInDLL);
		}
	}
}

void Shell::m_exec_mission(KString sMission, KBool bInDLL) {
	m_bNewLine = true;
	if (bInDLL) {
		KERR_CHK(KAI_ExecMission(m_hSession, sMission, m_sExecMode));
	}
	else {
		Mission* pMission = NULL;
		if (sMission == "abalone") pMission = new AbaloneMission(m_hSession, m_testLevel);	// based on MnistReaderMission + refactoring
		else if (sMission == "flower") pMission = new FlowerMission(m_hSession, m_testLevel);
		else if (sMission == "office31") pMission = new Office31Mission(m_hSession, m_testLevel);
		else if (sMission == "pulsar") pMission = new PulsarMission(m_hSession, m_testLevel);
		else if (sMission == "steel") pMission = new SteelMission(m_hSession, m_testLevel);
		else if (sMission == "water") pMission = new WaterMission(m_hSession, m_testLevel);
		else if (sMission == "urban") pMission = new UrbanSoundMission(m_hSession, m_testLevel);
		else if (sMission == "mnist_reader") pMission = new MnistReaderMission(m_hSession, m_testLevel);
		else if (sMission == "bert") pMission = new BertMission(m_hSession, "ptb_large", m_testLevel);
		else if (sMission == "bert_ptb_large") pMission = new BertMission(m_hSession, "ptb_large", m_testLevel);
		else if (sMission == "bert_ptb_small") pMission = new BertMission(m_hSession, "ptb_small", m_testLevel);
		else if (sMission == "bert_eng_mini") pMission = new BertMission(m_hSession, "eng_mini", m_testLevel);
		else if (sMission == "yolo3") pMission = new Yolo3Mission(m_hSession, "medium", m_testLevel);
		else if (sMission == "yolo3_large") pMission = new Yolo3Mission(m_hSession, "large", m_testLevel);
		else if (sMission == "yolo3_medium") pMission = new Yolo3Mission(m_hSession, "medium", m_testLevel);
		else if (sMission == "yolo3_small") pMission = new Yolo3Mission(m_hSession, "small", m_testLevel);
		else {
			printf("    [%s] unknown mission or not implemented yet\n", sMission.c_str());
			return;
		}
		pMission->Execute();
		delete pMission;
	}
}

int Shell::m_get_mission_idx(KString token) {
	char* p;
	long converted = strtol(token.c_str(), &p, 10);
	if (*p) {
		for (int nth = 0; nth < m_mission_count; nth++) {
			if (m_run_mission_names[nth] == token) return nth;
		}
		return 0;
	}
	else {
		return converted - 1;
	}
}

void Shell::m_set_cuda(KStrList tokens) {
	if (tokens.size() > 1) {
		KRetCode ret = KAI_SetCudaOption(m_hSession, tokens);
		if (ret != KRetOK) {
			m_usage_error();
			return;
		}
	}

	int device_cnt;
	int device_num;
	int block_size;

	KInt avail_mem;
	KInt using_mem;

	KERR_CHK(KAI_GetCudaOption(m_hSession, &device_cnt, &device_num, &block_size, &avail_mem, &using_mem));

	if (device_cnt == 0) {
		fprintf(stdout, "Sorry, no cuda device is found in your environment.\n\n");
	}
	else if (device_num < 0) {
		fprintf(stdout, "Cuda device is available. But You are not using cuda yet.\n\n");
	}
	else {
		fprintf(stdout, "You are using cuda device-%d among %d devices (blocksize:%d, memory %lld/%lld).\n\n", device_num, device_cnt, block_size, using_mem, avail_mem);
	}
}

void Shell::m_set_image(KStrList tokens) {
	if (tokens.size() > 1) {
		KRetCode ret = KAI_SetImageOption(m_hSession, tokens);
		if (ret != KRetOK) {
			m_usage_error();
			return;
		}
	}

	KBool img_screen;
	KString img_folder;

	KERR_CHK(KAI_GetImageOption(m_hSession, &img_screen, &img_folder));

	if (img_screen) {
		if (img_folder != "") fprintf(stdout, "Image output will be displayed on screen and saved on '%s' folder.\n\n", img_folder.c_str());
		else fprintf(stdout, "Image output will be displayed on screen but not will be saved\n\n");
	}
	else {
		if (img_folder != "") fprintf(stdout, "Image output will not be displayed on screen but will be saved on '%s' folder.\n\n", img_folder.c_str());
		else fprintf(stdout, "Image output will not be displayed on screen neither not will be saved\n\n");
	}
}

void Shell::m_set_test_level(KStrList tokens) {
	if (tokens.size() == 2) {
		if (tokens[1] == "core") m_testLevel = Ken_test_level::core;
		else if (tokens[1] == "brief") m_testLevel = Ken_test_level::brief;
		else if (tokens[1] == "detail") m_testLevel = Ken_test_level::detail;
		else if (tokens[1] == "full") m_testLevel = Ken_test_level::full;
		else m_usage_error();
	}
	else m_usage_error();
}

void Shell::m_set_select(KStrList tokens) {
	KRetCode ret = KAI_SetSelectOption(m_hSession, tokens);
	if (ret != KRetOK) {
		m_usage_error();
		return;
	}

	m_status_select(tokens[1]);
}

bool Shell::m_status_select(KString type_name) {
	KString type_desc;

	KRetCode ret = KAI_GetSelectDesc(m_hSession, type_name, &type_desc);
	if (ret != KRetOK) return false;

	fprintf(stdout, "%s: %s\n\n", type_name.c_str(), type_desc.c_str());
	return true;
}

void Shell::m_usage_error() {
	fprintf(stdout, "Sorry, unknown command. Hit help to see the usage.\n\n");
}

void Shell::ms_cbExecOutput(void* pInstance, KString sOutput, KBool bNewLine) {
	Shell* pShell = (Shell*)pInstance;

	if (pShell->m_bNewLine) fprintf(stdout, "    ");
	fprintf(stdout, "%s", sOutput.c_str());
	if (bNewLine) fprintf(stdout, "\n");
	pShell->m_bNewLine = bNewLine;
}
