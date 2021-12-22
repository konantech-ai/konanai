/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#ifdef EXE_VERSION

#include "../core/shell.h"
#include "../core/samples.h"
#include "../core/dataset.h"
#include "../core/log.h"
#include "../int_plugin/internal_plugin.h"

#include <stdio.h>
#include <sstream>
#include <boost/algorithm/string/join.hpp>
#include <iostream>

/*
#include "windows.h"
#define _CRTDBG_MAP_ALLOC //to get more details
#include <stdlib.h>  
#include <crtdbg.h>   //for malloc and free
*/

int main(int argc, char** argv) {
	/*
	_CrtMemState sOld;
	_CrtMemState sNew;
	_CrtMemState sDiff;
	_CrtMemCheckpoint(&sOld); //take a snapchot
	*/

	try {
		Shell shell;
		shell.run();
	}
	catch (KaiException ex) {
		logger.Print("KaiException %d: %s", ex.GetErrorCode(), ex.GetErrorMessage().c_str());
	}
	catch (...) {
		logger.Print("Unknown Exception On Shell Running");
	};

	/*
	_CrtMemCheckpoint(&sNew); //take a snapchot 
	if (_CrtMemDifference(&sDiff, &sOld, &sNew)) // if there is a difference
	{
		OutputDebugString("-----------_CrtMemDumpStatistics ---------");
		_CrtMemDumpStatistics(&sDiff);
		OutputDebugString("-----------_CrtMemDumpAllObjectsSince ---------");
		_CrtMemDumpAllObjectsSince(&sOld);
		OutputDebugString("-----------_CrtDumpMemoryLeaks ---------");
		_CrtDumpMemoryLeaks();
	}
	*/

	return 0;
}

Shell::Shell() : m_samples() {
	logger.open("kai", "shell");
	m_prompt = "kai> ";
	m_next_mission = 0;
	m_mission_names = m_samples.get_all_missions(&m_mission_count);
}

Shell::~Shell() {
}

void Shell::run() {
	CudaConn::OpenCuda();

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

bool Shell::m_exec_command(string cmd_line) {
	try {
		stringstream ss(cmd_line);
		string token;
		vector<string> tokens;

		while (ss >> token) tokens.push_back(token);

		if (tokens.size() == 0) return true;

		string cmd = tokens[0];

		if (cmd == "quit") return false;
		else if (cmd == "help") m_help();
		else if (cmd == "list") m_list_missions();
		else if (cmd == "cuda") m_set_cuda(tokens);
		else if (cmd == "image") m_set_image(tokens);
		else if (cmd == "run") m_run_mission(tokens);
		else if (cmd == "set") m_set_iplugin(tokens);
		else if (!m_status_iplugin(cmd)) m_usage_error();
	}
	catch (KaiException ex) {
		logger.Print("KaiException %d: %s", ex.GetErrorCode(), ex.GetErrorMessage().c_str());
	}
	catch (...) {
		logger.Print("Unknown Exception On Command Execution");
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
	fprintf(stdout, "        * <mission-i> can be either the name of mission or the number of mission.\n");

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

	vector<string> ipl_type_names = int_plugin_man.getTypeNames();
	int iptype_cnt = (int)ipl_type_names.size();

	for (int n = 0; n < iptype_cnt; n++) {
		string iptype_name = ipl_type_names[n];
		vector<string> iptype_noms = int_plugin_man.getTypeNoms(iptype_name);
		string noms = boost::algorithm::join(iptype_noms, " | ");
		fprintf(stdout, "    'set %s [%s]\n", iptype_name.c_str(), noms.c_str());
	}
	if (iptype_cnt > 0) fprintf(stdout, "\n");

	fprintf(stdout, "    'quit' -- will terminate this shell program.\n\n");
}
void Shell::m_list_missions() {
	fprintf(stdout, "You can run following missions in this shell.");

	for (int n = 0; n < m_mission_count; n++) {
		if (n % 4 == 0) fprintf(stdout, "\n   ");
		fprintf(stdout, " %2d. %-24s", n + 1, m_mission_names[n]);
	}

	fprintf(stdout, "\n\n");
}

void Shell::m_usage_error() {
	fprintf(stdout, "Sorry, unknown command. Hit help to see the usage.\n\n");
}

void Shell::m_set_cuda(vector<string> tokens) {
	if (tokens.size() == 1);
	else if (tokens.size() == 2) {
		if (tokens[1] == "on") CudaConn::OpenCuda();
		else if (tokens[1] == "off") CudaConn::CloseCuda();
		else m_usage_error();
	}
	else if (tokens.size() == 3) {
		if (tokens[1] == "device") {
			if (!CudaConn::SetDevice(std::stoi(tokens[2]))) fprintf(stdout, "Sorry, bad device number was given.\n\n");
		}
		else if (tokens[1] == "blocksize") CudaConn::SetBlockSize(std::stoi(tokens[2]));
	}
	else m_usage_error();

	if (!CudaConn::IsCudaAvailable()) {
		fprintf(stdout, "Sorry, no cuda device is found in your environment.\n\n");
	}
	else if (!CudaConn::UsingCuda()) {
		fprintf(stdout, "Cuda device is available. But You are not using cuda yet.\n\n");
	}
	else {
		int device_cnt = CudaConn::GetDeviceCount();
		int device_num = CudaConn::GetCurrDevice();
		int block_size = CudaConn::GetBlockSize();

		fprintf(stdout, "You are using cuda. Current device is %d among %d devices and blocksize is %d.\n\n", device_num, device_cnt, block_size);
	}
}

void Shell::m_set_image(vector<string> tokens) {
	if (tokens.size() == 1);
	else if (tokens.size() == 2) {
		if (tokens[1] == "off") {
			Dataset::set_img_display_mode(false);
			Dataset::set_img_save_folder("");
		}
		else if (tokens[1] == "screen") {
			Dataset::set_img_display_mode(true);
		}
		else m_usage_error();
	}
	else if (tokens.size() == 3) {
		if (tokens[1] == "screen") {
			if (tokens[2] == "on") Dataset::set_img_display_mode(true);
			else if (tokens[2] == "off") Dataset::set_img_display_mode(false);
			else m_usage_error();
		}
		else if (tokens[1] == "save") {
			if (tokens[2] == "off") Dataset::set_img_save_folder("");
			else Dataset::set_img_save_folder(tokens[2]);
		}
		else m_usage_error();
	}
	else m_usage_error();

	bool img_screen = Dataset::get_img_display_mode();
	string img_folder = Dataset::get_img_save_folder();

	if (img_screen) {
		if (img_folder != "") fprintf(stdout, "Image output will be displayed on screen and saved on '%s' folder.\n\n", img_folder.c_str());
		else fprintf(stdout, "Image output will be displayed on screen but not will be saved\n\n");
	}
	else {
		if (img_folder != "") fprintf(stdout, "Image output will not be displayed on screen but will be saved on '%s' folder.\n\n", img_folder.c_str());
		else fprintf(stdout, "Image output will not be displayed on screen neither not will be saved\n\n");
	}
}

bool Shell::m_status_iplugin(string cmd) {
	vector<string> ipl_type_names = int_plugin_man.getTypeNames();

	vector<string>::iterator it = find(ipl_type_names.begin(), ipl_type_names.end(), cmd);

	if (it != ipl_type_names.end()) {
		string intro  = int_plugin_man.introduce(cmd);
		fprintf(stdout, "%s: %s\n\n", cmd.c_str(), intro.c_str());
		return true;
	}

	return false;
}

void Shell::m_exec_mission(int nth) {
	const char* mission = m_mission_names[nth];
	int width = (70 - 16 - (int)strlen(mission)) / 2;
	logger.Print("");
	logger.Print("[%s]", mission);
	m_samples.execute(mission, m_exec_mode);
}

void Shell::m_run_mission(vector<string> tokens) {
	if (tokens.size() == 1) {
		int nth = m_next_mission++;
		m_next_mission = m_next_mission % m_mission_count;
		m_exec_mission(nth);
	}
	else if (tokens.size() == 2 && tokens[1] == "all") {
		for (int nth = 0; nth < m_mission_count; nth++) {
			m_exec_mission(nth);
		}
	}
	else if (tokens.size() == 5 && tokens[1] == "from" && tokens[3] == "to") {
		int nFrom = m_get_mission_idx(tokens[2]);
		int nTo = m_get_mission_idx(tokens[4]) + 1;

		for (int nth = nFrom; nth < nTo; nth++) {
			m_exec_mission(nth);
		}
	}
	else {
		for (int n = 1; n < tokens.size(); n++) {
			int nth = m_get_mission_idx(tokens[n]);
			m_exec_mission(nth);
		}
	}
}

void Shell::m_set_iplugin(vector<string> tokens) {
	string type_name = tokens[1];
	string component_name = tokens[2];

	if (int_plugin_man.set_plugin_component(type_name, component_name)) {
		m_status_iplugin(type_name);
	}
	else {
		fprintf(stdout, "Sorry execeution failed. Please check the details of your commands.\n\n");
	}
}

int Shell::m_get_mission_idx(string token) {
	char* p;
	long converted = strtol(token.c_str(), &p, 10);
	if (*p) {
		for (int nth = 0; nth < m_mission_count; nth++) {
			if (strcmp(m_mission_names[nth], token.c_str()) == 0) return nth;
		}
		return 0;
	}
	else {
		return converted - 1;
	}
}

#endif