/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#ifdef EXE_VERSION
#include "common.h"
#include "samples.h"

class Shell {
public:
	Shell();
	virtual ~Shell();

	virtual void run();

protected:
	virtual bool m_exec_command(string cmd_line);

	virtual void m_help();
	virtual void m_list_missions();
	virtual void m_usage_error();

	virtual void m_set_cuda(vector<string> tokens);
	virtual void m_set_image(vector<string> tokens);
	virtual void m_run_mission(vector<string> tokens);
	virtual void m_set_iplugin(vector<string> tokens);

	bool m_status_iplugin(string cmd);

	virtual void m_exec_mission(int nth);
	
	int m_get_mission_idx(string token);

protected:
	Samples m_samples;
	int m_next_mission;
	int m_mission_count;
	const char** m_mission_names;

	string m_prompt;
	string m_exec_mode;
};
#endif