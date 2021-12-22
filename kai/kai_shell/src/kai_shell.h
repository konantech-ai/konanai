/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../../kai_engine/src/include/kai_api.h"
#include "../../kai_engine/src/include/kai_api_shell.h"

class Shell {
public:
	Shell();
	virtual ~Shell();

	virtual void run();

protected:
	KHSession m_hSession;
	KString m_prompt;
	KString m_sVersion;
	KString m_sExecMode; // 아직 특별한 역할 없음

	KStrList m_run_mission_names;
	KStrList m_exec_mission_names;

	bool m_bNewLine;

	int m_next_mission;
	int m_mission_count;

	enum class Ken_test_level m_testLevel;

protected:
	virtual bool m_exec_command(KString cmd_line);

	virtual void m_help();
	virtual void m_list_missions();
	virtual void m_run_mission(KStrList tokens, KBool bInDLL);
	virtual void m_set_cuda(KStrList tokens);
	virtual void m_set_image(KStrList tokens);
	virtual void m_set_test_level(KStrList tokens);
	virtual void m_set_select(KStrList tokens);

	virtual bool m_status_select(KString type_name);

	virtual void m_exec_mission(KString sMission, KBool bInDLL);
	int m_get_mission_idx(KString token);

	virtual void m_usage_error();

	static void ms_cbExecOutput(void* pInstance, KString sOutput, KBool bNewLine);
};
