/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class MacroPack;

class Samples {
public:
	Samples();
	const char** get_all_missions(int* pnCount);
	int seek_mission_index(const char* mission);
	const char* get_nth_mission(int nIndex);

	void execute(const char* mission, string mode);

protected:
	static const char* m_ppSamples[];

	void m_create_resnet_macros(MacroPack& macro);
};

