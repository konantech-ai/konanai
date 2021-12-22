/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class Mission {
public:
	Mission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~Mission();

	virtual void Execute() = 0;

protected:
	static void ms_dumpStrList(KString sTitle, KStrList slList);

	static bool ms_isMember(KStrList slList, KString sData);

protected:
	KHSession m_hSession;

	KHDataset m_hDataset;
	KHNetwork m_hNetwork;
	KHOptimizer m_hOptimizer;
	KHExpression m_hLossExp, m_hAccExp, m_hVisExp, m_hPredExp;
	KHModel m_hModel;

	enum class Ken_test_level m_testLevel;

	static KString ms_data_root;
	static KString ms_cache_root;
};
