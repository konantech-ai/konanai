/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mission.h"
#include "utils/utils.h"

#include <algorithm>

KString Mission::ms_data_root = "../data/";
KString Mission::ms_cache_root = "../work/cache/";

Mission::Mission(KHSession hSession, enum Ken_test_level testLevel) {
	m_hSession = hSession;

	m_hDataset = 0;
	m_hOptimizer = 0;
	m_hLossExp = 0;
	m_hAccExp = 0;
	m_hVisExp = 0;
	m_hPredExp = 0;
	m_hNetwork = 0;
	m_hModel = 0;

	m_testLevel = testLevel;

	Utils::mkdir(ms_cache_root);
}

Mission::~Mission() {
	KERR_CHK(KAI_Dataset_close(m_hSession, m_hDataset));
	KERR_CHK(KAI_Optimizer_close(m_hSession, m_hOptimizer));
	KERR_CHK(KAI_Expression_close(m_hSession, m_hLossExp));
	KERR_CHK(KAI_Expression_close(m_hSession, m_hAccExp));
	KERR_CHK(KAI_Expression_close(m_hSession, m_hVisExp));
	KERR_CHK(KAI_Expression_close(m_hSession, m_hPredExp));
	KERR_CHK(KAI_Network_close(m_hSession, m_hNetwork));
	KERR_CHK(KAI_Model_close(m_hSession, m_hModel));
}

void Mission::ms_dumpStrList(KString sTitle, KStrList slList) {
	printf("%s: %d elements\n", sTitle.c_str(), (int)slList.size());
	for (int n = 0; n < (int)slList.size(); n++) {
		printf("  [%d] %s\n", n, slList[n].c_str());
	}
}

bool Mission::ms_isMember(KStrList slList, KString sData) {
	return (std::find(slList.begin(), slList.end(), sData) != slList.end());
}
