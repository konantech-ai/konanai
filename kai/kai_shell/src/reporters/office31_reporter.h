/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../reporter.h"
#include "../utils/utils.h"

class Office31Reporter : public Reporter {
public:
	Office31Reporter();
	virtual ~Office31Reporter();

	void setTargetNames(KaiList targetNames);

protected:
	KBool m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy);
	KBool m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy);
	KBool m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy);
	KBool m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy);

	virtual KBool m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs);

protected:
	int m_version;
	static int ms_checkCode;

protected:
	KaiList m_domainNames;
	KaiList m_productNames;
};
