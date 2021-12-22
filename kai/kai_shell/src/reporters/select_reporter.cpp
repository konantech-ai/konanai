/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "select_reporter.h"

int SelectReporter::ms_checkCode = 14249860;

SelectReporter::SelectReporter() : Reporter() {
	m_version = 1;
}

SelectReporter::~SelectReporter() {
}

void SelectReporter::setTargetNames(KaiList targetNames) {
	m_targetNames = targetNames;
}

KBool SelectReporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	FloatBuffer probs(m_hSession, vs["probs"]);
	FloatBuffer select(m_hSession, vs["select"]);
	FloatBuffer answer(m_hSession, vs["answer"]);

	KInt nsize = probs.axis(0);
	KInt psize = probs.axis(1);

	for (KInt n = 0; n < nsize; n++) {
		KInt nSelect = (KInt)select.at(n);
		KInt nAnswer = (KInt)answer.at(n);

		KString sSelect = m_targetNames[nSelect];
		KString sAnswer = m_targetNames[nAnswer];

		printf("   %lld: (%3.1f%%", n + 1, probs.at(n, 0));
		for (KInt m = 1; m < psize; m++) printf(", %3.1f", probs.at(n, m));

		printf(") => %s in %3.1f%% : ", sSelect.c_str(), probs.at(n, nSelect));

		if (nSelect == nAnswer) printf("Correct!\n");
		else printf("Wrong! Answer is %s (%3.1f%% in estimate)\n", sAnswer.c_str(), probs.at(n, nAnswer));
	}

	printf("\n");

	return false;
}
