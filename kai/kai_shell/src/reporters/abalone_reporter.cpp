/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "abalone_reporter.h"

int AbaloneReporter::ms_checkCode = 34661309;

AbaloneReporter::AbaloneReporter() : Reporter() {
	m_version = 1;
}

AbaloneReporter::~AbaloneReporter() {
}

KBool AbaloneReporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	FloatBuffer xs_def(m_hSession, xs["#default"]);
	FloatBuffer ys_def(m_hSession, ys["#default"]);
	FloatBuffer vs_def(m_hSession, vs["#default"]);

	KInt dsize = xs_def.axis(0);
	KInt vsize = xs_def.axis(1);

	for (KInt n = 0; n < dsize; n++) {
		printf("   %lld: [%5.3f", n + 1, xs_def.at(n, 0));
		for (KInt m = 1; m < vsize; m++) printf(" %5.3f", xs_def.at(n, m));
		printf("] => %3.1f(est) vs %3.1f\n", vs_def.at(n, 0), ys_def.at(n, 0));
	}

	printf("\n");

	return false;
}
