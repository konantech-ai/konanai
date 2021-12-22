/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "binary_reporter.h"

int BinaryReporter::ms_checkCode = 13125133;

BinaryReporter::BinaryReporter() : Reporter() {
	m_version = 1;
}

BinaryReporter::~BinaryReporter() {
}

KBool BinaryReporter::m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	int percent = (int)((batch_index + 1) * 100 / batch_count);
	fprintf(stdout, "     [%4lld/%4lld] %02d%% [", batch_index + 1, batch_count, percent);
	int n = 0;
	for (; n < percent; n += 2) printf("#");
	for (; n < 100; n += 2) printf(" ");
	printf("] loss = %5.3f, TP/FN/FP/TN = (%d,%d,%d,%d)\r", (KFloat)loss["#default"],
		(int)(KFloat)accuracy["TP"], (int)(KFloat)accuracy["FN"], (int)(KFloat)accuracy["FP"], (int)(KFloat)accuracy["TN"]);
	if (batch_count == batch_index + 1) printf("%c[2K", 27);
	return false;
}

KBool BinaryReporter::m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	KString acc_str = m_get_acc_string(accuracy);
	printf("   train epoch [%lld/%lld] ended at %s: loss = %5.3f, %s\n", epoch_index, epoch_count, sTimestamp.c_str(), (KFloat)loss["#default"], acc_str.c_str());
	return false;
}

KBool BinaryReporter::m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy) {
	KString acc_str = m_get_acc_string(accuracy);
	printf("> Model %s test ended at %s: %s\n\n", sName.c_str(), sTimestamp.c_str(), acc_str.c_str());
	return false;
}

KBool BinaryReporter::m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy) {
	KString acc_str = m_get_acc_string(accuracy);
	printf("%s\n", acc_str.c_str());
	return false;
}

KString BinaryReporter::m_get_acc_string(KaiDict accuracy) {
	KFloat TP = accuracy["TP"];
	KFloat TN = accuracy["TN"];
	KFloat FP = accuracy["FP"];
	KFloat FN = accuracy["FN"];

	KFloat acc = TP + TN;
	KFloat recall = TP / (TP + FN);
	KFloat precision = TP / (TP + FP);
	KFloat F1 = 2.0f * TP  / (2 * TP + FN + FP);

	char buffer[256];

	snprintf(buffer, 256, "accuracy=%5.3f, recall=%5.3f, precision=%5.3f, F1=%5.3f", acc, recall, precision, F1);

	return KString(buffer);
}


KBool BinaryReporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	FloatBuffer xs_def(m_hSession, xs["#default"]);
	FloatBuffer ys_def(m_hSession, ys["#default"]);
	FloatBuffer vs_def(m_hSession, vs["#default"]);

	KInt dsize = xs_def.axis(0);
	KInt vsize = xs_def.axis(1);

	for (KInt n = 0; n < dsize; n++) {
		printf("   %lld: [%5.3f", n + 1, xs_def.at(n, 0));
		for (KInt m = 1; m < vsize; m++) printf(" %5.3f", xs_def.at(n, m));
		printf("] => %3.1f%% (answer is %d)\n", vs_def.at(n, 0), (int)ys_def.at(n, 0));
	}

	printf("\n");

	return false;
}
