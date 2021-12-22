/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "mnist_reader_reporter.h"

int MnistReaderReporter::ms_checkCode = 14249860;

MnistReaderReporter::MnistReaderReporter() : Reporter() {
	m_version = 1;
}

MnistReaderReporter::~MnistReaderReporter() {
}

KBool MnistReaderReporter::m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	printf("   train epoch [%lld/%lld] ended at %s: loss = %5.3f, acc = %5.3f+%5.3f\n", epoch_index, epoch_count, sTimestamp.c_str(),
		(KFloat)loss["#default"], (KFloat)accuracy["char"], (KFloat)accuracy["word"]);
	return false;
}

KBool MnistReaderReporter::m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	int percent = (int)((batch_index + 1) * 100 / batch_count);
	fprintf(stdout, "     [%4lld/%4lld] %02d%% [", batch_index + 1, batch_count, percent);
	int n = 0;
	for (; n < percent; n += 2) printf("#");
	for (; n < 100; n += 2) printf(" ");
	printf("] loss = %5.3f, acc = %5.3f+%5.3f\r", (KFloat)loss["#default"], (KFloat)accuracy["char"], (KFloat)accuracy["word"]);
	if (batch_count == batch_index + 1) printf("%c[2K", 27);
	return false;
}

KBool MnistReaderReporter::m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy) {
	fprintf(stdout, "char accuracy = %5.3f, word accuracy = %5.3f\n", (KFloat)accuracy["char"], (KFloat)accuracy["word"]);
	return false;
}

KBool MnistReaderReporter::m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy) {
	printf("> Model %s test ended at %s: char accuracy = %5.3f, word accuracy = %5.3f\n\n",
		sName.c_str(), sTimestamp.c_str(), (KFloat)accuracy["char"], (KFloat)accuracy["word"]);
	return false;
}

KBool MnistReaderReporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	FloatBuffer probs(m_hSession, vs["char_probs"]);
	FloatBuffer select(m_hSession, vs["char_select"]);
	FloatBuffer answer(m_hSession, vs["char_answer"]);

	KInt nsize = probs.axis(0);
	KInt wsize = select.axis(1);
	KInt asize = answer.axis(1);

	for (KInt n = 0; n < nsize; n++) {
		printf("   %lld: answer is '", n);
		for (KInt m = 0; m < wsize; m++) {
			int c = (int)answer.at(n, m);
			printf("%c", (c == 0) ? ' ' : c - 1 + 'a');
		}
		printf(", estimate ");
		for (KInt m = 0; m < wsize; m++) {
			int c = (int)select.at(n, m);
			printf("%c", (c == 0) ? ' ' : c - 1 + 'a');
		}
		printf("\n");
	}

	return false;
}
