/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "yolo3_reporter.h"

int Yolo3Reporter::ms_checkCode = 16043907;

Yolo3Reporter::Yolo3Reporter(KString sub_model) : Reporter() {
	m_version = 1;
	m_sub_model = sub_model;
}

Yolo3Reporter::~Yolo3Reporter() {
}

KBool Yolo3Reporter::m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accu) {
	printf("   train epoch [%lld/%lld] ended at %s: loss = %5.3f+%5.3f+%5.3f+%5.3f, acc = %5.3f+%5.3f+%5.3f+%5.3f\n", epoch_index, epoch_count, sTimestamp.c_str(),
		(KFloat)loss["xy"], (KFloat)loss["wh"], (KFloat)loss["conf"], (KFloat)loss["class"],
		(KFloat)accu["xy"], (KFloat)accu["wh"], (KFloat)accu["conf"], (KFloat)accu["class"]);
	return false;
}

KBool Yolo3Reporter::m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accu) {
	int percent = (int)((batch_index + 1) * 100 / batch_count);
	fprintf(stdout, "     [%4lld/%4lld] %02d%% [", batch_index + 1, batch_count, percent);
	int n = 0;
	for (; n < percent; n += 2) printf("#");
	for (; n < 100; n += 2) printf(" ");
	printf("] loss = %5.3f+%5.3f+%5.3f+%5.3f, acc = %5.3f+%5.3f+%5.3f+%5.3f\r",
		(KFloat)loss["xy"], (KFloat)loss["wh"], (KFloat)loss["conf"], (KFloat)loss["class"],
		(KFloat)accu["xy"], (KFloat)accu["wh"], (KFloat)accu["conf"], (KFloat)accu["class"]);
	if (batch_count == batch_index + 1) printf("%c[2K", 27);
	return false;
}

KBool Yolo3Reporter::m_validateEnd(void* pAux, KString sTimestamp, KaiDict accu) {
	fprintf(stdout, "accuracy: xy = %5.3f, wh = %5.3f, conf = %5.3f, class = %5.3f\n",
		(KFloat)accu["xy"], (KFloat)accu["wh"], (KFloat)accu["conf"], (KFloat)accu["class"]);
	return false;
}

KBool Yolo3Reporter::m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accu) {
	printf("> Model %s test ended at %s: accuracy: xy = %5.3f, wh = %5.3f, conf = %5.3f, class = %5.3f\n",
		(KFloat)accu["xy"], (KFloat)accu["wh"], (KFloat)accu["conf"], (KFloat)accu["class"]);
	return false;
}

KBool Yolo3Reporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	THROW(KERR_UNIMPEMENTED_YET);
	/*
	IntBuffer tokens(m_hSession, xs["tokens"]);
	IntBuffer maskIndex(m_hSession, xs["mask_index"]);

	FloatBuffer ns_probs(m_hSession, vs["next_sent_probs"]);
	FloatBuffer ns_select(m_hSession, vs["next_sent_select"]);
	FloatBuffer ns_answer(m_hSession, vs["next_sent_answer"]);
	FloatBuffer mw_probs(m_hSession, vs["masked_word_probs"]);
	FloatBuffer mw_select(m_hSession, vs["masked_word_select"]);
	IntBuffer mw_answer(m_hSession, vs["masked_word_answer"]);

	KInt nsize = ns_probs.axis(0);
	KInt max_pos = tokens.axis(1);
	KInt m;

	for (KInt n = 0, mpos = 0; n < nsize; n++) {
		KString sSent1, sSent2;

		printf("   %lld: sent-1 '", n);
		for (m = 1; true; m++) {
			KString sWord = m_pFeeder->getTokenWord(tokens.at(n, m, 0));
			if (sWord == "[SEP]") break;
			printf(" %s", sWord.c_str());
			KString hiddenWord = m_pFeeder->getTokenWord(mw_answer.at(mpos));
			KString estimateWord = m_pFeeder->getTokenWord((KInt)mw_select.at(mpos));
			if (n * max_pos + m == maskIndex.at(mpos)) {
				printf("/%s/%s", hiddenWord.c_str(), estimateWord.c_str());
				mpos++;
			}
		}
		printf("'\n      sent-2 '");
		for (m++; true; m++) {
			KString sWord = m_pFeeder->getTokenWord(tokens.at(n, m, 0));
			if (sWord == "[SEP]") break;
			printf(" %s", sWord.c_str());
			KString hiddenWord = m_pFeeder->getTokenWord(mw_answer.at(mpos));
			KString estimateWord = m_pFeeder->getTokenWord((KInt)mw_select.at(mpos));
			if (n * max_pos + m == maskIndex.at(mpos)) {
				printf("/%s/%s", hiddenWord.c_str(), estimateWord.c_str());
				mpos++;
			}
		}
		float prob1 = ns_probs.at(n, 0);
		float prob2 = ns_probs.at(n, 1);
		KString estNextSent = (ns_select.at(n) == 0) ? "Yes" : "No";
		KString resultNextSent = (ns_select.at(n) == ns_answer.at(n)) ? "Correct" : "Wrog";
		printf("'\n      => next sentence => estimation: %s(%5.3f%%,%5.3f%%) : %s\n", estNextSent.c_str(), prob1, prob2, resultNextSent.c_str());
	}
	*/

	return false;
}
