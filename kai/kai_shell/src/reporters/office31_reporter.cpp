/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "office31_reporter.h"

int Office31Reporter::ms_checkCode = 14249860;

Office31Reporter::Office31Reporter() : Reporter() {
	m_version = 1;
}

Office31Reporter::~Office31Reporter() {
}

void Office31Reporter::setTargetNames(KaiList targetNames) {
	for (KInt n = 0; n < 3; n++) m_domainNames.push_back(targetNames[n]);
	for (KInt n = 4; n < 35; n++) m_productNames.push_back(targetNames[n]);
}

KBool Office31Reporter::m_epochEnd(void* pAux, KString sTimestamp, KInt epoch_count, KInt epoch_index, KInt dat_count, KaiDict loss, KaiDict accuracy) {
	printf("   train epoch [%lld/%lld] ended at %s: loss = %5.3f+%5.3f, acc = %5.3f+%5.3f\n", epoch_index, epoch_count, sTimestamp.c_str(),
		(KFloat)loss["domain"], (KFloat)loss["product"], (KFloat)accuracy["domain"], (KFloat)accuracy["product"]);
	return false;
}

KBool Office31Reporter::m_batchEnd(void* pAux, KString sTimestamp, KInt batch_count, KInt batch_index, KInt batch_size, KaiDict loss, KaiDict accuracy) {
	int percent = (int)((batch_index + 1) * 100 / batch_count);
	fprintf(stdout, "     [%4lld/%4lld] %02d%% [", batch_index + 1, batch_count, percent);
	int n = 0;
	for (; n < percent; n += 2) printf("#");
	for (; n < 100; n += 2) printf(" ");
	printf("] loss = %5.3f+%5.3f, acc = %5.3f+%5.3f\r", (KFloat)loss["domain"], (KFloat)loss["product"], (KFloat)accuracy["domain"], (KFloat)accuracy["product"]);
	if (batch_count == batch_index + 1) printf("%c[2K", 27);
	return false;
}

KBool Office31Reporter::m_validateEnd(void* pAux, KString sTimestamp, KaiDict accuracy) {
	fprintf(stdout, "domain accuracy = %5.3f, product accuracy = %5.3f\n", (KFloat)accuracy["domain"], (KFloat)accuracy["product"]);
	return false;
}

KBool Office31Reporter::m_testEnd(void* pAux, KString sName, KString sTimestamp, KaiDict accuracy) {
	printf("> Model %s test ended at %s: domain accuracy = %5.3f, product accuracy = %5.3f\n\n",
		sName.c_str(), sTimestamp.c_str(), (KFloat)accuracy["domain"], (KFloat)accuracy["product"]);
	return false;
}

KBool Office31Reporter::m_visualizeEnd(void* pAux, KString sName, KString sTimestamp, KaiDict xs, KaiDict ys, KaiDict os, KaiDict vs) {
	FloatBuffer dprobs(m_hSession, vs["domain_probs"]);
	FloatBuffer dselect(m_hSession, vs["domain_select"]);
	FloatBuffer danswer(m_hSession, vs["domain_answer"]);
	FloatBuffer pprobs(m_hSession, vs["product_probs"]);
	FloatBuffer pselect(m_hSession, vs["product_select"]);
	FloatBuffer panswer(m_hSession, vs["product_answer"]);

	KInt nsize = dprobs.axis(0);

	for (KInt n = 0; n < nsize; n++) {
		KInt nDomainSelect = (KInt)dselect.at(n);
		KInt nDomainAnswer = (KInt)danswer.at(n);

		KString sDomainSelect = m_domainNames[nDomainSelect];
		KString sDomainAnswer = m_domainNames[nDomainAnswer];

		printf("   %lld: domain  is %s in %4.1f%% prob: ", n, sDomainSelect.c_str(), dprobs.at(n, nDomainSelect));

		if (nDomainSelect == nDomainAnswer) printf("Correct!\n");
		else printf("Wrong! Answer is %s (%3.1f%% in estimate)\n", sDomainAnswer.c_str(), dprobs.at(n, nDomainAnswer));

		KInt nProductSelect = (KInt)pselect.at(n);
		KInt nProductAnswer = (KInt)panswer.at(n);

		KString sProductSelect = m_productNames[nProductSelect];
		KString sProductAnswer = m_productNames[nProductAnswer];

		printf("      product is %s in %4.1f%% prob: ", sProductSelect.c_str(), pprobs.at(n, nProductSelect));

		if (nProductSelect == nProductAnswer) printf("Correct!\n");
		else printf("Wrong! Answer is %s (%3.1f%% in estimate)\n", sProductAnswer.c_str(), pprobs.at(n, nProductAnswer));
	}

	printf("\n");

	return false;
}
