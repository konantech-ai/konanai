/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/mnist_reader_feeder.h"
#include "../reporters/mnist_reader_reporter.h"
#include "../utils/utils.h"

class MnistReaderMission : public Mission {
public:
	MnistReaderMission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~MnistReaderMission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute(KString sRnnCell);
	KHNetwork m_buildNetwork(KString sCellType);

protected:
	MnistReaderFeeder m_dataFeeder;
	MnistReaderReporter m_reporter;
};
