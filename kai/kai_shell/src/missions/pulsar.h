/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/csv_feeder.h"
#include "../reporters/binary_reporter.h"

class PulsarMission : public Mission {
public:
	PulsarMission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~PulsarMission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute(KString sRnnCell);
	KHNetwork m_buildNetwork(KString sCellType);

protected:
	CsvFeeder m_dataFeeder;
	BinaryReporter m_reporter;
};
