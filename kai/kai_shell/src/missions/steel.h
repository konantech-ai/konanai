/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/csv_feeder.h"
#include "../reporters/select_reporter.h"
//#include "../reporters/steel_reporter.h"

class SteelMission : public Mission {
public:
	SteelMission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~SteelMission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute(KString sRnnCell);
	KHNetwork m_buildNetwork(KString sCellType);

protected:
	CsvFeeder     m_dataFeeder;
	SelectReporter m_reporter;
	//SteelReporter m_reporter;
};
