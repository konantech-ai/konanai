/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/urban_feeder.h"
#include "../reporters/select_reporter.h"
#include "../utils/utils.h"

class UrbanSoundMission : public Mission {
public:
	UrbanSoundMission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~UrbanSoundMission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute(KString sRnnCell, KBool use_state=false);
	KHNetwork m_buildNetwork(KString sCellType, KBool use_state);

protected:
	UrbanDataFeeder m_urbanFeeder;
	SelectReporter m_reporter;
};
