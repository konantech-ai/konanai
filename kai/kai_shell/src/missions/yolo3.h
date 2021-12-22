/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/yolo3_feeder.h"
#include "../reporters/yolo3_reporter.h"
#include "../utils/utils.h"

class Yolo3Mission : public Mission {
public:
	Yolo3Mission(KHSession hSession, KString sub_model, enum Ken_test_level testLevel);
	virtual ~Yolo3Mission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute();
	KHNetwork m_buildNetwork();

protected:
	KString m_sub_model;

	Yolo3Feeder m_feeder;
	Yolo3Reporter m_reporter;
};
