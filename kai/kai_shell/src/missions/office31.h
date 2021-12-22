/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../mission.h"
#include "../data_feeders/folder_class_feeder.h"
#include "../reporters/office31_reporter.h"

class Office31Mission : public Mission {
public:
	Office31Mission(KHSession hSession, enum Ken_test_level testLevel);
	virtual ~Office31Mission();

	virtual void Execute();

protected:
	void m_createComponents();
	void m_execute(KString sNetwork, KString sSubModel = "");

	KHNetwork m_buildNetwork(KString sNetwork, KString sSubModel);

	KHNetwork m_buildMlpNetwork();
	KHNetwork m_buildCnnNetwork();
	KHNetwork m_buildInceptionNetwork(KString sSubModel);
	KHNetwork m_buildResnetNetwork(KString sSubModel);

	void m_regist_macro_conv1_LA();
	void m_regist_macro_conv1_LAB();
	void m_regist_macro_conv1_LBA();

	void m_regist_macro_conv_pair();

	void m_regist_macro_preproc();
	void m_regist_macro_resize();
	void m_regist_macro_inception1();
	void m_regist_macro_inception2();
	void m_regist_macro_postproc();

	void m_regist_macro_resnet_blocks();
	void m_regist_macro_resnet_model(KString sModel);

protected:
	FolderClassFeeder m_dataFeeder;
	Office31Reporter m_reporter;
};
