/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/engine.h"
#include "../core/dataset.h"
#include "../core/macro_pack.h"

#include "../int_plugin/layer.cuh"

class EncoderDecoder : public Engine {
public:
	EncoderDecoder(const char* name, Dataset& dataset, const char* conf, const char* options = NULL, MacroPack* pMacros = NULL);
	virtual ~EncoderDecoder();

	virtual Layer* seek_named_layer(string name);

protected:
	void m_build_neuralnet(const char* conf);
	void m_dump_structure();

	Layers m_elayers;
	Layers m_dlayers;
	List m_epms;
	List m_dpms;
};