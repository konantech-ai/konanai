/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "encdec.h"
#include "../core/log.h"

EncoderDecoder::EncoderDecoder(const char* name, Dataset& dataset, const char* conf, const char* options, MacroPack* pMacros)
	: Engine(name, dataset, "(none)", options, pMacros) {
	Dict hconfigs = Value::parse_dict(conf);
	
	List econf = hconfigs["encoder"];
	List dconf = hconfigs["decoder"];

	//string concated = Value::description(econf) + Value::description(dconf);
	
	m_build_neuralnet(conf);

	if ((bool)lookup_option("dump_structure")) {
		m_dump_structure();
		//Engine::m_dump_structure(m_layers, "Neuralnet", 0);
	}
}

EncoderDecoder::~EncoderDecoder() {
}

Layer* EncoderDecoder::seek_named_layer(string name) {
	for (vector<Layer*>::iterator it = m_elayers.begin(); it != m_elayers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}

	for (vector<Layer*>::iterator it = m_dlayers.begin(); it != m_dlayers.end(); it++) {
		Layer* pLayer = (*it)->seek_named_layer(name);
		if (pLayer) return pLayer;
	}

	throw KaiException(KERR_ASSERT);
	return NULL;
}

void EncoderDecoder::m_build_neuralnet(const char* conf) {
	Dict hconfigs = Value::parse_dict(conf);

	Shape shape = m_dataset.input_shape;
	bool seq = m_dataset.input_seq();

	build_hidden_net(hconfigs["encoder"], shape, seq, m_elayers, m_epms);
	build_hidden_net(hconfigs["decoder"], shape, seq, m_dlayers, m_dpms);

	if ((bool)lookup_option("use_output_layer")) {
		build_output_net(shape, seq, m_dlayers, m_dpms);
	}

	assert(m_dataset.output_seq() == seq);
	assert(m_dataset.output_shape == shape);

	m_layers.insert(m_layers.end(), m_elayers.begin(), m_elayers.end());
	m_layers.insert(m_layers.end(), m_dlayers.begin(), m_dlayers.end());

	m_pms.insert(m_pms.end(), m_epms.begin(), m_epms.end());
	m_pms.insert(m_pms.end(), m_dpms.begin(), m_dpms.end());
}

void EncoderDecoder::m_dump_structure() {
	logger.Print("Encoder-Decoder structure");

	int64 pm_count = 0;

	pm_count += Engine::m_dump_structure(m_elayers, "Encoder", 1);
	pm_count += Engine::m_dump_structure(m_dlayers, "Decoder", 1);

	logger.Print("Total parameter count: %lld pms", pm_count);
}

