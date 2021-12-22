/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "data_channel.h"
#include "dataset.h"
#include "random.h"
#include "host_math.h"

#include "../cuda/cuda_conn.cuh"

#include <time.h>

DataChannel::DataChannel(Dataset* pDataset, enum data_channel channel, int64* data_idx, int64 data_cnt, int64 epoch, int64 batch_count, int64 batch_size, int64 max_stack) {
	m_pDataset = pDataset;
	m_channel = channel;

	if (m_channel == data_channel::train) m_name = "train";
	else if (m_channel == data_channel::test) m_name = "test";
	else if (m_channel == data_channel::validate) m_name = "validate";
	else if (m_channel == data_channel::visualize) m_name = "visualize";
	else if (m_channel == data_channel::autoencode) m_name = "autoencode";

	m_data_idx = data_idx;
	m_data_cnt = data_cnt;
	m_epoch = epoch;
	m_batch_count = batch_count;
	m_batch_size = batch_size;
	m_max_stack = max_stack;

	m_max_buffer_size = max_stack ? max_stack : 10;

	m_continue = true;
	m_mu_buffer = new std::mutex();
	//m_producer = thread(ms_produce, this);

	//std::thread producer_thread(ms_produce, this);
	m_channel_seed = Random::dice(100000);

	m_producer = new std::thread(ms_produce, this);
}

DataChannel::~DataChannel() {
	m_continue = false;
	m_producer->join();
	delete m_producer;
	delete m_mu_buffer;
}

void DataChannel::get_data(Dict& xs, Dict& ys) {
	//m_producer.join();
	while (true) {
		m_mu_buffer->lock();
		if (m_data_Queue.size() > 0) {
			Dict minibatch = m_data_Queue.front();
			m_data_Queue.pop();
			xs = minibatch["xs"];
			ys = minibatch["ys"];
			m_mu_buffer->unlock();
			break;
		}
		m_mu_buffer->unlock();
		m_sleep();
	}
}

void DataChannel::m_sleep() {
#ifdef KAI2021_WINDOWS
	Util::nanosleep(100000);
#else
	int64 microsec = 100; // length of time to sleep, in miliseconds
	struct timespec req = { 0 };
	req.tv_sec = 0;
	req.tv_nsec = microsec * 1000L;
	nanosleep(&req, (struct timespec*)NULL);
#endif
}

#include <iostream>

void DataChannel::ms_produce(void* aux) {
	//std::thread::id this_id = std::this_thread::get_id();
	//std::cout << "DataChannel::ms_produce called in thread " << this_id << "...\n";
	CudaConn::SetDevice();

	DataChannel* pInstance = (DataChannel*)aux;
	Random::seed(pInstance->m_channel_seed);
	if (pInstance->m_epoch == 0) pInstance->m_produce_unlimited();
	else pInstance->m_produce_fixed();
}

void DataChannel::m_produce_fixed() {
	for (int64 n = 0; n < m_epoch; n++) {
		if (!m_continue) break;
		kmath->shuffle(m_data_cnt, m_data_idx);
		m_pDataset->save_data_index("_fixed");
		for (int64 m = 0, pos = 0; m < m_batch_count; m++, pos += m_batch_size) {
			if (!m_continue) break;
			Dict xs, ys;
			m_pDataset->gen_minibatch_data(m_channel, m_data_idx + pos, m_batch_size, xs, ys);
			Dict minibatch;
			minibatch["xs"] = xs;
			minibatch["ys"] = ys;
			while ((int64)m_data_Queue.size() >= m_max_stack) m_sleep();
			m_mu_buffer->lock();
			m_data_Queue.push(minibatch);
			m_mu_buffer->unlock();
		}
	}
}

void DataChannel::m_produce_unlimited() {
	int64 pos = 0;

	while (m_continue) {
		if (pos == 0) {
			kmath->shuffle(m_data_cnt, m_data_idx);
			m_pDataset->save_data_index("_"+m_name);
		}

		Dict xs, ys;
		m_pDataset->gen_minibatch_data(m_channel, m_data_idx + pos, m_batch_size, xs, ys);
		Dict minibatch;
		minibatch["xs"] = xs;
		minibatch["ys"] = ys;
		while ((int64)m_data_Queue.size() >= m_max_stack) {
			if (!m_continue) break;
			m_sleep();
		}
		m_mu_buffer->lock();
		m_data_Queue.push(minibatch);
		m_mu_buffer->unlock();
		pos += m_batch_size;
		if (pos + m_batch_size > m_data_cnt) pos = 0;
	}
}
