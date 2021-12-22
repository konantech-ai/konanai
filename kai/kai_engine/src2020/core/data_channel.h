/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class Dataset;

class DataChannel {
public:
	DataChannel(Dataset* pDataset, enum data_channel channel, int64* data_idx, int64 data_cnt, int64 epoch, int64 batch_count, int64 batch_size, int64 max_stack=10);
	virtual ~DataChannel();

	virtual void get_data(Dict& xs, Dict& ys);

protected:
	Dataset* m_pDataset;
	enum data_channel m_channel;

	string m_name;

	int64* m_data_idx;
	int64 m_data_cnt;
	int64 m_epoch;
	int64 m_batch_count;
	int64 m_batch_size;
	int64 m_max_stack;
	
	int64 m_max_buffer_size;

	bool m_continue;

	queue<Dict> m_data_Queue;
	//vector<Value> m_buffer;
	mutex* m_mu_buffer;

	thread* m_producer;
	int64 m_channel_seed;

	static void ms_produce(void* aux);

	void m_produce_fixed();
	void m_produce_unlimited();

	void m_sleep();
};