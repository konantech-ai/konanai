/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../components/kmodel_instance.h"
#include "../exec/callback.h"
#include "../math/karray.h"

//#include <iostream>
//#include <ctime>
//#include <ratio>
#include <chrono>

using namespace std::chrono;

class KaiModel;
class KaiDataset;
class KaiDataloader;
class KaiNetwork;
class KaiOptimizer;

class KaiParameters;
class KaiMath;

class KaiExecContext {
public:
	KaiExecContext(KaiSession* pSession, KaiModelInstance* pModelInst, exec_mode mode, KaiDict kwArgs, KaiCallbackAgent* pCbAgent);
	virtual ~KaiExecContext();

	KaiSession* get_session() { return m_pSession; }

	KInt get_int_property(KString sKey);
	KInt get_int_property(KString sKey, KInt nDefault);

	void set_property(KString sKey, KaiValue vValue);
	KaiValue get_property(KString sKey, KaiValue vDefault=KaiValue());
	KaiDict get_component_property(KString sKey);

	void train();
	void test();
	void visualize();
	
	KaiList predict();

	KaiMath* get_math() { return m_pMath; }

	KBool debugTraceOn() { return ms_debugTrace;  }
	/*
	KaiArray<KFloat> strap_default_forward(KaiDict xs);
	KaiArray<KFloat> strap_default_backprop(KaiDict gfs);

	KaiDict unstrap_default_forward(KaiArray<KFloat> f);
	KaiDict unstrap_default_backprop(KaiArray<KFloat> gx);

	KaiArray<KFloat> forward_neuralnet(KaiArray<KFloat> x);
	KaiArray<KFloat> backprop_neuralnet(KaiArray<KFloat> gf);

	void forward_postproc(KaiDict xs, KaiDict ys, KaiDict fs);
	KaiDict backprop_postproc(KaiDict xs, KaiDict ys, KaiDict fs);

	void update_parameter();
	*/

	KaiValue conv_to_cuda(KaiValue value);

protected:
	void m_initProperties(KaiDict kwArgs);
	
	KaiValue m_conv_to_cuda(KaiValue value, int depth);

	void m_fetch_data(KaiDict dataSection, KaiDict& xs, KaiDict& ys, KInt& mb_size);

	void m_train_minibatch(KInt batch_count, KInt batch_index, KaiDict xs, KaiDict ys, KInt mb_size);
	void m_test_minibatch(KaiDict xs, KaiDict ys, KInt mb_size);
	void m_validate_minibatch(KaiDict xs, KaiDict ys, KInt mb_size);
	void m_visualize_minibatch(KaiDict xs, KaiDict ys, KInt mb_size);

	KaiDict m_predict_minibatch(KaiDict xs);

	KString m_job_name();

	void m_train_job_start();
	void m_train_job_finish();
	void m_train_epoch_start(KInt epoch_count, KInt epoch_index);
	void m_train_epoch_finish(KInt epoch_count, KInt epoch_index);

	void m_exec_save(KInt nEpoch, KInt nBatch = -1);
	void m_exec_report(KBool validate, KBool bReport, KInt nEpoch, KInt nBatch = -1);
	void m_exec_visualize(Ken_visualize_mode mode, KInt nEpoch, KInt nBatch = -1);

	void m_test_job_start();
	void m_test_job_finish();

	void m_validate_start(KInt nEpoch, KInt nBatch);
	//void m_validate_finish(KInt nEpoch, KInt nBatch);

	void m_visualize_start(Ken_visualize_mode mode, KInt nEpoch, KInt nBatch);
	void m_visualize_finish(KInt nEpoch, KInt nBatch);

	void m_visualize_output(KaiDict xs, KaiDict ys, KaiDict outs);

	void m_predict_job_start();
	void m_predict_job_finish();

	void m_forward_neuralnet(KaiDict xs, KBool bIsTraining, KaiDict& pack);
	void m_backprop_neuralnet(KaiDict outs, KaiDict grads);

	KaiDict m_eval_loss_grad(KaiDict xs, KaiDict ys, KaiDict outs, KaiDict& grads, KInt mb_size);
	KaiDict m_eval_accuracy(KaiDict xs, KaiDict ys, KaiDict outs, KInt mb_size);

	KaiDict m_fetch_loss_values(KaiDict loss_arr);

	void m_update_parameter();

	bool m_invoke_check(KString sKey, KInt curr);
	bool m_not_invoke_check(KString sKey, KInt curr);

	void m_shuffle_data(KaiDict& dataSection);

	KFloat m_duration_milisec(KString sKey);

	void m_accumulate_loss(KaiDict& acc_loss, KaiDict loss, KInt mb_size);
	void m_accumulate_accs(KaiDict& acc_accs, KaiDict accs, KInt mb_size);

protected:
	KaiCallbackAgent* m_pCbAgent;

	KaiModelInstance* m_pModelInstance;
	KaiMath* m_pMath;

	KaiSession* m_pSession;

	exec_mode m_mode;

	KaiDict m_shortTermContextInfo;
	
	KaiDict m_tr_loss;

	KaiDict m_tr_accs;
	KaiDict m_va_accs;
	KaiDict m_te_accs;

	KaiDict m_tr_data;
	KaiDict m_te_data;
	KaiDict m_va_data;

	KaiList m_train_history;

	static KBool ms_debugTrace;

	std::map < KString, high_resolution_clock::time_point> m_timeMap;
};
