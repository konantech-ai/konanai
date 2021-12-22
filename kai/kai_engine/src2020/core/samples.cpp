/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>

#include "samples.h"
#include "dataset.h"
#include "engine.h"
#include "corpus.h"
#include "random.h"
#include "macro_pack.h"
#include "func_timer.h"
#include "log.h"

#include "../datasets/abalone.h"
#include "../datasets/pulsar.h"
#include "../datasets/steel.h"
#include "../datasets/flower.h"
#include "../datasets/office31.h"
#include "../datasets/dummy.h"
#include "../datasets/automata.h"
#include "../datasets/urbansound.h"
#include "../datasets/videoshot.h"
#include "../datasets/mnist_auto.h"
#include "../datasets/mnist_encdec.h"
#include "../datasets/mnist_gan.h"
#include "../datasets/cifar10_auto.h"
#include "../datasets/gan_cifar10.h"
#include "../datasets/word2vec.h"
#include "../datasets/word_seq.h"
#include "../datasets/bert.h"
#include "../datasets/yolo.h"
#include "../datasets/test_dataset.h"
#include "../datasets/coco.h"

#include "../datasets/corpus/ptb_corpus.h"
#include "../datasets/corpus/korean_news.h"

#include "../apps/autoencoder.h"
#include "../apps/encdec.h"
#include "../apps/gan.h"

#include "../utils/news_reformer.h"
#include "../utils/test.h"

#include "../server/connect.h"

// trouble: mysql_test, adam, flower_cnn(slow, leak)

const char* Samples::m_ppSamples[] = {
		"start_server", "stop_server", "mysql_test",
		"abalone", "pulsar", "steel", "pulsar_select", "flower", "office31",
		"flower_cnn", "office31_cnn", "show_params", "regs", "inception_v3", "inception_flower", "resnet", "resnet_flower",
		"automata", "automata_drop", "urban_sound", "video_shot",
		"autoencoder", "autoencoder_hash", "autoencoder_cifar", "autoencoder_hash_cifar", "encoder_decoder_eng", "encoder_decoder_kor", "gan", "gan_mnist", "gan_cifar10",
		"w2v_cbow14", "w2v_cbow25", "w2v_skip14", "w2v_skip25", "next_word", "bert_ptb",
		"bert_ptb_large", "bert_ptb_2step", "bert_korean_news", "bert_mid_korean_news", "bert_mid2_korean_news", "bert_large_korean_news", "bert_layer_test",
		"yolo", "image_fill", "image_fill_small",
		"test_alphgo", "play_alphgo", "gtp_alphgo", "train_alphgo", "test_reinforce", "reinforce",
		"news", 
		"gan_book", };

void Samples::execute(const char* mission, string mode) {
	//Random::seed(1234);


	if (strcmp(mission, "start_server") == 0) {
//hs.cho
#ifdef KAI2021_WINDOWS
		kai_server.openService();
#endif
	}
	else if (strcmp(mission, "stop_server") == 0) {
#ifdef KAI2021_WINDOWS
		kai_server.closeService();
#endif
	}
	else if (strcmp(mission, "mysql_test") == 0) {
#ifdef KAI2021_WINDOWS
		MysqlConn conn;
		conn.test();
#endif
	}
	else if (strcmp(mission, "abalone") == 0) {
		AbaloneDataset ad;
		Engine am("ch01-1.abalone_model", ad, "[]", "{'dump_structure': True}");
		am.exec_all("{'epoch_count':10, 'batch_size':10, 'report':2}");
		Engine am2("ch01-1.abalone_model-2", ad, "[16]", "{'dump_structure': True}");
		am2.exec_all("{'epoch_count':10, 'report':2}");
	}
	else if (strcmp(mission, "pulsar") == 0) {
		PulsarDataset pd;
		Engine pm("ch02-1.pulsar_model", pd, "[4]");
		pm.exec_all("{'epoch_count':10, 'report':2}");
		//pm.visualize(5);
	}
	else if (strcmp(mission, "steel") == 0) {
		SteelDataset sd;
		//Engine sm("ch03-1.steel_model", sd, "[4]");
		Engine sm("ch03-1.steel_model", sd, "[12, 7]");
		sm.exec_all("{'epoch_count':10, 'report':2}");
	}
	else if (strcmp(mission, "pulsar_select") == 0) {
		PulsarSelectDataset psd;
		Engine psm("ch02-2.pulsar_select_model", psd, "[4]");
		psm.exec_all("{'epoch_count':10, 'report':2}");
	}
	else if (strcmp(mission, "flower") == 0) {
		FlowerDataset fd("chap05/flowers", "flowerkodell.100_plain.dat", Shape(100, 100), Shape(100, 100, 3));

		Engine fm1("ch05-1.flowers_model_1", fd, "[10]");
		fm1.exec_all("{'epoch_count':5, 'batch_size':10, 'report':1}");

		Engine fm2("ch05-2.flowers_model_2", fd, "[30, 10]");
		fm2.exec_all("{'epoch_count':5, 'batch_size':10, 'report':1}");

		Engine fm3("ch05-2.flowers_model_clip_grad", fd, "[30, 10]", "{'epoch_show_grad_norm':5, 'clip_grad':1.0}");
		fm2.exec_all("{'epoch_count':5, 'batch_size':10, 'report':1}");
	}
	else if (strcmp(mission, "office31") == 0) {
		Office31Dataset od("chap06/office31", "office31.kodell.100_plain.dat", Shape(100, 100), Shape(100, 100, 3));
		Engine om1("ch06-1.office31_model_1", od, "[10]", "{'optimizer':'sgd'}");
		om1.exec_all("{'epoch_count':5, 'report':1}");
		Engine om2("ch06-2.office31_model_2", od, "[64,32,10]", "{'optimizer':'sgd'}");
		om2.exec_all("{'epoch_count':5, 'report':1, 'learning_rate':0.0001}");
		//Engine om3("ch06-3.office31_model_3", od, "[64,32,10]", "{'optimizer':'adam'}");
		//om3.exec_all("{'epoch_count':10, 'report':2, 'learning_rate':0.0001}");
	}
	else if (strcmp(mission, "flower_cnn") == 0) {
		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));
		Engine fm1("ch07-1.flowers_cnn_model_1", fd,
			"[['conv',{ 'ksize':5, 'chn' : 6 }], ['max', { 'stride':4 }], \
			 ['conv', { 'ksize':3, 'chn' : 12 }], ['avg', { 'stride':2 }]]",
			"{'dump_structure': True}");
		fm1.exec_all("{'epoch_count':5, 'report':1}");
		Engine fm2("ch07-2.flowers_cnn_model_1", fd,
			"[['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]",
			"{'dump_structure': True}");
		fm2.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "office31_cnn") == 0) {
		Office31Dataset od("chap06/office31", "office31.kodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));
		Engine om1("ch07-3.office31_cnn_model_1", od,
			"[['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]",
			"{'dump_structure': True}");
		om1.exec_all("{'epoch_count':5, 'report':1}");
		Engine om2("ch07-4.office31_cnn_model_2", od,
			"[['conv', { 'ksize':3, 'chn' : 6, 'actfunc' : 'sigmoid' }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12, 'actfunc' : 'sigmoid' }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 , 'actfunc' : 'sigmoid'}], ['avg', { 'stride':3 }]]",
			"{'dump_structure': True}");
		om2.exec_all("{'epoch_count':5, 'report':1}");
		/*
		Engine om3("ch07-5.office31_cnn_model_3", od,
			"[['conv', { 'ksize':3, 'chn' : 6, 'actfunc' : 'tanh' }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12, 'actfunc' : 'tanh' }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 , 'actfunc' : 'tanh'}], ['avg', { 'stride':3 }]]",
			"{'dump_structure': True}");
		om3.exec_all("{'epoch_count':10, 'report':2}");
		*/
	}
	else if (strcmp(mission, "show_params") == 0) {
		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));
		Engine fm1("ch08-1.flowers_model_1", fd, "[30, 10]");
		fm1.exec_all("{'epoch_count':10, 'report':2, 'show_params':True}");
		Engine fm2("ch08-2.flowers_model_2", fd, "[30, 10]", "{'l2_decay':0.1}");
		fm2.exec_all("{'epoch_count':10, 'report':2, 'show_params':True}");
		Engine fm3("ch08-3.flowers_model_3", fd, "[30, 10]", "{'l1_decay':0.01}");
		fm3.exec_all("{'epoch_count':10, 'report':2, 'show_params':True}");
	}
	else if (strcmp(mission, "regs") == 0) {
		string cnn1 = "[['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]";
		string cnn2 = "[['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }],	\
						['dropout', { 'keep_prob':0.6 }],								\
						['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }],	\
						['dropout', { 'keep_prob':0.6 }],								\
						['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }],	\
						['dropout', { 'keep_prob':0.6 }]]";
		string cnn3 = "[['noise', { 'type':'normal','mean' : 0,'std' : 0.01 }],			\
						['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }],	\
						['noise', { 'type':'normal','mean' : 0,'std' : 0.01 }],			\
						['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }],	\
						['noise', { 'type':'normal','mean' : 0,'std' : 0.01 }],			\
						['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]";
		string cnn4 = "[['batch_normal'],												\
						['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }],	\
						['batch_normal'],												\
						['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }],	\
						['batch_normal'],												\
						['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]";
		string cnn5 = "[['conv', { 'ksize':3, 'chn' : 6,  'actions' : 'BLA' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 12, 'actions' : 'BLA' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 24, 'actions' : 'BLA' }], ['avg', { 'stride':3 }]]";
		string cnn6 = "[['conv', { 'ksize':3, 'chn' : 6,  'actions' : 'LAB' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 12, 'actions' : 'LAB' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 24, 'actions' : 'LAB' }], ['avg', { 'stride':3 }]]";
		string cnn7 = "[['conv', { 'ksize':3, 'chn' : 6,  'actions' : 'LBA' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 12, 'actions' : 'LBA' }], ['max', { 'stride':2 }],	\
						['conv', { 'ksize':3, 'chn' : 24, 'actions' : 'LBA' }], ['avg', { 'stride':3 }]]";

		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));

		Engine fcnn1("ch08-4.flowers_cnn_baseline", fd, cnn1.c_str(), "{'dump_structure': True}");
		fcnn1.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn2("ch08-5.flowers_cnn_dropout", fd, cnn2.c_str(), "{'dump_structure': True}");
		fcnn2.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn3("ch08-6.flowers_cnn_noise", fd, cnn3.c_str(), "{'dump_structure': True}");
		fcnn3.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn4("ch08-7.flowers_cnn_batch_normal", fd, cnn4.c_str(), "{'dump_structure': True}");
		fcnn4.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn5("ch08-8.flowers_cnn_BLA", fd, cnn5.c_str(), "{'dump_structure': True}");
		fcnn5.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn6("ch08-9.flowers_cnn_LAB", fd, cnn6.c_str(), "{'dump_structure': True}");
		fcnn6.exec_all("{'epoch_count':10, 'report':2}");

		Engine fcnn7("ch08-10.flowers_cnn_LBA", fd, cnn7.c_str(), "{'dump_structure': True}");
		fcnn7.exec_all("{'epoch_count':10, 'report':2}");
	}
	else if (strcmp(mission, "inception_v3") == 0) {
		MacroPack macros;

		macros.set_macro("v3_preproc",
			"['serial', \
				['conv', {'ksize':3, 'stride':2, 'chn':32, 'padding':'VALID'}], \
				['conv', {'ksize':3, 'chn':32, 'padding':'VALID'}],				\
				['conv', {'ksize':3, 'chn':64, 'padding':'SAME'}],				\
				['max', {'ksize':3, 'stride':2, 'padding':'VALID'}],			\
				['conv', {'ksize':1, 'chn':80, 'padding':'VALID'}],				\
				['conv', {'ksize':3, 'chn':192, 'padding':'VALID'}],			\
				['max', {'ksize':3, 'stride':2, 'padding':'VALID'}]]");

		macros.set_macro("v3_inception1",
			"['parallel',										\
				['conv', {'ksize':1, 'chn':64}],				\
				['serial',										\
					['conv', {'ksize':1, 'chn':48}],			\
					['conv', {'ksize':5, 'chn':64}]],			\
				['serial',										\
					['conv', {'ksize':1, 'chn':64}],			\
					['conv', {'ksize':3, 'chn':96}],			\
					['conv', {'ksize':3, 'chn':96}]],			\
				['serial',										\
					['avg', {'ksize':3, 'stride':1}],			\
					['conv', {'ksize':1, 'chn':'#chn'}]]]");

		macros.set_macro("v3_resize1",
			"['parallel',											\
				['conv', {'ksize':3, 'stride':2, 'chn':384}],		\
				['serial',											\
					['conv', {'ksize':1, 'chn':64}],				\
					['conv', {'ksize':3, 'chn':96}],				\
					['conv', {'ksize':3, 'stride':2, 'chn':96}]],	\
				['max', {'ksize':3, 'stride':2}]]");

		macros.set_macro("v3_inception2",
			"['parallel',										\
				['conv', {'ksize':1, 'chn':192}],				\
				['serial',										\
					['conv', {'ksize':[1,1], 'chn':'#chn'}],	\
					['conv', {'ksize':[1,7], 'chn':'#chn'}],	\
					['conv', {'ksize':[7,1], 'chn':192}]],		\
				['serial',										\
					['conv', {'ksize':[1,1], 'chn':'#chn'}],	\
					['conv', {'ksize':[7,1], 'chn':'#chn'}],	\
					['conv', {'ksize':[1,7], 'chn':'#chn'}],	\
					['conv', {'ksize':[7,1], 'chn':'#chn'}],	\
					['conv', {'ksize':[1,7], 'chn':192}]],		\
				['serial',										\
					['avg', {'ksize':3, 'stride':1}],			\
					['conv', {'ksize':1, 'chn':192}]]]");

		macros.set_macro("v3_resize2",
			"['parallel',													\
				['serial',													\
					['conv', {'ksize':1, 'chn':192}],						\
					['conv', {'ksize':3, 'stride':2, 'chn':320}]],			\
				['serial',													\
					['conv', {'ksize':[1,1], 'chn':192}],					\
					['conv', {'ksize':[1,7], 'chn':192}],					\
					['conv', {'ksize':[7,1], 'chn':192}],					\
					['conv', {'ksize':[3,3], 'stride':[2,2], 'chn':192}]],	\
				['max', {'ksize':3, 'stride':2}]]");

		macros.set_macro("v3_inception3",
			"['parallel',													\
				['conv', {'ksize':1, 'chn':320}],							\
				['serial',													\
					['conv', {'ksize':[3,3], 'chn':384}],					\
					['parallel',											\
						['conv', {'ksize':[1,3], 'chn':384}],				\
						['conv', {'ksize':[3,1], 'chn':384}]]],				\
				['serial',													\
					['conv', {'ksize':[1,1], 'chn':448}],					\
					['conv', {'ksize':[3,3], 'chn':384}],					\
					['parallel',											\
						['conv', {'ksize':[1,3], 'chn':384}],				\
						['conv', {'ksize':[3,1], 'chn':384}]]],				\
				['serial',													\
					['avg', {'ksize':3, 'stride':1}],						\
					['conv', {'ksize':1, 'chn':192}]]]");

		macros.set_macro("v3_postproc",
			"['serial',														\
				['avg', {'stride':8}],										\
				['dropout', {'keep_prob':0.7}]]");

		macros.set_macro("inception_v3",
			"['serial',														\
				['custom', {'name':'v3_preproc'}],							\
				['custom', {'name':'v3_inception1', 'args':{'#chn':32}}],	\
				['custom', {'name':'v3_inception1', 'args':{'#chn':64}}],	\
				['custom', {'name':'v3_inception1', 'args':{'#chn':64}}],	\
				['custom', {'name':'v3_resize1'}],							\
				['custom', {'name':'v3_inception2', 'args':{'#chn':128}}],	\
				['custom', {'name':'v3_inception2', 'args':{'#chn':160}}],	\
				['custom', {'name':'v3_inception2', 'args':{'#chn':160}}],	\
				['custom', {'name':'v3_inception2', 'args':{'#chn':192}}],	\
				['custom', {'name':'v3_resize2'}],							\
				['custom', {'name':'v3_inception3'}],						\
				['custom', {'name':'v3_inception3'}],						\
				['custom', {'name':'v3_postproc'}]]");

		DummyDataset imagenet("imagenet", "classify", Shape(299, 299, 3), Shape(200));

		Engine inception_v3("ch09-0.inception_v3", imagenet,
			"[['custom', {'name':'inception_v3'}]]", "{'dump_structure':True}", &macros);
	}
	else if (strcmp(mission, "inception_flower") == 0) {
		MacroPack macros;
	
		macros.set_macro("flower_preproc",
			"['serial', ['conv', {'ksize':3, 'stride':2, 'chn':6, 'actions':'#act'}]]");

		macros.set_macro("flower_inception1",
			"['parallel', \
				['conv', {'ksize':1, 'chn':4, 'actions':'#act'}], \
				['conv', {'ksize':3, 'chn':6, 'actions':'#act'}], \
				['serial', \
					['conv', {'ksize':3, 'chn':6, 'actions':'#act'}], \
					['conv', {'ksize':3, 'chn':6, 'actions':'#act'}]], \
				['serial', \
					['avg', {'ksize':3, 'stride':1}], \
					['conv', {'ksize':1, 'chn':4, 'actions':'#act'}]]]");

		macros.set_macro("flower_resize",
			"['parallel', \
				['conv', {'ksize':3, 'stride':2, 'chn':12, 'actions':'#act'}], \
				['serial', \
					['conv', {'ksize':3, 'chn':12, 'actions':'#act'}], \
					['conv', {'ksize':3, 'stride':2, 'chn':12, 'actions':'#act'}]], \
				['avg', {'ksize':3, 'stride':2}]]");

		macros.set_macro("flower_inception2",
			"['parallel', \
				['conv', {'ksize':1, 'chn':8, 'actions':'#act'}], \
				['serial', \
					['conv', {'ksize':[3,3], 'chn':8, 'actions':'#act'}], \
					['parallel', \
						['conv', {'ksize':[1,3], 'chn':8, 'actions':'#act'}], \
						['conv', {'ksize':[3,1], 'chn':8, 'actions':'#act'}]]], \
				['serial', \
					['conv', {'ksize':[1,1], 'chn':8, 'actions':'#act'}], \
					['conv', {'ksize':[3,3], 'chn':8, 'actions':'#act'}], \
					['parallel', \
						['conv', {'ksize':[1,3], 'chn':8, 'actions':'#act'}], \
						['conv', {'ksize':[3,1], 'chn':8, 'actions':'#act'}]]], \
				['serial', \
					['avg', {'ksize':3, 'stride':1}], \
					['conv', {'ksize':1, 'chn':8, 'actions':'#act'}]]]");

		macros.set_macro("flower_postproc",
			"['serial', ['avg', {'stride':6}], ['dropout', {'keep_prob':0.7}]]");

		macros.set_macro("inception_flower",
			"['serial', \
				['custom', {'name':'flower_preproc', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_inception1', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_inception1', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_inception2', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_inception2', 'args':{'#act':'#act'}}], \
				['custom', {'name':'flower_postproc', 'args':{'#act':'#act'}}]]");

		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));

		string conf_flower_LA = "['custom', {'name':'inception_flower', 'args':{'#act':'LA'}}]";
		Engine model_flower_LA("ch09-1.model_flower_LA", fd, conf_flower_LA.c_str(), "{'dump_structure':True}", &macros);
		model_flower_LA.exec_all("{'epoch_count':5, 'report':1}");

		string conf_flower_LAB = "['custom', {'name':'inception_flower', 'args':{'#act':'LAB'}}]";
		Engine model_flower_LAB("ch09-2.model_flower_LAB", fd, conf_flower_LAB.c_str(), "{'dump_structure':True}", &macros);
		model_flower_LAB.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "resnet") == 0) {
		DummyDataset imagenet("imagenet", "classify", Shape(224, 224, 3), 1000);

		MacroPack macros;
		m_create_resnet_macros(macros);

		macros.set_macro("vgg_19",
			"['serial', \
				['custom', {'name':'p24', 'args':{'#repeat':2, '#chn':64}}], \
				['custom', {'name':'p24', 'args':{'#repeat':2, '#chn':128}}], \
				['custom', {'name':'p24', 'args':{'#repeat':4, '#chn':256}}], \
				['custom', {'name':'p24', 'args':{'#repeat':4, '#chn':512}}], \
				['custom', {'name':'p24', 'args':{'#repeat':4, '#chn':512}}], \
				['serial', {'repeat':2}, ['full', {'width':4096}]]]");

		Engine vgg19("ch09-1.vgg_19", imagenet, "['custom', {'name':'vgg_19'}]", "{'dump_structure':True}", &macros);

		macros.set_macro("plain_34",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':64, 'actions':'#act'}], \
				['max', {'stride':2}], \
				['serial', {'repeat':6}, ['conv', {'ksize':3, 'chn':64, 'actions':'#act'}]], \
				['custom', {'name':'pn', 'args':{'#cnt1':7, '#n':128, '#act':'#act'}}], \
				['custom', {'name':'pn', 'args':{'#cnt1':11, '#n':256, '#act':'#act'}}], \
				['custom', {'name':'pn', 'args':{'#cnt1':5, '#n':512, '#act':'#act'}}], \
				['avg', {'stride':7}]]");

		Engine plain_34("ch09-2.plain_34", imagenet, "['custom', {'name':'plain_34', 'args':{'#act':'LA'}}]", "{'dump_structure':True}", &macros);

		macros.set_macro("residual_34",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':64, 'actions':'#act'}], \
				['max', {'stride':2}], \
				['custom', {'name':'rfull', 'args':{'#cnt':3, '#n':64, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':3, '#n':128, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':5, '#n':256, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':2, '#n':512, '#act':'#act'}}], \
				['avg', {'stride':7}]]");

		Engine residual_34("ch09-3.residual_34", imagenet, "['custom', { 'name':'residual_34', 'args' : {'#act':'LA'} }]", "{'dump_structure':True}", &macros);

		macros.set_macro("bottleneck_152",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':64, 'actions':'#act'}], \
				['max', {'ksize':3, 'stride':2}], \
				['custom', {'name':'bfull','args':{'#cnt':3,'#n1':64,'#n4':256,'#act':'#act'}}], \
				['custom', {'name':'bhalf','args':{'#cnt1':7,'#n1':128,'#n4':512, '#act':'#act'}}], \
				['custom', {'name':'bhalf','args':{'#cnt1':35,'#n1':256,'#n4':1024, '#act':'#act'}}], \
				['custom', {'name':'bhalf','args':{'#cnt1':2,'#n1':512,'#n4':2048, '#act':'#act'}}], \
				['avg', {'stride':7}]]");

		Engine bottleneck_152("ch09-4.bottleneck_152", imagenet, "['custom', { 'name':'bottleneck_152', 'args' : {'#act':'LAB'} }]", "{'dump_structure':True}", &macros);
	}
	else if (strcmp(mission, "resnet_flower") == 0) {
		FlowerDataset fd("chap05/flowers", "flowerkodell.64_2d.dat", Shape(64, 64), Shape(64, 64, 3));

		MacroPack macros;
		m_create_resnet_macros(macros);

		macros.set_macro("plain_flower",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':16, 'actions':'#act'}], \
				['max', { 'stride':2 }], \
				['serial', { 'repeat':4 }, ['conv', { 'ksize':3, 'chn' : 16, 'actions' : '#act' }]], \
				['custom', { 'name':'pn', 'args' : {'#cnt1':3, '#n' : 32, '#act' : '#act'} }], \
				['custom', { 'name':'pn', 'args' : {'#cnt1':3, '#n' : 64, '#act' : '#act'} }], \
				['avg', { 'stride':4 }]]");

		Engine plain_flower("ch09-5.plain_flower", fd, "['custom', {'name':'plain_flower', 'args':{'#act':'LAB'}}]", "{'dump_structure':True}", &macros);
		plain_flower.exec_all("{'epoch_count':5, 'report':1}");

		macros.set_macro("residual_flower",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':16, 'actions':'#act'}], \
				['max', { 'stride':2 }], \
				['custom', { 'name':'rfull', 'args' : {'#cnt':2, '#n' : 16, '#act' : '#act'} }], \
				['custom', { 'name':'rhalf', 'args' : {'#cnt1':1, '#n' : 32, '#act' : '#act'} }], \
				['custom', { 'name':'rhalf', 'args' : {'#cnt1':1, '#n' : 64, '#act' : '#act'} }], \
				['avg', { 'stride':4 }]]");

		Engine residual_flower("ch09-6.residual_flower", fd, "['custom', {'name':'residual_flower', 'args':{'#act':'LAB'}}]", "{'dump_structure':True}", &macros);
		residual_flower.exec_all("{'epoch_count':5, 'report':1}");

		macros.set_macro("bottleneck_flower",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':16, 'actions':'#act'}], \
				['max', { 'ksize':3, 'stride' : 2 }], \
				['custom', { 'name':'bfull', 'args' : {'#cnt':1,'#n1' : 16,'#n4' : 64, '#act' : '#act'} }], \
				['custom', { 'name':'bhalf', 'args' : {'#cnt1':2,'#n1' : 32,'#n4' : 128, '#act' : '#act'} }], \
				['custom', { 'name':'bhalf', 'args' : {'#cnt1':1,'#n1' : 64,'#n4' : 256, '#act' : '#act'} }], \
				['avg', { 'stride':4 }]]");

		Engine bottleneck_flower("ch09-7.bottleneck_flower", fd, "['custom', {'name':'bottleneck_flower', 'args':{'#act':'LAB'}}]", "{'dump_structure':True}", &macros);
		bottleneck_flower.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "automata") == 0) {
		AutomataDataset ad;

		Engine am_1d("ch10-1.am_1d",  ad, "['rnn', {'recur_size':64,  'outseq':False, 'lstm':False}]");
		Engine am_2d("ch10-2.am_2d",  ad, "['rnn', {'recur_size':64,  'outseq':False, 'lstm':True}]");
		Engine am_3d("ch10-3.am_3d",  ad, "['rnn', {'recur_size':64,  'outseq':False, 'lstm':True, 'use_state':True}]");

		am_1d.exec_all("{'epoch_count':5, 'report':1}");
		am_2d.exec_all("{'epoch_count':5, 'report':1}");
		am_3d.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "automata_drop") == 0) {
		AutomataDataset add;

		Engine am_dd("ch10-7.am_dd", add, "[['rnn', {'recur_size':64,  'outseq':False, 'lstm':True, 'use_state':True}], ['dropout', {'keep_prob':0.5}]]");
		Engine am_sd("ch10-8.am_sd", add, "[['rnn', {'recur_size':64,  'outseq':False, 'lstm':True, 'use_state':True}], ['dropout', {'keep_prob':0.5}]]");

		am_dd.exec_all("{'epoch_count':5, 'report':1}");
		am_sd.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "urban_sound") == 0) {
		UrbanSoundDataset usd_10_10("chap11/urban-sound-classification", "urban.1"); // , 10, 20, 10, 10, 100);
		//UrbanSoundDataset usd_10_100("chap11/urban-sound-classification", "urban.2", 10, 20, 10, 20, 100);

		const char* conf_basic = "['rnn', {'recur_size':64, 'outseq':False}]";
		const char* conf_lstm = "['rnn', { 'recur_size':64, 'outseq' : False, 'lstm' : True }]";
		const char* conf_state = "['rnn', { 'recur_size':64, 'outseq' : False, 'lstm' : True, 'use_state' : True }]";

		Engine us_basic_10_10("ch11-1.us_basic_10_10", usd_10_10, conf_basic);
		Engine us_lstm_10_10("ch11-2.us_lstm_10_10", usd_10_10, conf_lstm);
		Engine us_state_10_10("ch11-3.us_state_10_10", usd_10_10, conf_state);

		us_basic_10_10.exec_all("{'epoch_count':5, 'report':1}");
		us_lstm_10_10.exec_all("{'epoch_count':5, 'report':1}");
		us_state_10_10.exec_all("{'epoch_count':5, 'report':1}");

		/*
		Engine us_basic_10_100("ch11-4.us_basic_10_100", usd_10_100, conf_basic);
		Engine us_lstm_10_100("ch11-5.us_lstm_10_100", usd_10_100, conf_lstm);
		Engine us_state_10_100("ch11-6.us_state_10_100", usd_10_100, conf_state);

		us_basic_10_10.exec_all("{'epoch_count':100, 'report':20}");
		us_lstm_10_10.exec_all("{'epoch_count':100, 'report':20}");
		us_state_10_10.exec_all("{'epoch_count':100, 'report':20}");

		us_basic_10_100.exec_all("{'epoch_count':100, 'report':20}");
		us_lstm_10_100.exec_all("{'epoch_count':100, 'report':20}");
		us_state_10_100.exec_all("{'epoch_count':100, 'report':20}");
		*/
	}
	else if (strcmp(mission, "video_shot") == 0) {
		VideoShotDataset vsd("movies", "video_shot.kodell.100.dat", 100, 10, Shape(100,100,3));

		/*
		const char* conf1 = " \
			[['avg',{'stride':30}], \
			 ['conv',{'ksize':3, 'chn':12}], \
			 ['full', { 'width':16 }], \
			 ['rnn', { 'recur_size':8 }]]";

		Engine vsm1("ch12-1.vsm1", vsd, conf1, "{'dump_structure':True}");
		vsm1.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf2 = " \
			[['avg', { 'stride':30 }], \
			 ['conv', { 'ksize':3, 'chn' : 12 }], \
			 ['full', { 'width':16 }], \
			 ['rnn', { 'recur_size':8 }], \
			 ['rnn', { 'recur_size':4 }]]";

		Engine vsm2("ch12-3.vsm2", vsd, conf2);
		vsm2.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf3 = " \
			[['conv', { 'ksize':3, 'chn' : 6 }], \
			 ['max', { 'stride':2 }], \
			 ['conv', { 'ksize':3, 'chn' : 12 }], \
			 ['max', { 'stride':2 }], \
			 ['conv', { 'ksize':3, 'chn' : 24 }], \
			 ['max', { 'stride':2 }], \
			 ['conv', { 'ksize':3, 'chn' : 48 }], \
			 ['avg', { 'stride':5 }], \
			 ['full', { 'width':32 }], \
			 ['rnn', { 'recur_size':8, 'lstm' : True, 'use_state' : True }]]";

		Engine vsm3("ch12-4.vsm3", vsd, conf3, "{'dump_structure':True}");
		vsm3.exec_all("{'epoch_count':5, 'report':1}");
		*/

		MacroPack macros;
		m_create_resnet_macros(macros);

		/*
		macros.set_macro("residual_videoshot",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':64, 'actions':'#act'}], \
				['custom', {'name':'rfull', 'args':{'#cnt':3, '#n':64, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':3, '#n':128, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':5, '#n':256, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':2, '#n':512, '#act':'#act'}}], \
				['avg', {'stride':7}], \
			    ['full', { 'width':32 }], \
			    ['rnn', { 'recur_size':8, 'lstm' : True, 'use_state' : True }]]");
		*/

		macros.set_macro("residual_videoshot",
			"['serial', \
				['conv', {'ksize':7, 'stride':2, 'chn':16, 'actions':'#act'}], \
				['custom', {'name':'rfull', 'args':{'#cnt':2, '#n':16, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':2, '#n':32, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':3, '#n':64, '#act':'#act'}}], \
				['custom', {'name':'rhalf', 'args':{'#cnt1':2, '#n':128, '#act':'#act'}}], \
				['globalavg'], \
			    ['full', { 'width':64 }], \
			    ['rnn', { 'recur_size':32, 'lstm' : True, 'use_state' : True }]]");

		Engine vsm4("ch12-4.vsm4", vsd, "['custom', { 'name':'residual_videoshot', 'args' : {'#act':'LAB'} }]", "{'dump_structure':True}", &macros);
		vsm4.exec_all("{'epoch_count':5, 'batch_size':8, 'report':1}");
		/*
		vsm1.exec_all("{'epoch_count':40, 'report':10}");
		vsm2.exec_all("{'epoch_count':40, 'report':10}");
		vsm3.exec_all("{'epoch_count':40, 'report':10}");
		*/
	}
	else if (strcmp(mission, "autoencoder") == 0) {
		MnistAutoDataset mset_all(1.00f);
		MnistAutoDataset mset_1p(0.10f);

		const char* conf_mlp = "[['full', {'width':128}], ['full', {'width':32}]]";

		Engine mnist_mlp_all("mnist_mlp_all", mset_all, conf_mlp, "{'dump_structure':True}");
		mnist_mlp_all.exec_all("{'epoch_count':5, 'report':1}");

		Engine mnist_mlp_1p("mnist_mlp_1p", mset_1p, conf_mlp);
		mnist_mlp_1p.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_auto_1 = " \
				 {'encoder': [['full', {'width':128}], ['full', {'width':32}]], \
				  'decoder': [['full', {'width':128}], ['full', {'width':784, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		Autoencoder mnist_auto_1("mnist_auto_1", mset_1p, conf_auto_1, "{'optimizer':'adam', 'dump_structure':True}");
		mnist_auto_1.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_auto_sig = " \
				 {'encoder': [['full', {'width':128}], ['full', {'width':32, 'actfunc':'sigmoid'}]], \
				  'decoder': [['full', {'width':128}], ['full', {'width':784, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		Autoencoder mnist_auto_sig("mnist_auto_sig", mset_1p, conf_auto_sig, "{'optimizer':'adam', 'dump_structure':True}");
		mnist_auto_sig.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_auto_2 = " \
				 {'encoder': [['full', {'width':128}], ['full', {'width':32}], ['full', {'width':8}]], \
				  'decoder': [['full', {'width':32}], ['full', {'width':128}], ['full', {'width':784, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		Autoencoder mnist_auto_2("mnist_auto_2", mset_1p, conf_auto_2, "{'optimizer':'adam', 'dump_structure':True}");
		mnist_auto_2.exec_all("{'epoch_count':5, 'report':1}");

		/*
		// 성능 안 나옴, 처리 과정 확인 필요
		const char* conf_auto_cnn = " \
				 {'encoder': [['reshape', {'shape':[28,28,1]}], \
						      ['conv', {'ksize':3, 'chn':4}], \
						      ['max', {'stride':2}], \
						      ['conv', {'ksize':3, 'chn':8}], \
						      ['max', {'stride':2}], \
						      ['conv', {'ksize':3, 'chn':16}], \
						      ['globalavg']], \
				  'decoder': [['expand', {'ratio':[7,7]}], \
						      ['conv', {'ksize':3, 'chn':8}], \
						      ['deconv', {'ksize':3, 'stride':2, 'chn':8}], \
						      ['conv', {'ksize':3, 'chn':4}], \
						      ['deconv', {'ksize':3, 'stride':2, 'chn':4}], \
						      ['conv', {'ksize':3, 'chn':1, 'actfunc':'tanh'}], \
						      ['reshape', {'shape':[784]}]], \
				  'supervised': []}";

		Autoencoder mnist_auto_cnn("mnist_auto_cnn", mset_1p, conf_auto_cnn, "{'optimizer':'adam', 'dump_structure':True}");
		mnist_auto_cnn.exec_all("{'epoch_count':5, 'report':1}");
		*/
	}
	else if (strcmp(mission, "autoencoder_hash") == 0) {
		MnistAutoDataset mset_all(1.00f);

		const char* conf_auto_1 = " \
				 {'encoder': [['full', {'width':128}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':32, 'actfunc':'sigmoid'}]], \
				  'decoder': [['full', {'width':128}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':784, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		AutoencoderHash mnist_hash_1("mnist_hash_1", mset_all, conf_auto_1);
		mnist_hash_1.exec_autoencode("{'epoch_count':5, 'report':1}");
		mnist_hash_1.save_param();
		mnist_hash_1.semantic_hasing_index();
		mnist_hash_1.semantic_hasing_search();
	}
	else if (strcmp(mission, "autoencoder_cifar") == 0) {
		Cifar10AutoDataset cset_all(1.00f);
		Cifar10AutoDataset cset_1p(0.10f);

		const char* conf_mlp = "[['full', {'width':512}], ['full', {'width':128}], ['full', {'width':32}]]";

		Engine cifar_mlp_all("cifar_mlp_all", cset_all, conf_mlp, "{'dump_structure':True}");
		cifar_mlp_all.exec_all("{'epoch_count':5, 'report':1}");

		Engine cifar_mlp_1p("cifar_mlp_1p", cset_1p, conf_mlp);
		cifar_mlp_1p.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_auto_1 = " \
				 {'encoder': [['full', {'width':512}], ['full', {'width':128}], ['full', {'width':32}]], \
				  'decoder': [['full', {'width':128}], ['full', {'width':512}], ['full', {'width':3072, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		Autoencoder cifar_auto_1("cifar_auto_1", cset_1p, conf_auto_1, "{'optimizer':'adam', 'dump_structure':True}");
		cifar_auto_1.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_auto_sig = " \
				 {'encoder': [['full', {'width':512}], ['full', {'width':128}], ['full', {'width':32, 'actfunc':'sigmoid'}]], \
				  'decoder': [['full', {'width':128}], ['full', {'width':512}], ['full', {'width':3072, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		Autoencoder cifar_auto_sig("cifar_auto_sig", cset_1p, conf_auto_sig, "{'optimizer':'adam', 'dump_structure':True}");
		cifar_auto_sig.exec_all("{'epoch_count':5, 'report':1}");

		/*
		// 성능 안 나옴, 처리 과정 확인 필요
		const char* conf_auto_cnn = " \
				 {'encoder': [['reshape', {'shape':[28,28,1]}], \
						      ['conv', {'ksize':3, 'chn':4}], \
						      ['max', {'stride':2}], \
						      ['conv', {'ksize':3, 'chn':8}], \
						      ['max', {'stride':2}], \
						      ['conv', {'ksize':3, 'chn':16}], \
						      ['globalavg']], \
				  'decoder': [['expand', {'ratio':[7,7]}], \
						      ['conv', {'ksize':3, 'chn':8}], \
						      ['deconv', {'ksize':3, 'stride':2, 'chn':8}], \
						      ['conv', {'ksize':3, 'chn':4}], \
						      ['deconv', {'ksize':3, 'stride':2, 'chn':4}], \
						      ['conv', {'ksize':3, 'chn':1, 'actfunc':'tanh'}], \
						      ['reshape', {'shape':[784]}]], \
				  'supervised': []}";

		Autoencoder cifar_auto_cnn("cifar_auto_cnn", cset_1p, conf_auto_cnn, "{'optimizer':'adam', 'dump_structure':True}");
		cifar_auto_cnn.exec_all("{'epoch_count':5, 'report':1}");
		*/
	}
	else if (strcmp(mission, "autoencoder_hash_cifar") == 0) {
		Cifar10AutoDataset cset_all(1.00f);

		const char* conf_auto_1 = " \
				 {'encoder': [['full', {'width':512}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':128}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':32, 'actfunc':'sigmoid'}]], \
				  'decoder': [['full', {'width':128}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':512}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':3072, 'actfunc':'tanh'}]], \
				  'supervised': []}";

		AutoencoderHash cifar_hash_1("cifar_hash_1", cset_all, conf_auto_1);

		for (int n = 0; n < 20; n++) {
			logger.Print("***************** CIFAR LOOP %d ***************", n);
			cifar_hash_1.exec_autoencode("{'epoch_count':5, 'report':1, 'learning_rate':0.001}");
			cifar_hash_1.semantic_hasing_index();
			cifar_hash_1.semantic_hasing_search();
		}

		for (int n = 0; n < 20; n++) {
			logger.Print("***************** CIFAR LOOP %d ***************", n + 20);
			cifar_hash_1.exec_autoencode("{'epoch_count':5, 'report':1, 'learning_rate':0.0005}");
			cifar_hash_1.semantic_hasing_index();
			cifar_hash_1.semantic_hasing_search();
		}

		for (int n = 0; n < 20; n++) {
			logger.Print("***************** CIFAR LOOP %d ***************", n + 40);
			cifar_hash_1.exec_autoencode("{'epoch_count':5, 'report':1, 'learning_rate':0.00025}");
			cifar_hash_1.semantic_hasing_index();
			cifar_hash_1.semantic_hasing_search();
		}

		for (int n = 0; n < 20; n++) {
			logger.Print("***************** CIFAR LOOP %d ***************", n + 60);
			cifar_hash_1.exec_autoencode("{'epoch_count':5, 'report':1, 'learning_rate':0.000125}");
			cifar_hash_1.semantic_hasing_index();
			cifar_hash_1.semantic_hasing_search();
		}

		for (int n = 0; n < 20; n++) {
			logger.Print("***************** CIFAR LOOP %d ***************", n + 80);
			cifar_hash_1.exec_autoencode("{'epoch_count':5, 'report':1, 'learning_rate':0.0001}");
			cifar_hash_1.semantic_hasing_index();
			cifar_hash_1.semantic_hasing_search();
		}
	}
	else if (strcmp(mission, "encoder_decoder_eng") == 0) {
		MnistEngDataset mnist_eng;

		const char* conf_eng0 = " \
				 {'encoder': [['full', {'width':10}]], \
				  'decoder': [['rnn', {'recur_size':32, 'outseq':True, 'timesteps':6}], \
							  ['full', {'width':27, 'actfunc':'none'}]]}";

		EncoderDecoder encdec_eng0("ch14-1.encdec_eng0", mnist_eng, conf_eng0, "{'use_output_layer':False, 'dump_structure':True}");
		encdec_eng0.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_eng1 = " \
				 {'encoder': [['full', {'width':128}], ['full', {'width':32}]], \
				  'decoder': [['rnn', {'recur_size':32, 'outseq':True, 'timesteps':6}], \
							  ['full', {'width':27, 'actfunc':'none'}]]}";

		EncoderDecoder encdec_eng1("ch14-1.encdec_eng1", mnist_eng, conf_eng1, "{'use_output_layer':False, 'dump_structure':True}");
		encdec_eng1.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_eng2 = " \
				 {'encoder': [['full', {'width':128}], ['batch_normal'], ['full', {'width':32}]], \
				  'decoder': [['rnn', {'recur_size':32, 'outseq':True, 'timesteps':6}], \
							  ['full', {'width':27, 'actfunc':'none'}]]}";

		EncoderDecoder encdec_eng2("ch14-1.encdec_eng2", mnist_eng, conf_eng1, "{'use_output_layer':False, 'dump_structure':True}");
		encdec_eng2.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_eng3 = " \
				 {'encoder': [['full', {'width':128}], ['dropout', { 'keep_prob':0.9 }], ['full', {'width':32}]], \
				  'decoder': [['rnn', {'recur_size':32, 'outseq':True, 'timesteps':6}], \
							  ['full', {'width':27, 'actfunc':'none'}]]}";

		EncoderDecoder encdec_eng3("ch14-1.encdec_eng3", mnist_eng, conf_eng3, "{'use_output_layer':False, 'dump_structure':True}");
		encdec_eng3.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "encoder_decoder_kor") == 0) {
		MnistKorDataset mnist_kor2(2);

		// 단어별 정확도, 철자별 정확도 구분 집계 필요
		const char* conf_kor2 = " \
			{'encoder': [['full', {'width':32, 'name':'digits'}], \
						 ['batch_normal'], \
						 ['rnn', { 'recur_size':10, 'outseq' : False }]], \
			 'decoder': [['rnn', { 'recur_size':32, 'outseq' : True, 'timesteps' : 4 }], \
						 ['batch_normal']] }";

		EncoderDecoder encdec_kor2("encdec_kor2", mnist_kor2, conf_kor2, "{'dump_structure':True}");
		encdec_kor2.exec_all("{'epoch_count':5, 'report':1}");

		MnistAutoDataset mset_all(1.00f);

		const char* conf_mlp = "[['full', {'width':32, 'name':'digit'}]]";

		Engine mnist_mlp_all("mnist_mlp_all", mset_all, conf_mlp, "{'dump_structure':True}");
		mnist_mlp_all.exec_all("{'epoch_count':5, 'report':1}");

		EncoderDecoder encdec_kor2_pretrain("encdec_kor2_pretrain", mnist_kor2, conf_kor2, "{'dump_structure':True}");
		encdec_kor2_pretrain.copy_params(mnist_mlp_all, "{'digits':'digit'}");
		encdec_kor2_pretrain.exec_all("{'epoch_count':5, 'report':1}");

		/*
		const char* conf_kor2_nobn = " \
			{'encoder': [['full', {'width':32}], \
						 ['rnn', { 'recur_size':10, 'outseq' : False }]], \
			 'decoder': [['rnn', { 'recur_size':32, 'outseq' : True, 'timesteps' : 4 }]] }";

		EncoderDecoder encdec_kor2_nobn("encdec_kor2_nobn", mnist_kor2, conf_kor2_nobn, "{'dump_structure':True}");
		encdec_kor2_nobn.exec_all("{'epoch_count':5, 'report':1}");

		const char* conf_kor2_drop = " \
			{'encoder': [['full', {'width':32}], \
						 ['dropout', { 'keep_prob':0.8 }], \
						 ['rnn', { 'recur_size':10, 'outseq' : False }]], \
			 'decoder': [['rnn', { 'recur_size':32, 'outseq' : True, 'timesteps' : 4 }], \
						 ['dropout', { 'keep_prob':0.9 }]] }";

		EncoderDecoder encdec_kor2_drop("encdec_kor2_drop", mnist_kor2, conf_kor2_drop, "{'dump_structure':True}");
		encdec_kor2_drop.exec_all("{'epoch_count':5, 'report':1}");
		*/

		MnistKorDataset mnist_kor3(3);

		const char* conf_kor3 = " \
			{'encoder': [['full', {'width':32}], \
						 ['batch_normal'], \
						 ['rnn', { 'recur_size':10, 'outseq' : False }]], \
			 'decoder': [['rnn', { 'recur_size':32, 'outseq' : True, 'timesteps' : 6 }], \
						 ['batch_normal']] }";

		EncoderDecoder encdec_kor3("encdec_kor3", mnist_kor3, conf_kor3, "{'dump_structure':True}");
		encdec_kor3.exec_all("{'epoch_count':5, 'report':1}");

		MnistKorDataset mnist_kor4(4);

		const char* conf_kor4 = " \
			{'encoder': [['full', {'width':32}], \
						 ['batch_normal'], \
						 ['rnn', { 'recur_size':10, 'outseq' : False }]], \
			 'decoder': [['rnn', { 'recur_size':32, 'outseq' : True, 'timesteps' : 8 }], \
						 ['batch_normal']] }";

		EncoderDecoder encdec_kor4("encdec_kor4", mnist_kor4, conf_kor4, "{'dump_structure':True}");
		encdec_kor4.exec_all("{'epoch_count':5, 'report':1}");
	}
	else if (strcmp(mission, "gan") == 0) {
		// 성능 안 나옴, 처리 과정 확인 필요
		GanDatasetPicture dset_pic_gogh("gogh", "gogh.jpg", "chap15/gan.kodell.gogh.dat");
		GanDatasetPicture dset_pic_jungsun("jungsun", "jungsun.jpg", "chap15/gan.kodell.jungsun.dat");

		const char* conf_pic = " \
			{'generator':   [['full', {'width':64}], \
							 ['full', {'width':3072, 'actfunc':'sigmoid'}]], \
			 'discriminor': [['full', {'width':64}], \
							 ['full', {'width':1, 'actfunc':'none'}]]}";

		Gan gan_pic_gogh("ch15-1.gan_pic_gogh", dset_pic_gogh, conf_pic, "{'dump_structure':True}");
		gan_pic_gogh.exec_all("{'epoch_count':100, 'report':20}");

		Gan gan_pic_jungsun("ch15-2.gan_pic_jungsun", dset_pic_jungsun, conf_pic);
		gan_pic_jungsun.exec_all("{'epoch_count':100, 'report':20}");

	}
	else if (strcmp(mission, "gan_mnist") == 0) {
		GanDatasetMnist dset_gan_mnist_full("dset_gan_mnist_full");
		//GanDatasetMnist dset_gan_mnist_68("dset_gan_mnist_68", "68");
		//GanDatasetMnist dset_gan_mnist_8("dset_gan_mnist_8", "8");

		/*
		const char* conf_gan_mnist = " \
			{'generator':   [['full', {'width':256, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}], \
							 ['full', {'width':512, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}], \
							 ['full', {'width':1024, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}], \
							 ['full', {'width':784, 'actfunc':'tanh'}]], \
			 'discriminor': [['full', {'width':1024, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}], \
							 ['full', {'width':512, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}], \
							 ['full', {'width':256, 'actfunc':'leaky_relu', 'leaky_alpha':0.2}], \
							 ['batch_normal', {'momentum':0.8}]]}";
		*/

		const char* conf_gan_mnist = " \
			{'generator':   [['full', {'width':16, 'actfunc':'leaky_relu', 'leaky_alpha':0.1}], \
							 ['batch_normal', {'momentum':0.9}], \
							 ['full', {'width':256, 'actfunc':'leaky_relu', 'leaky_alpha':0.1}], \
							 ['batch_normal', {'momentum':0.9}], \
							 ['full', {'width':784, 'actfunc':'tanh'}]], \
			 'discriminor': [['full', {'width':256, 'actfunc':'leaky_relu', 'leaky_alpha':0.1}], \
							 ['batch_normal', {'momentum':0.9}], \
							 ['full', {'width':16, 'actfunc':'leaky_relu', 'leaky_alpha':0.1}]]}";

		Gan gan_mnist_full("ch15-6.gan_mnist_full", dset_gan_mnist_full, conf_gan_mnist, "{'learning_rate':0.001, 'adam_ro1':0.9, 'dump_structure':True}");
		//gan_mnist_full.exec_all("{'epoch_count':100, 'report':1, 'pre_train_epoch':3, 'epoch_visualize':1, 'learning_rate':0.001, 'adam_ro1':0.9, 'epoch_save':1, 'epoch_show_grad_norm':5}");
		//gan_mnist_full.load_param(40);  // 'grad_clip':1000000.0
		//gan_mnist_full.load_param(120); // lr => 0.0005
		gan_mnist_full.load_param(190);   // lr => 0.00025
		gan_mnist_full.exec_all("{'epoch_count':1000, 'report':1, 'pre_train_epoch':0, 'epoch_visualize':1, 'learning_rate':0.00025, 'adam_ro1':0.9, \
								  'epoch_save':10, 'epoch_show_grad_norm':5, 'grad_clip':1000000.0}");
		/*
		Gan gan_mnist_68("ch15-3.gan_mnist_68", dset_gan_mnist_68, conf_gan_mnist);
		gan_mnist_68.exec_all("{'epoch_count':10, 'report':2}");

		Gan gan_mnist_no_adam = Gan("ch15-4.gan_mnist_no_adam", dset_gan_mnist_8, conf_gan_mnist, "{'optimizer':'sgd'}");
		gan_mnist_no_adam.exec_all("{'epoch_count':10, 'report':2}");

		Gan gan_mnist_adam("ch15-5.gan_mnist_adam", dset_gan_mnist_8, conf_gan_mnist, "{'optimizer':'adam'}");
		gan_mnist_adam.exec_all("{'epoch_count':10, 'report':2}");
		*/

		//Gan gan_mnist_full("ch15-6.gan_mnist_full", dset_gan_mnist_full, conf_gan_mnist, "{'learning_rate':0.0002, 'adam_ro1':0.5, 'dump_structure':True}");
		// 
		//gan_mnist_full.exec_all("{'epoch_count':10, 'report':1, 'pre_train_epoch':0, 'epoch_visualize':1, 'learning_rate':0.0001, 'adam_ro1':0.5}");
		//gan_mnist_full.exec_all("{'epoch_count':10, 'report':1, 'pre_train_epoch':0, 'epoch_visualize':1, 'learning_rate':0.0001, 'adam_ro1':0.5, 'epoch_show_grad_norm':5}");
		//gan_mnist_full.exec_all("{'epoch_count':1000, 'report':1, 'pre_train_epoch':0, 'epoch_visualize':1, 'learning_rate':0.0001, 'adam_ro1':0.5, 'epoch_show_grad_norm':5, 'clip_grad':1000000000000.0}");
		//gan_mnist_full.exec_all("{'epoch_count':10, 'report':1, 'pre_train_epoch':0, 'epoch_visualize':1, 'learning_rate':0.0001, 'adam_ro1':0.5, 'clip_grad':1.0}");

		//gan_mnist_full.load_param(40);
		//gan_mnist_full.exec_all("{'epoch_count':1000, 'report':5, 'pre_train_epoch':0, 'epoch_visualize':10, 'learning_rate':0.00005, 'adam_ro1':0.5, 'epoch_save':10}");
		// optimizer = Adam(0.0002, 0.5) 참고 예제에 나온 설정, 0.5는 무얼까? beta1?
	}
	else if (strcmp(mission, "gan_cifar10") == 0) {
		GanCifar10Dataset dset_cifar10("gan_cifar10");

		/*
		* // 25에포크 이후 판별기에서 자꾸 nan 발생
		const char* conf_mlp = " \
			{'generator':   [['full', {'width':64}], \
							 ['batch_normal'], \
							 ['full', {'width':3072, 'actfunc':'tanh'}], \
							 ['reshape', {'shape':[32,32,3]}]], \
			 'discriminor': [['full', {'width':64}], \
							 ['batch_normal'], \
							 ['full', {'width':1, 'actfunc':'none'}]]}";
		*/

		const char* conf_mlp = " \
			{'generator':   [['full', {'width':16}], \
							 ['full', {'width':256}], \
							 ['full', {'width':3072, 'actfunc':'tanh'}], \
							 ['reshape', {'shape':[32,32,3]}]], \
			 'discriminor': [['full', {'width':256}], \
						     ['full', {'width':16}], \
							 ['full', {'width':1, 'actfunc':'sigmoid'}]]}";

		Gan gan_cifar0("gan_cifar0", dset_cifar10, conf_mlp, "{'seed_size':4, 'dump_structure':True, 'use_output_layer':False}");
		gan_cifar0.exec_all("{'batch_size':10, 'epoch_count':500, 'report':1, 'pre_train_epoch': 3, 'epoch_visualize':5, 'epoch_save':5, 'clip_grad':500.0 }");

		/*
		const char* conf_cifar10_1 = " \
			{'seed_size':   1, \
			 'generator':   [['full', {'width':192}], \
							 ['reshape', {'shape':[2,2,48]}], \
							 ['batch_normal'], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':24}], \
							 ['batch_normal'], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':12}], \
							 ['batch_normal'], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':6}], \
							 ['batch_normal'], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':3, 'actfunc':'tanh'}]], \
			 'discriminor': [['conv', {'ksize':3, 'stride':2, 'chn':6}], \
							 ['batch_normal'], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':12}], \
							 ['batch_normal'], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':24}], \
							 ['batch_normal'], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':48}], \
							 ['batch_normal'], \
							 ['full', {'width':1, 'actfunc':'none'}]]}";

		Gan gan_cifar1("gan_cifar10_1", dset_cifar10, conf_cifar10_1, "{'dump_structure':True}");
		gan_cifar1.exec_all("{'epoch_count':100, 'report':5, 'epoch_visualize':5, 'pre_train_epoch':5, 'learning_rate':0.001}");

		const char* conf_cifar10_2 = " \
			{'seed_size':   1, \
			 'generator':   [['full', {'width':192}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['reshape', {'shape':[2,2,48]}], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':24}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':12}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':6}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['deconv', {'ksize':3, 'stride':2, 'chn':3, 'actfunc':'tanh'}]], \
			 'discriminor': [['conv', {'ksize':3, 'stride':2, 'chn':6}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':12}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':24}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['conv', {'ksize':3, 'stride':2, 'chn':48}], \
							 ['dropout', { 'keep_prob':0.9 }], \
							 ['full', {'width':1, 'actfunc':'none'}]]}";

		Gan gan_cifar_2("gan_cifar10_2", dset_cifar10, conf_cifar10_2, "{'dump_structure':True}");
		gan_cifar_2.exec_all("{'epoch_count':100, 'report':1, 'epoch_visualize':5, 'pre_train_epoch':5, 'learning_rate':0.001}");
		*/
	}
	else if (strcmp(mission, "w2v_cbow14") == 0) {
		PtbCorpus ptb_corpus("ptb_corpus", "ptb.train.cop", "ptb.train.txt");

		const char* conf_w2v = "['embedding', {'vec_size':32, 'name':'embed'}, ['merge', {'method':'mean'}]]";

		Word2VecDataset ptbcbow_14("ptbcbow_14", ptb_corpus, "cbow", 1, 4);
		Engine cbow_14("ex01-1.cbow_14", ptbcbow_14, conf_w2v, "{'use_output_layer':False, 'dump_structure':True}");
		cbow_14.exec_all("{'epoch_count':1, 'batch_size':32, 'report':1}");
		cbow_14.save_named_wvec_param("embed", "cbow_14.ptb_train_adam.dic");

		WordSeqDataset pseq_20("wordseq", ptb_corpus, 20);

		const char* conf_c14 = "[['embed', {'fileload':'cbow_14.ptb_train_adam.dic', 'name':'embed'}], \
							     ['rnn', {'recur_size':32, 'outseq':True}], \
								 ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]";
		Engine wseq_c14("ex02-1.wseq_c14", pseq_20, conf_c14, "{'dump_structure':True, 'use_output_layer':False}");
		wseq_c14.exec_all("{'epoch_count':1, 'batch_size':32, 'report':1}");
		wseq_c14.save_named_wvec_param("embed", "wseq_cbow_14.ptb_train_adam.dic");
	}
	else if (strcmp(mission, "w2v_cbow25") == 0) {
		PtbCorpus ptb_corpus("ptb_corpus", "ptb.short.cop", "ptb.short.txt");

		const char* conf_w2v = "['embedding', {'vec_size':32, 'name':'embed'}, ['merge', {'method':'mean'}]]";

		Word2VecDataset ptbcbow_25("ptbcbow_25", ptb_corpus, "cbow", 2, 5);
		Engine cbow_25("ex01-2.cbow_25", ptbcbow_25, conf_w2v, "{'use_output_layer':False, 'dump_structure':True}");
		cbow_25.visualize(3);
		cbow_25.exec_all("{'epoch_count':5, 'batch_size':100, 'report':1}");
		cbow_25.save_named_wvec_param("embed.win", "cbow_25.ptb_train_adam.dic");
	}
	else if (strcmp(mission, "w2v_skip14") == 0) {
		PtbCorpus ptb_corpus("ptb_corpus", "ptb.short.cop", "ptb.short.txt");

		const char* conf_w2v = "['embedding', {'vec_size':32, 'name':'embed'}, ['merge', {'method':'mean'}]]";

		Word2VecDataset ptbskip_14("ptbskip_14", ptb_corpus, "skip", 1, 4);
		Engine skip_14("ex01-3.skip_14", ptbskip_14, conf_w2v, "{'use_output_layer':False, 'dump_structure':True}");
		skip_14.visualize(3);
		skip_14.exec_all("{'epoch_count':5, 'batch_size':100, 'report':1}");
		skip_14.save_named_wvec_param("embed.win", "skip_14.ptb_train_adam.dic");
	}
	else if (strcmp(mission, "w2v_skip25") == 0) {
		PtbCorpus ptb_corpus("ptb_corpus", "ptb.short.cop", "ptb.short.txt");

		const char* conf_w2v = "['embedding', {'vec_size':32, 'name':'embed'}, ['merge', {'method':'mean'}]]";

		Word2VecDataset ptbskip_25("ptbskip_25", ptb_corpus, "skip", 2, 5);
		Engine skip_25("ex01-4.skip_25", ptbskip_25, conf_w2v, "{'use_output_layer':False, 'dump_structure':True}");
		skip_25.visualize(3);
		skip_25.exec_all("{'epoch_count':5, 'batch_size':100, 'report':1}");
		skip_25.save_named_wvec_param("embed.win", "skip_25.ptb_train_adam.dic");
	}
	else if (strcmp(mission, "next_word") == 0) {
		// 성능 안 나옴, 처리 과정 확인 필요
		// 사전 파라미터 업데이트의 순차적 처리를 병렬화하는 과정에서 문제 발생 가능성 뫂음
		PtbCorpus ptb_corpus("ptb_corpus", "ptb.cop", "ptb.train.txt");

		WordSeqDataset pseq_20("wordseq", ptb_corpus, 20);

		const char* conf_c14 = "[['embed', {'fileload':'cbow_14.ptb_train_adam.dic', 'name':'embed'}], \
							     ['rnn', {'recur_size':32, 'outseq':True}], \
								 ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]";
		Engine wseq_c14("ex02-1.wseq_c14", pseq_20, conf_c14, "{'dump_structure':True, 'use_output_layer':False}");
		wseq_c14.exec_all("{'epoch_count':10, 'batch_size':100, 'report':1}");
		wseq_c14.save_named_wvec_param("embed.w", "wseq_cbow_14.ptb_train_adam.dic");

		/*
		const char* conf_c25 = "[['embed', {'fileload':'cbow_25.ptb_32_adam_100.dic', 'name':'embed'}], \
							    ['rnn', {'recur_size':32, 'outseq':True}], \
								['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]";
		Engine wseq_c25("ex02-2.wseq_c25", pseq_20, conf_c25, "{'dump_structure':True, 'use_output_layer':False}");
		wseq_c25.exec_all("{'epoch_count':10, 'batch_size':100, 'report':1}");
		wseq_c25.save_named_wvec_param("embed.w", "wseq_cbow_25.ptb_32_adam_100.dic");

		const char* conf_s14 = "[['embed', {'fileload':'skip_14.ptb_32_adam_100.dic', 'name':'embed'}], \
							    ['rnn', {'recur_size':32, 'outseq':True}], \
								['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]";
		Engine wseq_s14("ex02-3.wseq_s14", pseq_20, conf_s14, "{'dump_structure':True, 'use_output_layer':False}");
		wseq_s14.exec_all("{'epoch_count':10, 'batch_size':100, 'report':1}");
		wseq_s14.save_named_wvec_param("embed.w", "wseq_skip_14.ptb_32_adam_100.dic");

		const char* conf_s25 = "[['embed', {'fileload':'skip_25.ptb_32_adam_100.dic', 'name':'embed'}], \
							    ['rnn', {'recur_size':32, 'outseq':True}], \
								['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]";
		Engine wseq_s25("ex02-3.wseq_s14", pseq_20, conf_s25, "{'dump_structure':True, 'use_output_layer':False}");
		wseq_s25.exec_all("{'epoch_count':10, 'batch_size':100, 'report':1}");
		wseq_s25.save_named_wvec_param("embed.w", "wseq_skip_25.ptb_32_adam_100.dic");
		*/
	}
	else if (strcmp(mission, "bert_layer_test") == 0) {
		Dict info;

		info["hidden_size"] = 128;		// 1024 for BERT-large
		info["stack_depth"] = 3;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;
		//info["l2_decay"] = 0.01f;
		//info["learning_rate"] = 0.0001f;

		/*
		info["hidden_size"] = 8;		// 1024 for BERT-large
		info["stack_depth"] = 1;		// 24 for BERT - large
		info["attention_heads"] = 2;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 2;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 16;
		*/

		PtbCorpus ptb_corpus("ptb_corpus", "ptb.cop", "ptb.train.txt");
		BertDataset bert_dataset("bert_dataset", ptb_corpus, max_position_embeddings);

		// 두 번째 add 계층과 pass 게층 밑의 seqwrap 계층 삭제 영향에 유의, 특히 둘 다 inseq=False 였음
		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
			     ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}], \
			    ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }], \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("ex03-1.bert_trainer", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");
		//bert.exec_all("{'batch_size':4, 'epoch_count':1, 'temp_batch_count':10, 'batch_report':2, 'valid_count': 7, 'test_batch_size':4, 'report':1}");
		bert.exec_all("{'batch_size':4, 'epoch_count':2, 'batch_report':1, 'valid_count': 3, 'test_batch_size':3, 'report':1}");
	}
	else if (strcmp(mission, "bert_ptb") == 0) {
		Dict info;

		info["hidden_size"] = 128;		// 1024 for BERT-large
		info["stack_depth"] = 3;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;
		
		//info["l2_decay"] = 0.01f;
		//info["learning_rate"] = 0.0001f;

		/*
		info["hidden_size"] = 8;		// 1024 for BERT-large
		info["stack_depth"] = 1;		// 24 for BERT - large
		info["attention_heads"] = 2;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 2;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 16;
		*/

		PtbCorpus ptb_corpus("ptb_corpus", "ptb.cop", "ptb.train.txt");
		BertDataset bert_dataset("bert_dataset", ptb_corpus, max_position_embeddings);

		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
			     ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}], \
			    ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("ex03-1.bert_trainer", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");
		//bert.exec_all("{'batch_size':4, 'epoch_count':1, 'temp_batch_count':10, 'batch_report':2, 'valid_count': 7, 'test_batch_size':4, 'report':1}");
		bert.exec_all("{'batch_size':10, 'epoch_count':10, 'batch_report':10, 'valid_count': 10, 'test_batch_size':5, 'report':1, 'learning_rate':0.0001}");

		//bert.visualize(3);
		//bert.exec_all("{'batch_size':20, 'epoch_count':10, 'batch_report':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'temp_batch_count':1, 'skip_test':True, 'report':0, 'show_cnt':0}");
	}
	else if (strcmp(mission, "bert_ptb_2step") == 0) {
		Dict info;

		info["hidden_size"] = 128;		// 1024 for BERT-large
		info["stack_depth"] = 3;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;
		
		//info["l2_decay"] = 0.01f;
		//info["learning_rate"] = 0.0001f;

		/*
		info["hidden_size"] = 8;		// 1024 for BERT-large
		info["stack_depth"] = 1;		// 24 for BERT - large
		info["attention_heads"] = 2;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 2;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 16;
		*/

		PtbCorpus ptb_corpus("ptb_corpus", "ptb.cop", "ptb.train.txt");
		BertDataset bert_dataset("bert_dataset", ptb_corpus, max_position_embeddings);

		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
			     ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}], \
			    ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("ex03-1.bert_trainer", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");
		//bert.exec_all("{'batch_size':4, 'epoch_count':1, 'temp_batch_count':10, 'batch_report':2, 'valid_count': 7, 'test_batch_size':4, 'report':1}");
		bert.exec_all("{'batch_size':10, 'epoch_count':1, 'batch_report':10, 'valid_count': 10, 'skip_test':True, 'report':1, 'learning_rate':0.001}");
		bert.exec_all("{'batch_size':10, 'epoch_count':10, 'batch_report':10, 'valid_count': 10, 'test_batch_size':5, 'report':1, 'learning_rate':0.0001}");

		//bert.visualize(3);
		//bert.exec_all("{'batch_size':20, 'epoch_count':10, 'batch_report':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'temp_batch_count':1, 'skip_test':True, 'report':0, 'show_cnt':0}");
	}
	else if (strcmp(mission, "bert_ptb_large") == 0) {
		Dict info;

		info["hidden_size"] = 1024;
		info["stack_depth"] = 24;
		info["attention_heads"] = 16;
		info["intermediate_size"] = (int)info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;
		//info["l2_decay"] = 0.01f;
		//info["learning_rate"] = 0.0001f;

		/*
		info["hidden_size"] = 8;		// 1024 for BERT-large
		info["stack_depth"] = 1;		// 24 for BERT - large
		info["attention_heads"] = 2;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 2;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 16;
		*/

		PtbCorpus ptb_corpus("ptb_corpus", "ptb.cop", "ptb.train.txt");
		BertDataset bert_dataset("bert_dataset", ptb_corpus, max_position_embeddings);

		const char* bert_conf = " \
				[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent']}], \
				 ['batch_normal'], \
				 ['serial', {'repeat':%stack_depth%}, \
				  ['add', \
				   ['serial', \
					 ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
					 ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
				  ['batch_normal'], \
				  ['add', \
				   ['serial', {'wrap_axis':0}, \
					['full', {'width':%hidden_size%}], \
					['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
					['full', {'width':%hidden_size%}], \
					['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
				  ['batch_normal']], \
				 ['pass', \
				  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
				  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
				  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
				 ['serial', {'wrap_axis':0}, \
				   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("ex03-1.bert_trainer", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");
		//bert.exec_all("{'batch_size':4, 'epoch_count':1, 'temp_batch_count':10, 'batch_report':2, 'valid_count': 7, 'test_batch_size':4, 'report':1}");
		bert.exec_all("{'batch_size':10, 'epoch_count':10, 'batch_report':1, 'valid_count': 10, 'test_batch_size':5, 'report':1, 'learning_rate':0.0001}");

		//bert.visualize(3);
		//bert.exec_all("{'batch_size':20, 'epoch_count':10, 'batch_report':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'report':1}");
		//bert.exec_all("{'batch_size':10, 'epoch_count':1, 'temp_batch_count':1, 'skip_test':True, 'report':0, 'show_cnt':0}");
	}
	else if (strcmp(mission, "yolo") == 0) {
		MacroPack macros;

		if (mode == "step_debug") {
			YoloCocoDataset ydset("coco", mode);
			ydset.step_debug();
			throw KaiException(KERR_ASSERT);
		}

		// 5e-4, 1e-5 등의 숫자 표현 지원 필요
		macros.set_macro("DBL",
			"['conv', {'ksize':'#ksize', 'chn':'#chn', 'stride':'#stride', \
					   'actions':'LBA', 'actfunc' : 'leaky_relu', 'leaky_alpha' : 0.1, 'l2_decay' : 0.0005, \
					   'batch_norm_decay' : 0.999, 'batch_norm_epsilon' : 0.00001, 'batch_norm_scale' : True}]");

		macros.set_macro("res_unit", \
			"['add', {'x':True}, \
				['serial', {}, \
					['DBL', { 'ksize':1, 'stride' : 1, 'chn' : '#chn' }], \
					['DBL', { 'ksize':3, 'stride' : 1, 'chn' : '#chn*2' }]]]");

		macros.set_macro("res_chain",
			"[['DBL', {'ksize':3, 'chn':'#chn*2', 'stride':2}], \
			  ['serial', { 'repeat':'#num', 'set' : '#set' }, \
					['res_unit', { 'chn':'#chn' }]]]");

		macros.set_macro("yolo_block",
			"[['DBL', {'ksize':1, 'stride':1, 'chn':'#chn'}], \
			  ['DBL', { 'ksize':3, 'stride' : 1, 'chn' : '#chn*2' }], \
			  ['DBL', { 'ksize':1, 'stride' : 1, 'chn' : '#chn' }], \
			  ['DBL', { 'ksize':3, 'stride' : 1, 'chn' : '#chn*2' }], \
			  ['DBL', { 'ksize':1, 'stride' : 1, 'chn' : '#chn' }], \
			  ['pass', { 'set':'#inter' }], \
			  ['DBL', { 'ksize':3, 'stride' : 1, 'chn' : '#chn*2' }]]");

		macros.set_macro("yolo_head",
			"[['yolo_block', {'chn':'#chn', 'inter':'#inter'}], \
			  ['conv', { 'ksize':1, 'actfunc' : 'none', 'chn' : '#cn', 'set' : '#map' }]]");

		macros.set_macro("yolo_merge",
			"['parallel', \
				['pass', { 'get':'#direct' }], \
				['serial', {}, \
					['pass', { 'get':'#up_sample' }], \
					['DBL', { 'ksize':1, 'stride' : 1, 'chn' : '#chn' }], \
					['expand', { 'ratio':2 }]]]");

		macros.set_macro("yolo_v3",
			"[['DBL', {'ksize':3, 'chn':32, 'stride':1}], \
			  ['res_chain', { 'num':1, 'chn' : 32, 'set' : '' }], \
			  ['res_chain', { 'num':2, 'chn' : 64, 'set' : ''}], \
			  ['res_chain', { 'num':8, 'chn' : 128, 'set' : 'route_1' }], \
			  ['res_chain', { 'num':8, 'chn' : 256, 'set' : 'route_2' }], \
			  ['res_chain', { 'num':4, 'chn' : 512, 'set' : '' }], \
			  ['yolo_head', { 'chn':512, 'cn' : '#vec_size', 'inter' : 'route_3', 'map' : 'feature_map_1' }], \
			  ['yolo_merge', { 'chn':256, 'direct' : 'route_2', 'up_sample' : 'route_3' }], \
			  ['yolo_head', { 'chn':256, 'cn' : '#vec_size', 'inter' : 'inter_1', 'map' : 'feature_map_2' }], \
			  ['yolo_merge', { 'chn':128, 'direct' : 'route_1', 'up_sample' : 'inter_1' }], \
			  ['yolo_head', { 'chn':128, 'cn' : '#vec_size', 'inter' : '', 'map' : 'feature_map_3' }]]");

		string datetime = (mode == "cont") ? "20201124-145513" : "";

		YoloCocoDataset ydset("coco", mode, datetime);
		
		string yolo_conf = "['yolo_v3', { '#vec_size': %vec_size% }]"; 

		Dict info;
		info["vec_size"] = ydset.nvec_ancs;

		string conf = Util::externd_conf(yolo_conf, info);

		// l2_decay를 0.001에서 0.01로 수정
		
		Engine yolo("yolo", ydset, conf.c_str(), "{'use_output_layer': False, 'dump_structure': True, 'l2_decay':0.01}", &macros);

		if (mode == "map_test") {
			yolo.load_param("20201001/yolo-epoch-0000-batch-00000003_0.pmk");
			yolo.visualize(10);
			throw KaiException(KERR_ASSERT);
		}

		string options = "{";

		options += "'batch_size':8, 'epoch_count' : 10, 'report' : 1";
		options += ", 'batch_report' : 10, 'valid_count' : 10, 'in_batch_valid' : 500";
		options += ", 'in_batch_save' : 10, 'in_batch_visualize' : 400, 'test_count' : 256, 'test_batch_size' : 8, 'learning_rate' : 0.0001";

		/*
		if (mode == "cont") {
			yolo.load_param("keep/ex04-1.yolo-epoch-0-batch-1540_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 1540";
		}

		if (mode == "cont") {
			yolo.load_param("keep/ex04-1.yolo-epoch-0-batch-1960_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 1960";
		}

		if (mode == "cont") {
			yolo.load_param("keep/ex04-1.yolo-epoch-0-batch-2640.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 2640";
		}

		if (mode == "cont") {
			yolo.load_param("keep/yolo-epoch-0000-batch-00000090_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 90, 'acc_time': 11334";
		}

		if (mode == "cont") {
			yolo.load_param("keep/yolo-epoch-0000-batch-00000990_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 0990, 'acc_time': 124830";
		}

		if (mode == "cont") {
			yolo.load_param("keep/yolo-epoch-0000-batch-00002310_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 2310, 'acc_time': 293803";
		}

		if (mode == "cont") {
			yolo.load_param("keep/yolo-epoch-0000-batch-00000210_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 210, 'acc_time': 26904";
		}
		*/

		if (mode == "cont") {
			yolo.load_param("keep/yolo-epoch-0000-batch-00000090_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 90, 'acc_time': 7963"; 
			logger.change_to_cont_file("20201124", "cuda_yolo_mission_yolo20201124-145513.log");
		}

		options += ", 'in_batch_valid' : 20, 'in_batch_save' : 10, 'in_batch_visualize' : 91";
		//options += ", 'in_batch_valid' : 20, 'in_batch_save' : 1, 'batch_report' : 1, 'in_batch_visualize' : 100";
		options += "}";

		yolo.exec_all(options.c_str());

		//yolo.exec_all("{'epoch_count':1, 'report':1, 'debug_bypass_neuralnet':True}");
		//yolo.exec_all("{'epoch_count':10, 'report':2}");
	}
	else if (strcmp(mission, "news") == 0) {
		NewsReformer new_reformer;
		//new_reformer.exec_all();
		new_reformer.exec_create_dict();
		new_reformer.exec_replace_words(false);
	}
	else if (strcmp(mission, "bert_korean_news") == 0) {
		Dict info;

		info["hidden_size"] = 128;		// 1024 for BERT-large
		info["stack_depth"] = 3;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;

		KoreanNewsCorpus korean_news("korean_news_corpus", "v0.1");
		BertDataset bert_dataset("bert_dataset", korean_news, max_position_embeddings);

		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent', 'tag']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}]]], \
			  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}]]], \
			  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("bert_korean_news_dropout_after_add", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");

		string options = "{";
		
		options += "'batch_size':10, 'epoch_count' : 10, 'batch_report' : 10, 'valid_count' : 10, 'in_batch_valid' : 500";
		options += ", 'in_batch_save' : 100, 'in_batch_visualize' : 600, 'test_count' : 256, 'test_batch_size' : 8, 'report' : 1, 'learning_rate' : 0.0001";

		if (mode == "cont") {
			bert.load_param("keep/bert_korean_news_dropout_after_add-epoch-0000-batch-00611600_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 611600, 'acc_time': 2291758"; 
		}

		options += "}";

		bert.exec_all(options.c_str());
	}
	else if (strcmp(mission, "bert_mid_korean_news") == 0) {
		Dict info;

		info["hidden_size"] = 256;		// 1024 for BERT-large
		info["stack_depth"] = 6;		// 24 for BERT - large
		info["attention_heads"] = 4;	// 16 for BERT - large
		info["intermediate_size"] = (int)info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;

		KoreanNewsCorpus korean_news("korean_news_corpus", "v0.1");
		BertDataset bert_dataset("bert_dataset", korean_news, max_position_embeddings);

		const char* bert_conf = " \
				[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent', 'tag']}], \
				 ['batch_normal'], \
				 ['serial', {'repeat':%stack_depth%}, \
				  ['add', \
				   ['serial', \
					 ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}]]], \
				  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
				  ['batch_normal'], \
				  ['add', \
				   ['serial', {'wrap_axis':0}, \
					['full', {'width':%hidden_size%}], \
					['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
					['full', {'width':%hidden_size%}]]], \
				  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
				  ['batch_normal']], \
				 ['pass', \
				  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
				  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
				  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
				 ['serial', {'wrap_axis':0}, \
				   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("bert_mid_korean_news", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");

		string options = "{";

		options += "'batch_size':10, 'epoch_count' : 10, 'batch_report' : 10, 'valid_count' : 10, 'in_batch_valid' : 500";
		options += ", 'in_batch_save' : 100, 'in_batch_visualize' : 600, 'test_count' : 256, 'test_batch_size' : 8, 'report' : 1, 'learning_rate' : 0.0001";

		options += "}";

		bert.exec_all(options.c_str());
	}
	else if (strcmp(mission, "bert_mid2_korean_news") == 0) {
		Dict info;

		info["hidden_size"] = 256;		// 1024 for BERT-large
		info["stack_depth"] = 8;		// 24 for BERT - large
		info["attention_heads"] = 8;	// 16 for BERT - large
		info["intermediate_size"] = (int)info["hidden_size"] * 8;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;

		KoreanNewsCorpus korean_news("korean_news_corpus", "v0.1");
		BertDataset bert_dataset("bert_dataset", korean_news, max_position_embeddings);

		const char* bert_conf = " \
					[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent', 'tag']}], \
					 ['batch_normal'], \
					 ['serial', {'repeat':%stack_depth%}, \
					  ['add', \
					   ['serial', \
						 ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}]]], \
					  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
					  ['batch_normal'], \
					  ['add', \
					   ['serial', {'wrap_axis':0}, \
						['full', {'width':%hidden_size%}], \
						['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
						['full', {'width':%hidden_size%}]]], \
					  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
					  ['batch_normal']], \
					 ['pass', \
					  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
					  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
					  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
					 ['serial', {'wrap_axis':0}, \
					   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("bert_mid2_korean_news", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");

		string options = "{";

		options += "'batch_size':10, 'epoch_count' : 10, 'batch_report' : 10, 'valid_count' : 10, 'in_batch_valid' : 500";
		options += ", 'in_batch_save' : 100, 'in_batch_visualize' : 600, 'test_count' : 256, 'test_batch_size' : 8, 'report' : 1, 'learning_rate' : 0.0001";

		options += "}";

		bert.exec_all(options.c_str());
	}
	else if (strcmp(mission, "bert_large_korean_news") == 0) {
		Dict info;

		info["hidden_size"] = 1024;		// 1024 for BERT-large
		info["stack_depth"] = 24;		// 24 for BERT - large
		info["attention_heads"] = 16;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 4;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 512;

		//info["l2_decay"] = 0.01f;
		//info["learning_rate"] = 0.0001f;

		/*
		info["hidden_size"] = 8;		// 1024 for BERT-large
		info["stack_depth"] = 1;		// 24 for BERT - large
		info["attention_heads"] = 2;	// 16 for BERT - large
		info["intermediate_size"] = (int) info["hidden_size"] * 2;
		info["hidden_dropout_ratio"] = 0.1f;
		info["attention_dropout_ratio"] = 0.1f;

		int max_position_embeddings = 16;
		*/

		KoreanNewsCorpus korean_news("korean_news_corpus", "v0.1");
		BertDataset bert_dataset("bert_dataset", korean_news, max_position_embeddings);

		/*
		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent', 'tag']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}], \
			     ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}], \
			    ['dropout', {'drop_ratio':%hidden_dropout_ratio%}]]], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";
		*/

		// for cuda_dropout_after_add
		const char* bert_conf = " \
			[['embed', {'vec_size':%hidden_size%, 'voc_size':'dataset.voc_size', 'plugin':['pos', 'sent', 'tag']}], \
			 ['batch_normal'], \
			 ['serial', {'repeat':%stack_depth%}, \
			  ['add', \
			   ['serial', \
			     ['attention', {'multi-head':True, 'attention_heads':%attention_heads%, 'dropout':%attention_dropout_ratio%}]]], \
			  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
			  ['batch_normal'], \
			  ['add', \
			   ['serial', {'wrap_axis':0}, \
			    ['full', {'width':%hidden_size%}], \
			    ['full', {'width':%intermediate_size%, 'actfunc':'gelu'}], \
			    ['full', {'width':%hidden_size%}]]], \
			  ['dropout', {'drop_ratio':%hidden_dropout_ratio%}], \
			  ['batch_normal']], \
			 ['pass', \
			  ['extract', { 'filter': [{'axis':1,'index' : 0}] }] , \
			  ['full', {'width':%hidden_size%, 'actfunc':'tanh'}], \
			  ['full', { 'width':2, 'actfunc' : 'none', 'set':'is_next_sent'}]], \
			 ['serial', {'wrap_axis':0}, \
			   ['full', {'width':'dataset.voc_size', 'actfunc':'none', 'bias':False}]]]";

		string bert_conf_ext = Util::externd_conf(bert_conf, info);

		Engine bert("bert_large_korean_news", bert_dataset, bert_conf_ext.c_str(), "{'l2_decay':0.01, 'use_output_layer':False, 'dump_structure':True}");

		string options = "{";
		
		options += "'batch_size':10, 'epoch_count' : 10, 'batch_report' : 10, 'valid_count' : 10, 'in_batch_valid' : 500";
		options += ", 'in_batch_save' : 100, 'in_batch_visualize' : 600, 'test_count' : 256, 'test_batch_size' : 8, 'report' : 1, 'learning_rate' : 0.0001";

		if (mode == "cont") {
			bert.load_param("keep/bert_large_korean_news-epoch-0000-batch-00037600_0.pmk");
			options += ", 'start_epoch' : 0, 'start_batch' : 37600, 'acc_time': 3100160"; 
			logger.change_to_cont_file("20201012", "cuda_yolo_mission_bert_large_korean_news20201012-103637.log");
		}

		options += "}";

		bert.exec_all(options.c_str());
	}
	else if (strcmp(mission, "gan_book") == 0) {
		//Random::seed(1234);

		GanDatasetMnist dset_gan_mnist_68("dset_gan_mnist_68", "68");
		GanDatasetMnist dset_gan_mnist_full("dset_gan_mnist_full");
		GanDatasetMnist dset_gan_mnist_8("dset_gan_mnist_8", "8");

		string desc = dset_gan_mnist_8.description();
		logger.Print("dataset: %s", desc.c_str());

		const char* conf_gan_mnist = " \
			{'seed_shape':  [16], \
			 'generator':   [['full', {'width':64}], \
							 ['full', {'width':784, 'actfunc':'tanh'}]], \
			 'discriminor': [['conv',{ 'inshape':[28,28,1], 'ksize':3, 'chn':4 }], \
							 ['conv',{ 'ksize':3, 'chn':4 }], \
							 ['max', { 'stride':2 }], \
							 ['conv',{ 'ksize':3, 'chn':8 }], \
							 ['conv',{ 'ksize':3, 'chn':8 }], \
							 ['max', { 'stride':2 }], \
							 ['full', {'width':1, 'actfunc':'none'}]]}";

		Gan gan_mnist_adam("ch15-5.gan_mnist_adam", dset_gan_mnist_8, conf_gan_mnist, "{'optimizer':'adam', 'dump_structure':True}");
		gan_mnist_adam.exec_all("{'epoch_count':100, 'report':2, 'learning_rate':0.001, 'epoch_visualize':2}");

		throw KaiException(KERR_ASSERT);

		Gan gan_mnist_68("ch15-3.gan_mnist_68", dset_gan_mnist_68, conf_gan_mnist);
		gan_mnist_68.exec_all("{'epoch_count':10, 'report':2}");

		Gan gan_mnist_no_adam = Gan("ch15-4.gan_mnist_no_adam", dset_gan_mnist_8, conf_gan_mnist, "{'optimizer':'sgd'}");
		gan_mnist_no_adam.exec_all("{'epoch_count':10, 'report':2}");

		Gan gan_mnist_full("ch15-6.gan_mnist_full", dset_gan_mnist_full, conf_gan_mnist);
		gan_mnist_full.exec_all("{'epoch_count':10, 'report':2}");

		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));
		Engine fm1("gan_book_ch01_1", fd,
			"[['conv',{ 'ksize':3, 'chn' : 6 }], \
			  ['conv', { 'ksize':3, 'chn' : 6 }], \
			  ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12 }], \
			  ['conv', { 'ksize':3, 'chn' : 12 }], \
			  ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 }], \
			  ['max', { 'stride':2 }]]", \
			"{'dump_structure': True}");
		//fm1.exec_all("{'batch_size':10, 'epoch_count':1, 'temp_batch_count':10, 'skip_test':True, 'report':0, 'show_cnt':0}");
		fm1.exec_all("{'epoch_count':10, 'epoch_count':20, 'report':2, 'learning_rate' : 0.001}");

		/*
		FlowerDataset fd("chap05/flowers", "flowerkodell.96_2d.dat", Shape(96, 96), Shape(96, 96, 3));
		Engine fm1("ch07-1.flowers_cnn_model_1", fd,
			"[['conv',{ 'ksize':5, 'chn' : 6 }], ['max', { 'stride':4 }], \
			 ['conv', { 'ksize':3, 'chn' : 12 }], ['avg', { 'stride':2 }]]",
			"{'dump_structure': True}");
		//fm1.exec_all("{'batch_size':10, 'epoch_count':1, 'temp_batch_count':10, 'skip_test':True, 'report':0, 'show_cnt':0}");
		fm1.exec_all("{'epoch_count':10, 'report':2}");
		Engine fm2("ch07-2.flowers_cnn_model_1", fd,
			"[['conv', { 'ksize':3, 'chn' : 6 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 12 }], ['max', { 'stride':2 }], \
			  ['conv', { 'ksize':3, 'chn' : 24 }], ['avg', { 'stride':3 }]]",
			"{'dump_structure': True}");
		fm2.exec_all("{'epoch_count':10, 'report':2}");
		*/
	}
	else if (strcmp(mission, "aaa") == 0) {
	/*
	*/
	}
	else {
		logger.Print("execute %s mission : not implemented yet...", mission);
	}

	CudaConn::GarbageCheck();
}

Samples::Samples() {
}

const char** Samples::get_all_missions(int* pnCount) {
	*pnCount = sizeof(m_ppSamples) / sizeof(m_ppSamples[0]);
	return m_ppSamples;
}

int Samples::seek_mission_index(const char* mission) {
	int size = sizeof(m_ppSamples) / sizeof(m_ppSamples[0]);
	for (int n = 0; n < size; n++) {
		if (strcmp(mission, m_ppSamples[n]) == 0) return n;
	}
	return -1;
}

const char* Samples::get_nth_mission(int nIndex) {
	return m_ppSamples[nIndex];
}


void Samples::m_create_resnet_macros(MacroPack& macros) {
	macros.set_macro("p24",
		"['serial', \
			['serial', {'repeat':'#repeat'}, ['conv', {'ksize':3, 'chn':'#chn'}]], \
			['max', {'stride':2}]]");

	macros.set_macro("pn",
		"['serial', \
			['conv', {'ksize':3, 'stride':2, 'chn':'#n', 'actions':'#act'}], \
			['serial', {'repeat':'#cnt1'}, \
					 ['conv', {'ksize':3, 'chn':'#n', 'actions':'#act'}]]]");

	macros.set_macro("rf",
		"['add', {'x':True}, \
			['serial', ['conv', {'ksize':3, 'chn':'#n', 'actions':'#act'}], \
					   ['conv', {'ksize':3, 'chn':'#n', 'actions':'#act'}]]]");

	macros.set_macro("rh",
		"['add', {'x':True, 'stride':2}, \
			['serial', ['conv', {'ksize':3, 'stride':2, 'chn':'#n', 'actions':'#act'}], \
					   ['conv', {'ksize':3, 'chn':'#n', 'actions':'#act'}]]]");

	macros.set_macro("rfull",
		"['serial', \
			['serial', {'repeat':'#cnt'}, \
					 ['custom', {'name':'rf', 'args':{'#n':'#n', '#act':'#act'}}]]]");

	macros.set_macro("rhalf",
		"['serial', \
			['custom', {'name':'rh', 'args':{'#n':'#n', '#act':'#act'}}], \
			['serial', {'repeat':'#cnt1'}, \
					 ['custom', {'name':'rf', 'args':{'#n':'#n', '#act':'#act'}}]]]");

	macros.set_macro("bf",
		"['add', {'x':True}, \
			['serial', \
				['conv', {'ksize':1, 'chn':'#n1', 'actions':'#act'}], \
				['conv', {'ksize':3, 'chn':'#n1', 'actions':'#act'}], \
				['conv', {'ksize':1, 'chn':'#n4', 'actions':'#act'}]]]");

	macros.set_macro("bh",
		"['add', {'x':True, 'stride':2}, \
			['serial', \
				['conv', {'ksize':1, 'stride':2, 'chn':'#n1', 'actions':'#act'}], \
				['conv', {'ksize':3, 'chn':'#n1', 'actions':'#act'}], \
				['conv', {'ksize':1, 'chn':'#n4', 'actions':'#act'}]]]");

	macros.set_macro("bfull",
		"['serial', {'repeat':'#cnt'}, ['custom', {'name':'bf', 'args':{'#n1':'#n1', '#n4':'#n4', '#act':'#act'}}]]");

	macros.set_macro("bhalf",
		"['serial', \
			['custom', {'name':'bh', 'args':{'#n1':'#n1', '#n4':'#n4', '#act':'#act'}}], \
			['serial', {'repeat':'#cnt1'}, ['custom', {'name':'bf', 'args':{'#n1':'#n1', '#n4':'#n4', '#act':'#act'}}]]]");
}
