/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/common.h"
#include "data_channel.h"
#include "value.h"
#include "util.h"
#include "host_math.h"
#include "shape.h"
#include "idx.h"

#include <stdlib.h>  
#include <stdio.h>
#include <string>

using namespace std;

class Engine;

class Dataset {
public:
    Dataset(string name, string mode, bool x_seq = false, bool y_seq = false, string datetime="");
    virtual ~Dataset();

    virtual string description();

    virtual int64 train_count();
    virtual int64 test_count();
    virtual int64 validate_count();

    virtual void open_data_channel_all(enum data_channel channel);  // 전체 데이터를 하나의 미니배치로 제공, 데이터 뒤섞기 없음
    virtual void open_data_channel_once(enum data_channel channel, int64 data_count);  // 데이터 뒤섞기 후에 data_count 크기의 미니배치 하나만 제공
    virtual void open_data_channel_repeat(enum data_channel channel, int64 batch_size, int64 max_stack=3);  // (데이터 뒤섞기 + data_count 크기의 미니배치 제공) 기능을 반복해 수행, max_stack 이내의 재고 유자
    virtual void open_data_channel_batchs(enum data_channel channel, int64 batch_count, int64 batch_size);  // 데이터 뒤섞기 후 batch_size 크기의 미니배치를 batch_count 회만큼 반복 제공
    virtual int64 open_data_channel_epochs(enum data_channel channel, int64 epoch_count, int64 batch_size);  // 전체 데이터를 batch_size 크기의 미니배치로 분할해 epoch_count 회만큼 반복 제공, 매 에포크 시작시 데이터 뒤섞기 수행, 에포크당 미니배치 개수를 반환

    virtual void get_data(enum data_channel channel, Dict& xs, Dict& ys);  // 채널 오픈시 예약된 내용에 따라 멀티스레드 방식으로 데이터를 준비해 호출 당 하나의 미니배치 제공

    virtual void close_data_channel(enum data_channel channel);

    Dict forward_postproc_sys(Dict xs, Dict ys, Dict outs);
    Dict forward_postproc_autoencode_sys(Dict xs, Dict outs);
    Dict backprop_postproc_sys(Dict ys, Dict outs);
    Dict backprop_postproc_autoencode_sys(Dict xs, Dict outs);
    
    Dict eval_accuracy_sys(Dict xs, Dict ys, Dict outs);
    Dict eval_accuracy_autoencode_sys(Dict xs, Dict outs);

    float forward_postproc_base(Dict y, Dict out, string mode);
    Dict backprop_postproc_base(Dict y, Dict out, string mode);
    float eval_accuracy_base(Dict xs, Dict ys, Dict outs, string mode);

    float m_forward_postproc_no_cuda(Dict y, Dict out, enum loss_mode enumMode);
    Dict m_backprop_postproc_no_cuda(Dict y, Dict out, enum loss_mode enumMode);
    float m_eval_accuracy_no_cuda(Dict x, Dict y, Dict out, enum loss_mode enumMode);

    float m_forward_postproc_cuda(Dict y, Dict out, enum loss_mode enumMode);
    Dict m_backprop_postproc_cuda(Dict y, Dict out, enum loss_mode enumMode);
    float m_eval_accuracy_cuda(Dict x, Dict y, Dict out, enum loss_mode enumMode);

    virtual void log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2);
    virtual void log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2);
    virtual void log_test(string name, Dict acc, int64 tm1, int64 tm2);

    virtual string get_acc_str(string& acc_keys, Dict acc);
    virtual string get_loss_str(Dict loss);

    virtual void gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys);

    virtual void prepare_minibatch_data(int64* data_idxs, int64 size);

    virtual void gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);
    virtual void gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    virtual void visualize_main(Dict xs, Dict ys, Dict outs);
    virtual void visualize(Value xs, Value estimates, Value answers) = 0;

    Shape input_shape;
    Shape output_shape;

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

    virtual int64 input_timesteps() { throw KaiException(KERR_ASSERT);  return 0; }
    virtual int64 output_timesteps() { throw KaiException(KERR_ASSERT); return 0; }

    virtual Value get_ext_param(string key) { throw KaiException(KERR_ASSERT); return 0; }

    virtual Dict forward_postproc(Dict xs, Dict ys, Dict outs, string mode);
    virtual Dict backprop_postproc(Dict ys, Dict outs, string mode);
    virtual Dict eval_accuracy(Dict xs, Dict ys, Dict outs, string mode);

    virtual bool use_custom_data_format() { return false; }

    #ifdef KAI2021_WINDOWS
    virtual void load_data_index(string suffix="");
    virtual void save_data_index(string suffix="");
    #else
    //hs.cho
    virtual void load_data_index(string suffix = "");
    virtual void save_data_index(string suffix = "");
    //virtual void load_data_index();
    //virtual void save_data_index();
    #endif

    static bool get_img_display_mode() { return ms_img_display_mode; }
    static string get_img_save_folder() { return ms_img_save_folder; }

    static void set_img_display_mode(bool mode) { ms_img_display_mode = mode; }
    static void set_img_save_folder(string fname) { ms_img_save_folder = fname; }

protected:
    static string ms_img_save_folder;
    static bool ms_img_display_mode;

    map<enum data_channel, DataChannel*> m_channels;
    map<enum data_channel, int64> m_data_begin;
    map<enum data_channel, int64> m_data_count;

    enum loss_mode m_enumMode;

    string m_name;
    string m_custom_mode;

    Array<float> m_default_xs;
    Array<float> m_default_ys;

    int64 m_data_cnt;

    Array<int64> m_data_idx_arr;
    int64* m_data_idx;  // m_data_idx_arr 배열의 내부 메모리 포인터, 자동 삭제되므로 삭제 금지

    bool m_need_to_load_data_index;
    string m_data_index_path;

    mutex* m_gen_dat_mutex;

    static Array<int64> ms_cuda_compress_flags(int64* cuda_flag, int64 size);

    void m_shuffle_index(int64 size, float tr_ratio = 0.75f, float va_ratio = 0.05f, float vi_ratio = 0.05f);

    void m_show_select_results(Value cest, Value cans, vector<string> names);
    void m_show_seq_binary_results(Value cest, Value cans);
    void m_draw_images_horz(Value csx, Shape image_shape, int ratio=1);
    void m_dump_mnist_image_data(Array<float> xs);

    void m_load_mnist_data(string path, Array<unsigned char>& images, Array<unsigned char>& labels, vector<string>& target_names);
    void m_load_cifar10_data(string path, Array<unsigned char>& images, Array<unsigned char>& labels, vector<string>& target_names);

    string m_get_mode_str() const;

    static enum loss_mode ms_str_to_mode(string mode, string* custom_mode=NULL);
};

class AutoencodeDataset : public Dataset {
public:
    AutoencodeDataset(string name, string mode, float ratio=1.0, bool x_seq = false, bool y_seq = false);
    virtual ~AutoencodeDataset();

    virtual int64 train_count(); // { return int64((float)m_tr_cnt * m_ratio); }
    virtual int64 autoencode_count(); // { return m_tr_cnt; }

    virtual void visualize_autoencode(Dict xs, Dict code, Dict repl, Dict outs, Dict ys);
    virtual void visualize_hash(Array<int64> rank1, Array<int64> rank2, Array<int64> key_labels, Array<int64> dat_label, Array<float> distance, Array<float> keys, Array<float> repl, Array<float> xs);

protected:
    float m_ratio;
};

class Gan;

class GanDataset : public Dataset {
public:
    GanDataset(string name, bool x_seq = false, bool y_seq = false) : Dataset(name, "binary", x_seq, y_seq) {}
    virtual ~GanDataset() {}

    void m_gan_shuffle_index(int64 size) { m_shuffle_index(size, 1.0f, 0.0f, 0.0f); }

    void gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    virtual void visualize(Value xs, Value estimates, Value answers);
    virtual void visualize(Gan* model, Dict real_xs, Dict fake_xs) = 0;

    virtual string get_loss_str(Dict loss);

    virtual void log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2);
    virtual void log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2);
    virtual void log_test(string name, Dict acc, int64 tm1, int64 tm2);
};

