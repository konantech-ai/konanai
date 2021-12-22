/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class MnistEncDecDataset : public Dataset {
public:
    MnistEncDecDataset(string name);
    virtual ~MnistEncDecDataset();

    virtual Dict forward_postproc(Dict xs, Dict ys, Dict outs, string mode);
    virtual Dict backprop_postproc(Dict ys, Dict outs, string mode);
    virtual Dict eval_accuracy(Dict xs, Dict ys, Dict outs, string mode);

protected:
    Array<unsigned char> m_images;
    Array<unsigned char> m_labels;

    vector<string> m_target_names;
};

class MnistEngDataset : public MnistEncDecDataset {
public:
    MnistEngDataset();
    virtual ~MnistEngDataset();

    virtual void gen_plain_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    void visualize(Value xs, Value estimates, Value answers);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return true; }

    virtual int64 output_timesteps() { return m_word_len; }

protected:
    int64 m_word_len;

    Array<float> m_captions;

protected:
    void m_set_captions();
    string m_eng_prob_to_caption(Array<int64> arr, int64 nth);
};

class MnistKorDataset : public MnistEncDecDataset {
public:
    MnistKorDataset(int64 length);
    virtual ~MnistKorDataset();

    virtual void gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_seq_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    void visualize(Value xs, Value estimates, Value answers);

    virtual bool input_seq() { return true; }
    virtual bool output_seq() { return true; }

    virtual int64 input_timesteps() { return m_length; }
    virtual int64 output_timesteps() { return 2 * m_length; }

protected:
    static string ms_alphabet;

    int64 m_length;

    void m_add_digit_pair(float* py, int64& m, int64& num, int64 unit);
    void m_set_digit(float* py, int64& m, int64 digit);

    string m_kor_prob_to_caption(Array<int64> arr, int64 nth);
};
