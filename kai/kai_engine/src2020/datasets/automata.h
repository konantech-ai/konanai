/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/engine.h"

class AutomataDataset : public Dataset {
public:
    AutomataDataset();
    virtual ~AutomataDataset();

    virtual void prepare_minibatch_data(int64* data_idxs, int64 size);
    virtual void gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y);
    virtual void gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x);

    void visualize(Value xs, Value estimates, Value answers);

    virtual bool input_seq() { return true; }
    virtual int64 input_timesteps() { return m_max_length; }

protected:
    map<string, string> m_alphabet;

    Dict m_rules;
    
    Dict m_action_table;
    Dict m_goto_table;

    int64 m_alphabet_size;

    int64 m_max_length;
    int64 m_min_length;

    Array<float> m_batch_xs;
    Array<float> m_batch_ys;

    Array<int64> m_batch_xlen;

protected:
    string m_generate_sent();
    string m_gen_node(string symbol, int64 depth);
    bool m_is_correct_sent(string sent);
};
