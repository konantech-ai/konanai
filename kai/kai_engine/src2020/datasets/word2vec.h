/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/corpus.h"

class Word2VecDataset : public Dataset {
public:
	Word2VecDataset(string name, Corpus& corpus, string mode, int64 window_size = 1, int64 sample_cnt = 4);
	virtual ~Word2VecDataset();

    virtual Value get_ext_param(string key);

    virtual void gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys);

    virtual void visualize_main(Dict xs, Dict ys, Dict outs);
    void visualize(Value xs, Value estimates, Value answers);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

    virtual bool use_custom_data_format() { return true; }

protected:
    Corpus& m_corpus;
    
    int64 m_window_size;
    int64 m_sample_cnt;

    int64 m_voc_size;

    int64 m_in_cnt;
    int64 m_out_cnt;

    string m_w2v_mode;
};