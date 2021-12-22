/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/corpus.h"

class WordSeqDataset : public Dataset {
public:
    WordSeqDataset(string name, Corpus& corpus, int64 seq_len);
    virtual ~WordSeqDataset();

    virtual bool use_custom_data_format() { return true; }
    virtual void gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys);

    virtual void visualize_main(Dict xs, Dict ys, Dict outs);
    void visualize(Value xs, Value estimates, Value answers);

    virtual Value get_ext_param(string key);

    virtual bool input_seq() { return true; }
    virtual bool output_seq() { return true; }

    virtual int64 input_timesteps() { return m_seq_len; }
    virtual int64 output_timesteps() { return m_seq_len; }

protected:
    Corpus& m_corpus;

    int64 m_seq_len;
    int64 m_voc_size;
};