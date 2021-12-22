/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../core/dataset.h"
#include "../core/corpus.h"

class BertDataset : public Dataset {
public:
    BertDataset(string name, Corpus& corpus, int64 max_sent_len);
    virtual ~BertDataset();

    virtual bool use_custom_data_format() { return true; }
    virtual void gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys);

    virtual void visualize_main(Dict xs, Dict ys, Dict outs);
    void visualize(Value xs, Value estimates, Value answers);

    virtual Value get_ext_param(string key);

    virtual bool input_seq() { return false; }
    virtual bool output_seq() { return false; }

    virtual Dict forward_postproc(Dict xs, Dict ys, Dict estimate, string mode);
    virtual Dict backprop_postproc(Dict ys, Dict estimate, string mode);
    virtual Dict eval_accuracy(Dict x, Dict y, Dict out, string mode);
        
    virtual void log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2);
    virtual void log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2);
    virtual void log_test(string name, Dict acc, int64 tm1, int64 tm2);

protected:
    Corpus& m_corpus;

    bool m_trace;
    bool m_use_tag;

    int64 m_max_sent_len;
    int64 m_voc_size;
    
    int64 m_sent_cnt;

    int64 m_unk_count;

    const int64 EMT = 0;
    const int64 CLS = 1;
    const int64 SEP = 2;
    const int64 UNK = 3;

    void m_hide_word(int64 wid, Array<int64>& xwids, Array<int64>& ywids, int64 n, int64 nt);
    string m_word(int64 wid);
};
