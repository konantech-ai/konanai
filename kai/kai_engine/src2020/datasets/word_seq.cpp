/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "word_seq.h"
#include "../core/log.h"
#include "../core/host_math.h"
#include "../cuda/cuda_math.h"

WordSeqDataset::WordSeqDataset(string name, Corpus& corpus, int64 seq_len) : Dataset(name, "class_idx"), m_corpus(corpus) {
    m_seq_len = seq_len;
    m_voc_size = m_corpus.voc_size();

    int64 data_cnt = m_corpus.corpus_word_count() - seq_len + 1;

    input_shape = Shape(1);

    m_shuffle_index(data_cnt);
}

WordSeqDataset::~WordSeqDataset() {
}

Value WordSeqDataset::get_ext_param(string key) {
    if (key == "voc_sizes") {
        List voc_sizes;
        voc_sizes.push_back(m_voc_size);
        return voc_sizes;
    }
    else if (key == "vec_size") return 128;
    else throw KaiException(KERR_ASSERT);
    return 0;
}

void WordSeqDataset::gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys) {
    Array<int64> words(Shape(size, m_seq_len));
    Array<int64> nexts(Shape(size, m_seq_len));
    Array<int64> length = kmath->ones_int(Shape(size), m_seq_len);

    int64* wp = words.data_ptr();
    int64* np = nexts.data_ptr();

    for (int64 n = 0; n < size; n++) {
        int64 cidx = data_idxs[n];
        for (int64 m = 0; m < m_seq_len; m++) {
            *wp++ = m_corpus.get_nth_word(cidx++);
            *np++ = m_corpus.get_nth_word(cidx);
        }
    }

    Dict def_xs = Value::wrap_dict("wids", words);
    Dict def_ys = Value::wrap_dict("wids", nexts);

    xs = Value::wrap_dict("default", def_xs);
    ys = Value::wrap_dict("default", def_ys);
}

void WordSeqDataset::visualize_main(Dict xs, Dict ys, Dict outs) {
    Dict xs_def = xs["default"], ys_def = ys["default"], out_def = outs["default"];
    Array<int64> word = xs_def["wids"];
    Array<int64> next = ys_def["wids"];
    Array<float> out_data = out_def["data"];
    Array<float> est_next = CudaConn::ToHostArray(out_data, "wordseq out");

    int64 mb_size = est_next.axis_size(0);
    int64 vec_size = est_next.axis_size(-1);

    Array<int64> est = kmath->argmax(est_next.reshape(Shape(-1, vec_size)), 0).reshape(est_next.shape().remove_end());

    for (int64 n = 0; n < mb_size; n++) {
        logger.PrintWait("Input sequence-%d:", n);
        for (int64 m = 0; m < 5; m++) {
            string curr_word = m_corpus.get_word_to_visualize(word[Idx(n, m)]);
            logger.PrintWait(" %s", curr_word.c_str());
        }
        logger.Print("...");
        for (int64 m = 0; m < m_seq_len; m++) {
            string curr_word = m_corpus.get_word_to_visualize(word[Idx(n, m)]);
            string next_word = m_corpus.get_word_to_visualize(next[Idx(n, m)]);
            string est_word = m_corpus.get_word_to_visualize(est[Idx(n, m)]);
            logger.Print("    %s => %s : %s", curr_word.c_str(), next_word.c_str(), est_word.c_str());
        }
    }
}

void WordSeqDataset::visualize(Value cxs, Value cest, Value cans) {
    throw KaiException(KERR_ASSERT);
}
