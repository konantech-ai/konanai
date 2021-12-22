/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "word2vec.h"
#include "../core/log.h"
#include "../core/host_math.h"
#include "../cuda/cuda_math.h"

Word2VecDataset::Word2VecDataset(string name, Corpus& corpus, string mode, int64 window_size, int64 sample_cnt) : Dataset(name, "class_1st"), m_corpus(corpus){
    m_w2v_mode = mode;
    
    m_window_size = window_size;
    m_sample_cnt = sample_cnt;
    m_voc_size = m_corpus.voc_size();

    int64 data_cnt = m_corpus.pred_word_size(m_window_size);

    if (mode == "cbow") {
        m_in_cnt = 2 * window_size;
        m_out_cnt = sample_cnt;
    }
    else if (mode == "skip") {
        m_in_cnt = 1;
        m_out_cnt = 2 * window_size * sample_cnt;
    }
    else {
        throw KaiException(KERR_ASSERT);
    }

    input_shape = Shape(m_in_cnt + m_out_cnt, 1);
    output_shape = Shape(m_out_cnt, 1);

    m_shuffle_index(data_cnt);
}

Word2VecDataset::~Word2VecDataset() {
}

Value Word2VecDataset::get_ext_param(string key) {
    if (key == "voc_sizes") {
        List voc_sizes;
        voc_sizes.push_back(m_voc_size);
        return voc_sizes;
    }
    else if (key == "win_cnt") return m_in_cnt;
    else if (key == "wout_cnt") return m_out_cnt;
    else throw KaiException(KERR_ASSERT);
    return 0;
}

void Word2VecDataset::gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys) {
    if (m_w2v_mode == "cbow") {
        Array<int64> context(Shape(size, 2 * m_window_size, 1));
        Array<int64> keyword(Shape(size, m_sample_cnt, 1));

        int64* cp = context.data_ptr();
        int64* kp = keyword.data_ptr();

        for (int64 n = 0; n < size; n++) {
            int64 cidx = data_idxs[n];

            for (int64 m = 0; m < m_window_size; m++) {
                m_corpus.fill_word(cp, cidx);
            }

            m_corpus.fill_word_noms(kp, cidx, m_sample_cnt);

            for (int64 m = 0; m < m_window_size; m++) {
                m_corpus.fill_word(cp, cidx);
            }
        }

        xs["hint"] = context;
        xs["noms"] = keyword;
    }
    else if (m_w2v_mode == "skip") {
        Array<int64> keyword(Shape(size, 1, 1));
        Array<int64> context(Shape(size, 2 * m_window_size * m_sample_cnt, 1));

        int64* kp = keyword.data_ptr();
        int64* cp = context.data_ptr();

        for (int64 n = 0; n < size; n++) {
            int64 cidx = data_idxs[n];

            for (int64 m = 0; m < m_window_size; m++) {
                m_corpus.fill_word_noms(cp, cidx, m_sample_cnt);
            }

            m_corpus.fill_word(kp, cidx);

            for (int64 m = 0; m < m_window_size; m++) {
                m_corpus.fill_word_noms(cp, cidx, m_sample_cnt);
            }
        }

        xs["hint"] = keyword;
        xs["noms"] = context;
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
}

void Word2VecDataset::visualize(Value xs, Value estimates, Value answers) {
    throw KaiException(KERR_ASSERT);
}

void Word2VecDataset::visualize_main(Dict xs, Dict ys, Dict outs) {
    Array<int64> hint = xs["hint"];
    Array<int64> noms = xs["noms"];

    hint = hint.to_host()[Axis(_all_, _all_, Ax(0))];
    noms = noms.to_host()[Axis(_all_, _all_, Ax(0))];

    hint = hint.reshape(hint.shape().remove_end());
    noms = noms.reshape(noms.shape().remove_end());

    Dict def_outs = outs["default"];

    Array<float> logits = def_outs["data"];
    Array<float> probs = kmath->softmax(logits);
    
    probs = probs.to_host();

    int64 mb_size = hint.axis_size(0);

    if (m_w2v_mode == "cbow") {
        Array<int64> est = hmath.argmax(probs, 0);

        for (int64 n = 0; n < mb_size; n++) {
            for (int64 m = 0; m < m_window_size; m++) {
                logger.PrintWait("%s ", m_corpus.get_word_to_visualize(hint[Idx(n, m)]).c_str());
            }
            logger.PrintWait("_%s_", m_corpus.get_word_to_visualize(noms[Idx(n, est[Idx(n)])]).c_str());
            for (int64 m = 0; m < m_window_size; m++) {
                logger.PrintWait(" %s", m_corpus.get_word_to_visualize(hint[Idx(n, m + m_window_size)]).c_str());
            }
            logger.PrintWait(" ==>");
            for (int64 m = 0; m < m_sample_cnt; m++) {
                float prob = probs[Idx(n, m)];
                logger.PrintWait(" %s(%d%%)", m_corpus.get_word_to_visualize(noms[Idx(n, m)]).c_str(), int64(prob*100));
            }
            logger.Print(" ==> %s", (est[Idx(n)] == 0) ? "Correct" : "Wrong");
        }
    }
    else if (m_w2v_mode == "skip") {
        probs = probs.reshape(Shape(mb_size * 2 * m_window_size, m_sample_cnt));
        Array<int64> est = hmath.argmax(probs, 0);

        noms = noms.reshape(Shape(mb_size, 2 * m_window_size, m_sample_cnt));
        probs = probs.reshape(Shape(mb_size, 2 * m_window_size, m_sample_cnt));
        est = est.reshape(Shape(mb_size, 2 * m_window_size));
        
        probs.print("probs");
        est.print("est");

        for (int64 n = 0; n < mb_size; n++) {
            for (int64 m = 0; m < m_window_size; m++) logger.PrintWait("_%d_ ", m);
            logger.PrintWait("%s", m_corpus.get_word_to_visualize(hint[Idx(n, 0)]).c_str());
            for (int64 m = 0; m < m_window_size; m++) logger.PrintWait(" _%d_", m_window_size+m);
            logger.Print(" ==>");
            for (int64 m = 0; m < 2*m_window_size; m++) {
                logger.PrintWait("   _%d_:", m);
                for (int64 k = 0; k < m_sample_cnt; k++) {
                    float prob = probs[Idx(n, m, k)];
                    logger.PrintWait(" %s(%d%%)", m_corpus.get_word_to_visualize(noms[Idx(n, m, k)]).c_str(), int(prob*100));
                }
                logger.Print(" ==> %s", (est[Idx(n, m)] == 0) ? "Correct" : "Wrong");
            }
        }
    }
    else {
        throw KaiException(KERR_ASSERT);
    }
}
