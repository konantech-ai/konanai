/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "bert.h"
#include "../core/host_math.h"
#include "../core/engine.h"
#include "../core/random.h"
#include "../core/log.h"
#include "../cuda/cuda_math.h"
#include "../cuda/cuda_conn.cuh"

BertDataset::BertDataset(string name, Corpus& corpus, int64 max_sent_len) : Dataset(name, "bert"), m_corpus(corpus) {
    m_trace = false;

    m_max_sent_len = max_sent_len;
    m_use_tag = m_corpus.use_tag();
    m_voc_size = m_corpus.voc_size() + 4; // 'EMP', 'CLS', 'SEP', 'UNK'

    m_sent_cnt = m_corpus.corpus_sent_count();

    input_shape = Shape(max_sent_len, m_use_tag ? 4 : 3);
    output_shape = Shape(max_sent_len, m_voc_size);

    //m_shuffle_index(m_sent_cnt, 0.95f, 0.01f, 0.01f);

    //m_sent_cnt = 20;
    m_shuffle_index(m_sent_cnt, 0.997f, 0.001f, 0.001f);

    logger.Print("tr_cnt = %d, va_cnt = %d, vi_cnt = %d, te_cnt = %d", m_data_count[data_channel::train], m_data_count[data_channel::validate], m_data_count[data_channel::visualize], m_data_count[data_channel::test]);

    //throw KaiException(KERR_ASSERT);


    //m_trace = true;
}

BertDataset::~BertDataset() {
}

void BertDataset::m_hide_word(int64 wid, Array<int64>& xwids, Array<int64>& ywids, int64 n, int64 nt) {
    //if (rnd_babo() % 100 >= 15) return;
    //if (!m_trace) throw KaiException(KERR_ASSERT); // 40% to 15% È¯¿ø
    if (Random::dice(100) >= 15) return;

    int64 dice = Random::dice(100);
    
    ywids[Idx(n, nt)] = wid;

    m_unk_count++;

    if (dice < 80) {
        xwids[Idx(n, nt, 0)] = UNK;
        if (m_use_tag) xwids[Idx(n, nt, 3)] = 0;
    }
    else if (dice < 90) {
        int64 fake_wid, fake_tag_id;
        fake_wid = m_corpus.get_random_word(wid, &fake_tag_id);
        xwids[Idx(n, nt, 0)] = fake_wid + 4;
        if (m_use_tag) xwids[Idx(n, nt, 3)] = fake_tag_id;
    }
}

void BertDataset::gen_minibatch_data(enum data_channel channel, int64* data_idxs, int64 size, Dict& xs, Dict& ys) {
    m_unk_count = 0;

    Array<int64> xwids = kmath->zeros_int(input_shape.add_front(size));     // initialize with EMT
    Array<int64> ywids = kmath->zeros_int(Shape(size, m_max_sent_len));
    Array<float> ynext = kmath->zeros(Shape(size, 2));

    for (int64 n = 0; n < size; n++) {
        bool next_sent; // = rnd_babo() % 2 == 0;

        int64 sidx1 = data_idxs[n];
        int64 sidx2 = m_corpus.get_next_sent(sidx1, &next_sent, 0.5f, m_max_sent_len-3); // next_sent ? sidx1 + 1 : (sidx1 + 2 + rnd_babo() % (m_sent_cnt - 1)) % m_sent_cnt;

        int64 leng1 = m_corpus.get_sent_length(sidx1);
        int64 leng2 = m_corpus.get_sent_length(sidx2);

        int64 nt = 0;

        xwids[Idx(n, nt, 0)] = CLS;
        ynext[Idx(n, next_sent ? 1 : 0)] =  1.0;

        for (int64 m = 0; m < leng1; m++, nt++) {
            int64 wid = m_corpus.get_wid(sidx1, m) + 4;
            xwids[Idx(n, nt, 0)] = wid;
            xwids[Idx(n, nt, 1)] = nt;
            xwids[Idx(n, nt, 2)] = 0;
            if (m_use_tag) xwids[Idx(n, nt, 3)] = m_corpus.get_tag_id(sidx1, m);
            m_hide_word(wid, xwids, ywids, n, nt);
        }

        xwids[Idx(n, nt, 0)] = SEP;
        xwids[Idx(n, nt, 1)] = nt;
        xwids[Idx(n, nt++, 2)] = 0;

        for (int64 m = 0; m < leng2; m++, nt++) {
            int64 wid = m_corpus.get_wid(sidx2, m) + 4;
            xwids[Idx(n, nt, 0)] = wid;
            xwids[Idx(n, nt, 1)] = nt;
            xwids[Idx(n, nt, 2)] = 1;
            if (m_use_tag) xwids[Idx(n, nt, 3)] = m_corpus.get_tag_id(sidx2, m);
            m_hide_word(wid, xwids, ywids, n, nt);
        }

        xwids[Idx(n, nt, 0)] = SEP;
        xwids[Idx(n, nt, 1)] = nt;
        xwids[Idx(n, nt++, 2)] = 1;

        for (; nt < m_max_sent_len; nt++) {
            xwids[Idx(n, nt, 1)] = nt;
        }
    }

    xs["wids"] = xwids;
    ys["wids"] = ywids;
    ys["next"] = ynext;

    if (m_trace) xwids.print("BertDataset::generate_data::xwids");
}

void BertDataset::visualize_main(Dict xs, Dict ys, Dict outs) {
    Dict estimate = outs["default"];
    Dict dnext = outs["is_next_sent"];

    Array<float> elogits = estimate["data"];
    Array<float> ans_next_sent = ys["next"];
    Array<float> est_next_sent = dnext["data"];

    Array<int64> xwids = xs["wids"];
    Array<int64> ywids = ys["wids"];

    int64 mb_size = elogits.axis_size(0);

    Array<int64> ewids = kmath->argmax(elogits.merge_time_axis(), 0).split_time_axis(mb_size);

    ewids = CudaConn::ToHostArray(ewids, "ewids");

    if (m_trace) elogits.print("BertDataset::visualize_main::elogits");
    if (m_trace) ans_next_sent.print("BertDataset::visualize_main::ans_next_sent");
    if (m_trace) est_next_sent.print("BertDataset::visualize_main::est_next_sent");

    if (m_trace) xwids.print("BertDataset::visualize_main::xwids");
    if (m_trace) ywids.print("BertDataset::visualize_main::ywids");
    if (m_trace) ewids.print("BertDataset::visualize_main::ewids");

    Array<float> host_ans_next_sent = CudaConn::ToHostArray(ans_next_sent, "ans_next_sent");
    Array<float> host_est_next_sent = CudaConn::ToHostArray(est_next_sent, "est_next_sent");

    Array<float> prob_next_sent = kmath->softmax(host_est_next_sent);

    if (m_trace) prob_next_sent.print("BertDataset::visualize_main::prob_next_sent");

    for (int64 n = 0; n < mb_size; n++) {
#ifdef KAI2021_WINDOWS
        throw KaiException(KERR_ASSERT);
        float ans_ns = 0.0f;
#else
        float ans_ns = host_ans_next_sent[Idx(n, 0)] ? 0.0 : 1.0;
#endif
        float est_ns = prob_next_sent[Idx(n, 1)];
        string correct = ((ans_ns > 0.5) == (est_ns > 0.5)) ? "Correct" : "Wrong";
        logger.Print("Sampple-%d [Is_Next_Sent] answer %4.2f vs. estimate %4.2f : %s", n + 1, ans_ns, est_ns, correct.c_str());
        for (int64 m = 0; m < m_max_sent_len; ) {
            int64 xwid, ywid;
            logger.PrintWait("    ");
            for (int64 k = 0; k < 16; k++, m++) {
                xwid = xwids[Idx(n, m, 0)];
                ywid = ywids[Idx(n, m)];

                if (xwid == EMT) break;
                else if (xwid == CLS) logger.PrintWait("[CLS] ");
                else if (xwid == SEP) logger.PrintWait("[SEP] ");
                else if (xwid == UNK) logger.PrintWait("[UNK] ");
                else if (ywid == 0) logger.PrintWait("%s ", m_word(xwid).c_str());
                else logger.PrintWait("[%d %s] ", m+1, m_word(xwid).c_str());
            }
            logger.Print("");
            if (xwid == EMT) break;
        }

        for (int64 m = 0; m < m_max_sent_len; m++) {
            int64 xwid, ywid, ewid;

            xwid = xwids[Idx(n, m, 0)];
            ywid = ywids[Idx(n, m)];
            ewid = ewids[Idx(n, m)];

            if (xwid != UNK && ywid == 0) continue;

            string result = (ywid == ewid) ? "O" : "X";

            if (xwid == UNK) {
                logger.Print("    [%d] %s UNK ===> %s vs %s", m+1, result.c_str(), m_word(ywid).c_str(), m_word(ewid).c_str());
            }
            else if (ywid != 0) {
                logger.Print("    [%d] %s %s ===> %s vs. %s", m+1, result.c_str(), m_word(xwid).c_str(), m_word(ywid).c_str(), m_word(ewid).c_str());
            }
        }
    }
}

void BertDataset::visualize(Value xs, Value estimates, Value answers) {
    throw KaiException(KERR_ASSERT);
}

string BertDataset::m_word(int64 wid) {
    string lemma = m_corpus.get_word_to_visualize(wid - 4);
    string tag = lemma.substr(1, 2);
    string word = lemma.substr(4, lemma.length()-5);

    string exp = word + "/" + tag;

    return exp;
}

Value BertDataset::get_ext_param(string key) {
    if (key == "voc_size") return m_voc_size;
    else if (key == "vec_size") return 0; // unspecify
    else if (key == "pos_size") return m_max_sent_len;
    else if (key == "sent_size") return 2;
    else if (key == "tag_size") return m_corpus.tag_size();
    else throw KaiException(KERR_ASSERT);
    return 0;
}

Dict BertDataset::forward_postproc(Dict xs, Dict ys, Dict outs, string mode) {
    Dict estimate = outs["default"];
    Dict dnext = outs["is_next_sent"];

    Array<float> ans_next_sent = ys["next"];
    Array<float> est_next_sent = dnext["data"];

    Array<int64> ywids = ys["wids"];
    Array<float> ewids = estimate["data"];

    if (m_trace) ywids.print("BertDataset::forward_postproc::ywids");
    if (m_trace) ans_next_sent.print("BertDataset::forward_postproc::ans_next_sent");
    if (m_trace) est_next_sent.print("BertDataset::forward_postproc::est_next_sent");
    if (m_trace) ewids.print("BertDataset::forward_postproc::ewids");
    if (m_trace) ans_next_sent.print("BertDataset::forward_postproc::ans_next_sent");

    ewids = ewids.merge_time_axis();
    ywids = ywids.merge_time_axis();

    if (m_trace) ewids.print("BertDataset::forward_postproc::output merged");
    if (m_trace) ywids.print("BertDataset::forward_postproc::ywids merged");

    Array<bool> selector = ywids > UNK;

    Array<float> est_marked_words = kmath->extract_selected(ewids, selector);
    Array<int64> ans_marked_words = kmath->extract_selected(ywids, selector);

    if (m_trace) est_marked_words.print("BertDataset::forward_postproc::est_marked_words");
    if (m_trace) ans_marked_words.print("BertDataset::forward_postproc::ans_marked_words");

    Dict ymark = Value::wrap_dict("wids", ans_marked_words);
    Dict emark = Value::wrap_dict("data", est_marked_words);

    Dict ynext = Value::wrap_dict("data", ans_next_sent);
    Dict enext = Value::wrap_dict("data", est_next_sent);

    float loss_marked_word = Dataset::forward_postproc_base(ymark, emark, "class_idx");
    float loss_next_sent = Dataset::forward_postproc_base(ynext, enext, "classify");

    if (m_trace) logger.Print("BertDataset::forward_postproc::loss_marked_word = %f", loss_marked_word);
    if (m_trace) logger.Print("BertDataset::forward_postproc::loss_next_sent = %f", loss_next_sent);

    Dict loss;

    loss["next_sent"] = loss_next_sent;
    loss["marked_word"] = loss_marked_word;

    logger.Bookeep("loss = %f, loss_next_sent = %f, loss_marked_word = %f\n", loss_next_sent+loss_marked_word, loss_next_sent, loss_marked_word);

    if (m_trace) logger.Print("BertDataset::forward_postproc::loss = %f", loss_next_sent + loss_marked_word);

    return loss;
}

Dict BertDataset::backprop_postproc(Dict ys, Dict outs, string mode) {
    Dict estimate = outs["default"];
    Dict dnext = outs["is_next_sent"];

    Array<int64> ywids = ys["wids"];
    Array<float> ewids = estimate["data"];
    Array<float> ans_next_sent = ys["next"];
    Array<float> est_next_sent = dnext["data"];

    if (m_trace) ywids.print("BertDataset::backprop_postproc::ywids");
    if (m_trace) ans_next_sent.print("BertDataset::backprop_postproc::ans_next_sent");
    if (m_trace) est_next_sent.print("BertDataset::backprop_postproc::est_next_sent");
    if (m_trace) ewids.print("BertDataset::backprop_postproc::ewids");

    Shape eshape = ewids.shape();

    ewids = ewids.merge_time_axis();
    ywids = ywids.merge_time_axis();
    
    if (m_trace) ewids.print("BertDataset::backprop_postproc::output merged");
    if (m_trace) ywids.print("BertDataset::backprop_postproc::ywids merged");

    Array<bool> selector = ywids > UNK;
    //aux["selector"] = selector;

    Array<float> est_marked_words = kmath->extract_selected(ewids, selector);
    Array<int64> ans_marked_words = kmath->extract_selected(ywids, selector);

    if (m_trace) est_marked_words.print("BertDataset::backprop_postproc::est_marked_words");
    if (m_trace) ans_marked_words.print("BertDataset::backprop_postproc::ans_marked_words");

    Dict ymark = Value::wrap_dict("wids", ans_marked_words);
    Dict emark = Value::wrap_dict("data", est_marked_words);

    Dict ynext = Value::wrap_dict("data", ans_next_sent);
    Dict enext = Value::wrap_dict("data", est_next_sent);

    Dict G_is_next_sent = Dataset::backprop_postproc_base(ynext, enext, "classify"); 
    Dict G_marked_word = Dataset::backprop_postproc_base(ymark, emark, "class_idx");

    if (m_trace) logger.Print("BertDataset::backprop_postproc::G_is_next_sent = %s", Value::description(G_is_next_sent).c_str());
    if (m_trace) logger.Print("BertDataset::backprop_postproc::G_marked_word = %s", Value::description(G_marked_word).c_str());

    Array<float> G_selected = G_marked_word["data"];
    Array<float> G_estimate = kmath->zeros(eshape);

    G_estimate = G_estimate.merge_time_axis();
    G_estimate = kmath->fill_selected(G_estimate, G_selected, selector);
    G_estimate = G_estimate.reshape(eshape);

    Dict G_output;

    G_output["default"] = Value::wrap_dict("data", G_estimate);
    G_output["is_next_sent"] = G_is_next_sent;

    return G_output;
}

Dict BertDataset::eval_accuracy(Dict xs, Dict ys, Dict outs, string mode) {
    m_trace = false;

    if (m_trace) {
        logger.Print("*** Visualize for eval_accuracy debugging ***");
        visualize_main(xs, ys, outs);
    }

    Dict estimate = outs["default"];
    Dict dnext = outs["is_next_sent"];

    Array<int64> ywids = ys["wids"];
    Array<float> ans_next_sent = ys["next"];
    Array<float> est_next_sent = dnext["data"];
    Array<float> ewids = estimate["data"];

    if (m_trace) ywids.print("BertDataset::forward_postproc::ywids");
    if (m_trace) ans_next_sent.print("BertDataset::eval_accuracy::ans_next_sent");
    if (m_trace) est_next_sent.print("BertDataset::eval_accuracy::est_next_sent");
    if (m_trace) ewids.print("BertDataset::eval_accuracy::ewids");

    ewids = ewids.merge_time_axis();
    ywids = ywids.merge_time_axis();

    if (m_trace) ewids.print("BertDataset::eval_accuracy::output merged");
    if (m_trace) ywids.print("BertDataset::eval_accuracy::ywids merged");

    Array<bool> selector = ywids > UNK;

    Array<float> est_marked_words = kmath->extract_selected(ewids, selector);
    Array<int64> ans_marked_words = kmath->extract_selected(ywids, selector);

    if (m_trace) est_marked_words.print("BertDataset::eval_accuracy::est_marked_words");
    if (m_trace) ans_marked_words.print("BertDataset::eval_accuracy::ans_marked_words");

    Dict ymark = Value::wrap_dict("wids", ans_marked_words);
    Dict emark = Value::wrap_dict("data", est_marked_words);

    Dict ynext = Value::wrap_dict("data", ans_next_sent);
    Dict enext = Value::wrap_dict("data", est_next_sent);

    float acc_marked_word = Dataset::eval_accuracy_base(xs, ymark, emark, "class_idx");
    float acc_next_sent = Dataset::eval_accuracy_base(xs, ynext, enext, "classify");

    static int64 nth = 0;
    nth++;
    if (m_trace) logger.Print("BertDataset::eval_accuracy[%d]::acc_marked_word = %f", nth, acc_marked_word);
    if (m_trace) logger.Print("BertDataset::eval_accuracy[%d]::acc_next_sent = %f", nth, acc_next_sent);

    Dict acc;

    acc["next_sent"] = acc_next_sent;
    acc["marked_word"] = acc_marked_word;

    return acc;
}

void BertDataset::log_train(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, Dict acc, int64 tm1, int64 tm2) {
    float loss_mean_next_sent = loss_mean["next_sent"];
    float acc_mean_next_sent = acc_mean["next_sent"];
    float acc_next_sent = acc["next_sent"];

    float loss_mean_marked_word = loss_mean["marked_word"];
    float acc_mean_marked_word = acc_mean["marked_word"];
    float acc_marked_word = acc["marked_word"];

    float loss_mean_sum = loss_mean_next_sent + loss_mean_marked_word;

    if (batch_count == 0)
        logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
    else
        logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

    logger.Print("loss=%16.9e(%16.9e+%16.9e), accuracy=(%16.9e,%16.9e)/(%16.9e,%16.9e) (%d/%d secs)", loss_mean_sum, loss_mean_next_sent, loss_mean_marked_word,
        acc_mean_next_sent, acc_mean_marked_word, acc_next_sent, acc_marked_word, tm1, tm2);
}

void BertDataset::log_train_batch(int64 epoch, int64 epoch_count, int64 batch, int64 batch_count, Dict loss_mean, Dict acc_mean, int64 tm1, int64 tm2) {
    float loss_mean_next_sent = loss_mean["next_sent"];
    float acc_mean_next_sent = acc_mean["next_sent"];

    float loss_mean_marked_word = loss_mean["marked_word"];
    float acc_mean_marked_word = acc_mean["marked_word"];

    float loss_mean_sum = loss_mean_next_sent + loss_mean_marked_word;

    if (batch_count == 0)
        logger.PrintWait("    Epoch %lld/%lld: ", epoch, epoch_count);
    else
        logger.PrintWait("    Batch %lld/%lld(in Epoch %lld): ", batch, batch_count, epoch);

    logger.Print("loss=%16.9e(%16.9e+%16.9e), accuracy=(%16.9e,%16.9e) (%lld/%lld secs)", loss_mean_sum, loss_mean_next_sent, loss_mean_marked_word,
        acc_mean_next_sent, acc_mean_marked_word, tm1, tm2);
}

void BertDataset::log_test(string name, Dict acc, int64 tm1, int64 tm2) {
    float acc_next_sent = acc["next_sent"];
    float acc_marked_word = acc["marked_word"];

    logger.Print("Model %s test report: accuracy = %16.9e/%16.9e, (%lld/%lld secs)", name.c_str(), acc_next_sent, acc_marked_word, tm2, tm1);
    logger.Print("");
}
