/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class Corpus {
public:
    Corpus(string name, bool use_tag) {m_name = name; m_use_tag = use_tag; }
    virtual ~Corpus() {}

    virtual bool use_tag() { return m_use_tag; }

    virtual int64 voc_size() = 0; 
    virtual int64 tag_size() = 0;
    virtual int64 corpus_sent_count() = 0;

    virtual int64 corpus_word_count() = 0;
    virtual int64 get_nth_word(int64 nth) = 0;
    virtual int64 pred_word_size(int64 window_size) = 0;

    virtual string get_word_to_visualize(int64 wid) = 0;
    virtual int64 get_next_sent(int64 sidx, bool* p_next_sent, float next_ratio, int64 max_len) = 0;
    virtual int64 get_sent_length(int64 sidx) = 0;
    virtual int64 get_wid(int64 sidx, int64 nth) = 0;

    virtual int64 get_tag_id(int64 sidx, int64 nth) = 0;
    virtual int64 get_random_word(int64 real_wid, int64* p_fake_tag_id) = 0;

    virtual void fill_word(int64*& wp, int64& cidx) = 0;
    virtual void fill_word_noms(int64*& wp, int64& cidx, int64 sample_cnt) = 0;

protected:
    string m_name;
    bool m_use_tag;
};