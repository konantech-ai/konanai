/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../../core/corpus.h"

class PtbCorpus : public Corpus {
public:
	PtbCorpus(string name, string dic_path, string src_path);
	virtual ~PtbCorpus();

    virtual int64 voc_size() { return (int64)m_w2i.size(); }
    virtual int64 tag_size() { return 0; }

    virtual int64 corpus_word_count() { return (int64)m_words.size(); }
    virtual int64 corpus_sent_count() { return (int64)m_sents.size() - 1; }

    virtual string get_word_to_visualize(int64 wid) { return m_i2w[wid]; }
    virtual int64 get_sent_length(int64 sidx) { return (int64) m_sents[sidx].size(); }
    virtual int64 get_wid(int64 sidx, int64 nth) { return m_sents[sidx][nth]; }
    virtual int64 get_tag_id(int64 sidx, int64 nth) { return 0; }
    virtual int64 get_nth_word(int64 nth) { return m_words[nth]; }

    virtual int64 pred_word_size(int64 window_size);

    virtual int64 get_next_sent(int64 sidx, bool* p_next_sent, float next_ratio, int64 max_len);
    virtual int64 get_random_word(int64 real_wid, int64* p_fake_tag_id);
    
    virtual void fill_word(int64*& wp, int64& cidx);
    virtual void fill_word_noms(int64*& wp, int64& cidx, int64 sample_cnt);

protected:
    map<string, int64> m_w2i;
    map<int64, string> m_i2w;
    map<int64, int64> m_freq;

    vector<int64> m_words;
    vector<vector<int64>> m_sents;

    //int64 get_random_word() { return m_words[rnd_babo() % m_words.size()]; }

protected:
    void create_cache(string data_path);
    void load_cache(string cache_path);
    void save_cache(string cache_path);
};