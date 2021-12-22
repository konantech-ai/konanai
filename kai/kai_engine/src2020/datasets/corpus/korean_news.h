/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "../../core/corpus.h"

class KoreanNewsCorpus : public Corpus {
public:
	KoreanNewsCorpus(string name, string folder_name);
	virtual ~KoreanNewsCorpus();

    virtual int64 voc_size();
    virtual int64 tag_size();

    virtual int64 corpus_word_count();
    virtual int64 corpus_sent_count();

    virtual string get_word_to_visualize(int64 wid);
    virtual int64 get_next_sent(int64 sidx, bool* p_next_sent, float next_ratio, int64 max_len);
    virtual int64 get_sent_length(int64 sidx);
    virtual int64 get_wid(int64 sidx, int64 nth);
    virtual int64 pred_word_size(int64 window_size);
    virtual int64 get_nth_word(int64 nth);

    virtual int64 get_tag_id(int64 sidx, int64 nth);
    virtual int64 get_random_word(int64 real_wid, int64* p_fake_tag_id);

    virtual void fill_word(int64*& wp, int64& cidx);
    virtual void fill_word_noms(int64*& wp, int64& cidx, int64 sample_cnt);

protected:
    int64 m_voc_count;
    int64 m_tag_count;

    int64 m_article_cnt;
    int64 m_sentence_cnt;
    int64 m_word_cnt;
    
    int64 m_string_pool_size;

    int64* m_arts_corpus;
    int64* m_arts_length;
    int64* m_sents_corpus;
    int64* m_sents_length;
    int64* m_words_corpus;

    char* m_is_last_sent;

    int64* m_word_tags;
    int64* m_wstr_index;

    char* m_string_pool;
    
    map<int64, string> m_tag_to_str;
    Dict m_str_to_tag;

    float m_adjust_ratio;

protected:
    void m_load_corpus_contents(string folder_name);
    void m_load_corpus_cache(FILE* fid);
    void m_save_corpus_cache(FILE* fid);
};