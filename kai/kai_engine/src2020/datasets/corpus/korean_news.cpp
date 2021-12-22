/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "korean_news.h"

#include "../../core/value.h"
#include "../../core/random.h"

KoreanNewsCorpus::KoreanNewsCorpus(string name, string folder_name) : Corpus(name, true) {
    m_voc_count = 0;
    m_tag_count = 0;

    m_article_cnt = 0;
    m_sentence_cnt = 0;
    m_word_cnt = 0;
    m_string_pool_size = 0;

    m_word_tags = NULL;
    m_wstr_index = NULL;
    m_string_pool = NULL;

    m_arts_corpus = NULL;
    m_arts_length = NULL;
    m_sents_corpus = NULL;
    m_sents_length = NULL;
    m_words_corpus = NULL;

    m_is_last_sent = NULL;

#ifdef KAI2021_WINDOWS
    throw KaiException(KERR_ASSERT);
#else
    string cache_path = KArgs::cache_root + folder_name + ".cache";

    FILE* fid = fopen(cache_path.c_str(), "rb");

    if (fid != NULL) {
        m_load_corpus_cache(fid);
        fclose(fid);
    }
    else {
        m_load_corpus_contents(folder_name);
        FILE* fid = fopen(cache_path.c_str(), "wb");
        assert(fid != NULL);
        m_save_corpus_cache(fid);
        fclose(fid);
    }

    m_adjust_ratio = (float) m_sentence_cnt / (float) (m_sentence_cnt - m_article_cnt);
#endif
}

KoreanNewsCorpus::~KoreanNewsCorpus() {
    delete[] m_word_tags;
    delete[] m_wstr_index;
    delete[] m_string_pool;

    delete[] m_arts_corpus;
    delete[] m_arts_length;
    delete[] m_sents_corpus;
    delete[] m_sents_length;
    delete[] m_words_corpus;

    delete[] m_is_last_sent;
}

void KoreanNewsCorpus::m_load_corpus_contents(string folder_name) {
#ifdef KAI2021_WINDOWS
    throw KaiException(KERR_ASSERT);
#else
    char* chunk = NULL;
    size_t len = 0;

    string data_path = KArgs::data_root + "kbert/" + folder_name + "/";
    string map_path = data_path + "map.dat";

    //temp_gen_map(map_path);
    /*
    {'article_count':1371939, 'sentence_count' : 22571749, 'string_pool_size' : 232970, 'tags' : ['AD', 'AJ', 'AX', 'CP', 'DN', 'DT', 'EF', 'EP', 'IJ', 'NN', 'NP', 'NU', 'NX', 'PF
        ','PP','SF','SJ','SN','SV','SY','VV','VX','ZZ'],'wid_files':['wids.hangyure.2010_00.json','wids.kyunghyang
        .2010_00.json','wids.hangyure.2011_00.json','wids.kyunghyang.2011_00.json','wids.hangyure.2012_00.json','wids.kyunghyang.2012_00.json','wids.hangyure.201
        3_00.json','wids.kyunghyang.2013_00.json','wids.hangyure.2014_00.json','wids.kyunghyang.2014_00.json','wids.hangyure.2015_00.json','wids.kyunghyang.2015_
        00.json','wids.hangyure.2016_00.json','wids.kyunghyang.2016_00.json','wids.hangyure.2017_00.json','wids.kyunghyang.2017_00.json','wids.hangyure.2018_00.j
        son','wids.kyunghyang.2018_00.json','wids.hangyure.2019_00.json','wids.kyunghyang.2019_00.json'],'word_count':784,725,814}
        */

    // {'freq':479,'string':'???','tag':'SY','wid':35}

    FILE* fmap = fopen(map_path.c_str(), "rt");
    getline(&chunk, &len, fmap);
    Dict map= Value::parse_dict(chunk);
    fclose(fmap);

    m_voc_count = map["voc_count"];
    m_string_pool_size = map["string_pool_size"];

    m_article_cnt = map["article_count"];
    m_sentence_cnt = map["sentence_count"];
    m_word_cnt = map["word_count"];

    m_word_tags = new int64[m_voc_count];
    m_wstr_index = new int64[m_voc_count];
    m_string_pool = new char [m_string_pool_size];

    memset(m_string_pool, 0, m_string_pool_size);
    memset(m_wstr_index, 0, sizeof(int64)* m_voc_count);

    List tags = map["tags"];
    m_tag_count = (int64)tags.size();

    int64 tag_idx = 0;
    for (List::iterator it = tags.begin(); it != tags.end(); it++) {
        string tag = *it;
        m_tag_to_str[tag_idx] = tag;
        m_str_to_tag[tag] = tag_idx;
        tag_idx++;
    }

    string dic_path = data_path + (string) map["voc_file"];

    FILE* fdic = fopen(dic_path.c_str(), "rt");

    int64 count_chk = 0;
    int64 string_pool_idx = 0;

    while ((getline(&chunk, &len, fdic)) != -1) {
        Dict voc_term = Value::parse_dict(chunk);
        int64 wid = voc_term["wid"];
        string tag = voc_term["tag"];
        string word = voc_term["string"];
        assert(wid >= 0 && wid < m_voc_count);
        m_word_tags[wid] = m_str_to_tag[tag];
        m_wstr_index[wid] = string_pool_idx;
        strcpy(m_string_pool + string_pool_idx, word.c_str());
        string_pool_idx += (int64)word.size() + 1;
        count_chk++;
    }

    assert(count_chk == m_voc_count);
    assert(string_pool_idx == m_string_pool_size);

    fclose(fdic);

    m_arts_corpus = new int64[m_article_cnt];
    m_arts_length = new int64[m_article_cnt];
    m_sents_corpus = new int64[m_sentence_cnt];
    m_sents_length = new int64[m_sentence_cnt];
    m_words_corpus = new int64[m_word_cnt];
    
    m_is_last_sent = new char[m_sentence_cnt];

    memset(m_is_last_sent, 0, m_sentence_cnt);

    int64 art_idx = 0;
    int64 sent_idx = 0;
    int64 word_idx = 0;

    List files = map["wid_files"];

    for (List::iterator it = files.begin(); it != files.end(); it++) {
        string file_path = data_path + (string) *it;
        logger.Print("loading %s...", file_path.c_str());

        FILE* fin = fopen(file_path.c_str(), "rt");

        while ((getline(&chunk, &len, fin)) != -1) {
            List article_wids = Value::parse_list(chunk);
            List article_vids;

            m_arts_corpus[art_idx] = sent_idx;

            for (int64 ns = 0; ns < (int64)article_wids.size(); ns++) {
                List sent_wids = article_wids[ns];
                List sent_vids;

                if ((int64)sent_wids.size() > 300) continue; // skip too long sentence

                m_sents_corpus[sent_idx] = word_idx;
                m_sents_length[sent_idx++] = (int64)sent_wids.size();

                for (int64 nw = 0; nw < (int64)sent_wids.size(); nw++) {
                    m_words_corpus[word_idx++] = sent_wids[nw];
                }
            }

            if (sent_idx == m_arts_corpus[art_idx]) continue; // empty article due to remove of too long sentences

            m_arts_length[art_idx] = sent_idx - m_arts_corpus[art_idx];
            m_is_last_sent[sent_idx - 1] = 1;

            art_idx++;
        }
        fclose(fin);
    }

    assert(art_idx <= m_article_cnt);    // due to skip too long sentence
    assert(sent_idx <= m_sentence_cnt);  // due to skip too long sentence
    assert(word_idx <= m_word_cnt);      // due to skip too long sentence

    m_article_cnt = art_idx;
    m_sentence_cnt = sent_idx;
    m_word_cnt = word_idx;
#endif
}

void KoreanNewsCorpus::m_load_corpus_cache(FILE* fid) {
    fread(&m_voc_count, sizeof(int64), 1, fid);
    fread(&m_tag_count, sizeof(int64), 1, fid);
    fread(&m_article_cnt, sizeof(int64), 1, fid);
    fread(&m_sentence_cnt, sizeof(int64), 1, fid);
    fread(&m_word_cnt, sizeof(int64), 1, fid);
    fread(&m_string_pool_size, sizeof(int64), 1, fid);

    m_arts_corpus = new int64[m_article_cnt];
    m_arts_length = new int64[m_article_cnt];
    m_sents_corpus = new int64[m_sentence_cnt];
    m_sents_length = new int64[m_sentence_cnt];
    m_words_corpus = new int64[m_word_cnt];

    fread(m_arts_corpus, sizeof(int64), m_article_cnt, fid);
    fread(m_arts_length, sizeof(int64), m_article_cnt, fid);
    fread(m_sents_corpus, sizeof(int64), m_sentence_cnt, fid);
    fread(m_sents_length, sizeof(int64), m_sentence_cnt, fid);
    fread(m_words_corpus, sizeof(int64), m_word_cnt, fid);

    m_word_tags = new int64[m_voc_count];
    m_wstr_index = new int64[m_voc_count];

    fread(m_word_tags, sizeof(int64), m_voc_count, fid);
    fread(m_wstr_index, sizeof(int64), m_voc_count, fid);

    m_string_pool = new char[m_string_pool_size];

    fread(m_string_pool, sizeof(char), m_string_pool_size, fid);

    string tag_desc;
    Value::serial_load(fid, tag_desc);
    m_str_to_tag = Value::parse_dict(tag_desc.c_str());

    for (Dict::iterator it = m_str_to_tag.begin(); it != m_str_to_tag.end(); it++) {
        string tag = it->first;
        int64 tag_id = it->second;
        m_tag_to_str[tag_id] = tag;
    }

    m_is_last_sent = new char[m_sentence_cnt];
    memset(m_is_last_sent, 0, m_sentence_cnt);

    for (int64 n = 0; n < m_article_cnt; n++) {
        m_is_last_sent[m_arts_corpus[n] + m_arts_length[n] - 1] = 1;
    }
}

void KoreanNewsCorpus::m_save_corpus_cache(FILE* fid) {
    fwrite(&m_voc_count, sizeof(int64), 1, fid);
    fwrite(&m_tag_count, sizeof(int64), 1, fid);
    fwrite(&m_article_cnt, sizeof(int64), 1, fid);
    fwrite(&m_sentence_cnt, sizeof(int64), 1, fid);
    fwrite(&m_word_cnt, sizeof(int64), 1, fid);
    fwrite(&m_string_pool_size, sizeof(int64), 1, fid);

    fwrite(m_arts_corpus, sizeof(int64), m_article_cnt, fid);
    fwrite(m_arts_length, sizeof(int64), m_article_cnt, fid);
    fwrite(m_sents_corpus, sizeof(int64), m_sentence_cnt, fid);
    fwrite(m_sents_length, sizeof(int64), m_sentence_cnt, fid);
    fwrite(m_words_corpus, sizeof(int64), m_word_cnt, fid);

    fwrite(m_word_tags, sizeof(int64), m_voc_count, fid);
    fwrite(m_wstr_index, sizeof(int64), m_voc_count, fid);

    fwrite(m_string_pool, sizeof(char), m_string_pool_size, fid);

    string tag_desc = Value::description(m_str_to_tag);
    Value::serial_save(fid, tag_desc);
}


int64 KoreanNewsCorpus::voc_size() {
    return m_voc_count;
}

int64 KoreanNewsCorpus::tag_size() {
    return m_tag_count;
}

int64 KoreanNewsCorpus::corpus_word_count() {
    return m_word_cnt;
}

int64 KoreanNewsCorpus::corpus_sent_count() {
    return m_sentence_cnt;
}

string KoreanNewsCorpus::get_word_to_visualize(int64 wid) {
    int64 tag = m_word_tags[wid];
    return "(" + m_tag_to_str[tag] + " " + (string)(m_string_pool + m_wstr_index[wid]) + ")";
}

int64 KoreanNewsCorpus::get_next_sent(int64 sidx, bool* p_next_sent, float next_ratio, int64 max_len) {
    float ratio = next_ratio * m_adjust_ratio;
    
    if (m_is_last_sent[sidx] == 0 && Random::uniform() < ratio) {
        if (m_sents_length[sidx] + m_sents_length[sidx + 1] <= max_len) {
            *p_next_sent = true;
            return sidx + 1;
        }
    }
    
    for (int64 n = 0; n < 1000; n++) {
        int64 sidx2 = Random::dice(m_sentence_cnt);
        if (sidx2 == sidx + 1) continue;
        if (m_sents_length[sidx] + m_sents_length[sidx2] > max_len) continue;
        *p_next_sent = false;
        return sidx2;
    }

    throw KaiException(KERR_ASSERT);

    return 0;
}

int64 KoreanNewsCorpus::get_sent_length(int64 sidx) {
    return m_sents_length[sidx];
}

int64 KoreanNewsCorpus::get_wid(int64 sidx, int64 nth) {
    int64 widx = m_sents_corpus[sidx] + nth;
    return m_words_corpus[widx];
}

// 문장 경계 고려되지 않은 상태임
int64 KoreanNewsCorpus::pred_word_size(int64 window_size) {
    return m_word_cnt - 2 * window_size;
}

int64 KoreanNewsCorpus::get_nth_word(int64 nth) {
    return m_words_corpus[nth];
}

int64 KoreanNewsCorpus::get_tag_id(int64 sidx, int64 nth) {
    int64 widx = m_sents_corpus[sidx] + nth;
    int64 wid = m_words_corpus[widx];
    return m_word_tags[wid];
}

int64 KoreanNewsCorpus::get_random_word(int64 real_wid, int64* p_fake_tag_id) {
    while (true) {
        int64 widx = Random::dice(m_word_cnt);
        int64 wid = m_words_corpus[widx];

        if (wid == real_wid) continue;

        *p_fake_tag_id = m_word_tags[wid];

        return wid;
    }
}

void KoreanNewsCorpus::fill_word(int64*& wp, int64& cidx) {
    throw KaiException(KERR_ASSERT);
}

void KoreanNewsCorpus::fill_word_noms(int64*& wp, int64& cidx, int64 sample_cnt) {
    throw KaiException(KERR_ASSERT);
}
