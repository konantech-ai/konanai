/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "ptb_corpus.h"
#include "../../core/common.h"
#include "../../core/util.h"
#include "../../core/random.h"
#include "../../core/log.h"


//hs.cho
#ifdef KAI2021_WINDOWS
#else
#define strtok_s strtok_r
#endif 

PtbCorpus::PtbCorpus(string name, string dic_path, string src_path) : Corpus(name, false) {
    logger.Print("corpus %s loading...", name.c_str());

    string data_path = KArgs::data_root + "corpus/" + src_path;
    string cache_path = KArgs::cache_root + "corpus/" + name ;

    Util::mkdir(KArgs::cache_root);
    Util::mkdir(KArgs::cache_root + "corpus");
    Util::mkdir(cache_path);

    cache_path += "/" + dic_path;

    try {
        load_cache(cache_path);
    }
    catch (exception err) {
        create_cache(data_path);
        save_cache(cache_path);
    }

    logger.Print("corpus loaded");
}

void PtbCorpus::create_cache(string data_path) {
    int64 next_wid = 0;
    m_w2i["<eos>"] = next_wid;
    m_i2w[next_wid++] = "<eos>";

    FILE* fid = Util::fopen(data_path.c_str(), "rt");

    char buffer[1024];

    while (!feof(fid)) {
        char* p = fgets(buffer, 1024, fid);
        if (p == NULL) break;
        vector<int64> sent;
        if (p[strlen(p) - 1] == '\n') p[strlen(p) - 1] = 0;
        char* context = NULL;
        char* token = strtok_s(buffer, " ", &context);
        while (token) {
            string word = token;
            if (m_w2i.find(word) == m_w2i.end()) {
                m_w2i[word] = next_wid;
                m_i2w[next_wid] = word;
                m_freq[next_wid] = 1;
                m_words.push_back(next_wid);
                sent.push_back(next_wid++);
            }
            else {
                int64 wid = m_w2i[word];
                m_freq[wid]++;
                m_words.push_back(wid);
                sent.push_back(wid);
            }
            token = strtok_s(NULL, " ", &context);
        }
        m_freq[0]++;
        sent.push_back(0);
        m_sents.push_back(sent);
    }
    fclose(fid);
}

void PtbCorpus::load_cache(string cache_path) {
    FILE* fid = Util::fopen(cache_path.c_str(), "rb");

    Util::read_map_si64(fid, m_w2i);
    Util::read_map_i64s(fid, m_i2w);
    Util::read_map_i64i64(fid, m_freq);
    Util::read_vv_i64(fid, m_sents);
    Util::read_v_i64(fid, m_words);

    fclose(fid);
}

void PtbCorpus::save_cache(string cache_path) {
    FILE* fid = Util::fopen(cache_path.c_str(), "wb");

    Util::save_map_si64(fid, m_w2i);
    Util::save_map_i64s(fid, m_i2w);
    Util::save_map_i64i64(fid, m_freq);
    Util::save_vv_i64(fid, m_sents);
    Util::save_v_i64(fid, m_words);

    fclose(fid);
}

PtbCorpus::~PtbCorpus() {
}

int64 PtbCorpus::get_next_sent(int64 sidx, bool* p_next_sent, float next_ratio, int64 max_len) {
    int64 sidx_next;

    for (int64 n = 0; n < 1000; n++) {
        bool b_next_sent = Random::uniform() <= next_ratio;

        if (b_next_sent) {
            sidx_next = sidx + 1;
        }
        else {
            sidx_next = Random::dice((int64)m_sents.size());
            if (sidx_next == sidx || sidx_next == sidx + 1) continue;
        }

        if (get_sent_length(sidx) + get_sent_length(sidx_next) > max_len) continue;

        *p_next_sent = b_next_sent;
        return sidx_next;
    }
    
    throw KaiException(KERR_ASSERT);
    return 0;
}

int64 PtbCorpus::get_random_word(int64 real_wid, int64* p_fake_tag_id) {
    while (true) {
        int64 wid = m_words[Random::dice((int64)m_words.size())];
        if (wid != real_wid) return wid;
    }
}

int64 PtbCorpus::pred_word_size(int64 window_size) {
    return (int64)m_words.size() - 2 * window_size;
}

void PtbCorpus::fill_word(int64*& wp, int64& cidx) {
    *wp++ = m_words[cidx++];
}

void PtbCorpus::fill_word_noms(int64*& wp, int64& cidx, int64 sample_cnt) {
    *wp++ = m_words[cidx++];
    for (int64 n = 1; n < sample_cnt; n++) {
        *wp++ = m_words[Random::dice((int64)m_words.size())];
    }
}
