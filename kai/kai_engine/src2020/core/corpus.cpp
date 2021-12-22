/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "corpus.h"

/*
#include "args.h"
#include "util.h"

Corpus::Corpus(string name, string dic_path, string src_path) {
    string cache_path = KArgs::result_root + dic_path;

    FILE* fid = fopen(cache_path.c_str(), "rb");

    if (fid != NULL) {
        m_load_corpus_cache(fid);
        fclose(fid);
    }
    else {
        m_load_corpus_text(src_path);
        FILE* fid = fopen(cache_path.c_str(), "wb");
        assert(fid != NULL);
        m_save_corpus_cache(fid);
        fclose(fid);
    }
}

void Corpus::m_load_corpus_text(string src_path) {
    string text_path = KArgs::data_root + "chap16/corpus/" + src_path;

    int next_wid = 0;
    m_w2i["<eos>"] = next_wid;
    m_i2w[next_wid++] = "<eos>";

    FILE* fid = fopen(text_path.c_str(), "rt");

    char buffer[1024];

    while (!feof(fid)) {
        char* p = fgets(buffer, 1024, fid);
        if (p == NULL) break;
        vector<int> sent;
        if (p[strlen(p) - 1] == '\n') p[strlen(p) - 1] = 0;
        char* token = strtok(buffer, " ");
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
                int wid = m_w2i[word];
                m_freq[wid]++;
                m_words.push_back(wid);
                sent.push_back(wid);
            }
            token = strtok(NULL, " ");
        }
        m_freq[0]++;
        sent.push_back(0);
        m_sents.push_back(sent);
    }
    fclose(fid);
}

void Corpus::m_load_corpus_cache(FILE* fid) {
    Util::read_map_si(fid, m_w2i);
    Util::read_map_is(fid, m_i2w);
    Util::read_map_ii(fid, m_freq);
    Util::read_vv_int(fid, m_sents);
    Util::read_v_int(fid, m_words);
}

void Corpus::m_save_corpus_cache(FILE* fid) {
    Util::save_map_si(fid, m_w2i);
    Util::save_map_is(fid, m_i2w);
    Util::save_map_ii(fid, m_freq);
    Util::save_vv_int(fid, m_sents);
    Util::save_v_int(fid, m_words);
}

Corpus::~Corpus() {
}
*/