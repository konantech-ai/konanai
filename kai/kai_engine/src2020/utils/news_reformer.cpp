/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "news_reformer.h"
#include "../core/value.h"
#include "../core/util.h"
#include "../core/log.h"

#include <stdio.h>
#include <sys/stat.h>

NewsReformer::NewsReformer() {
	for (int n = 0; n < 10; n++) {
		char fname1[128], fname2[128];

#ifdef KAI2021_WINDOWS
		throw KaiException(KERR_ASSERT);
#else
		sprintf(fname1, "hangyure.%d", 2010 + n);
		sprintf(fname2, "kyunghyang.%d", 2010 + n);
#endif
		
		m_fnames.push_back(fname1);
		m_fnames.push_back(fname2);
	}

	m_nMaxFreqCnt = 100000;
	m_nDictSize = 30000;
	m_pFreqDist = new int[m_nMaxFreqCnt];
	memset(m_pFreqDist, 0, sizeof(int) * m_nMaxFreqCnt);

	m_time1 = time(NULL);

	m_seq_wid = 0;

#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	setlocale(LC_NUMERIC, "");
#endif
}

NewsReformer::~NewsReformer() {
	for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
		struct DicTermInfo* term = it->second;
		delete term;
	}
}

void NewsReformer::exec_all() {
	exec_collect_words();
	exec_create_dict(false);
	exec_replace_words(false);
}

void NewsReformer::exec_collect_words() {
	for (vector<string>::iterator it = m_fnames.begin(); it != m_fnames.end(); it++) {
		m_time2 = time(NULL);

		string fname = *it;
		logger.Print("fname: %s", fname.c_str());

		char text_path[1024];
		char full_path[1024];
		char temp_path[1024];
		char vocs_path[1024];

#ifdef KAI2021_WINDOWS
		throw KaiException(KERR_ASSERT);
		mysprintf(text_path, "/home/ubuntu/work/cuda/textdata/text/news.%s_00.csv", fname.c_str());
		mysprintf(full_path, "/home/ubuntu/work/cuda/textdata/full/anal.%s_00.json", fname.c_str());
		mysprintf(temp_path, "/home/ubuntu/work/cuda/textdata/temp/temp.%s_00.json", fname.c_str());
		mysprintf(vocs_path, "/home/ubuntu/work/cuda/textdata/temp/vocs.%s_00.json", fname.c_str());
#else
		sprintf(text_path, "/home/ubuntu/work/cuda/textdata/text/news.%s_00.csv", fname.c_str());
		sprintf(full_path, "/home/ubuntu/work/cuda/textdata/full/anal.%s_00.json", fname.c_str());
		sprintf(temp_path, "/home/ubuntu/work/cuda/textdata/temp/temp.%s_00.json", fname.c_str());
		sprintf(vocs_path, "/home/ubuntu/work/cuda/textdata/temp/vocs.%s_00.json", fname.c_str());
#endif

		struct stat st;

		if (stat(text_path, &st) == 0) {
			int64 text_size = st.st_size;
			logger.Print("%s: %'lld bytes", text_path, text_size);
		}
		else {
			logger.Print("Error: open %s failure", text_path);
		}

		if (stat(full_path, &st) == 0) {
			int64 full_size = st.st_size;
			logger.Print("%s: %'lld bytes", full_path, full_size);
		}
		else {
			logger.Print("Error: open %s failure", full_path);
		}

#ifdef KAI2021_WINDOWS
		throw KaiException(KERR_ASSERT);
		FILE* fin = NULL;
		FILE* fout = NULL;
#else
		FILE* fin = fopen(full_path, "rt");
		FILE* fout = fopen(temp_path, "wt");
#endif

		m_proc_collect_words(fin, fout);

		fclose(fin);
		fclose(fout);

		FILE* fvoc = Util::fopen(vocs_path, "wb");
		
		fwrite(&m_seq_wid, sizeof(int), 1, fvoc);
		fwrite(&m_nDictSize, sizeof(int), 1, fvoc);
		fwrite(&m_nMaxFreqCnt, sizeof(int), 1, fvoc);
		fwrite(m_pFreqDist, sizeof(int), m_nMaxFreqCnt, fvoc);

		int dic_size = (int)m_dic.size();
		fwrite(&dic_size, sizeof(int), 1, fvoc);

		for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
			string word = it->first;
			DicTermInfo* term = it->second;

			int str_leng = (int)word.length();
			fwrite(&str_leng, sizeof(int), 1, fvoc);
			fwrite(word.c_str(), sizeof(char), str_leng, fvoc);

			fwrite(&term->m_nth, sizeof(int), 1, fvoc);
			fwrite(&term->m_freq, sizeof(int), 1, fvoc);

			str_leng = (int)term->m_tag.length();
			fwrite(&str_leng, sizeof(int), 1, fvoc);
			fwrite(term->m_tag.c_str(), sizeof(char), str_leng, fvoc);

			str_leng = (int)term->m_lemma.length();
			fwrite(&str_leng, sizeof(int), 1, fvoc);
			fwrite(term->m_lemma.c_str(), sizeof(char), str_leng, fvoc);
		}

		fclose(fvoc);
	}

	/*
	throw KaiException(KERR_ASSERT);

	m_time2 = time(NULL);

	int acc_cnt = 0;

	for (int n = m_nMaxFreqCnt - 1; n > 0; n--) {
		if (acc_cnt + m_pFreqDist[n] > m_nDictSize) {
			m_nFreqThreshold = n;
			break;
		}
		acc_cnt += m_pFreqDist[n];
	}

	delete[] m_pFreqDist;

	logger.Print("FreqThreshold = %d", m_nFreqThreshold);

	int next_voc_id = 0;

	char dict_path[1024];

	sprintf(dict_path, "/home/ubuntu/work/cuda/textdata/dict/knews.voc");

	FILE* fdic = fopen(dict_path, "wt");

	map<string, int> void_map;

	for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
		struct DicTermInfo* term = it->second;

		if (term->m_freq <= m_nFreqThreshold) {
			term->m_lemma = "";
			if (void_map.find(term->m_tag) != void_map.end()) {
				term->m_voc_id = void_map[term->m_tag];
				continue;
			}
			void_map[term->m_tag] = next_voc_id;
		}

		term->m_voc_id = next_voc_id++;

		Dict dic_term;

		dic_term["wid"] = term->m_voc_id;
		dic_term["freq"] = term->m_freq;
		dic_term["tag"] = term->m_tag;
		dic_term["string"] = term->m_lemma;

		Value vterm = dic_term;

		fprintf(fdic, "%s\n", vterm.description().c_str());
	}

	fclose(fdic);

	time_t tm = time(NULL);

	logger.Print("total word counts in data, %d entries in dictionary: %d (%d, %d secs)", (int) m_dic.size(), next_voc_id, (int)(tm - m_time2), (int)(tm - m_time1));

	m_time2 = time(NULL);

	for (vector<string>::iterator it = m_fnames.begin(); it != m_fnames.end(); it++) {
		m_time3 = time(NULL);

		string fname = *it;

		char temp_path[1024];
		char term_path[1024];

		sprintf(temp_path, "/home/ubuntu/work/cuda/textdata/temp/temp.%s_00.json", fname.c_str());
		sprintf(term_path, "/home/ubuntu/work/cuda/textdata/term/term.%s_00.json", fname.c_str());

		FILE* fin = fopen(temp_path, "rt");
		FILE* fout = fopen(term_path, "wt");

		m_replace_words(fin, fout);

		fclose(fin);
		fclose(fout);

		logger.Print("final term file %s created", fname.c_str());
	}

	for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
		struct DicTermInfo* term = it->second;
		delete term;
	}
	*/
}

void NewsReformer::exec_create_dict(bool need_load, string fname) {
	if (need_load) {
		if (fname == "") {
			int idx = (int) m_fnames.size() - 1;
			fname = m_fnames[idx];
		}

		char vocs_path[1024];

#ifdef KAI2021_WINDOWS
		throw KaiException(KERR_ASSERT);
		mysprintf(vocs_path, "/home/ubuntu/work/cuda/textdata/temp/vocs.%s_00.json", fname.c_str());

		FILE* fvoc = Util::fopen(vocs_path, "rb");
#else
		sprintf(vocs_path, "/home/ubuntu/work/cuda/textdata/temp/vocs.%s_00.json", fname.c_str());

		FILE* fvoc = fopen(vocs_path, "rb");
#endif

		fread(&m_seq_wid, sizeof(int), 1, fvoc);
		fread(&m_nDictSize, sizeof(int), 1, fvoc);
		fread(&m_nMaxFreqCnt, sizeof(int), 1, fvoc);
		fread(m_pFreqDist, sizeof(int), m_nMaxFreqCnt, fvoc);

		int dic_size;
		fread(&dic_size, sizeof(int), 1, fvoc);

		char buffer[1024];

		for (int n = 0; n < dic_size; n++) {
			int str_leng;
			fread(&str_leng, sizeof(int), 1, fvoc);
			assert(str_leng < 1024);
			fread(buffer, sizeof(char), str_leng, fvoc);
			buffer[str_leng] = 0;
			string word = buffer;

			DicTermInfo* term = new DicTermInfo;

			fread(&term->m_nth, sizeof(int), 1, fvoc);
			fread(&term->m_freq, sizeof(int), 1, fvoc);

			fread(&str_leng, sizeof(int), 1, fvoc);
			assert(str_leng < 1024);
			fread(buffer, sizeof(char), str_leng, fvoc);
			buffer[str_leng] = 0;
			term->m_tag = buffer;

			fread(&str_leng, sizeof(int), 1, fvoc);
			assert(str_leng < 1024);
			fread(buffer, sizeof(char), str_leng, fvoc);
			buffer[str_leng] = 0;
			term->m_lemma = buffer;

			m_dic[word] = term;
		}

		fclose(fvoc);
	}

	m_time2 = time(NULL);

	int acc_cnt = 0;

	for (int n = m_nMaxFreqCnt - 1; n > 0; n--) {
		if (acc_cnt + m_pFreqDist[n] > m_nDictSize) {
			m_nFreqThreshold = n;
			break;
		}
		acc_cnt += m_pFreqDist[n];
	}

	delete[] m_pFreqDist;

	logger.Print("FreqThreshold = %d", m_nFreqThreshold);

	int next_voc_id = 0;

	string voc_file_name = "knews_paper.voc";
	char dic_path[1024];
#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
	mysprintf(dic_path, "/home/ubuntu/work/cuda/textdata/v0.1/%s", voc_file_name.c_str());
#else
	sprintf(dic_path, "/home/ubuntu/work/cuda/textdata/v0.1/%s", voc_file_name.c_str());
#endif

	map<string, int> void_map;

	for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
		struct DicTermInfo* term = it->second;

		if (term->m_freq <= m_nFreqThreshold) {
			if (void_map.find(term->m_tag) == void_map.end()) {
				void_map[term->m_tag] = next_voc_id++;
			}
		}
		else {
			term->m_voc_id = next_voc_id++;
		}
	}

	int rest = m_nDictSize - next_voc_id;
	int string_pool_size = 0;

	set<string> tag_names;

	FILE* fdic = Util::fopen(dic_path, "wt");

	for (map<string, DicTermInfo*>::iterator it = m_dic.begin(); it != m_dic.end(); it++) {
		struct DicTermInfo* term = it->second;

		if (term->m_freq <= m_nFreqThreshold) {
			if (term->m_freq < m_nFreqThreshold || rest <= 0) {
				term->m_voc_id = void_map[term->m_tag];
				m_index[term->m_nth] = term->m_voc_id;
				continue;
			}

			rest--;
			term->m_voc_id = next_voc_id++;
		}

		Dict dic_term;

		dic_term["wid"] = term->m_voc_id;
		dic_term["freq"] = term->m_freq;
		dic_term["tag"] = term->m_tag;
		dic_term["string"] = term->m_lemma;

		string_pool_size += (int) term->m_lemma.size() + 1;
		tag_names.insert(term->m_tag);

		m_index[term->m_nth] = term->m_voc_id;

		fprintf(fdic, "%s\n", Value::description(dic_term).c_str());
	}

	for (map<string, int>::iterator it = void_map.begin(); it != void_map.end(); it++) {
		string tag = it->first;
		int voc_id = it->second;

		Dict dic_term;

		dic_term["wid"] = voc_id;
		dic_term["freq"] = 0;
		dic_term["tag"] = tag;
		dic_term["string"] = "";

		string_pool_size++;
		tag_names.insert(tag);

		fprintf(fdic, "%s\n", Value::description(dic_term).c_str());
	}

	fclose(fdic);

	List tag_list;
	
	for (set<string>::iterator it = tag_names.begin(); it != tag_names.end(); it++) {
		tag_list.push_back(*it);
	}

	m_dic_map["voc_file"] = voc_file_name;
	m_dic_map["voc_count"] = next_voc_id;
	m_dic_map["string_pool_size"] = string_pool_size;
	m_dic_map["tags"] = tag_list;

	time_t tm = time(NULL);

	logger.Print("total word counts in data, %d entries in dictionary: %d (%d, %d secs)", (int)m_dic.size(), next_voc_id, (int)(tm - m_time2), (int)(tm - m_time1));
}

void NewsReformer::m_replace_words(FILE* fin, FILE* fout, int& art_cnt, int& sent_cnt, int& word_cnt) {
#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	char* chunk = NULL;
	size_t len = 0;
	int64 chunk_cnt = 0;

	while ((getline(&chunk, &len, fin)) != -1) {
		if (strcmp(chunk, "]\n") == 0) {
			logger.Print("    * Bad chunk in %lld-th article: %s", chunk_cnt, chunk);
			continue;
		}

		List article_wids = Value::parse_list(chunk);
		List article_vids;

		chunk_cnt++;
		art_cnt++;

		for (int ns = 0; ns < (int)article_wids.size(); ns++) {
			List sent_wids = article_wids[ns];
			List sent_vids;
			sent_cnt++;
			word_cnt += (int)sent_wids.size();

			for (int nw = 0; nw < (int)sent_wids.size(); nw++) {
				int wid = sent_wids[nw];
				int voc_id = m_index[wid];
				sent_vids.push_back(voc_id);
			}

			article_vids.push_back(sent_vids);
		}

		Value art_vids = article_vids;
		fprintf(fout, "%s\n", art_vids.description().c_str());

		time_t tm = time(NULL);
		if (chunk_cnt % 10000 == 0) logger.Print("    %lld chunks are processed (%d, %d, %d secs)", chunk_cnt, (int)(tm - m_time3), (int)(tm - m_time2), (int)(tm-m_time1));
		m_time3 = tm;
	}
#endif
}

void NewsReformer::m_proc_collect_words(FILE* fin, FILE* fout) {
#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	m_time3 = time(NULL);

	int64 total_size = 0;
	int64 chunk_cnt = 0;

	char* chunk = NULL;
	size_t len = 0;

	while ((getline(&chunk, &len, fin)) != -1) {
		total_size += (int64) strlen(chunk);
		chunk_cnt++;

		List json = Value::parse_list(chunk);
		List article_wids;

		for (int n = 0; n < (int)json.size(); n++) {
			Dict entry = json[n];
			List sents = entry["sents"];
			
			for (int ns = 0; ns < (int)sents.size(); ns++) {
				Dict sent = sents[ns];
				assert(sent.size() == 1);
				List words = sent["words"];
				List sent_wids;

				for (int nw = 0; nw < (int)words.size(); nw++) {
					Dict word = words[nw];
					List nbest = word["nbest"];
					Dict best = nbest[0];
					List lemmas = best["lemmas"];

					string str_word = word["string"];

					for (int nm = 0; nm < (int)lemmas.size(); nm++) {
						Dict lemma = lemmas[nm];
						string str_lemma = lemma["string"];
						string str_tag = lemma["tag"];

						if (str_tag == "NU") str_lemma = "";
						//ms_trim(str_lemma);

						string lemma_pair = str_lemma + str_tag;

						if (m_dic.find(lemma_pair) != m_dic.end()) {
							struct DicTermInfo* term = m_dic[lemma_pair];
							term->m_freq++;
							if (term->m_freq < m_nMaxFreqCnt) {
								m_pFreqDist[term->m_freq - 1]--;
								m_pFreqDist[term->m_freq]++;
							}
							sent_wids.push_back(term->m_nth);
						}
						else {
							DicTermInfo* term = new DicTermInfo;;
							term->m_nth = m_seq_wid++;
							term->m_freq = 1;
							term->m_voc_id = 0;
							term->m_tag = str_tag;
							term->m_lemma = str_lemma;
							m_dic[lemma_pair] = term;
							sent_wids.push_back(term->m_nth);
							m_pFreqDist[1]++;
						}
					}
				}
				article_wids.push_back(sent_wids);
			}
		}

		Value art_wids = article_wids;
		fprintf(fout, "%s\n", art_wids.description().c_str());

		if (chunk_cnt == 2) {
			break;
			throw KaiException(KERR_ASSERT);
		}
		
		logger.Print("chunk %lld: length of json list = %lld", chunk_cnt, (int64) json.size());
		//hs.cho
		//logger.Print("%lld: %lld, %lld => %'lld", chunk_cnt, (int64)len, (int64)strlen(line), total_size);
		logger.Print("%lld: %lld, %lld => %'lld", chunk_cnt, (int64)len, (int64)strlen(chunk), total_size);
		time_t tm = time(NULL);

		if (chunk_cnt % 1000 == 0) {
			logger.Print("    %lld chunks are checked: %'lld bytes (%d, %d, %d secs): %'d words",
				chunk_cnt, total_size, (int)(tm - m_time3), (int)(tm - m_time2), (int)(tm - m_time1), m_seq_wid);
			m_time3 = tm;
		}
	}
	
	logger.Print("total %'lld chunks: %'lld bytes", chunk_cnt, total_size);

	if (chunk) free(chunk);
#endif
}

void NewsReformer::exec_replace_words(bool need_load) {
#ifdef KAI2021_WINDOWS
	throw KaiException(KERR_ASSERT);
#else
	if (need_load) throw KaiException(KERR_ASSERT);

	m_time2 = time(NULL);

	List wid_files;

	int art_cnt = 0;
	int sent_cnt = 0;
	int word_cnt = 0;

	for (vector<string>::iterator it = m_fnames.begin(); it != m_fnames.end(); it++) {
		m_time3 = time(NULL);

		string fname = *it;

		char temp_path[1024];
		char wids_path[1024];

		sprintf(wids_path, "wids.%s_00.json", fname.c_str());
		wid_files.push_back((string) wids_path);

		sprintf(temp_path, "/home/ubuntu/work/cuda/textdata/temp/temp.%s_00.json", fname.c_str());
		sprintf(wids_path, "/home/ubuntu/work/cuda/textdata/v0.1/wids.%s_00.json", fname.c_str());

		FILE* fin = fopen(temp_path, "rt");
		FILE* fout = fopen(wids_path, "wt");

		assert(fin != NULL);
		assert(fout != NULL);

		logger.Print("replace_words(%s) started", fname.c_str());

		m_replace_words(fin, fout, art_cnt, sent_cnt, word_cnt);

		fclose(fin);
		fclose(fout);

		logger.Print("final term file %s created", fname.c_str());
	}

	m_dic_map["wid_files"] = wid_files;
	m_dic_map["article_count"] = art_cnt;
	m_dic_map["sentence_count"] = sent_cnt;
	m_dic_map["word_count"] = word_cnt;

	string map_file_name = "map.dat";
	char map_path[1024];
	sprintf(map_path, "/home/ubuntu/work/cuda/textdata/v0.1/%s", map_file_name.c_str());

	FILE* fmap = fopen(map_path, "wt");
	fprintf(fmap, "%s", Value::description(m_dic_map).c_str());
	fclose(fmap);
#endif
}
