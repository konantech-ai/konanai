/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "automata.h"
#include "../core/log.h"
#include "../core/random.h"

AutomataDataset::AutomataDataset() : Dataset("automata", "binary") {
    m_max_length = 30;
    m_min_length = 10;

    m_alphabet["eos"] = "$";
    m_alphabet["add_op"] = "+-";
    m_alphabet["mult_op"] = "*/";
    m_alphabet["lparen"] = "(";
    m_alphabet["rparen"] = ")";
    m_alphabet["alpha"] = "abcdefghijklmnopqrstuvwxyz";
    m_alphabet["digit"] = "0123456789";
    m_alphabet["symbols"] = m_alphabet["eos"] + m_alphabet["add_op"] + m_alphabet["mult_op"] + m_alphabet["lparen"] + m_alphabet["rparen"];
    m_alphabet["alphanum"] = m_alphabet["alpha"] + m_alphabet["digit"];
    m_alphabet["alphabet"] = m_alphabet["symbols"] + m_alphabet["alphanum"];

    m_alphabet_size =  m_alphabet["alphabet"].length();

    m_rules = Value::parse_dict("{ \
                'S' : [['E']], \
                'E' : [['T'], ['E', 'add_op', 'T']], \
                'T' : [['F'], ['T', 'mult_op', 'F']], \
                'F' : [['V'], ['N'], ['lparen', 'E', 'rparen']], \
                'V' : [['alpha'], ['alpha', 'V2']], \
                'V2': [['alphanum'], ['alphanum', 'V2']], \
                'N' : [['digit'], ['digit', 'N']]}");

    m_alphabet["E_next"] = m_alphabet["eos"] + m_alphabet["add_op"] + m_alphabet["rparen"];
    m_alphabet["T_next"] = m_alphabet["E_next"] + m_alphabet["mult_op"];
    m_alphabet["F_next"] = m_alphabet["T_next"];
    m_alphabet["V_next"] = m_alphabet["F_next"];
    m_alphabet["N_next"] = m_alphabet["F_next"];

    m_action_table = Value::parse_dict("{ \
                '0': [['alpha', 6], ['digit', 7], ['lparen', 8]], \
                '1': [['add_op', 9], ['eos', 0]], \
                '2' : [['mult_op', 10], ['E_next', -1, 'E']], \
                '3' : [['T_next', -1, 'T']], \
                '4' : [['F_next', -1, 'F']], \
                '5' : [['F_next', -1, 'F']], \
                '6' : [['alphanum', 6], ['V_next', -1, 'V']], \
                '7' : [['digit', 7], ['N_next', -1, 'N']], \
                '8' : [['alpha', 6], ['digit', 7], ['lparen', 8]], \
                '9' : [['alpha', 6], ['digit', 7], ['lparen', 8]], \
                '10' : [['alpha', 6], ['digit', 7], ['lparen', 8]], \
                '11' : [['V_next', -2, 'V']], \
                '12' : [['N_next', -2, 'N']], \
                '13' : [['rparen', 16], ['add_op', 9]], \
                '14' : [['mult_op', 10], ['T_next', -3, 'T']], \
                '15' : [['F_next', -3, 'F']], \
                '16' : [['F_next', -3, 'F']]}");

    m_goto_table = Value::parse_dict("{ \
                '0': { 'E' : '1', 'T' : '2', 'F' : '3', 'V' : '4', 'N' : '5' }, \
                '6': { 'V' : '11' }, \
                '7' : { 'N' : '12' }, \
                '8' : { 'E' : '13', 'T' : '2', 'F' : '3', 'V' : '4', 'N' : '5' }, \
                '9' : { 'T' : '14', 'F' : '3', 'V' : '4', 'N' : '5' }, \
                '10' : { 'F' : '15', 'V' : '4', 'N' : '5' }}");

	input_shape = Shape(m_alphabet_size);
	output_shape = Shape(1);

	m_shuffle_index(10000);
}

AutomataDataset::~AutomataDataset() {
}

void AutomataDataset::prepare_minibatch_data(int64* data_idxs, int64 size) {
    m_batch_xs = hmath.zeros(Shape(size, m_max_length, m_alphabet_size));
    m_batch_xlen = Array<int64>(Shape(size));
    m_batch_ys = Array<float>(Shape(size));

    for (int64 n = 0; n < size; n++) {
        string alphabet = m_alphabet["alphabet"];

        bool is_correct = Random::dice(2);

        m_batch_ys[Idx(n)] = is_correct ? 1.0f : 0.0f;

        string sent;

        if (is_correct) {
            sent = m_generate_sent();
            if (!m_is_correct_sent(sent)) throw KaiException(KERR_ASSERT);
        }
        else {
            while (true) {
                sent = m_generate_sent();
                int64 touch = 1 + Random::dice(3);
                for (int64 m = 0; m < touch; m++) {
                    int64 sent_pos = Random::dice(sent.length());
                    int64 char_idx = Random::dice(m_alphabet_size);
                    sent[sent_pos] = alphabet[char_idx];
                }
                if (!m_is_correct_sent(sent)) break;
            }
        }

        m_batch_xlen[Idx(n)] = sent.length();

        for (int64 m = 0; m < (int64) sent.length(); m++) {
            int64 pos = alphabet.find(sent[m]);
            m_batch_xs[Idx(n, m, pos)] = 1.0;
        }
    }
}

void AutomataDataset::gen_seq_xdata(int64 nth, int64 data_idx, int64 xsize, float* px, Value& to_y) {
    assert(xsize == m_max_length * m_alphabet_size);
    memcpy(px, m_batch_xs.data_ptr() + nth * xsize, sizeof(float)*xsize);
}

void AutomataDataset::gen_plain_ydata(int64 nth, int64 data_idx, int64 ysize, float* py, Value from_x) {
    assert(ysize == 1);
    *py = m_batch_ys[Idx(nth)];
}

void AutomataDataset::visualize(Value cxs, Value cest, Value cans) {
    Array<float> xs = cxs, est = cest, ans = cans;
    
    int64 mb_size = xs.shape()[0], vec_size = xs.shape()[1];
    xs = xs.reshape(Shape(mb_size * vec_size, -1));

    Array<int64> xalpha = kmath->argmax(xs, 0).to_host();
    xalpha = xalpha.reshape(Shape(mb_size, vec_size));

    string alphabet = m_alphabet["alphabet"];

    for (int64 n = 0; n < mb_size; n++) {
        string sent;
        for (int64 i = 0; i < vec_size; i++) {
            int64 alpha_idx = xalpha[Idx(n, i)];
            if (alpha_idx == 0) break;
            sent += alphabet[alpha_idx];
        }

        string answer = "Wrong-Pattern", guess = "Guess-wrong", result = "X";

        if (ans[Idx(n, 0)] > 0.5) answer = "Correct-pattern";
        if (est[Idx(n, 0)] > 0.5) guess = "Guess-good";
        if (ans[Idx(n, 0)] > 0.5 && est[Idx(n, 0)] > 0.5) result = "O";
        if (ans[Idx(n, 0)] < 0.5 && est[Idx(n, 0)] < 0.5) result = "O";

        logger.Print("%s: %s => %s(%4.2f) : %s", sent.c_str(), answer.c_str(), guess.c_str(), est[Idx(n, 0)], result.c_str());
    }
}

string AutomataDataset::m_generate_sent() {
    while (true) {
        string sent;
        try {
            sent = m_gen_node("S", 0);
        }
        catch (int64) {
            continue;
        }
        if ((int64)sent.length() < m_min_length) continue;
        if ((int64)sent.length() > m_max_length) continue;
        return sent;
    }
}

string AutomataDataset::m_gen_node(string symbol, int64 depth) {
    if (depth > 30) throw depth;
    List rules = m_rules[symbol];
    List rule = rules[Random::dice(rules.size())];
    
    string exp = "";

    for (List::iterator it = rule.begin(); it != rule.end(); it++) {
        string term = *it;
        if (m_rules.find(term) != m_rules.end()) {
            exp += m_gen_node(term, depth + 1);
        }
        else {
            string noms = m_alphabet[term];
            int64 pos = Random::dice(noms.size());
            exp += noms[pos];
        }
    }
            
    return exp;
}

bool AutomataDataset::m_is_correct_sent(string sent) {
    vector<string> states;
    states.push_back("0");

    sent += '$';

    int64 pos = 0;
    char nextch = sent[pos++];
    
    while (true) {
        List actions = m_action_table[states.back()];
        bool found = false;
        for (List::iterator it = actions.begin(); it != actions.end(); it++) {
            List action = *it;
            string lookaheads = m_alphabet[action[0]];
            if (lookaheads.find(nextch) == string::npos) continue;
            found = true;
            int64 state = action[1];
            if (state == 0) return true; // accept
            if (state > 0) { // shift
                states.push_back(to_string(state));
                nextch = sent[pos++];
                break;
            }
            else { // reduce
                while (state++ < 0) states.pop_back();
                Dict goto_map = m_goto_table[states.back()];
                string symbol = action[2];
                string goto_state = goto_map[symbol];
                states.push_back(goto_state);
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
}
