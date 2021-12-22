/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "macro_pack.h"
#include "value.h"

MacroPack::MacroPack() {
}

MacroPack::~MacroPack() {
}

void MacroPack::set_macro(string name, string config) {
	m_macros[name] = Value::parse_list(config.c_str());
}

bool MacroPack::in_macro(string name) {
    if (this == NULL) return false;
    return m_macros.find(name) != m_macros.end();
}

List MacroPack::get_macro(string name, Dict options) {
    if (this == NULL) throw KaiException(KERR_ASSERT);

    if (m_macros.find(name) == m_macros.end()) throw KaiException(KERR_ASSERT);

    List fetched = m_macros[name];

    if (options.find("args") != options.end()) {
        Dict args = options["args"];
        fetched = m_replace_arg(fetched, args);
    }

    return fetched;
}

List MacroPack::m_replace_arg(List list, Dict args) {
    for (List::iterator it = list.begin(); it != list.end(); it++) {
        if (it->type() == vt::string) {
            string term = *it;
            if (term[0] == '#') *it = m_parse_arg(term, args);
        }
        else if (it->type() == vt::list) {
            List term = *it;
            *it = m_replace_arg(term, args);
        }
        else if (it->type() == vt::dict) {
            Dict term = *it;
            *it = m_replace_arg(term, args);
        }
    }

    return list;
}

Dict MacroPack::m_replace_arg(Dict dict, Dict args) {
    for (Dict::iterator it = dict.begin(); it != dict.end(); it++) {
        if (it->second.type() == vt::string) {
            string term = it->second;
            it->second = m_parse_arg(term, args);
        }
        else if (it->second.type() == vt::list) {
            List term = it->second;
            it->second = m_replace_arg(term, args);
        }
        else if (it->second.type() == vt::dict) {
            Dict term = it->second;
            it->second = m_replace_arg(term, args);
        }
    }

    return dict;
}

Value MacroPack::m_parse_arg(string term, Dict args) {
    // 수식처리 : 일단 yolo 에제에 나타나는 #chn * 2 형태의 처리만 지원

    if (term[0] == '#') {
        if (args.find(term) != args.end()) {
            return args[term];
        }

        int coef = 1;

        if (term.find('*') != string::npos) {
            size_t pos = term.find('*');
            coef = std::stoi(term.substr(pos + 1));
            term = term.substr(0, pos);
        }

        string term_rest = term.substr(1);

        if (args.find(term_rest) != args.end()) {
            Value val = args[term_rest];
            if (coef != 1) {
                if (val.type() == vt::kint) {
                    val = coef * (int)val;
                }
                else {
                    throw KaiException(KERR_ASSERT);
                }
            }
            return val;
        }
    }
    else {

    }

    return term;
}
