/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class MacroPack {
public:
	MacroPack();
	virtual ~MacroPack();

	void set_macro(string name, string config);
	bool in_macro(string name);
	List get_macro(string name, Dict options);

protected:
	Dict m_macros;

	List m_replace_arg(List list, Dict args);
	Dict m_replace_arg(Dict dict, Dict args);

	Value m_parse_arg(string term, Dict args);
};