/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "idx.h"

string Idx::desc() {
	string exp = "(";
	for (int n = 0; n < size(); n++) {
		if (n > 0) exp += ",";
		exp += to_string((*this)[n]);
	}
	return exp + ")";
}

