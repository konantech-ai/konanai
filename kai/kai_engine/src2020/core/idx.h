/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class IdxCore {
public:
	IdxCore() {
		m_nCount = 0;
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
	}
	virtual ~IdxCore() {}
	void destroy() { if (--m_nRefCount <= 0) { delete this; } }

protected:
	friend class Idx;

	int m_nRefCount;
	int64 m_nCount;
	int64 m_element[KAI_MAX_DIM];
};

class Idx {
public:
	Idx() {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 0;
	}
	Idx(int64 a1) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 1;
		m_core->m_element[0] = a1;
	}
	Idx(int64 a1, int64 a2) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 2;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
	}
	Idx(int64 a1, int64 a2, int64 a3) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 3;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
	}
	Idx(int64 a1, int64 a2, int64 a3, int64 a4) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 4;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
	}
	Idx(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 5;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
		m_core->m_element[4] = a5;
	}
	Idx(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5, int64 a6) {
		m_core = new IdxCore();
		m_core->m_nRefCount = 1;
		m_core->m_nCount = 6;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
		m_core->m_element[4] = a5;
		m_core->m_element[5] = a6;
	}
	Idx(const Idx& src) {
		m_core = src.m_core;
		m_core->m_nRefCount++;
	}

	Idx& operator = (const Idx& src) {
		if (this == &src) return *this;
		m_core->destroy();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		return *this;
	}

	virtual ~Idx() { m_core->destroy(); }

	int64 size() const { return m_core->m_nCount; }
	int64& operator [](int64 nth) { return m_core->m_element[ax_mod(nth, size())]; }
	int64 operator [](int64 nth) const { return m_core->m_element[ax_mod(nth, size())]; }

	string desc();

	void set_size(int64 size) { m_core->m_nCount = size; }

protected:
	IdxCore* m_core;
};

