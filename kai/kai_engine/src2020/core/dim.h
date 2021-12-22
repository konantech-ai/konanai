/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include <cstring>

#include "common.h"
#include "shape.h"

class DimCore {
public:
	DimCore() {
		m_nRefCount = 1;
		m_nDim = 0;
		memset(&m_element, 0, sizeof(int64)*KAI_MAX_DIM);
	}
	virtual ~DimCore() {}
	void destroy() { if (--m_nRefCount <= 0) { delete this; } }


protected:
	friend class Dim;
	friend class Value;
	friend class HostMath;

	int m_nRefCount;
	int64 m_nDim;
	int64 m_element[KAI_MAX_DIM];
	int64 m_pack_size[KAI_MAX_DIM];
};

class Dim {
public:
	Dim() {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 0;
	}
	Dim(int64 a1) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 1;
		m_core->m_element[0] = a1;
		m_core->m_pack_size[0] = 1;
	}
	Dim(int64 a1, int64 a2) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 2;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_pack_size[0] = a2;
		m_core->m_pack_size[1] = 1;
	}
	Dim(int64 a1, int64 a2, int64 a3) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 3;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_pack_size[0] = a2 * a3;
		m_core->m_pack_size[1] = a3;
		m_core->m_pack_size[2] = 1;
	}
	Dim(int64 a1, int64 a2, int64 a3, int64 a4) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 4;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
		m_core->m_pack_size[0] = a2 * a3 * a4;
		m_core->m_pack_size[1] = a3 * a4;
		m_core->m_pack_size[2] = a4;
		m_core->m_pack_size[3] = 1;
	}
	Dim(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 5;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
		m_core->m_element[4] = a5;
		m_core->m_pack_size[0] = a2 * a3 * a4 * a5;
		m_core->m_pack_size[1] = a3 * a4 * a5;
		m_core->m_pack_size[2] = a4 * a5;
		m_core->m_pack_size[3] = a5;
		m_core->m_pack_size[4] = 1;
	}
	Dim(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5, int64 a6) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = 6;
		m_core->m_element[0] = a1;
		m_core->m_element[1] = a2;
		m_core->m_element[2] = a3;
		m_core->m_element[3] = a4;
		m_core->m_element[4] = a5;
		m_core->m_element[5] = a6;
		m_core->m_pack_size[0] = a2 * a3 * a4 * a5 * a6;
		m_core->m_pack_size[1] = a3 * a4 * a5 * a6;
		m_core->m_pack_size[2] = a4 * a5 * a6;
		m_core->m_pack_size[3] = a5 * a6;
		m_core->m_pack_size[4] = a6;
		m_core->m_pack_size[5] = 1;
	}
	Dim(const Dim& src) {
		ms_mu_ref.lock();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_ref.unlock();
	}
	Dim(Shape shape) {
		m_core = new DimCore();
		m_core->m_nRefCount = 1;
		m_core->m_nDim = shape.size();
		int64 prod = 1;
		for (int64 n = m_core->m_nDim - 1; n >= 0; n--) {
			m_core->m_element[n] = shape[n];
			m_core->m_pack_size[n] = prod;
			prod *= shape[n];
		}
	}

	Dim& operator = (const Dim& src) {
		if (this == &src) return *this;
		ms_mu_ref.lock();
		m_core->destroy();
		m_core = src.m_core;
		m_core->m_nRefCount++;
		ms_mu_ref.unlock();
		return *this;
	}

	virtual ~Dim() {
		ms_mu_ref.lock();
		m_core->destroy();
		ms_mu_ref.unlock();
	}

	Dim deepcopy();

	int64 get_offset(Idx idx) const;
	int64 total_size() const;
	int64 dim() const { return m_core->m_nDim; }
	int64 axis_size(int64 nth) const { return m_core->m_element[nth]; }
	int64 pack_size(int64 nth) { return m_core->m_pack_size[nth]; }

	//int64& operator [](int64 nth) { nth = ax_mod(nth, m_core->m_nDim);  return m_core->m_element[nth]; }
	int64 operator [](int64 nth) const { nth = ax_mod(nth, m_core->m_nDim); return m_core->m_element[nth]; }

	Shape axis_select(Idx axis);
	void eval_idx(Idx axis, int64 nth, Idx& idx, int64& rest);

	//Shape get_shape();

	bool same_internal_structure(Dim dim);
	
	void transpose();

	string desc();

	bool operator ==(const Dim& other) const;

	void prod_check();

protected:
	friend class Value;
	friend class HostMath;

	DimCore* m_core;

	static mutex_wrap ms_mu_ref;
};
