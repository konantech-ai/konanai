/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "dim.h"
#include "idx.h"
#include "shape.h"

mutex_wrap Dim::ms_mu_ref;

Dim Dim::deepcopy() {
	Dim clone;
	clone.m_core->m_nDim = m_core->m_nDim;
	memcpy(clone.m_core->m_element, m_core->m_element, sizeof(int64) * KAI_MAX_DIM);
	memcpy(clone.m_core->m_pack_size, m_core->m_pack_size, sizeof(int64) * KAI_MAX_DIM);
	return clone;
}

int64 Dim::get_offset(Idx idx) const {
	assert(idx.size() == m_core->m_nDim);

	int64 offset = idx[0];
	
	for (int n = 1; n < idx.size(); n++) {
		offset = offset * m_core->m_element[n] + idx[n]; // *m_core->m_product[m_core->m_xd[n]];;
	}

	return offset;
}

bool Dim::same_internal_structure(Dim dim) {
	if (dim.m_core->m_nDim != m_core->m_nDim) return false;
	if (memcmp(dim.m_core->m_element, m_core->m_element, sizeof(int)*KAI_MAX_DIM) != 0) return false;
	return true;
}

void Dim::transpose() {
	throw KaiException(KERR_ASSERT);
	//swap<unsigned char>(m_core->m_xd[0], m_core->m_xd[1]);
}

string Dim::desc() {
	string shape;
	string delimeter = "(";

	for (int n = 0; n < m_core->m_nDim; n++) {
		shape += delimeter + to_string(m_core->m_element[n]);
		delimeter = ",";
	}

	return shape + ")";
}

bool Dim::operator ==(const Dim& other) const {
	if (dim() != other.dim()) return false;

	for (int n = 0; n < m_core->m_nDim; n++) {
		if (axis_size(n) != other.axis_size(n)) return false;
	}

	return true;
}

Shape Dim::axis_select(Idx axis) {
	Shape shape;

	shape.m_nCount = axis.size();
	for (int n = 0; n < axis.size(); n++) {
		shape.m_element[n] = m_core->m_element[axis[n]];
	}

	return shape;
}

int64 Dim::total_size() const {
	int64 prod = 1;

	for (int n = 0; n < m_core->m_nDim; n++) {
		prod *= m_core->m_element[n];
	}

	return prod;
}

void Dim::prod_check() {
	int64 prod = 1;

	for (int64 n = m_core->m_nDim - 1; n >= 0; n--) {
		assert(prod == m_core->m_pack_size[n]);
		prod *= m_core->m_element[n];
	}
}

void Dim::eval_idx(Idx axis, int64 nth, Idx& idx, int64& rest) {
	throw KaiException(KERR_ASSERT);
	/*
	int64 cood[KAI_MAX_DIM];

	for (int64 n = 0; n < dim(); n++) {
		cood[n] = nth / m_core->m_product[n];
		nth = nth % m_core->m_product[n];
	}

	idx.set_size(axis.size());

	for (int64 n = 0; n < axis.size(); n++) {
		idx[n] = cood[m_core->m_xd[axis[n]]];
		cood[m_core->m_xd[axis[n]]] = -1;
	}

	rest = 0;

	for (int64 n = dim()-1, unit = 1; n >= 0; n--) {
		if (cood[m_core->m_xd[n]] != -1) {
			rest += unit * cood[m_core->m_xd[n]];
			unit *= m_core->m_element[m_core->m_xd[n]];
		}
	}
	*/
}

