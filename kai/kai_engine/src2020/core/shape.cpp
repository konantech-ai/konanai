/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "common.h"
#include "shape.h"
#include "dim.h"
#include "idx.h"
#include "value.h"

Shape::Shape(List lst) {
	m_nCount = (int) lst.size();
	for (int n = 0; n < m_nCount; n++) {
		m_element[n] = lst[n];
	}
}

Shape::Shape(Dim dim) {
	m_nCount = dim.dim();
	for (int n = 0; n < m_nCount; n++) {
		m_element[n] = dim.axis_size(n);
	}
}

int64 Shape::total_size() {
	int64 size_prod = 1;
	for (int n = 0; n < size(); n++) {
		size_prod *= (*this)[n];
	}
	return size_prod;
}

Shape Shape::add_front(int64 a) {
	assert(size() < KAI_MAX_DIM);
	Shape clone;
	clone.m_element[0] = a;
	for (int n = 0; n < size(); n++) {
		clone.m_element[n + 1] = m_element[n];
	}
	clone.m_nCount = m_nCount + 1;
	return clone;
}

Shape Shape::add_front(int64 a1, int64 a2) {
	assert(size() < KAI_MAX_DIM-1);
	Shape clone;
	clone.m_element[0] = a1;
	clone.m_element[1] = a2;
	for (int n = 0; n < size(); n++) {
		clone.m_element[n + 2] = m_element[n];
	}
	clone.m_nCount = m_nCount + 2;
	return clone;
}

Shape Shape::add_nth(int64 pos, int64 ax) {
	assert(size() < KAI_MAX_DIM-1);
	Shape clone;
	for (int64 n = 0; n < pos; n++) {
		clone.m_element[n] = m_element[n];
	}
	clone.m_element[pos] = ax;
	for (int64 n = pos; n < size(); n++) {
		clone.m_element[n + 1] = m_element[n];
	}
	clone.m_nCount = m_nCount + 1;
	return clone;
}

Shape Shape::append(int64 a) {
	assert(size() < KAI_MAX_DIM);
	Shape clone;
	clone.m_nCount = m_nCount + 1;
	for (int n = 0; n < size(); n++) {
		clone.m_element[n] = m_element[n];
	}
	clone.m_element[size()] = a;
	return clone;
}

Shape Shape::remove_front() {
	assert(size() > 0);
	Shape clone;
	for (int n = 0; n < size() - 1; n++) {
		clone.m_element[n] = m_element[n + 1];
	}
	clone.m_nCount = m_nCount - 1;
	return clone;
}

Shape Shape::remove_end() {
	assert(size() > 0);
	Shape clone;
	for (int n = 0; n < size() - 1; n++) {
		clone.m_element[n] = m_element[n];
	}
	clone.m_nCount = m_nCount - 1;
	return clone;
}

Shape Shape::remove_tail(int64 tail_len) {
	assert(size() >= tail_len);
	Shape clone;
	for (int n = 0; n < size() - tail_len; n++) {
		clone.m_element[n] = m_element[n];
	}
	clone.m_nCount = m_nCount - tail_len;
	return clone;
}

Shape Shape::remove_nth(int64 nth) {
	assert(size() > 0);
	assert(nth >= -size() && nth < size());
	nth = ax_mod(nth, size());
	Shape clone;
	for (int64 n = 0, m = 0; n < size(); n++) {
		if (n == nth) continue;
		clone.m_element[m++] = m_element[n];
	}
	clone.m_nCount = m_nCount - 1;
	return clone;
}

Shape Shape::replace_end(int64 a) {
	assert(size() > 0);
	Shape clone = *this;
	clone.m_element[size() - 1] = a;
	return clone;
}

Shape Shape::replace_end(Shape shape) {
	assert(size() > shape.size());
	Shape clone = *this;
	for (int64 n = 0; n < shape.size(); n++) {
		clone.m_element[size() - shape.size() + n] = shape[n];
	}
	return clone;
}

Shape Shape::replace_tail(int64 tail_len, Shape shape) {
	assert(size() > tail_len);
	assert(size() - tail_len + shape.size() <= KAI_MAX_DIM);
	Shape clone = *this;
	for (int64 n = 0; n < shape.size(); n++) {
		clone.m_element[size() - tail_len + n] = shape[n];
	}
	clone.m_nCount = size() - tail_len + shape.size();
	return clone;
}

Shape Shape::append(Shape shape) {
	assert(size() + shape.size() <= KAI_MAX_DIM);
	Shape clone = *this;
	for (int64 n = 0; n < shape.size(); n++) {
		clone.m_element[size() + n] = shape[n];
	}
	clone.m_nCount = size() + shape.size();
	return clone;
}

Shape Shape::replace_nth(int64 nth, int64 a) {
	assert(size() > 0);
	assert(nth >= -size() && nth < size());
	nth = ax_mod(nth, size());
	Shape clone = *this;
	clone.m_element[nth] = a;
	return clone;
}

Shape Shape::merge_time_axis() {
	Shape clone;
	clone.m_nCount = m_nCount - 1;
	clone.m_element[0] = m_element[0] * m_element[1];
	for (int n = 1; n < size()-1; n++) {
		clone.m_element[n] = m_element[n+1];
	}
	return clone;
}

Shape Shape::split_time_axis(int64 mb_size) {
	assert(size() < KAI_MAX_DIM);
	Shape clone;
	for (int n = 0; n < size(); n++) {
		clone.m_element[n + 1] = m_element[n];
	}
	clone.m_nCount = m_nCount + 1;
	clone.m_element[0] = mb_size;
	clone.m_element[1] /= mb_size;
	return clone;
}

Shape Shape::fix_unknown(int64 total_size) {
	Shape clone;
	clone.m_nCount = m_nCount;
	
	int neg_idx = -1;
	int64 prod = 1;
	
	for (int n = 0; n < size(); n++) {
		if (m_element[n] == -1) {
			assert(neg_idx == -1);
			neg_idx = n;
		}
		else {
			clone.m_element[n] = m_element[n];
			prod *= m_element[n];
		}
	}

	if (neg_idx == -1) {
		assert(prod == total_size);
	}
	else {
		assert(total_size % prod == 0);
		clone.m_element[neg_idx] = total_size / prod;
	}

	return clone;
}

bool Shape::operator == (const Shape& other) {
	if (size() != other.size()) return false;
	for (int n = 0; n < size(); n++) {
		if ((*this)[n] != other[n]) return false;
	}
	return true;
}

bool Shape::operator != (const Shape& other) {
	if (size() != other.size()) return true;
	for (int n = 0; n < size(); n++) {
		if ((*this)[n] != other[n]) return true;
	}
	return false;
}

bool Shape::operator == (int64 other) {
	for (int n = 0; n < size(); n++) {
		if ((*this)[n] != other) return false;
	}
	return true;
}

string Shape::desc() {
	string exp = "(";
	for (int n = 0; n < size(); n++) {
		if (n > 0) exp += ",";
		exp += to_string((*this)[n]);
	}
	return exp + ")";
}

Shape Shape::add(const Shape other) {
	assert(size() == other.size());

	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] + other[n];
	return result;
}

Shape Shape::sub(const Shape other) {
	assert(size() == other.size());

	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] - other[n];
	return result;
}

Shape Shape::mul(const Shape other) {
	assert(size() == other.size());

	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] * other[n];
	return result;
}

Shape Shape::div(const Shape other) {
	assert(size() == other.size());

	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] / other[n];
	return result;
}

Shape Shape::mod(const Shape other) {
	assert(size() == other.size());

	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] % other[n];
	return result;
}

Shape Shape::operator +(int64 other) {
	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] + other;
	return result;
}

Shape Shape::operator - (int64 other) {
	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] - other;
	return result;
}

Shape Shape::operator *(int64 other) {
	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] * other;
	return result;
}

Shape Shape::operator /(int64 other) {
	Shape result = *this;
	for (int n = 0; n < size(); n++) result[n] = (*this)[n] / other;
	return result;
}

ShapeCounter::ShapeCounter(Shape shape) {
	m_shape = shape;
	m_idx.set_size(shape.size());
	
	for (int n = 0; n < shape.size(); n++) m_idx[n] = 0;
}

ShapeCounter::~ShapeCounter() {
}

void ShapeCounter::operator ++() {
	for (int64 n = m_shape.size() - 1; n >= 0; n--) {
		if (m_idx[n] < m_shape[n] - 1) {
			m_idx[n]++;
			break;
		}
		m_idx[n] = 0;
	}
}

int64 ShapeCounter::operator [](int64 index) {
	return m_idx[index];
}
