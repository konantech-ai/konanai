/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"
#include "idx.h"

class Shape {
public:
	Shape() {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 0;
	}
	Shape(int64 a1) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 1;
		m_element[0] = a1;
	}
	Shape(int64 a1, int64 a2) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 2;
		m_element[0] = a1;
		m_element[1] = a2;
	}
	Shape(int64 a1, int64 a2, int64 a3) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 3;
		m_element[0] = a1;
		m_element[1] = a2;
		m_element[2] = a3;
	}
	Shape(int64 a1, int64 a2, int64 a3, int64 a4) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 4;
		m_element[0] = a1;
		m_element[1] = a2;
		m_element[2] = a3;
		m_element[3] = a4;
	}
	Shape(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 5;
		m_element[0] = a1;
		m_element[1] = a2;
		m_element[2] = a3;
		m_element[3] = a4;
		m_element[4] = a5;
	}
	Shape(int64 a1, int64 a2, int64 a3, int64 a4, int64 a5, int64 a6) {
		memset(m_element, 0, sizeof(int64) * KAI_MAX_DIM);
		m_nCount = 6;
		m_element[0] = a1;
		m_element[1] = a2;
		m_element[2] = a3;
		m_element[3] = a4;
		m_element[4] = a5;
		m_element[5] = a6;
	}
	Shape(int64* sizes, int64 cnt) {
		m_nCount = cnt;
		memcpy(m_element, sizes, sizeof(int64) * cnt);
	}
	Shape(int* sizes, int64 cnt) {
		m_nCount = cnt;
		for (int64 n = 0; n < cnt; n++) m_element[n] = (int64) sizes[n];
	}
	Shape(List lst);
	Shape(Dim dim);

	virtual ~Shape() {}

	int64 size() const { return m_nCount; }
	int64 total_size();
	int64& operator [](int64 nth) { return m_element[ax_mod(nth, size())]; }
	int64 operator [](int64 nth) const { return m_element[ax_mod(nth, size())]; }

	bool operator == (const Shape& other);
	bool operator != (const Shape& other);

	bool operator == (int64 other);

	string desc();

	Shape add_front(int64 a);
	Shape add_front(int64 a1, int64 a2);
	Shape add_nth(int64 pos, int64 size);
	Shape append(int64 a);
	Shape append(Shape shape);
	Shape remove_front();
	Shape remove_end();
	Shape remove_tail(int64 tail_len);
	Shape remove_nth(int64 nth);
	Shape replace_nth(int64 nth, int64 a);
	Shape replace_end(int64 a);
	Shape replace_end(Shape shape);
	Shape replace_tail(int64 tail_len, Shape shape);

	Shape fix_unknown(int64 total_size);

	Shape add(const Shape other);
	Shape sub(const Shape other);
	Shape mul(const Shape other);
	Shape div(const Shape other);
	Shape mod(const Shape other);

	Shape operator +(int64 other);
	Shape operator -(int64 other);
	Shape operator *(int64 other);
	Shape operator /(int64 other);

	Shape merge_time_axis();
	Shape split_time_axis(int64 mb_size);

protected:
	friend class Dim;

	int64 m_nCount;
	int64 m_element[KAI_MAX_DIM];
};

class Idx;

class ShapeCounter {
public:
	ShapeCounter(Shape shape);
	~ShapeCounter();

	void operator ++();
	int64 operator [](int64 index);
protected:
	Shape m_shape;
	Idx m_idx;
};