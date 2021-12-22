/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#ifdef NOT_PRECIATED
#include "../session/kcommon.h"
#include "../include/kai_api.h"

class KShapeUtil {
public:
	/*
	KaiShape create(KInt* sizes, KInt cnt) {
		KaiShape shape;
		for (KInt n = 0; n < cnt; n++) shape.push_back(sizes[n]);
		return shape;
	}

	KString desc(KaiShape shape);

	KInt total_size(KaiShape shape);

	KaiShape insert_head(KInt head, KaiShape tailShape);
	KaiShape replace_end(KaiShape srcShape, KInt end);
	KaiShape replace_tail(KaiShape headShape, KInt cut_len, KaiShape tailShape);
	KaiShape replace_tail(KaiShape headShape, KaiShape cutShape, KaiShape tailShape);
	*/

	/*
	KaiShape(KInt a1=0, KInt a2 = 0, KInt a3 = 0, KInt a4 = 0, KInt a5 = 0, KInt a6 = 0, KInt a7 = 0, KInt a8 = 0) {
		m_element[0] = a1;
		m_element[1] = a2;
		m_element[2] = a3;
		m_element[3] = a4;
		m_element[4] = a5;
		m_element[5] = a6;
		m_element[6] = a7;
		m_element[7] = a8;

		m_nCount = 1;
		while (m_nCount < KAI_MAX_DIM) {
			if (m_element[m_nCount] == 0) break;
			m_nCount++;
		}

		m_checkCode = ms_checkCode;
	}
	KaiShape(KInt* sizes, KInt cnt) {
		m_nCount = cnt;
		memcpy(m_element, sizes, sizeof(KInt) * cnt);
		m_checkCode = ms_checkCode;
	}
	KaiShape(int* sizes, KInt cnt) {
		m_nCount = cnt;
		for (KInt n = 0; n < cnt; n++) m_element[n] = (KInt)sizes[n];
		m_checkCode = ms_checkCode;
	}
	KaiShape(KaiList lst) {
		m_nCount = lst.size();
		for (KInt n = 0; n < m_nCount; n++){
			m_element[n] = (KInt)lst[n];
		}
		m_checkCode = ms_checkCode;
	}

	virtual ~KaiShape() {}

	KInt size() const { return m_nCount; }
	
	KInt& operator [](KInt nth) { return m_element[ax_mod(nth, size())]; }
	KInt operator [](KInt nth) const { return m_element[ax_mod(nth, size())]; }

	bool operator == (const KaiShape& other);
	bool operator != (const KaiShape& other);

	bool operator == (KInt other);

	KaiShape add_front(KInt a);
	KaiShape add_front(KInt a1, KInt a2);
	KaiShape add_nth(KInt pos, KInt size);
	KaiShape append(KInt a);
	KaiShape append(KaiShape shape);
	KaiShape remove_front();
	KaiShape remove_end();
	KaiShape remove_tail(KInt tail_len);
	KaiShape remove_nth(KInt nth);
	KaiShape replace_nth(KInt nth, KInt a);
	KaiShape replace_end(KInt a);
	KaiShape replace_end(KaiShape shape);
	KaiShape replace_tail(KInt tail_len, KaiShape shape);

	KaiShape fix_unknown(KInt total_size);

	KaiShape add(const KaiShape other);
	KaiShape sub(const KaiShape other);
	KaiShape mul(const KaiShape other);
	KaiShape div(const KaiShape other);
	KaiShape mod(const KaiShape other);

	KaiShape operator +(KInt other);
	KaiShape operator -(KInt other);
	KaiShape operator *(KInt other);
	KaiShape operator /(KInt other);

	KaiShape merge_time_axis();
	KaiShape split_time_axis(KInt mb_size);

protected:
	//friend class Dim;

	KInt m_nCount;
	KInt m_element[KAI_MAX_DIM];

	int m_checkCode;

	static int ms_checkCode;
	*/
};

extern KShapeUtil kshape;

/*
class Idx;

class ShapeCounter {
public:
	ShapeCounter(KaiShape shape);
	~ShapeCounter();

	void operator ++();
	KInt operator [](KInt index);
protected:
	KaiShape m_shape;
	Idx m_idx;
};
*/
#endif