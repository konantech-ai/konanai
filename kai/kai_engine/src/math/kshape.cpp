/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#ifdef NOT_PRECIATED
#include "kshape.h"

KShapeUtil kshape;

KString KShapeUtil::desc(KaiShape shape) {
	KString exp = "(";
	KInt size = (KInt)shape.size();
	for (KInt n = 0; n < size; n++) {
		if (n > 0) exp += ",";
		exp += to_string(shape[n]);
	}
	return exp + ")";
}

KInt KShapeUtil::total_size(KaiShape shape) {
	KInt size_prod = 1;
	KInt size = (KInt)shape.size();
	for (KInt n = 0; n < size; n++) {
		size_prod *= shape[n];
	}
	return size_prod;
}

KaiShape KShapeUtil::insert_head(KInt head, KaiShape tailShape) {
	KInt tlen = tailShape.size();

	if (tlen + 1 > KAI_MAX_DIM) throw KaiException(KERR_TOO_LONG_TAIL_IN_SHAPE_INSERT_HEAD);

	KaiShape shape;

	shape.push_back(head);

	for (KInt n = 0; n < tlen; n++) {
		shape.push_back(tailShape[n]);
	}

	return shape;
}

KaiShape KShapeUtil::replace_end(KaiShape srcShape, KInt end) {
	KaiShape shape = srcShape;
	shape[shape.size() - 1] = end;
	return shape;
}

KaiShape KShapeUtil::replace_tail(KaiShape headShape, KInt clen, KaiShape tailShape) {
	KInt hlen = headShape.size();
	KInt tlen = tailShape.size();

	if (hlen < clen) throw KaiException(KERR_TOO_LARGE_CUT_IN_SHAPE_REPLACE_TAIL);
	if (hlen + tlen - clen > KAI_MAX_DIM) throw KaiException(KERR_TOO_LONG_TAIL_IN_SHAPE_REPLACE_TAIL);

	KaiShape shape;

	for (KInt n = 0; n < hlen - clen; n++) {
		shape.push_back(headShape[n]);
	}

	for (KInt n = 0; n < tlen; n++) {
		shape.push_back(tailShape[n]);
	}

	return shape;
}

KaiShape KShapeUtil::replace_tail(KaiShape headShape, KaiShape cutShape, KaiShape tailShape) {
	KInt hlen = headShape.size();
	KInt clen = cutShape.size();
	KInt tlen = tailShape.size();

	if (hlen < clen) throw KaiException(KERR_TOO_LARGE_CUT_IN_SHAPE_REPLACE_TAIL);
	if (hlen + tlen - clen > KAI_MAX_DIM) throw KaiException(KERR_TOO_LONG_TAIL_IN_SHAPE_REPLACE_TAIL);

	KaiShape shape;

	for (KInt n = 0; n < hlen - clen; n++) {
		shape.push_back(headShape[n]);
	}

	for (KInt n = 0; n < clen; n++) {
		if (headShape[hlen-clen+n] != cutShape[n]) throw KaiException(KERR_SHAPE_MISMATCHT_IN_SHAPE_REPLACE_TAIL);
	}

	for (KInt n = 0; n < tlen; n++) {
		shape.push_back(tailShape[n]);
	}

	return shape;
}

#endif