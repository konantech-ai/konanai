/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <gtest/gtest.h>
#include "session/kcommon.h"
#include "include/kai_api.h"
#include "include/kai_api_shell.h"
#include "session/session.h"
#include "math/karray.h"
#include "math/kcudamath.h"


TEST(TEST_KAI_ARRAY, test)
{	
	KaiHostMath host;
	KaiValue kshape = KaiShape({ 1,2,3,4,5,6,7 });
	KaiValue kshape2 = KaiShape({ 9,8,7,6,5,4,3 });
	KaiArray<KFloat> array(kshape);
	KaiArray<KFloat> array2(kshape2);
	KaiArray<KFloat> sum;
	
	array = host.transpose(array2);
	KString a;
	array.dump(a);
	EXPECT_EQ(a," ");
}