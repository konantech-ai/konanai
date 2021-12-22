/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include <gtest/gtest.h>
#include "session/kcommon.h"
#include "include/kai_api.h"
#include "include/kai_api_shell.h"

TEST(TEST_KAI_VALUE, init)
{
    KaiValue kint = 3434;
    KaiValue kint2 = (KInt(3434));
    EXPECT_EQ(kint, kint2);


    KaiValue kfloat = 1.5f;
    KaiValue kfloat2 = KFloat(1.5f);
    EXPECT_EQ(kfloat, kfloat2);


    KaiValue kstring = "dsdsdsd";
    KaiValue kstring2 = string("dsdsd");
    EXPECT_EQ(kstring == kstring2, 0);



    KaiValue klist = KaiList{ KaiList{{ 0,1} , {1,10} }, KaiList{ 1,3,20,2  } };
    KaiValue klist2 = KaiList{ 0,10, 1,20 };
    KaiValue klist3 = KaiList(klist2);
   // KaiList   klist4 = ((KaiList)klist2). + (KaiList)klist3;   //list operator ??????
    EXPECT_EQ(klist.desc(), std::string("[[[0,1],[1,10]],[1,3,20,2]]"));
    EXPECT_EQ(klist2.desc(), klist3.desc());


    KaiValue kdic = KaiDict();
    KaiDict  kdic2 = kdic;
    kdic2["test1"] = KaiList{ {10,2},{3,4} };
    ((KaiDict)kdic)["test1"] = KaiList{ {10,2},{3,4} };
    // KaiDict kdic3 = kdic + kdic2   // map operator??
    EXPECT_EQ(kdic.desc(),"{test1:[[10,2],[3,4]]}");
    EXPECT_EQ(kdic2.desc(), "{test1:[[10,2],[3,4]]}");


    KaiValue kshape = KaiShape({ 1,2,3,4,5,6,7 });
    KInt list[7] = { 1,2,3,4,5,6,7 };
    KaiValue kshape2 = KaiShape(7, list);
    KaiValue kshape3 = KaiShape(kshape);
    KaiValue kshape4 = ((KaiShape)kshape).append(KaiShape{ 8 });
    KaiValue kshape5 = ((KaiShape)kshape).replace_tail_by_size({ 6,7 }, { 1,2 }); //whatif old tail does not match?
    EXPECT_EQ(kshape4.desc(), "<1,2,3,4,5,6,7,8>");
    EXPECT_EQ(kshape5.desc(), "<1,2,3,4,5,1,2>");
    


}
TEST(TEST_KAI_VALUE, arithmetic)
{

  // not implemented yet
    EXPECT_EQ(1, 1);
}

//KAI_TEST_MAIN("");