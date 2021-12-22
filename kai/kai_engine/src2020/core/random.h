/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#pragma once

#include "common.h"

class Random {
public:
	static void seed(int64 nSeed);

	static Array<float> uniform(Shape shape);
	static Array<float> bernoulli(Shape shape, float prob);
	static Array<float> normal(float mean, float std, Shape shape);

	static int64 dice(int64 nom_cnt);

	static float uniform(float start=0.0f, float end=1.0f);
	static int64 dice_except(int64 among, int64 except);
	static float beta(float alpha, float beta);

protected:
	static std::default_random_engine& gen() { return map_gen[std::this_thread::get_id()]; }

protected:
	static map< std::thread::id, std::default_random_engine> map_gen;
};