/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "random.h"
#include "array.h"
#include "func_timer.h"

//std::default_random_engine Random::gen;
//mt19937 Random::gen;

map< std::thread::id, std::default_random_engine> Random::map_gen;

void Random::seed(int64 nSeed) {
	gen().seed((int)nSeed);
}

Array<float> Random::uniform(Shape shape) {
	FuncTimer func_timer("random_uniform");
	Array<float> array(shape);

	std::uniform_real_distribution<float> coin(0.0f, 1.0f); 

	float* pData = array.data_ptr();
	int64 size = array.shape().total_size();

	for (int64 i = 0; i < size; i++) {
#ifdef NORANDOM
		* pData++ = 0.5f;
#else
		* pData++ = coin(gen());
#endif

	}

	return array;
}

Array<float> Random::bernoulli(Shape shape, float prob) {
	FuncTimer func_timer("random_bernoulli");
	Array<float> array(shape);

	std::bernoulli_distribution coin(prob);

	float* pData = array.data_ptr();
	int64 size = array.shape().total_size();

	for (int64 i = 0; i < size; i++) {
#ifdef NORANDOM
		* pData++ = 1.0f;
#else
		* pData++ = coin(gen()) ? 1.0f : 0.0f;
#endif
	}
	return array;
}

Array<float> Random::normal(float mean, float std, Shape shape) {
	FuncTimer func_timer("random_normal");
	Array<float> array(shape);

	std::normal_distribution<float> coin(mean, std);

	float* pData = array.data_ptr();
	int64 size = array.shape().total_size();

	for (int64 i = 0; i < size; i++) {

#ifdef NORANDOM
		* pData++ = std/0.5f;
#else
		* pData++ = coin(gen());
#endif
	}

	return array;
}

int64 Random::dice(int64 nom_cnt) {
	std::uniform_int_distribution<int64> coin(0, nom_cnt - 1);
	return coin(gen());
}

int64 Random::dice_except(int64 among, int64 except) {
	std::uniform_int_distribution<int64> coin(1, among - 1);
	int64 rnd = coin(gen());
	return (except + rnd) % among;
}

float Random::uniform(float start, float end) {
	std::uniform_real_distribution<float> coin(start, end);
	return coin(gen());
}

float Random::beta(float alpha, float beta) {
	throw KaiException(KERR_ASSERT);
	// 임시 처리임
	std::uniform_real_distribution<float> coin(0.0f, 1.0f);
	std::default_random_engine& g = gen();
	return (coin(g) + coin(g) + coin(g)) / 3.0f;
}
