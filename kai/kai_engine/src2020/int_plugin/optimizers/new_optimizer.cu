/*K
This file is part of Konantech AI Framework project
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution
*/
#include "new_optimizer.cuh"


//hs.cho
#ifndef KAI2021_WINDOWS
#define sprintf_s snprintf
#endif

// 생성자: 필요한 설정값이 있는 경우 전달받은 kwArgs 정보를 조사해 설정
TemplateOptimizer::TemplateOptimizer(Dict kwArgs) : Optimizer("new", kwArgs) {
	m_test = (float)Value::seek_option(kwArgs, "test", 0.9f);	// 알맞은 디폴트 값(ex.0.9f)을 알려주면서 필요한 하이퍼 파라미터를 챙깁니다.
}

TemplateOptimizer::~TemplateOptimizer() {
}

void TemplateOptimizer::setup(Dict kwArgs) {
	m_test = (float)Value::seek_option(kwArgs, "test", m_test);	// 생성자에서 설정된 기존 값을 디폴트 값으로 알려주면서 필요한 하이퍼 파라미터를 챙깁니다.
}

// shell의 optimizer 소개 문구에 추가할 하이퍼파라미터 값을 출력합니다.
string TemplateOptimizer::introduce_extra() {
	char buffer[128];
	sprintf_s(buffer,128,", test:%f", m_test);
	return  (string)buffer;
}

void TemplateOptimizer::m_alloc_affine_param(Dict& param, Shape shape, bool use_cuda, Dict kwArgs) {
	// Optimizer::alloc_affine_para() 함수m에서 파라미터 기본 구조를 생성한 후 호출합니다.
	// 모멘텀 등 추가 파라미터가 필요한 경우 이 함수를 재정의해 param 내용을 수정하거나 추가하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::m_alloc_embed_param(Dict& param, vector<int64> voc_sizes, int64 vec_size, Dict kwArgs) {
	// Optimizer::alloc_embed_param() 함수에서 파라미터 기본 구조를 생성한 후 호출합니다.
	// 모멘텀 등 추가 파라미터가 필요한 경우 이 함수를 재정의해 param 내용을 수정하거나 추가하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::m_forward_affine(Dict param, Array<float> x, Array<float>& output) {
	// Optimizer::forward_affine() 함수에서 파라미터를 이용해 선형 연산 순전파 처리를 수행한 후 호출합니다.
	// 처리 결과에 수정이 필요하거나 정보 수집 등 다른 추가적인 처리가 필요한 경의 이 함수에서 알맞은 작업을 수행하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::m_backprop_affine(Dict param, Array<float> x, Array<float> G_affine, Array<float>& G_input) {
	// Optimizer::backprop_affine() 함수에서 파라미터를 이용해 선형 연산 역전파 처리를 수행한 후 호출합니다.
	// 처리 결과에 수정이 필요하거나 정보 수집 등 다른 추가적인 처리가 필요한 경의 이 함수에서 알맞은 작업을 수행하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::m_forward_embed(Dict param, Array<int64> selector, Array<float>& output) {
	// Optimizer::forward_embed() 함수에서 파라미터를 이용해 임베드 연산 순전파 처리를 수행한 후 호출합니다.
	// 처리 결과에 수정이 필요하거나 정보 수집 등 다른 추가적인 처리가 필요한 경의 이 함수에서 알맞은 작업을 수행하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::m_forward_embed_cuda(Dict param, Array<float> word_vecs, Array<int64> selector) {
	// Optimizer::forward_embed_cuda() 함수에서 파라미터를 이용해 임베드 연산 순전파 처리를 수행한 후 호출합니다.
	// 처리 결과에 수정이 필요하거나 정보 수집 등 다른 추가적인 처리가 필요한 경의 이 함수에서 알맞은 작업을 수행하세요.
	// 추가적인 처리가 필요하지 않은 경우 함수 선언 및 정의를 삭제하거나 이 정의를 빈 함수로 두세요.
}

void TemplateOptimizer::update_weight(Dict pm_w, Array<float> G_weight) {
	// 알고리즘에 맞게 선형 연산의 가중치 파라미터에 대한 업데이트를 비쿠다 방식으로 수행합니다.
}

void TemplateOptimizer::update_bias(Dict pm_b, Array<float> G_bias) {
	// 알고리즘에 맞게 선형 연산의 바이어스 파라미터에 대한 업데이트를 비쿠다 방식으로 수행합니다.
}

void TemplateOptimizer::update_embed(Dict param, Array<float> G_Words, Array<int64> selector) {
	// 알고리즘에 맞게 임베드 연산의 시전 파라미터에 대한 업데이트를 비쿠다 방식으로 수행합니다.
}

// 필요한 커널 함수 선언
// 아래의 세 함수에서 이용하는 커널 함수들을 선언합니다.

void TemplateOptimizer::update_weight_cuda(Dict param, Array<float> G_weight) {
	// 알고리즘에 맞게 선형 연산의 가중치 파라미터에 대한 업데이트를 쿠다 방식으로 수행합니다.
}

void TemplateOptimizer::update_bias_cuda(Dict param, Array<float> G_bias) {
	// 알고리즘에 맞게 선형 연산의 바이어스 파라미터에 대한 업데이트를 쿠다 방식으로 수행합니다.
}

void TemplateOptimizer::update_embed_cuda(Dict param, Array<float> G_words, Array<int64> selector) {
	// 알고리즘에 맞게 임베드 연산의 시전 파라미터에 대한 업데이트를 쿠다 방식으로 수행합니다.
}

// 커널 함수 및 디바이스 함수 정의
// 앞에서 선언한 커널 함수를 정의합니다. 이들 함수에서 호출하는 디바이스 함수가 있는 경우 이들도 정의합니다.
