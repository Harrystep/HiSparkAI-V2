/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/int8/arithmetic_self_int8.h"

namespace mindspore {
class ErfInt8Test : public ::testing::Test {
 public:
  ErfInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
extern float accuracy_threshold;

// Testcase1: ErfInt8 with scale=1.0, zp=0
TEST_F(ErfInt8Test, Testcase01) {
  std::vector<int8_t> input = {0, 1, -1, 2, -2, 50, -50, 127};
  std::vector<int8_t> benchmark = {0, 1, -1, 1, -1, 1, -1, 1};
  const int length = static_cast<int>(input.size());
  std::vector<int8_t> output(length);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 1.0f;
  para.in_args_.zp_ = 0;
  para.out_args_.scale_ = 1.0f;
  para.out_args_.zp_ = 0;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Testcase01 output:\n";
  std::for_each(output.begin(), output.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Testcase01 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: ErfInt8 with scale=0.1, zp=0
TEST_F(ErfInt8Test, Testcase02) {
  std::vector<int8_t> input = {0, 10, -10, 50, -50, -100, 100, 127};
  std::vector<int8_t> benchmark = {0, 8, -8, 10, -10, -10, 10, 10};
  const int length = static_cast<int>(input.size());
  std::vector<int8_t> output(length);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 0.1f;
  para.in_args_.zp_ = 0;
  para.out_args_.scale_ = 0.1f;
  para.out_args_.zp_ = 0;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Testcase02 output:\n";
  std::for_each(output.begin(), output.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Testcase02 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: ErfInt8 with more values
TEST_F(ErfInt8Test, Testcase03) {
  std::vector<int8_t> input = {-1, 0, 1, -5, 5, 10, -10, 100};
  std::vector<int8_t> benchmark = {-1, 0, 1, -1, 1, 1, -1, 1};
  const int length = static_cast<int>(input.size());
  std::vector<int8_t> output(length);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 1.0f;
  para.in_args_.zp_ = 0;
  para.out_args_.scale_ = 1.0f;
  para.out_args_.zp_ = 0;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Testcase03 output:\n";
  std::for_each(output.begin(), output.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Testcase03 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
