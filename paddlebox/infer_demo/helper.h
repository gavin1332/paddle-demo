// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <glog/logging.h>
#include <fstream>
#if !defined(_WIN32)
#include <sys/time.h>
#endif
#include <algorithm>
#include <chrono>  // NOLINT
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/include/paddle_inference_api.h"

#define lylog LOG(ERROR) << "[liuyi05] "

namespace paddle {
namespace inference {

// Timer for timer
class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

static int GetUniqueId() {
  static int id = 0;
  return id++;
}

static void split(const std::string &str,
                  char sep,
                  std::vector<std::string> *pieces,
                  bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}
static void split_to_float(const std::string &str,
                           char sep,
                           std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
}
static void split_to_int64(const std::string &str,
                           char sep,
                           std::vector<int64_t> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*is),
                 [](const std::string &v) { return std::stoi(v); });
}
static void split_to_int(const std::string &str,
                         char sep,
                         std::vector<int> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(),
                 pieces.end(),
                 std::back_inserter(*is),
                 [](const std::string &v) { return std::stoi(v); });
}

static void PrintTime(int batch_size,
                      int repeat,
                      int num_threads,
                      int tid,
                      double batch_latency,
                      int epoch = 1) {
  double sample_latency = batch_latency / batch_size;
  LOG(INFO) << "====== threads: " << num_threads << ", thread id: " << tid
            << " ======";
  LOG(INFO) << "====== batch_size: " << batch_size << ", iterations: " << epoch
            << ", repetitions: " << repeat << " ======";
  LOG(INFO) << "====== batch latency: " << batch_latency
            << "ms, number of samples: " << batch_size * epoch
            << ", sample latency: " << sample_latency
            << "ms, fps: " << 1000.f / sample_latency << " ======";
}

}  // namespace inference
}  // namespace paddle

namespace helper {

template <typename T>
struct Slot {
  std::string id;
  std::vector<T> data;
};

class Data {
 public:
  Data(std::string file_name,
       size_t start,
       size_t end) : _total_length(0) {
    _file.open(file_name);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
    read_file_to_vec(start, end);
    reset_current_line();
  }
  void reset_current_line();
  void read_file_to_vec(const size_t start, const size_t end);
  void get_slots(std::vector<Slot<float>> *slots);
  const std::vector<std::string>& get_lines() {
    return _lines;
  };

 private:
  std::fstream _file;
  size_t _total_length;
  size_t _inputs_size;
  std::vector<std::string> _lines;
  size_t _current_line;
};

void Data::read_file_to_vec(const size_t start, const size_t end) {
  std::string line;
  size_t count = 0;
  _lines.clear();
  while (std::getline(_file, line)) {
    if (count >= start && count <= end) {
      _lines.push_back(line);
    }
    count++;
  }
  _inputs_size = _lines.size();
}

void Data::reset_current_line() { _current_line = 0; }

void Data::get_slots(std::vector<Slot<float>> *slots) {
  slots->clear();
  while (_current_line < _lines.size()) {
    // 1. Split current line to slot_name and data.
    std::vector<std::string> line;
    paddle::inference::split(_lines[_current_line], '\t', &line);
    CHECK_EQ(line.size(), 2);

    // 2. Construct the slot object.
    Slot<float> slot;
    slot.id = line[0];
    paddle::inference::split_to_float(line[1], ' ', &slot.data);
    for (const auto& data: slot.data) {
      DLOG(INFO) << "[data]: " << data;
    }
    slots->push_back(slot);
    _current_line++;
  }
}

void link_slots_and_tensors(
    paddle::PaddlePredictor *predictor,
    std::unordered_map<std::string, std::unique_ptr<paddle::ZeroCopyTensor>>
        *slot_id_tensor_map) {
  const auto &input_names = predictor->GetInputNames();
  for (const auto &name : input_names) {
    auto input = predictor->GetInputTensor(name);
    slot_id_tensor_map->insert(std::make_pair(name, std::move(input)));
  }
}

template <typename T>
int slots_to_tensors(
    const std::vector<Slot<T>> &slots,
    std::unordered_map<std::string, std::unique_ptr<paddle::ZeroCopyTensor>>
        *slot_id_tensor_map,
    const paddle::PaddlePlace &place,
    int batch_size = 1) {
  int cnt = 0;
  for (const auto &slot : slots) {
    if (slot_id_tensor_map->count(slot.id) == 0) {
      ++cnt;
      continue;
    }
    auto tensor = (*slot_id_tensor_map)[slot.id].get();
    int ins_size = slot.data.size() / batch_size;
    tensor->Reshape({batch_size, ins_size});

    T *dst = tensor->template mutable_data<T>(place);
    memcpy(dst, &slot.data[0], sizeof(T) * slot.data.size());
  }
  LOG(INFO) << "skip " << cnt << " slots";
}

template <typename T>
int tensor_to_vector(paddle::ZeroCopyTensor *tensor_p, std::vector<T>* vec) {
  CHECK(tensor_p->type() == paddle::PaddleDType::FLOAT32);
  std::vector<int> shape = tensor_p->shape();
  std::stringstream ss;
  for (auto& s: shape) {
    ss << s << ",";
  }
  LOG(INFO) << "Shape of " << tensor_p->name() << " [" << ss.str() << "]";
  int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  vec->resize(num);
  tensor_p->copy_to_cpu(vec->data());
}

void fill_output_tensors_by_names(paddle::PaddlePredictor* predictor,
  std::unordered_map<std::string, std::vector<float>>* tensors,
  const std::vector<std::string>& names) {
    for (const auto& name: names) {
      std::vector<float> vec;
      tensor_to_vector(predictor->GetOutputTensor(name).get(), &vec);
      tensors->insert(std::make_pair(name, vec));
    }
}

}  // namespace helper
