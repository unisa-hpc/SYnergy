#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include "types.hpp"
#include "belated_kernel.hpp"


namespace synergy {
inline namespace v1 {
namespace detail {

class phase_manager {
private:
  target_metric metric;
  float freq_change_cost = 1.0f; // TODO understand this value
  std::vector<belated_kernel> kernels;

protected:
  template<typename Iterator>
  double calculate_cost(Iterator begin, Iterator end) {
    double cost = 0.0;
    for (auto it = begin; it != end; ++it) {
      cost += it->get_cost(metric);
    }
    return cost;
  }

  template<typename Iterator>
  double calculate_cost(frequency freq, Iterator begin, Iterator end) {
    double cost = 0.0;
    for (auto it = begin; it != end; ++it) {
      cost += it->get_cost(freq, metric);
    }
    return cost;
  }

  template<typename Iterator>
  size_t one_change(frequency first_freq, frequency second_freq, Iterator begin, Iterator end) const {
    double min = this->calculate_cost(begin, end);
    size_t best = end - begin;

    for (int i = end - begin - 1; i >= 0; --i) {
      auto mit = begin + i;
      auto cost_l = this->calculate_cost(first_freq, begin, mit);
      auto cost_r = this->calculate_cost(second_freq, mit + 1, end);
      auto cost = cost_l + cost_r + freq_change_cost;
      if (cost < min) {
        min = cost;
        best = i;
      }
    }
    return best;
  }

public:
  phase_manager(target_metric metric) : metric{metric} {}

};

} // namespace detail
} // namespace v1
} // namespace synergy