#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include "types.hpp"
#include "belated_kernel.hpp"


namespace synergy {
inline namespace v1 {
namespace detail {

struct freq_change_t {
  size_t index;
  frequency left_freq;
  frequency right_freq;
};

class phase_manager {
private:
  target_metric metric;
  float freq_change_cost = 1.0f; // TODO understand this value
  std::vector<belated_kernel> kernels;

protected:

  /**
   * @brief Finds the best frequency within the given range.
   * @param begin An iterator pointing to the beginning of the range.
   * @param end An iterator pointing to the end of the range.
   * @return The best frequency found within the range.
   */
  template<typename Iterator>
  frequency find_best_frequency(Iterator begin, Iterator end) {
    return begin->get_best_core_frequency(); // TODO implement some euristic that finds the best frequency.
  }
  
  /**
   * @brief Calculates the cost of a given range.
   * @param begin The beginning iterator.
   * @param end The ending iterator.
   * @param freq The frequency to be used in the calculation. If 0, the best frequency will be used.
   * @return The cost of the given range.
  */
  template<typename Iterator>
  double calculate_cost(Iterator begin, Iterator end, frequency freq = 0) {
    double cost = 0.0;
    for (auto it = begin; it != end; ++it) {
      cost += freq ? it->get_cost(freq, metric) : it->get_cost(metric);
    }
    return cost;
  }  

  /**
   * @brief Calculates the best frequency change in a given range.
   * @param begin The beginning iterator.
   * @param end The ending iterator.
   * @return The frequency change between the two iterators.
   */
  template<typename Iterator>
  freq_change_t one_change(Iterator begin, Iterator end) const {
    double min = std::numeric_limits<double>::max(); // TODO check if it needs another value
    
    freq_change_t best {
      .index = 0,
      .left_freq = 0,
      .right_freq = 0
    }; // TODO maybe setting a default value is a good idea

    for (int it = begin; it < end - 1; it++) {
      auto mit = begin + it;
      frequency best_freq_l = this->find_best_frequency(begin, mit);
      frequency best_freq_r = this->find_best_frequency(mit + 1, end);
      auto cost_l = this->calculate_cost(begin, mit, best_freq_l);
      auto cost_r = this->calculate_cost(mit + 1, end, best_freq_r);
      auto cost = cost_l + cost_r + freq_change_cost;
      if (cost < min) {
        min = cost;
        best.index = it;
        best.left_freq = best_freq_l;
        best.right_freq = best_freq_r;
      }
    }
    return best;
  }

  /**
   * @brief Returns a the best frequency change in the kernel vector.
   * @return The freq_change_t representing a single change.
   */
  freq_change_t one_change() {
    return this->one_change(kernels.begin(), kernels.end());
  }

public:
  phase_manager(target_metric metric) : metric{metric} {}

};

} // namespace detail
} // namespace v1
} // namespace synergy