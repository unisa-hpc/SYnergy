#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <set>
#include "types.hpp"
#include "belated_kernel.hpp"


namespace synergy {
namespace detail {

struct freq_change_t {
  size_t index;
  double cost;
};

/**
 * @class phase_manager
 * @brief Manages the phases and frequency changes for a set of kernels.
 *
 * The `phase_manager` class is responsible for managing the phases and frequency changes
 * for a set of kernels. It provides functions to calculate the best frequency changes,
 * calculate the cost of a given range, and retrieve the phases.
 *
 * The class also allows adding kernels, checking the consistency of the phase manager,
 * and flushing any pending changes.
 * @warning The phase manager is not thread-safe.
 */
class phase_manager {
private:
  target_metric metric;
  float freq_change_overhead = 1.0f; // TODO understand this value
  std::vector<belated_kernel> kernels;
  bool consistent = true;

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
      cost += get_cost(metric, freq);
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
  freq_change_t one_change(Iterator begin, Iterator end) {
    double min = std::numeric_limits<double>::max(); // TODO check if it needs another value

    freq_change_t best {
      .index = 0,
      .cost = min
    }; // TODO maybe setting a default value is a good idea

    for (auto it = begin; it < end - 1; it++) {
      auto mit = begin + it;
      frequency best_freq_l = this->find_best_frequency(begin, mit);
      frequency best_freq_r = this->find_best_frequency(mit + 1, end);
      auto cost_l = this->calculate_cost(begin, mit, best_freq_l);
      auto cost_r = this->calculate_cost(mit + 1, end, best_freq_r);
      auto cost = cost_l + cost_r + freq_change_overhead;
      if (cost < min) {
        min = best.cost = cost;
        best.index = it;
      }
    }

    return best;
  }

  struct phase_t {
    size_t start;
    size_t end;
    frequency target_freq;
  };

  /**
   * Calculates the flexible change in frequency for a given range of elements.
   *
   * @param begin The iterator pointing to the beginning of the range.
   * @param end The iterator pointing to the end of the range.
   * @param curr_change The current change in frequency.
   * @return A vector of freq_change_t representing the flexible change in frequency.
   */
  template<typename Iterator>
  std::vector<freq_change_t> flex_change(Iterator begin, Iterator end, frequency curr_change) {
    if (end - begin < 2) {
      return {};
    }

    auto cost_no_change = this->calculate_cost(begin, end, curr_change);

    freq_change_t mid_change = this->one_change(begin, end);
    auto left = this->flex_change(begin, begin + mid_change.index, mid_change.left_freq);
    auto right = this->flex_change(begin + mid_change.index + 1, end, mid_change.right_freq);

    double left_cost = 0.0;
    for (auto& change : left) {
      left_cost += change.cost;
    }
    double right_cost = 0.0;
    for (auto& change : right) {
      right_cost += change.cost;
    }

    auto cost_change = mid_change.cost + left_cost + right_cost + this->freq_change_overhead;

    if (cost_change < cost_no_change) {
      left.push_back(mid_change);
      left.insert(left.end(), right.begin(), right.end());
      return left;
    } else {
      return {};
    }
  }

  /**
   * @brief Calculates the best changes to perform in a given range.
   * @details This function is used to calculate the best changes to perform in a given range of elements.
   * It Uses the find_best_frequency and calculate_cost functions to calculate the best frequency and the cost of the given starting range.
   * @param begin The beginning iterator.
   * @param end The ending iterator.
   * @return A vector of freq_change_t representing the best changes.
  */
  template<typename Iterator>
  std::vector<freq_change_t> flex_change(Iterator begin, Iterator end) {
    auto freq = this->find_best_frequency(begin, end);
    return this->flex_change(begin, end, freq);
  }

  /**
   * @brief Calculates the best changes to perform in the kernel vector.
   * @return A vector of freq_change_t representing the best changes.
  */
  std::vector<freq_change_t> flex_change() {
    return this->flex_change(kernels.begin(), kernels.end());
  }

  /**
   * @brief Returns a the best frequency change in the kernel vector.
   * @return The freq_change_t representing a single change.
   */
  freq_change_t one_change() {
    return this->one_change(kernels.begin(), kernels.end());
  }

  /**
   * @brief Calculates the overhead of a frequency change.
   * @todo Understand how to calculate the overhead.
   */
  void compute_overhead() {
    this->freq_change_overhead = 1.0f;
  }

public:
  phase_manager(target_metric metric) : metric{metric} {}

  /**
   * Retrieves the phases.
   * 
   * @return A vector of phase_t objects representing the phases.
   */
  std::vector<phase_t> get_phases() {
    // set the consistency to false in order to avoid adding kernels after the phases are calculated
    consistent = false;

    // compute the overhead of a frequency change
    compute_overhead();

    // calculate the best changes
    std::vector<phase_t> phases;
    auto changes = this->flex_change();
    size_t start = 0;
    std::set<size_t> change_points;
    for (auto& change : changes) {
      change_points.insert(change.index);
    }
    for (auto& change : change_points) {
      phases.push_back({
        .start = start,
        .end = change,
        .target_freq = this->find_best_frequency(kernels.begin() + start, kernels.begin() + change)
      });
      start = change + 1;
    }
    return phases;
  }

  /**
   * @brief Adds a kernel to the phase manager.
   * @details This function is used to add a kernel to the phase manager.
   * @param kernel The kernel to be added.
   * @throw std::runtime_error if the phase manager is not consistent.
  */
  inline void add_kernel(belated_kernel kernel) {
    if (!consistent) {
      throw std::runtime_error("synergy::phase_manager error: cannot add kernel to inconsistent phase manager");
    }
    kernels.push_back(kernel);
  }

  /**
   * @brief Get the kernels.
   *
   * This function returns a constant reference to the vector of belated_kernel objects.
   *
   * @return A constant reference to the vector of belated_kernel objects.
   */
  inline const std::vector<belated_kernel>& get_kernels() const {
    return kernels;
  }

  /**
   * @brief Checks if the phase manager is consistent.
   * @return A boolean value representing the consistency of the phase manager.
  */
  inline const bool is_consistent() const {
    return consistent;
  }

  /**
   * @brief Flushes any pending changes in the phase manager.
   * 
   * This function is responsible for flushing any pending changes in the phase manager.
   * It ensures that all changes made to the phase manager are applied and reflected in the system.
   */
  inline void flush() {
    kernels.clear();
    consistent = true;
  }
};

} // namespace detail
} // namespace synergy