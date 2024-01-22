#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <set>
#include "types.hpp"
#include "belated_kernel.hpp"
#include "task_graph_state.hpp"


namespace synergy {
namespace detail {

struct freq_change_t {
  size_t index;
  double cost;
};

struct phase_t {
  size_t start;
  size_t end;
  frequency target_freq;
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
  synergy::detail::task_graph& task_graph;
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
      cost += it->get_cost(metric, freq);
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
      frequency best_freq_l = this->find_best_frequency(begin, it);
      frequency best_freq_r = this->find_best_frequency(it + 1, end);
      auto cost_l = this->calculate_cost(begin, it, best_freq_l);
      auto cost_r = this->calculate_cost(it + 1, end, best_freq_r);
      auto cost = cost_l + cost_r + freq_change_overhead;
      if (cost < min) {
        min = best.cost = cost;
        best.index = std::distance(begin, it);
      }
    }

    return best;
  }

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
    auto left_freq = this->find_best_frequency(begin, begin + mid_change.index);
    auto right_freq = this->find_best_frequency(begin + mid_change.index + 1, end);
    auto left = this->flex_change(begin, begin + mid_change.index, left_freq);
    auto right = this->flex_change(begin + mid_change.index + 1, end, right_freq);

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
   */
  void compute_overhead() { // TODO: Understand how to calculate the overhead.
    this->freq_change_overhead = 1.0f;
  }

public:
  phase_manager(target_metric metric, 
                synergy::detail::task_graph& task_graph) : 
                  metric{metric}, task_graph{task_graph} {
    kernels = task_graph.get_kernels();
  }

  /**
   * Retrieves the phases.
   * 
   * @return A vector of phase_t objects representing the phases.
   */
  std::vector<phase_t> get_phases() {
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
      auto end = change + 1;
      auto target_freq = this->find_best_frequency(kernels.begin() + start, kernels.begin() + end);
      phases.push_back(synergy::detail::phase_t{
        .start = start,
        .end = end,
        .target_freq = target_freq
      });
      start = change + 1;
    }
    return phases;
  }
};

} // namespace detail
} // namespace synergy