#pragma once

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <map>
#include "../types.hpp"


namespace synergy {
inline namespace v1 {
namespace detail {

/**
 * @brief Energy consumption
*/
struct consumption_t {
  frequency frequency;
  energy energy;
  time time;
};

/**
 * @brief Kernel information
*/
class belated_kernel {
protected:
  std::map<frequency, consumption_t> core_freq_predictions; // target core frequency -> predicted energy consumption
  frequency best_core_frequency; // predicted core frequency that satisfies target metric
  frequency setted_core_frequency; // actual core frequency
public:
  std::function<void(sycl::handler&)> cgh; // command group handler
  belated_kernel(std::function<void(sycl::handler&)> cgh) : cgh{cgh} {}

  inline void add_core_freq_prediction(frequency frequency, consumption_t consumption) noexcept { core_freq_predictions[frequency] = consumption; }
  inline void add_core_freq_prediction(frequency frequency, energy energy, time time) noexcept { core_freq_predictions[frequency] = consumption_t{frequency, energy, time}; }
  inline void del_core_freq_prediction(frequency frequency) { core_freq_predictions.erase(frequency); }
  inline consumption_t get_core_freq_prediction(frequency frequency) const { return core_freq_predictions.at(frequency); }
  inline void set_best_core_frequency(frequency best_core_frequency) { this->best_core_frequency = best_core_frequency; }
  inline void set_actual_core_frequency(frequency setted_core_frequency) { this->setted_core_frequency = setted_core_frequency; }
  inline const std::map<frequency, consumption_t>& get_core_predictions() const noexcept { return core_freq_predictions; }
  inline frequency get_best_core_frequency() const noexcept { return best_core_frequency; }
  inline frequency get_actual_core_frequency() const noexcept { return setted_core_frequency; }
  
  /**
   * @brief Calculate the time speedup for a given frequency
   * @details This function is used to calculate the time speedup between a certain core frequency and the currently setted core frequency
   * @param freq Frequency to calculate the time speedup
   * @return A value representing the time speedup
   * @throw std::runtime_error if the core frequency prediction is not found
  */
  inline float calculate_time_speedup(frequency freq) const {
    auto curr_time = core_freq_predictions.at(setted_core_frequency).time;
    auto new_time = core_freq_predictions.at(freq).time;

    return curr_time / new_time;
  }

  /**
   * @brief Calculate the energy saving for a given frequency
   * @details This function is used to calculate the energy saving between a certain core frequency and the currently setted core frequency
   * @param freq Frequency to calculate the energy speedup
   * @return A float value representing the energy saving: if the value is positive, the energy consumption is lower than the currently setted core frequency, otherwise it is higher
   * @throw std::runtime_error if the core frequency prediction is not found
  */
  inline float calculate_energy_saving(frequency freq) const {
    auto curr_energy = core_freq_predictions.at(setted_core_frequency).energy;
    auto new_energy = core_freq_predictions.at(freq).energy;

    return curr_energy - new_energy;
  }

  /**
   * @brief Calculates the metric value for a given target metric and frequency.
   * 
   * @param metric The target metric.
   * @param freq The frequency.
   * @return The calculated metric value.
   * @throw std::out_of_range if the core frequency prediction is not found
   * @todo Refine the calculation of the metric values.
   */
  inline float get_metric_value(target_metric metric, frequency freq) {
    if (core_freq_predictions.find(freq) == core_freq_predictions.end()) {
      throw std::out_of_range("synergy::belated_kernel error: core frequency prediction not found");
    }
    
    switch (metric) {
      case target_metric::EDP:
        return core_freq_predictions.at(freq).energy * core_freq_predictions.at(freq).time;
      case target_metric::ED2P:
        return core_freq_predictions.at(freq).energy * core_freq_predictions.at(freq).time * core_freq_predictions.at(freq).time;
      case target_metric::ES_25:
        return calculate_energy_saving(freq) / core_freq_predictions.at(freq).energy;
      case target_metric::ES_50:
        return calculate_energy_saving(freq) / core_freq_predictions.at(freq).energy;
      case target_metric::ES_75:
        return calculate_energy_saving(freq) / core_freq_predictions.at(freq).energy;
      case target_metric::ES_100:
        return calculate_energy_saving(freq) / core_freq_predictions.at(freq).energy;
      case target_metric::PL_00:
        return calculate_time_speedup(freq) / core_freq_predictions.at(freq).time;
      case target_metric::PL_25:
        return calculate_time_speedup(freq) / core_freq_predictions.at(freq).time;
      case target_metric::PL_50:
        return calculate_time_speedup(freq) / core_freq_predictions.at(freq).time;
      case target_metric::PL_75:
        return calculate_time_speedup(freq) / core_freq_predictions.at(freq).time;
      default:
        return 0;
    }
  }

  /**
   * Calculates the cost associated with a given frequency comparing the provided frequency values with the best predicted.
   *
   * @param target The target metric.
   * @param freq The frequency for which to calculate the cost. If not specified, the actual core frequency is used.
   * @return A float value representing the cost. If the value is positive, the cost is higher than the best predicted, otherwise it is lower.
   */
  inline float get_cost(target_metric target, frequency freq = 0) {
    auto comp_freq = freq ? freq : this->get_actual_core_frequency();
    return get_metric_value(target, freq) - get_metric_value(target, this->get_best_core_frequency());
  }
};

} // namespace detail
} // namespace v1
} // namespace synergy