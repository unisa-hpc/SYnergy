#ifndef SYNERGY_QUEUE_H
#define SYNERGY_QUEUE_H

#include <sycl/sycl.hpp>

#include "profiling_interface.hpp"
#include "scaling_interface.hpp"
#include "types.h"
#include "utils.hpp"
#include "vendor_implementations.hpp"

namespace synergy {

class queue : public sycl::queue {
public:
  using base = sycl::queue;

  template <typename... Args, std::enable_if_t<!::details::is_present_v<sycl::property_list, Args...> && !::details::has_property_v<Args...>, bool> = true>
  queue(Args &&...args)
      : base(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{})
  {
    initialize_queue();
  }

  template <typename... Args, std::enable_if_t<::details::is_present_v<sycl::property_list, Args...> || ::details::has_property_v<Args...>, bool> = true>
  queue(Args &&...args)
      : base(std::forward<Args>(args)...)
  {
    auto &&args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
    if constexpr (::details::is_present_v<sycl::property_list, Args...>) {
      sycl::property_list &&prop = std::get<::details::Index<sycl::property_list, Args...>::value>(args_tuple);
      if (!prop.has_property<sycl::property::queue::enable_profiling>()) {
        throw std::runtime_error("synergy::queue: enable_profiling property is required");
      }
    } else {
      if constexpr (!::details::is_present_v<sycl::property::queue::enable_profiling, Args...>) {
        throw std::runtime_error("synergy::queue: enable_profiling property is required");
      }
    }

    initialize_queue();
  }

  template <typename... Args>
  inline sycl::event submit(Args &&...args)
  {
    auto &&event = sycl::queue::submit(std::forward<Args>(args)...);
    m_energy->profile(event);
    return event;
  }

  inline double energy_consumption()
  {
    return m_energy->consumption();
  }

  inline std::vector<frequency> query_supported_frequencies()
  {
    return m_scaling->query_supported_frequencies();
  }

  inline std::vector<frequency> query_supported_core_frequencies(frequency memory_frequency)
  {
    return m_scaling->query_supported_core_frequencies(memory_frequency);
  }

private:
  std::unique_ptr<profiling_interface> m_energy;
  std::unique_ptr<scaling_interface> m_scaling;

  void initialize_queue();
};

} // namespace synergy

#endif // SYNERGY_QUEUE_H