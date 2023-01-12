#ifndef _SYNERGY_QUEUE_H_
#define _SYNERGY_QUEUE_H_

#include <chrono>
#include <functional>
#include <sycl/sycl.hpp>
#include <utility>

#include "energy_implementations.hpp"
#include "scaling_interface.hpp"
#include "utils.hpp"

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
  sycl::event submit(Args &&...args)
  {
    auto &&event = sycl::queue::submit(std::forward<Args>(args)...);
    m_energy->process(event);
    return event;
  }

  double get_queue_consumption()
  {
    m_energy->consumption();
  }

private:
  std::unique_ptr<energy_interface> m_energy;
  std::unique_ptr<scaling_interface> m_scaling;

  inline void initialize_queue()
  {
    if (get_device().is_gpu())
      throw std::runtime_error("synergy::queue: only GPUs are supported");

    std::string vendor = get_device().get_info<sycl::info::device::vendor>();
    if (vendor.find("nvidia")) {
#ifdef SYNERGY_CUDA_SUPPORT
      m_energy = std::make_unique<energy_nvidia>();
      m_scaling = std::make_unique<scaling_nvidia>();
#else
      throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
#endif
    } else if (vendor.find("amd")) {
#ifdef SYNERGY_ROCM_SUPPORT
      m_energy = std::make_unique<energy_amd>();
#else
      throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
#endif
    } else {
      throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
    }

    m_scaling->change_frequency(frequency::default_frequency, frequency::max_frequency);
  }
};

} // namespace synergy

#endif // _SYNERGY_QUEUE_H_