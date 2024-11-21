#pragma once

#include <future>
#include <thread>
#include <vector>

#include "device.hpp"
#include "kernel.hpp"
#include "profilers.hpp"

namespace synergy {

namespace detail {

class profiling_manager {
public:
  friend class concurrent_kernel_profiler<profiling_manager>;
  friend class sequential_kernel_profiler<profiling_manager>;
  friend class device_profiler<profiling_manager>;
#ifdef SYNERGY_HOST_PROFILING
  friend class host_device_profiler<profiling_manager>;
#endif
  profiling_manager(device& device) : device{device} {
#ifdef SYNERGY_DEVICE_PROFILING
#ifdef SYNERGY_HOST_PROFILING
    device_profiler = std::thread{detail::host_device_profiler<profiling_manager>{*this}};
#else
    device_profiler = std::thread{detail::device_profiler<profiling_manager>{*this}};
#endif
#endif
  }

  ~profiling_manager() {
    finished.store(true, std::memory_order_release);
#ifdef SYNERGY_DEVICE_PROFILING
    device_profiler.join();
#endif
  }

#ifdef SYNERGY_KERNEL_PROFILING
  void profile_kernel(sycl::event event, energy start_energy) {
    kernels.push_back(kernel{event});

    auto future = std::async(std::launch::async, sequential_kernel_profiler<profiling_manager>{*this, kernels.back(), start_energy});
    // this will automatically serialize all profiled kernels
    // since the std::future destructor will wait until the async thread ends
  }

  double kernel_energy(const sycl::event& event) const {
    kernel dummy{event};
    auto it = std::find(kernels.begin(), kernels.end(), dummy);
    if (it == kernels.end()) {
      throw std::runtime_error("synergy::queue error: kernel was not submitted to the queue");
    }

    return it->energy;
  }
#endif

#ifdef SYNERGY_DEVICE_PROFILING
  double device_energy() const {
    return device_energy_consumption;
  }
#ifdef SYNERGY_HOST_PROFILING
  double host_energy() const {
    return host_energy_consumption;
  }
#endif
#endif

private:
  device device;
  double device_energy_consumption = 0.0;
  double host_energy_consumption = 0.0;
  std::atomic<bool> finished = false;
#ifdef SYNERGY_KERNEL_PROFILING
  std::vector<kernel> kernels;
#endif

#ifdef SYNERGY_DEVICE_PROFILING
  std::thread device_profiler;
#endif
};

} // namespace detail

} // namespace synergy
