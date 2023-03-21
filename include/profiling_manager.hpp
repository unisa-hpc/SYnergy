#pragma once

#include <thread>
#include <vector>

#include "device.hpp"
#include "kernel.hpp"
#include "profilers.hpp"

namespace synergy {

namespace detail {

class profiling_manager {
public:
  friend class fine_grained_profiler<profiling_manager>;
  friend class coarse_grained_profiler<profiling_manager>;

  profiling_manager(device& device) : device{device} {
    kernels_profiler = std::thread{fine_grained_profiler<profiling_manager>{*this}};
    device_profiler = std::thread{coarse_grained_profiler<profiling_manager>{*this}};
  }

  ~profiling_manager() {
    finished = true;
    kernels_profiler.join();
    device_profiler.join();
  }

  void profile_kernel(sycl::event event) {
    kernels.push_back(kernel{event});
  }

  double kernel_energy(const sycl::event& event) const {
    kernel dummy{event};
    auto it = std::find(kernels.begin(), kernels.end(), dummy);
    if (it == kernels.end()) {
      throw std::runtime_error("synergy::queue error: kernel was not submitted to the queue");
    }

    return it->energy;
  }

  double device_energy() const {
    return device_energy_consumption;
  }

private:
  device device;
  double device_energy_consumption = 0.0;
  std::vector<kernel> kernels;
  bool finished = false;

  std::thread kernels_profiler;
  std::thread device_profiler;
};

} // namespace detail

} // namespace synergy
