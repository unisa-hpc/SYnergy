#ifndef SYNERGY_ENERGY_NVIDIA_H
#define SYNERGY_ENERGY_NVIDIA_H

#include <nvml.h>
#include <sycl/sycl.hpp>

#include "profiling_interface.hpp"

namespace synergy {

class profiling_nvidia : public profiling_interface {
public:
  profiling_nvidia();
  ~profiling_nvidia();
  void profile(sycl::event &e);
  double consumption();

private:
  nvmlDevice_t device_handle;
  std::function<void(sycl::event)> energy_function;
  static constexpr int intervals = 100000;
  static constexpr int sampling_rate = 15; // ms

  double energy_consumption = 0.0;
};

} // namespace synergy

#endif