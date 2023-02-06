#ifndef SYNERGY_ENERGY_AMD_H
#define SYNERGY_ENERGY_AMD_H

#include <rocm_smi/rocm_smi.h>
#include <sycl/sycl.hpp>

#include "../profiling_interface.hpp"
#include "utils.hpp"

namespace synergy {

class profiling_amd : public profiling_interface {
public:
  profiling_amd();
  ~profiling_amd();
  void profile(sycl::event &e);
  double consumption();

private:
  uint32_t device_handle = 0;
  std::function<void(sycl::event)> energy_function;
  static constexpr int intervals_length = 15; // ms

  double energy_consumption = 0.0;
};

} // namespace synergy

#endif