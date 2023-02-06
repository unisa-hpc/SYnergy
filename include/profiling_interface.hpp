#ifndef SYNERGY_PROFILING_INTERFACE_H
#define SYNERGY_PROFILING_INTERFACE_H

#include <sycl/sycl.hpp>

namespace synergy {

class profiling_interface {
public:
  virtual void profile(sycl::event &event) = 0;
  virtual double consumption() = 0;
  virtual ~profiling_interface() = default;
};

} // namespace synergy

#endif