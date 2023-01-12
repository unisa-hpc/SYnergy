#ifndef _SYNERGY_INTERFACE_H_
#define _SYNERGY_INTERFACE_H_
#include <sycl/sycl.hpp>

namespace synergy {

class energy_interface {
public:
  virtual void process(sycl::event &event) = 0;
  virtual double consumption() = 0;
  virtual ~energy_interface() = default;
};

} // namespace synergy

#endif