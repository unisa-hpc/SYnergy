#include "queue.hpp"

namespace synergy {

// TODO: implement this
inline device<std::any> queue::get_synergy_device() const
{
  sycl::event e;
}

double queue::kernel_energy_consumption(sycl::event& event)
{
  auto search = kernels_energy.find(event);
  if (search == kernels_energy.end())
    throw std::runtime_error("synergy::queue error: kernel was not submitted to the queue");
}

} // namespace synergy