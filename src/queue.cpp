#include "../include/queue.hpp"

namespace synergy {

void queue::initialize_queue()
{
  if (!get_device().is_gpu())
    throw std::runtime_error("synergy::queue: only GPUs are supported");

  std::string vendor = get_device().get_info<sycl::info::device::vendor>();
  std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

  if (vendor.find("nvidia") != std::string::npos) {
#ifdef SYNERGY_CUDA_SUPPORT
    m_energy = std::make_unique<profiling_nvidia>();
    m_scaling = std::make_unique<scaling_nvidia>();
#else
    throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
#endif
  } else if (vendor.find("amd") != std::string::npos) {
#ifdef SYNERGY_ROCM_SUPPORT
    m_energy = std::make_unique<profiling_amd>();
    m_scaling = std::make_unique<scaling_amd>();
#else
    throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
#endif
  } else {
    throw std::runtime_error("synergy::queue: vendor \"" + vendor + "\" not supported");
  }

  m_scaling->set_device_frequency(frequency_preset::default_frequency, frequency_preset::max_frequency);
}

} // namespace synergy