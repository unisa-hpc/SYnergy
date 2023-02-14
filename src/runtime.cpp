#include <cassert>
#include <memory>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "runtime.hpp"
#include "vendor_device.hpp"
#include "vendor_implementations.hpp"

namespace synergy {
std::unique_ptr<runtime> runtime::instance = nullptr;

void runtime::initialize()
{
  assert(instance == nullptr);
  instance = std::unique_ptr<runtime>(new runtime());
}

runtime& runtime::get_instance()
{
  if (instance == nullptr) {
    throw std::runtime_error("synergy::runtime was not initialized");
  }
  return *instance;
}

// TODO: handle the case where different platform may expose the same device (very-low priority, since there is no way to do it properly in SYCL)
// TODO: make sure that index given to synergy::device constructor is the "same" of the sycl::device
runtime::runtime()
{
  using namespace sycl;

  for (platform& plat : platform::get_platforms()) {

    std::string vendor = plat.get_info<info::platform::vendor>();
    std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

#ifdef SYNERGY_CUDA_SUPPORT
    if (vendor.find("nvidia") != std::string::npos) {
      auto nvidia_devices = plat.get_devices(info::device_type::gpu);

      for (int i = 0; i < nvidia_devices.size(); i++) {
        auto ptr = std::make_shared<vendor_device<synergy::management::nvml>>(i);
        devices.insert({nvidia_devices[i], ptr});
      }
    }
#endif

#ifdef SYNERGY_ROCM_SUPPORT
    if (vendor.find("amd") != std::string::npos || vendor.find("advanced micro devices") != std::string::npos) {
      auto amd_devices = plat.get_devices(info::device_type::gpu);

      for (int i = 0; i < amd_devices.size(); i++) {
        auto ptr = std::make_shared<vendor_device<synergy::management::rsmi>>(i);
        devices.insert({amd_devices[i], ptr});
      }
    }
#endif
  }
}

std::shared_ptr<device> runtime::assign_device(const sycl::device& sycl_device)
{
  auto search = devices.find(sycl_device);
  if (search == devices.end())
    throw std::runtime_error("error while assigning synergy::device to queue: sycl::device not supported");

  return search->second;
}

} // namespace synergy
