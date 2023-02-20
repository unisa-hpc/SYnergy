#include <cassert>
#include <memory>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "runtime.hpp"
#include "vendor_device.hpp"
#include "vendor_implementations.hpp"

namespace synergy {

runtime& runtime::get_instance()
{
  static runtime r;
  return r;
}

// TODO: handle the case where different platform may expose the same device (very-low priority, since there is no way to do it properly in SYCL)
// TODO: make sure that index given to synergy::device constructor is the "same" of the sycl::device
runtime::runtime()
{
  using namespace sycl;

  auto platforms = platform::get_platforms();
  int count_hip = 0;

  for (int i = 0; i < platforms.size(); i++) {

    std::string platform_name = platforms[i].get_info<info::platform::name>();
    std::transform(platform_name.begin(), platform_name.end(), platform_name.begin(), ::tolower);

#ifdef SYNERGY_PROOF
    std::cout << "\nplatform: " << platforms[i].get_info<info::platform::name>() << " ";
#endif

#ifdef SYNERGY_CUDA_SUPPORT
    if (platform_name.find("cuda") != std::string::npos) {
      auto nvidia_devices = platforms[i].get_devices(info::device_type::gpu);

      for (int j = 0; j < nvidia_devices.size(); j++) {
        auto ptr = std::make_shared<vendor_device<synergy::management::nvml>>(j);
        devices.insert({nvidia_devices[j], ptr});
      }
    }
#endif

#ifdef SYNERGY_ROCM_SUPPORT
    if (platform_name.find("hip") != std::string::npos) {
      auto amd_devices = platforms[i].get_devices(info::device_type::gpu);

      for (int j = 0; j < amd_devices.size(); j++) {
        auto ptr = std::make_shared<vendor_device<synergy::management::rsmi>>(count_hip); // passing count_hip is not an error: compile with SYNERGY_PROOF
        count_hip++;                                                                      // there is one platform for each AMD HIP GPU
        devices.insert({amd_devices[j], ptr});
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
