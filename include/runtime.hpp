#pragma once

#include <memory>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "device.hpp"
#include "vendor_implementations.hpp"

namespace synergy {

namespace detail {

class runtime {
public:
  static synergy::device synergy_device_from(const sycl::device& sycl_device) {
    static runtime r;

    auto search = r.devices.find(sycl_device);
    if (search == r.devices.end())
      throw std::runtime_error("error while assigning synergy::device to queue: sycl::device not supported");

    return search->second;
  }

  runtime(runtime const&) = delete;
  runtime(runtime&&) = delete;
  runtime& operator=(runtime const&) = delete;
  runtime& operator=(runtime&&) = delete;

private:
  std::unordered_map<sycl::device, synergy::device> devices;

  // TODO: handle the case where different platform may expose the same device (very-low priority, since there is no way to do it properly in SYCL)
  // TODO: make sure that index given to synergy::device constructor is the "same" of the sycl::device
  runtime() {
    using namespace sycl;

    auto platforms = platform::get_platforms();

#ifdef SYNERGY_ROCM_SUPPORT
    int count_hip = 0;
#endif
    for (size_t i = 0; i < platforms.size(); i++) {

      std::string platform_name = platforms[i].get_info<info::platform::name>();
      std::transform(platform_name.begin(), platform_name.end(), platform_name.begin(), ::tolower);

#ifdef SYNERGY_PROOF
      std::cout << "\nplatform: " << platforms[i].get_info<info::platform::name>() << " ";
#endif

#ifdef SYNERGY_CUDA_SUPPORT
      if (platform_name.find("cuda") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);

        for (size_t j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::nvml>>(j);
          devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif

#ifdef SYNERGY_ROCM_SUPPORT
      if (platform_name.find("hip") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);

        for (size_t j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::rsmi>>(count_hip); // passing count_hip is not an error: compile with SYNERGY_PROOF
          count_hip++;                                                             // there is one platform for each AMD HIP GPU
          devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif

#ifdef SYNERGY_LZ_SUPPORT
      if (platform_name.find("level-zero") != std::string::npos ||
          platform_name.find("level zero") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);
        for (int j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::lz>>(j);
          devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif
    }
  }
};
} // namespace detail

} // namespace synergy
