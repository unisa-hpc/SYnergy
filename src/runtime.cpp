#include <cassert>
#include <memory>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "runtime.hpp"
#include "vendors/nvml_wrapper.hpp"
#include "vendors/rsmi_wrapper.hpp"

namespace synergy {
std::unique_ptr<runtime> runtime::instance = nullptr;

void runtime::initialize()
{
  assert(instance == nullptr);
  instance = std::make_unique<runtime>();
}

runtime& runtime::get_instance()
{
  if (instance == nullptr) {
    throw std::runtime_error("synergy::runtime was not initialized");
  }
  return *instance;
}

// TODO: handle the case where different platform may have the same device
// TODO: make sure that index given to synergy::device constructor is the "same" of the sycl::device
runtime::runtime()
{
  using namespace sycl;

  for (platform& plat : platform::get_platforms()) {

    std::string vendor = plat.get_info<info::platform::vendor>();
    std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

    if (vendor.find("nvidia") != std::string::npos) {
      auto nvidia_devices = plat.get_devices(info::device_type::gpu);

      for (int i = 0; i < nvidia_devices.size(); i++)
        devices.push_back(synergy::device<synergy::management::nvml>(i));
    }

    if (vendor.find("amd") != std::string::npos || vendor.find("advanced micro devices") != std::string::npos) {
      auto amd_devices = plat.get_devices(info::device_type::gpu);

      for (int i = 0; i < amd_devices.size(); i++)
        devices.push_back(synergy::device<synergy::management::rsmi>(i));
    }
  }
}

} // namespace synergy
