#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include "../management_wrapper.hpp"

namespace synergy {
namespace detail {
namespace management {
struct lz {
  static constexpr std::string_view name = "LZ";
  static constexpr unsigned int max_frequencies = 256;
  static constexpr unsigned int sampling_rate = 5; // ms
  using device_identifier = unsigned int;
  using device_handle = zes_device_handle_t;
  using return_type = ze_result_t;
  static constexpr return_type return_success = ZE_RESULT_SUCCESS;
};
}; // namespace management

template <>
class management_wrapper<management::lz> {

public:
  inline unsigned int get_devices_count() const {
    return get_devices().size();
  }

  inline void initialize() const {
    check(zeInit(0));
  }

  inline void shutdown() const {}

  using lz = management::lz;

  inline lz::device_handle get_device_handle(lz::device_identifier id) const {
    auto ret = (lz::device_handle)get_devices()[id];
    return ret;
  }

  inline power get_power_usage(lz::device_handle handle) const {
    throw std::runtime_error{"synergy " + std::string(lz::name) + " wrapper error: get_power_usage is not supported"};
  }

  inline energy get_energy_usage(lz::device_handle handle) const {
    zes_pwr_handle_t hPwr;
    check(zesDeviceGetCardPowerDomain(handle, &hPwr));

    zes_power_energy_counter_t counter;
    check(zesPowerGetEnergyCounter(hPwr, &counter));
    return counter.energy;
  }

  inline std::vector<frequency> get_supported_core_frequencies(const lz::device_handle handle) const {
    return get_supported_frequency<ZES_FREQ_DOMAIN_GPU>(handle);
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(const lz::device_handle handle) const {
    return get_supported_frequency<ZES_FREQ_DOMAIN_MEMORY>(handle);
  }

  inline frequency get_core_frequency(const lz::device_handle handle) const {
    return get_frequency<ZES_FREQ_DOMAIN_GPU>(handle);
  }

  inline frequency get_uncore_frequency(const lz::device_handle handle) const {
    return get_frequency<ZES_FREQ_DOMAIN_MEMORY>(handle);
  }

  inline void set_core_frequency(const lz::device_handle handle, frequency target) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_GPU>(handle);
    double freq = static_cast<double>(target);
    zes_freq_range_t range{freq, freq};
    check(zesFrequencySetRange(h_freq, &range));
  }

  inline void set_uncore_frequency(const lz::device_handle handle, frequency target) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_MEMORY>(handle);
    double freq = static_cast<double>(target);
    zes_freq_range_t range{freq, freq};
    check(zesFrequencySetRange(h_freq, &range));
  }

  inline void set_all_frequencies(lz::device_handle handle, frequency core, frequency uncore) const {
    set_core_frequency(handle, core);
    set_uncore_frequency(handle, uncore);
  }

  inline void setup_profiling(lz::device_handle) const {}

  inline void setup_scaling(lz::device_handle) const {}

  inline std::string error_string(lz::return_type return_value) const {
    switch (return_value) {
    case ZE_RESULT_NOT_READY:
      return "synchronization primitive not signaled";
    case ZE_RESULT_ERROR_DEVICE_LOST:
      return "device hung, reset, was removed, or driver update occurred";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "insufficient host memory to satisfy call";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "insufficient device memory to satisfy call";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      return "error occurred when building module, see build log for details";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
      return "error occurred when linking modules, see build log for details";
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
      return "device requires a reset";
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
      return "device currently in low power state";
    case ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX:
      return "device is not represented by a fabric vertex";
    case ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE:
      return "fabric vertex does not represent a device";
    case ZE_RESULT_EXP_ERROR_REMOTE_DEVICE:
      return "fabric vertex represents a remote device or subdevice";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      return "access denied due to permission level";
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
      return "resource already in use and simultaneous access not allowed or resource was removed";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
      return "external required dependency is unavailable or missing";
    case ZE_RESULT_WARNING_DROPPED_DATA:
      return "data may have been dropped";
    case ZE_RESULT_ERROR_UNINITIALIZED:
      return "driver is not initialized";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      return "generic error code for unsupported versions";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "generic error code for unsupported features";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "generic error code for invalid arguments";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "handle argument is not valid";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      return "object pointed to by handle still in-use by device";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "pointer argument may not be nullptr";
    case ZE_RESULT_ERROR_INVALID_SIZE:
      return "size argument is invalid (e.g., must not be zero)";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      return "size argument is not supported by the device (e.g., too large)";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      return "alignment argument is not supported by the device (e.g., too small)";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      return "synchronization object in invalid state";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      return "enumerator argument is not valid";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      return "enumerator argument is not supported by the device";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      return "image format is not supported by the device";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      return "native binary is not supported by the device";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      return "global variable is not found in the module";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      return "kernel name is not found in the module";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      return "function name is not found in the module";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      return "group size dimension is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      return "global width dimension is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      return "kernel argument index is not valid for kernel";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      return "kernel argument size does not match kernel";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      return "value of kernel attribute is not valid for the kernel or device";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
      return "module with imports needs to be linked before kernels can be created from it";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      return "command list type does not match command queue type";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      return "copy operations do not support overlapping regions of memory";
    case ZE_RESULT_WARNING_ACTION_REQUIRED:
      return "an action is required to complete the desired operation";
    case ZE_RESULT_ERROR_UNKNOWN:
      return "unknown or internal error";
    default:
      return "code " + std::to_string(return_value);
    }
  }

private:
  error_checker<management::lz> check{*this};

  inline std::vector<ze_device_handle_t> get_devices() const {
    unsigned int drivers_count = 0;
    check(zeDriverGet(&drivers_count, nullptr));

    if (drivers_count < 1) {
      throw std::runtime_error{"synergy " + std::string(lz::name) + " wrapper error: could not get Level Zero drivers"};
    }
    std::vector<ze_driver_handle_t> drivers(drivers_count);
    check(zeDriverGet(&drivers_count, drivers.data()));

    unsigned int total_devices_count = 0;
    unsigned int devices_per_driver = 0;

    for (unsigned i = 0; i < drivers.size(); i++) {
      check(zeDeviceGet(drivers[i], &devices_per_driver, nullptr));
      total_devices_count += devices_per_driver;
    }

    std::vector<ze_device_handle_t> devices(total_devices_count);
    for (unsigned i = 0, offset = 0; i < drivers.size(); i++) {
      check(zeDeviceGet(drivers[i], &devices_per_driver, &devices.data()[offset]));
      offset += devices_per_driver;
    }

    return devices;
  }

  template <zes_freq_domain_t domain>
  zes_freq_handle_t get_frequency_handle(const management::lz::device_handle handle) const {
    zes_freq_handle_t ret = nullptr;
    unsigned handles_count = 0;
    check(zesDeviceEnumFrequencyDomains(handle, &handles_count, nullptr));

    if (handles_count > 0) {
      std::vector<zes_freq_handle_t> handles(handles_count);
      check(zesDeviceEnumFrequencyDomains(handle, &handles_count, handles.data()));
      for (unsigned i = 0; i < handles_count; i++) {
        zes_freq_properties_t props{};
        props.stype = ZES_STRUCTURE_TYPE_FREQ_PROPERTIES;
        if (zesFrequencyGetProperties(handles[i], &props) == lz::return_success) {
          if (props.type == domain) {
            ret = handles[i];
            break;
          }
        }
      }
    }
    return ret;
  }

  template <zes_freq_domain_t domain>
  std::vector<frequency> get_supported_frequency(const lz::device_handle handle) const {
    std::vector<frequency> freqs;
    zes_freq_handle_t h_freq = get_frequency_handle<domain>(handle);
    if (h_freq != nullptr) {
      unsigned count = 0;
      check(zesFrequencyGetAvailableClocks(h_freq, &count, nullptr));
      if (count) {
        std::vector<double> freq_arr(count);
        check(zesFrequencyGetAvailableClocks(h_freq, &count, freq_arr.data()));
        for (unsigned i = 0; i < count; i++) {
          freqs.push_back(static_cast<frequency>(freq_arr[i]));
        }
      }
    }
    return freqs;
  }

  template <zes_freq_domain_t domain>
  inline frequency get_frequency(const lz::device_handle handle) const {
    auto h_freq = get_frequency_handle<ZES_FREQ_DOMAIN_GPU>(handle);
    zes_freq_state_t state{};
    state.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;

    check(zesFrequencyGetState(h_freq, &state));
    return state.actual;
  }
};

}; // namespace detail
}; // namespace synergy
