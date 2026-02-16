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
    //TODO: this code is wrong. Add GEOPM DOMAIN type to handle different GPU domain case
    auto search_root = r.root_devices.find(sycl_device);
    if (search_root == r.root_devices.end()) {
      synergy::log::synergy_log(synergy::log::LogLevel::Debug, "synergy: sycl device not found as root device");
      auto search_subdevice = r.sub_devices.find(sycl_device);
      if (search_subdevice == r.sub_devices.end()) {
        throw std::runtime_error("error while assigning synergy::device to queue: sycl::device not supported");
      }
      return search_subdevice->second;
    }
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, "synergy: sycl device found as root device");

    return search_root->second;
  }

  runtime(runtime const&) = delete;
  runtime(runtime&&) = delete;
  runtime& operator=(runtime const&) = delete;
  runtime& operator=(runtime&&) = delete;

private:
  std::unordered_map<sycl::device, synergy::device> root_devices;
  std::unordered_map<sycl::device, synergy::device> sub_devices;


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

#ifdef SYNERGY_GEOPM_SUPPORT
      // TODO: Geopm backend consider the FLAT mode where each tile is a different GPUs
      /* 
         When executing in COMPOSITE mode there are two different error scenario:
         
         1. we create a synergy device for each sycl device composed of two or more tile. 
            However in the geopm_wrapper interface we are always using the
            GPU_CHIP_DOMAIN that consider each tile has a single device. 
            In that case synergy will change the frequency or profile energy of the wrong gpu.
            For example in a scenarion like this: COMPOSITE mode with 6 GPUs (0 to 5) each with two tiles.
            if I select the GPU with index 1 synergy will change the frequency of tile 1 that is the second tile of the GPU 0.

         2. If th user in COMPOSITE mode generate sub devices and try to build a synergy queue with the subdevice synergy throw a
            runtime error. 
         Solution: we can wrap the geopm command into a generic command that is specilized at runtime according to the type of sycl::device 
                   used to generate the synergy::queue.
                   We can have a firt loop that iterate on the root device and a second loop that iterate over the subdevice.
                   if the device is found as root device the command of the SYnergy queue can be specilized for the COMPOSITE DOMAIN
                   while if the device uses to build the synergy queue match with a subdevice the GEOPM command is specialized as GPU_DOMAIN_CHIP
      */
      auto devs = platforms[i].get_devices(info::device_type::gpu);

      // Loop over root devices
      for (size_t j = 0, k=0; j < devs.size(); j++) {
        synergy::log::synergy_log(synergy::log::LogLevel::Debug, "synergy: root device "+ std::to_string(j));
        // Check for subdevice
        auto tiles = devs[j].create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                      sycl::info::partition_affinity_domain::next_partitionable);
        bool is_subdevice = !tiles.empty();
        if(!is_subdevice){
          auto ptr = std::make_shared<vendor_device<management::geopm>>(j, synergy::gpu_domain::flat);
          root_devices.insert({devs[j], synergy::device{ptr}});   
        }
        else{
          auto ptr = std::make_shared<vendor_device<management::geopm>>(j, synergy::gpu_domain::composite);
          root_devices.insert({devs[j], synergy::device{ptr}});
        }

        for(auto tile : tiles){
          auto ptr = std::make_shared<vendor_device<management::geopm>>(k, synergy::gpu_domain::flat);
          sub_devices.insert({tile, synergy::device{ptr}});   
          k++;
        }
      }
#else

#ifdef SYNERGY_CUDA_SUPPORT
      if (platform_name.find("cuda") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);

        for (size_t j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::nvml>>(j);
          root_devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif

#ifdef SYNERGY_ROCM_SUPPORT
      if (platform_name.find("hip") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);

        for (size_t j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::rsmi>>(count_hip); // passing count_hip is not an error: compile with SYNERGY_PROOF
          count_hip++;                                                             // there is one platform for each AMD HIP GPU
          root_devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif

#ifdef SYNERGY_LZ_SUPPORT
      if (platform_name.find("level-zero") != std::string::npos ||
          platform_name.find("level zero") != std::string::npos) {
        auto devs = platforms[i].get_devices(info::device_type::gpu);
        for (size_t j = 0; j < devs.size(); j++) {
          auto ptr = std::make_shared<vendor_device<management::lz>>(j);
          root_devices.insert({devs[j], synergy::device{ptr}});
        }
      }
#endif
#endif
    }
  }
};
} // namespace detail

} // namespace synergy
