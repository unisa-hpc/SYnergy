#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <iostream>
#include <cmath>
#include <geopm/PlatformIO.hpp>
#include <geopm/PlatformTopo.hpp>
#include "../types.hpp"
#include "../management_wrapper.hpp"
#include "../log_msg.hpp"

namespace g = geopm;


namespace synergy {

namespace detail {

namespace management {
  const int NUM_TILES = 2; // We assume two tile in composite mode. TODO: handle more then 2 tiles.
  enum class synergy_signal {
    GET_POWER,
    GET_ENERGY,
    GET_TEMPERATURE,
    GET_CORE_FREQUENCY,
    GET_UNCORE_FREQUENCY,
    SET_MIN_CORE_FREQ,
    SET_MAX_CORE_FREQ

  };

 

  // Map synergy signal to geopm signal according to the GEOPM GPU domain
  inline std::string geopm_signal_name(synergy_signal signal, geopm_domain_e domain){
      // domain = GEOPM_DOMAIN_GPU or GEOPM_DOMAIN_GPU_CHIP
      switch (signal) {
          case synergy_signal::GET_POWER:
              return (domain == GEOPM_DOMAIN_GPU) ? "LEVELZERO::GPU_POWER" : "GPU_CORE_POWER";
          case synergy_signal::GET_ENERGY:
              return (domain == GEOPM_DOMAIN_GPU) ? "LEVELZERO::GPU_ENERGY" : "DRM::HWMON::ENERGY1_INPUT::GPU_CHIP";
          case synergy_signal::GET_TEMPERATURE:
              return (domain == GEOPM_DOMAIN_GPU) ? "LEVELZERO::GPU_CORE_TEMPERATURE_MAXIMUM" : "LEVELZERO::GPU_CORE_TEMPERATURE_MAXIMUM";
          case synergy_signal::GET_CORE_FREQUENCY:
              return (domain == GEOPM_DOMAIN_GPU) ? "DRM::BASE_ACT_FREQ" : "DRM::BASE_ACT_FREQ";
          case synergy_signal::SET_MAX_CORE_FREQ:
              return (domain == GEOPM_DOMAIN_GPU) ? "": "DRM::RPS_MAX_FREQ";
          case synergy_signal::SET_MIN_CORE_FREQ:
              return (domain == GEOPM_DOMAIN_GPU) ? "" : "DRM::RPS_MIN_FREQ";
          default:
              throw std::logic_error("Unknown synergy_signal");
      }
  }
  
  struct geopm_device_handle {
    unsigned int id;
    geopm_domain_e domain;  // GEOPM_DOMAIN_GPU or GEOPM_DOMAIN_GPU_CHIP
  };
  
  struct geopm {
    static constexpr std::string_view name = "GEOPM";
    static constexpr unsigned int max_frequencies = 256;
    static constexpr unsigned int sampling_rate = 5; // ms
    using device_identifier = unsigned int;
    using device_handle = geopm_device_handle;
    using return_type = unsigned int;
    static constexpr return_type return_success = 0;
  };

  /********* Geopm read and geopm write *********/
  double geopm_read(synergy_signal synergy_signal, geopm_domain_e domain, geopm::device_identifier id){
    return g::platform_io().read_signal(
      geopm_signal_name(synergy_signal, domain), 
      static_cast<int>(domain), 
      id);
  }
 
  void geopm_write(synergy_signal synergy_signal, geopm_domain_e domain, geopm::device_identifier id, double val){
    g::platform_io().write_control(
          geopm_signal_name(synergy_signal, domain), 
          domain, 
          id, 
          val);
  }
  /******************************** */

} // namespace management
 


template <>
struct vendor_traits<management::geopm> {
  // GEOPM requires domain selection (GPU vs GPU_CHIP)
  static constexpr bool needs_gpu_domain = true;
};

template <>
class management_wrapper<management::geopm> {

public:
  inline unsigned int get_devices_count() const {
    return g::platform_topo().num_domain(GEOPM_DOMAIN_GPU_CHIP);
  }

  inline void initialize() const {
  }

  inline void shutdown() const { }

  using geopm = management::geopm;

  inline geopm::device_handle get_device_handle(geopm::device_identifier id, synergy::gpu_domain domain) const {
    std::string debug_msg = "Vendor id: " + std::to_string(id) + " domain: " + std::to_string(static_cast<int>(domain));
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, debug_msg);
    return geopm::device_handle{id, map_domain(domain)};
  }

  //TODO: check if there is a more accurate command in geopm for power
  inline synergy::power get_power_usage(geopm::device_handle handle) const {
    std::string debug_msg = "get_power_usage -> Vendor id: " + std::to_string(handle.id) + " domain: " + std::to_string(static_cast<int>(handle.domain));
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, debug_msg);
    
    double power_val = management::geopm_read(
        management::synergy_signal::GET_POWER, 
        handle.domain, 
        handle.id);
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, "Power read: " + std::to_string(power_val));
    return static_cast<synergy::power>(power_val * 1e6); // from W to uW
  }

  inline energy get_energy_usage(geopm::device_handle handle) const {
    unsigned int energy = management::geopm_read(management::synergy_signal::GET_ENERGY,
      handle.domain, 
      handle.id
    );
    return energy * 1e6; // from J to uJ
  }

  inline std::vector<frequency> get_supported_core_frequencies(geopm::device_handle handle) const {
    auto min = g::platform_io().read_signal("GPU_CORE_FREQUENCY_MIN_AVAIL", handle.domain, handle.id);
    auto max = g::platform_io().read_signal("GPU_CORE_FREQUENCY_MAX_AVAIL", handle.domain, handle.id);
    auto step = g::platform_io().read_signal("GPU_CORE_FREQUENCY_STEP", handle.domain, handle.id);

    std::vector<frequency> frequencies;
    for (auto i = min; i <= max; i += step) {
      frequency freq = std::round(i * 1e-6);
      frequencies.push_back(freq);
    }
    return frequencies;
  }

  inline std::vector<frequency> get_supported_uncore_frequencies(geopm::device_handle handle) const {
    return {}; // TODO: we need this, but it is not supported by GEOPM
  }

  inline frequency get_core_frequency(geopm::device_handle handle) const {
    std::string debug_msg = "get_core_frequency -> Vendor id: " + std::to_string(handle.id) + " domain: " + std::to_string(static_cast<int>(handle.domain));
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, debug_msg);

    /* 
      In FLAT mode SYnergy return the core frequnecy of the selected tile. Differently in COMPOSITE mode we can have two scenarios:
       1. The device used by the syenergy::queue is a subdevice of the root. In that case SYnergy automatically set the GEOPM domain to FLAT,
          so that each subdevice is managed has a single independent device.
       2. The device use byt the synergy::queue is a root device composed of two tiles. 
          In that case the GEOPM domain is set to composite but we have to query the core frquency of both tile: for now we return a single value that 
          is the mean of the core freq.
          TODO: return an std::vector that contains the frequnecy of each tile.
    */
    if ( handle.domain == GEOPM_DOMAIN_GPU_CHIP){
      frequency freq = management::geopm_read(
        management::synergy_signal::GET_CORE_FREQUENCY,
        handle.domain, 
        handle.id) * 1e-6;
        
        synergy::log::synergy_log(synergy::log::LogLevel::Debug, "Core frequency: " + std::to_string(freq) + " MHz");      
        return freq;
    }
    else{
        frequency freq_0 = management::geopm_read(
        management::synergy_signal::GET_CORE_FREQUENCY, 
        GEOPM_DOMAIN_GPU_CHIP, // Domain here must be GPU_CHIP since each tile can have a diffrent frequency 
        handle.id * management::NUM_TILES) * 1e-6;
        
        frequency freq_1 = management::geopm_read(
          management::synergy_signal::GET_CORE_FREQUENCY, 
          GEOPM_DOMAIN_GPU_CHIP, // Domain here must be GPU_CHIP since each tile can have a diffrent frequency 
          (handle.id* management::NUM_TILES)+1) * 1e-6;
        
      return (freq_0 + freq_1) / 2;
    }
  }

  inline frequency get_uncore_frequency(geopm::device_handle handle) const {
    return 0; // TODO: we need this, but it is not supported by GEOPM
  }

  inline temperature_t get_temperature(geopm::device_handle handle) const{
    temperature_t temperature=0;
    if(handle.domain == GEOPM_DOMAIN_GPU){
      temperature +=  management::geopm_read(
        management::synergy_signal::GET_TEMPERATURE, 
        GEOPM_DOMAIN_GPU_CHIP, // Domain here must be GPU_CHIP since each tile can have a diffrent frequency 
        handle.id * management::NUM_TILES);
        temperature +=  g::platform_io().read_signal(
        management::geopm_signal_name(management::synergy_signal::GET_TEMPERATURE, handle.domain), 
        GEOPM_DOMAIN_GPU_CHIP, // Domain here must be GPU_CHIP since each tile can have a diffrent frequency 
        handle.id * management::NUM_TILES + 1);  
    }else{
        temperature =  management::geopm_read(
          management::synergy_signal::GET_TEMPERATURE, 
          handle.domain, // handle.domain is GPU_CHIP 
          handle.id);
      }
      return temperature;
  }
  
  inline void set_core_frequency(geopm::device_handle handle, frequency target) const {
    std::string debug_msg = "set_core_frequency -> Vendor id: " + std::to_string(handle.id) + " domain: " + std::to_string(static_cast<int>(handle.domain));
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, debug_msg);
    // MIN_CORE_FREQ should always be less equal then MAX_CORE_FREQ. This check is to avoid wrong min max core frequency settings.
    frequency curr_freq = get_core_frequency(handle);
    synergy::log::synergy_log(synergy::log::LogLevel::Debug, "Current freq: " + std::to_string(curr_freq) + " MHz");
    /* 
        In SYnergy with Intel COMPOSITE mode we can have two scenarion:
        1. The device provided to the synergy queue is the root device composes of two tiles. In that case SYnergy will set the GEOPM domain to composite.
           And when the set_core frequency function is called SYnergy set the frequency of both tile. Root device i will have tile with index i*2, i*2+1.
        2. The device provided to the SYnergy queue is a subdevice of the root. In that case we set the GEOPM domain to flat. So that we can mange each
           subdevice independently.
    */
    
    if (target >= curr_freq){ // handle min/max freq control
      if(handle.domain == geopm_domain_e::GEOPM_DOMAIN_GPU_CHIP){ // handle FLAT mode and Composite mode with subdeiveces
        management::geopm_write(
          management::synergy_signal::SET_MAX_CORE_FREQ,
          handle.domain,
          handle.id, 
          target * 1e6);
        management::geopm_write(
          management::synergy_signal::SET_MIN_CORE_FREQ, 
          handle.domain, 
          handle.id, 
          target * 1e6);
      }
      else{ // handle composite mode with synergy::queue built with a root device
        /***************** First tile freq change **********/
        management::geopm_write(
          management::synergy_signal::SET_MAX_CORE_FREQ, 
          GEOPM_DOMAIN_GPU_CHIP, // here we need GPU_CHIP domain since we have to change frequency for both tile 
          handle.id * management::NUM_TILES, 
          target * 1e6);

        management::geopm_write(
            management::synergy_signal::SET_MIN_CORE_FREQ,
            GEOPM_DOMAIN_GPU_CHIP,
            handle.id * management::NUM_TILES, 
            target * 1e6);
        /******************************************************/

        /***************** Second tile freq change **********/
        management::geopm_write(
          management::synergy_signal::SET_MAX_CORE_FREQ, 
          GEOPM_DOMAIN_GPU_CHIP, // here we need GPU_CHIP domain since we have to change frequency for both tile 
          handle.id * management::NUM_TILES + 1, 
          target * 1e6);

        management::geopm_write(
            management::synergy_signal::SET_MIN_CORE_FREQ,
            GEOPM_DOMAIN_GPU_CHIP,
            handle.id * management::NUM_TILES + 1, 
            target * 1e6);
            /******************************************************/
      }
    }
    else{  // handl min/max freq control
      if(handle.domain == geopm_domain_e::GEOPM_DOMAIN_GPU_CHIP){ // handle FLAT mode and Composite mode with subdeiveces

        management::geopm_write(
          management::synergy_signal::SET_MIN_CORE_FREQ, 
          handle.domain, 
          handle.id, 
          target * 1e6);

        management::geopm_write(
          management::synergy_signal::SET_MAX_CORE_FREQ, 
          handle.domain, 
          handle.id, 
          target * 1e6);
    
        }
        else{ // handle COMPOSITE mode with root device
          /********** First tile freq change ***********/
          management::geopm_write(
            management::synergy_signal::SET_MIN_CORE_FREQ, 
            handle.domain, 
            handle.id * management::NUM_TILES, 
            target * 1e6);

          management::geopm_write(
            management::synergy_signal::SET_MAX_CORE_FREQ, 
            handle.domain, 
            handle.id * management::NUM_TILES, 
            target * 1e6);
          /**********************************************/
          /********** Second tile freq change ***********/
          management::geopm_write(
            management::synergy_signal::SET_MIN_CORE_FREQ, 
            handle.domain, 
            handle.id * management::NUM_TILES + 1, 
            target * 1e6);

          management::geopm_write(
            management::synergy_signal::SET_MAX_CORE_FREQ, 
            handle.domain, 
            handle.id * management::NUM_TILES + 1, 
            target * 1e6);
            /**********************************************/
        }
      }

    return ;
  }

  inline void set_uncore_frequency(geopm::device_handle handle, frequency target) const {
    throw std::runtime_error{"synergy " + std::string(geopm::name) + " wrapper error: set_uncore_frequency is not supported"};
  }

  inline void set_all_frequencies(geopm::device_handle handle, frequency core, frequency uncore) const {
    set_core_frequency(handle, core);
    std::cerr << "synergy " << geopm::name << " wrapper warning: set_all_frequencies does not support uncore frequency" << std::endl;
  }

  inline void setup_profiling(geopm::device_handle) const {}

  inline void setup_scaling(geopm::device_handle handle) const {}

  inline std::string error_string(geopm::return_type return_value) const {
    return std::string{""};
  }

private:
  error_checker<management::geopm> check{*this};
  inline geopm_domain_e map_domain(synergy::gpu_domain d) const {
    switch (d) {
      case synergy::gpu_domain::composite:
        return GEOPM_DOMAIN_GPU;
      case synergy::gpu_domain::flat:
        return GEOPM_DOMAIN_GPU_CHIP;
    }
    throw std::logic_error("invalid gpu_domain");
  }
};

} // namespace detail

} // namespace synergy
