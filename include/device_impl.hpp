#pragma once

#include "management_wrapper.hpp"
#include "types.hpp"

#include <unordered_map>
#include <optional>
#include <exception>

namespace synergy {

using snap_id = unsigned;

namespace detail {

class device_impl {

public:
  virtual ~device_impl() = default;

  virtual std::vector<frequency> supported_core_frequencies() = 0;

  virtual std::vector<frequency> supported_uncore_frequencies() = 0;

  virtual frequency get_core_frequency(bool cached = true) = 0;

  virtual frequency get_uncore_frequency(bool cached = true) = 0;

  virtual void set_core_frequency(frequency target) = 0;

  virtual void set_uncore_frequency(frequency target) = 0;

  virtual void set_all_frequencies(frequency core, frequency uncore) = 0;

  virtual power get_power_usage() = 0;

  virtual unsigned get_power_sampling_rate() = 0;

  volatile virtual synergy::snap_id init_power_snapshot() = 0;

  virtual void begin_power_snapshot(synergy::snap_id id) = 0;
  
  virtual void end_power_snapshot(synergy::snap_id id) = 0;

  virtual power get_snapshot_avarage_power(unsigned id) = 0;
};

template <typename vendor>
class vendor_device : public device_impl {

public:
  inline vendor_device(typename vendor::device_identifier id) {
    library.initialize();
    handle = library.get_device_handle(id);

    current_core_frequency = library.get_core_frequency(handle);
    current_uncore_frequency = library.get_uncore_frequency(handle);
  }

  inline virtual ~vendor_device() { library.shutdown(); }

  inline virtual std::vector<frequency> supported_core_frequencies() { return library.get_supported_core_frequencies(handle); }

  inline virtual std::vector<frequency> supported_uncore_frequencies() { return library.get_supported_uncore_frequencies(handle); }

  inline virtual frequency get_core_frequency(bool cached = true) { return cached ? current_core_frequency : library.get_core_frequency(handle); }

  inline virtual frequency get_uncore_frequency(bool cached = true) { return cached ? current_uncore_frequency : library.get_uncore_frequency(handle); }

  inline virtual void set_core_frequency(frequency target) {
    library.set_core_frequency(handle, target);
    current_core_frequency = target;
    current_uncore_frequency = library.get_uncore_frequency(handle);
  }

  inline virtual void set_uncore_frequency(frequency target) {
    library.set_uncore_frequency(handle, target);
    current_uncore_frequency = target;
    current_core_frequency = library.get_core_frequency(handle);
  }

  inline virtual void set_all_frequencies(frequency core, frequency uncore) {
    library.set_all_frequencies(handle, core, uncore);
    current_core_frequency = core;
    current_uncore_frequency = uncore;
  }

  inline virtual power get_power_usage() {
    return library.get_power_usage(handle);
  }

  inline virtual unsigned get_power_sampling_rate() {
    return vendor::sampling_rate;
  }

  volatile virtual snap_id init_power_snapshot() {
    std::optional<typename vendor::power_snap_type> first;
    std::optional<typename vendor::power_snap_type> second;
    std::pair<std::optional<typename vendor::power_snap_type>, 
      std::optional<typename vendor::power_snap_type>> pair {first, second};

    snap_id id = snaps.size();
    snaps.push_back(pair);
    return id;
  }

  inline virtual void begin_power_snapshot(synergy::snap_id id) {
    typename vendor::power_snap_type snap = library.get_power_snap(handle);

    if (id < snaps.size()) {
      snaps[id].first.emplace(snap);
    } else {
      throw std::runtime_error("Element not found");
    }
  }

  inline virtual void end_power_snapshot(synergy::snap_id id) {
    typename vendor::power_snap_type snap = library.get_power_snap(handle);

    if (id < snaps.size()) {
      snaps[id].second.emplace(snap);
    } else {
      throw std::runtime_error("Element not found");
    }
  }

  inline power get_snapshot_avarage_power(synergy::snap_id id) {
    if (id < snaps.size()) {
      auto first = snaps[id].first;
      auto second = snaps[id].second;
      
      if (first.has_value() && second.has_value()) {
        return library.get_snapshot_avarage_power(first.value(), second.value());
      }
    }
    throw std::runtime_error{"Element not found"};
  }

private:
  management_wrapper<vendor> library;
  typename vendor::device_handle handle;
  frequency current_core_frequency;
  frequency current_uncore_frequency;
  std::vector<std::pair<std::optional<typename vendor::power_snap_type>, std::optional<typename vendor::power_snap_type>>> snaps;
};

} // namespace detail

} // namespace synergy
