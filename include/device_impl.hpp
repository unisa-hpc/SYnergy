#pragma once

#include "management_wrapper.hpp"
#include "types.hpp"

namespace synergy {

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

  virtual void init_power_snapshot() = 0;

  virtual void finalize_power_snapshot() = 0;

  virtual power get_snapshot_avarage_power() = 0;
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

  inline virtual void init_power_snapshot() {
    init_snap = library.get_power_snap();
  }

  inline virtual void finalize_power_snapshot() {
    final_snap = library.get_power_snap();
  }

  inline power get_snapshot_avarage_power() {
    return library.get_snapshot_avarage_power(init_snap, final_snap);
  }

private:
  management_wrapper<vendor> library;
  typename vendor::device_handle handle;
  frequency current_core_frequency;
  frequency current_uncore_frequency;
  typename vendor::power_snap_type init_snap;
  typename vencor::power_snap_type final_snap;
};

} // namespace detail

} // namespace synergy
