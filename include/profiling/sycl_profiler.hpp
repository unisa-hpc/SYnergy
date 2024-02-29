#include <chrono>
#include <numeric>
#include <synergy.hpp>
#include <vector>

namespace synergy {
// TODO: add license
#define EVENT_VEC_SIZE 1024
using wall_clock_t = std::chrono::high_resolution_clock;
using time_point_t = std::chrono::time_point<wall_clock_t>;
template <typename T, class Period>
using time_interval_t = std::chrono::duration<T, Period>;

template <typename T>
class Profiler final {
  using event_list = std::vector<sycl::event>;
  using time_point_list = std::vector<time_point_t>;

public:
  Profiler() = default;
  Profiler(synergy::queue& q, event_list& events, time_point_t start) : q(q) {
    profile(q, events, start);
  }

  void profile(synergy::queue& q, event_list& eventList, time_point_t start) {
    const auto end = wall_clock_t::now();

    T realExecutionTime = 0;

    const auto eventCount = eventList.size();
    for (size_t i = 0; i < eventCount; ++i) {
      auto& curEvent = eventList.at(i);
      curEvent.wait();

      const auto cgSubmissionTimePoint = curEvent.get_profiling_info<
          sycl::info::event_profiling::command_submit>();
      const auto startKernExecutionTimePoint =
          curEvent.get_profiling_info<
              sycl::info::event_profiling::command_start>();
      const auto endKernExecutionTimePoint =
          curEvent.get_profiling_info<
              sycl::info::event_profiling::command_end>();

      // Collect the submisson and computation time of each kernel
      m_profData.cgSubmissionTimes.push_back(to_milli(startKernExecutionTimePoint - cgSubmissionTimePoint));
      m_profData.kernExecutionTimes.push_back(to_milli(endKernExecutionTimePoint - startKernExecutionTimePoint));

#ifdef SYNERGY_KERNEL_PROFILING
      m_profData.kernelEnergyConsumptions.push_back(q.kernel_energy_consumption(curEvent));
#endif
    }

    time_interval_t<T, std::milli> curRealExecutionTime = end - start;
    realExecutionTime = curRealExecutionTime.count();

    // set the total energy and time consumed by each kernel
    set_total_command_group_submission_time();
    set_total_kernel_execution_time();
    set_total_kernel_execution_energy();
    set_real_execution_time(realExecutionTime);
#ifdef SYNERGY_DEVICE_PROFILING
    set_device_energy(q.device_energy_consumption());
#endif
  }
  // get times
  inline std::vector<T> get_command_group_submission_times() const {
    return m_profData.cgSubmissionTimes;
  }

  inline T get_total_command_group_submission_times() const {
    return m_profData.totalSubmissionTime;
  }

  inline std::vector<T> get_kernel_execution_times() const {
    return m_profData.kernExecutionTimes;
  }

  inline T get_total_kernel_execution_times() const {
    return m_profData.totalKernelTime;
  }

  inline T get_real_execution_time() const {
    return m_profData.realExecutionTime;
  }

  // get energy
  inline std::vector<T> get_kernel_execution_energies() const {
    return m_profData.kernelEnergyConsumptions;
  }

  inline T get_total_kernel_execution_energies() const {
    return m_profData.totalKernelEnergy;
  }

  inline T get_device_energy() const {
    return m_profData.totalDeviceEnergy;
  }

  // pass the index of the kernel to profile
  inline void print_all_profiling_info(size_t index) {
    std::cout << q.get_synergy_device().get_uncore_frequency() << ", "
              << q.get_synergy_device().get_core_frequency() << ", "
              << get_kernel_execution_times()[index] << ", "
              << get_kernel_execution_energies()[index] << ", "
              << get_real_execution_time() << ", "
              << get_total_kernel_execution_times() << ", "
              << get_device_energy() << ", "
              << get_total_kernel_execution_energies()
              << std::endl;
    return;
  }

private:
  struct profiling_data {
    std::vector<T> cgSubmissionTimes; // command group submission time

    std::vector<T> kernExecutionTimes; // exact computation time on the device of each kernel

    std::vector<T> kernelEnergyConsumptions; // energy consumption of each kernel

    T totalSubmissionTime{0};
    T totalKernelTime{0};

    T totalKernelEnergy{0};

    T realExecutionTime{0}; // wall clock time
    T totalDeviceEnergy{0}; // wall energy consumption
  };

  profiling_data m_profData;
  synergy::queue q;
  inline void set_total_command_group_submission_time() {
    for (double val : m_profData.cgSubmissionTimes) {
      m_profData.totalSubmissionTime += val;
    }
  }

  inline void set_total_kernel_execution_time() {

    for (double val : m_profData.kernExecutionTimes) {
      m_profData.totalKernelTime += val;
    }
  }

  inline void set_total_kernel_execution_energy() {
    for (double val : m_profData.kernelEnergyConsumptions) {
      m_profData.totalKernelEnergy += val;
    }
  }

  inline void set_real_execution_time(T realExecutionTime) {
    m_profData.realExecutionTime = realExecutionTime;
  }

  inline void set_device_energy(T deviceEnergy) {
    m_profData.totalDeviceEnergy = deviceEnergy;
  }

  inline T to_milli(T timeValue) const {
    return timeValue * static_cast<T>(1e-6);
  }
};

// Add energy profiling and expand the output of the profiler for multiple kernels
} // namespace synergy