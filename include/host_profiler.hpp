/**
 * @file host_profiler.hpp
 * @brief Host profiler implementation
 * @details Implementation of the host profiler functions. The profiler uses the Powercap interface
 * to get the energy consumption of the host. To access the profiler file, root privileges are
 * required. The host profiler is only available on Linux.
*/

#include <fstream>
#include <istream>
#include <vector>
#include <filesystem>
#include <unistd.h>

namespace synergy {
namespace host_profiler {
namespace detail {

  constexpr auto POWERCAP_ROOT_DIR = "/sys/class/powercap/";
  constexpr auto POWERCAP_ENERGY_FILE = "energy_uj";
  constexpr auto POWERCAP_UNCORE_NAME = "dram";
  constexpr auto POWERCAP_CORE_NAME = "core";
  constexpr auto POWERCAP_PACKAGE_NAME = "package";
  
  inline void do_root() { setreuid(0, 0); }

  inline void undo_root() { setreuid(geteuid(), getuid()); }

  /**
   * @brief Get the names of the host's packages
   * @param base_path The base path of the Powercap interface
   * @return A vector containing the names
  */
  std::vector<std::string> get_packages(std::string base_path = POWERCAP_ROOT_DIR) {
    std::vector<std::string> packages;
    std::filesystem::path path(base_path);

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      if (entry.is_directory()) {
        std::string name = entry.path().filename().string();
        if (name.find(":") == name.length() - 2) {
          packages.push_back(name);
        }
      }
    }

    return packages;
  }

  template <typename... Args>
  std::string build_path(Args... args) {
    std::string path;
    ((path += args) + ...); // fold expression to concatenate the strings
    return path;
  }
} // namespace detail
  using namespace detail;
  
  /**
   * @brief Get the energy consumption of the host in microjoules
   * @details Get the energy consumption of the host in microjoules. The function uses the Powercap
   * interface to get the energy consumption of the host.
   * @return A monotonically increasing value representing the energy consumption of the host in
   * microjoules
   * @throws std::runtime_error if the energy file(s) cannot be opened
  */
  double get_host_energy() {
    double energy = 0;
    unsigned long long e;

    do_root(); // we need root privileges to access the energy files

    // if it's a multi-cpu architecture, we want to sum the energy of all the cpus
    for (const auto& p : get_packages()) {
      std::string path = build_path(POWERCAP_ROOT_DIR, "/", p, "/", POWERCAP_ENERGY_FILE);
      std::ifstream file {path, std::ios::in};
      if (!file.is_open()) {
        throw std::runtime_error("synergy::device error: could not open energy file");
      }
      file >> e;
      energy += static_cast<double>(e);
    }

    undo_root(); // reset privileges

    return energy;
  }

} // namespace host_profiler
} // namespace synergy
