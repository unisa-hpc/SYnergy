#include <fstream>
#include <istream>
#include <unistd.h>

namespace synergy {
namespace detail {
  
  void do_root() {
    setreuid(0, 0);
  }

  void undo_root() {
    setreuid(geteuid(), getuid());
  }

  double get_host_energy() {
    unsigned long long e;

    do_root();
    std::ifstream file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", std::ios::in);
    if (!file.is_open()) {
      throw std::runtime_error("synergy::device error: could not open energy file");
    }

    file >> e;
    undo_root();

    return static_cast<double>(e);
  }
} // namespace detail
} // namespace synergy
