#include <any>
#include <memory>

#include "device.hpp"

namespace synergy {
class runtime {

  static void initialize();
  static runtime& get_instance();
  inline static bool is_initialized() { return instance != nullptr; }

private:
  runtime();
  runtime(const runtime&) = delete;
  runtime(runtime&&) = delete;

  static std::unique_ptr<runtime> instance;
  std::vector<device<std::any>> devices;
};

} // namespace synergy
