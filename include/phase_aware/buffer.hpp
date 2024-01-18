#pragma once

#include <sycl/sycl.hpp>

namespace synergy {

/**
 * @brief A buffer that can be used with SYnergy.
 * @attention This buffer sets the write_back to off by default in order to predict the data dependencies.
*/
template<typename DataT, int Dimensions = 1>
class buffer : public sycl::buffer<DataT, Dimensions> {
private:
  uint64_t id;

public:
  buffer(DataT* ptr, size_t size) : sycl::buffer<DataT, Dimensions>(ptr, size), id(reinterpret_cast<uint64_t>(ptr)) {
    this->set_write_back(false);
  }

  uint64_t get_id() const { return id; }
};

} // namespace synergy