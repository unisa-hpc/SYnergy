#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <string>
#include <map>
#include "buffer.hpp"
#include "task_graph_state.hpp"

namespace synergy {

template <typename DataT, int Dimensions = 1,
          sycl::access_mode AccessMode =
            (std::is_const_v<DataT> ? sycl::access_mode::read
                                    : sycl::access_mode::read_write),
          sycl::target AccessTarget = sycl::target::device,
          sycl::access::placeholder isPlaceholder = sycl::access::placeholder::false_t>
class accessor : public sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, isPlaceholder>{
private:
  void updateTaskGraph(uint64_t buffer_id) {
    auto task_graph = detail::TaskGraphState::getInstance();
    task_graph->add_dependency(buffer_id, AccessMode);
  }

public:
  accessor(synergy::buffer<DataT, Dimensions> buffer, sycl::handler &cgh) : sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, isPlaceholder>(buffer, cgh){
    updateTaskGraph(buffer.get_id());
  }
};

} // namespace synergy
