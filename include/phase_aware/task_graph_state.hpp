#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "buffer.hpp"
#include "accessor.hpp"
#include "belated_kernel.hpp"

namespace sycl_ext = sycl::ext::oneapi::experimental;

namespace synergy {
namespace detail {

struct dependency_t {
  uint64_t buffer_id;
  sycl::access::mode mode;
};

class TaskGraphState {
private:
  static TaskGraphState* _instance;
public:
  static TaskGraphState* getInstance() {
    if (_instance == nullptr) {
      _instance = new TaskGraphState();
    }
    return _instance;
  }

protected:
  std::vector<std::vector<dependency_t>> dependencies;
  size_t current_kernel = 0;

public:
  void next_kernel() {
    _instance->dependencies.push_back(std::vector<dependency_t>());
    _instance->current_kernel = dependencies.size() - 1;
  }

  void add_dependency(size_t kernel_id, uint64_t buffer_id, sycl::access::mode mode) {
    _instance->dependencies[kernel_id].push_back({buffer_id, mode});
  }

  void add_dependency(uint64_t buffer_id, sycl::access::mode mode) {
    _instance->dependencies[_instance->current_kernel].push_back({buffer_id, mode});
  }

  void clear_state() {
    _instance->dependencies.clear();
    _instance->current_kernel = 0;
  }

  const std::vector<std::vector<dependency_t>>& get_dependencies() const {
    return _instance->dependencies;
  }
};

synergy::detail::TaskGraphState* synergy::detail::TaskGraphState::_instance = nullptr;

struct node_t {
  size_t id;
  belated_kernel& kernel;
};

struct edge_t {
  node_t& src;
  node_t& dst;
};

class TaskGraphBuilder {
private:
  std::vector<belated_kernel>& kernels;
  std::vector<node_t> nodes;
  std::vector<edge_t> edges;
  bool consistent = false;

protected:
  void compute_dependencies() {
    auto task_graph = detail::TaskGraphState::getInstance();

    sycl::queue q;
    sycl_ext::command_graph<sycl_ext::graph_state::modifiable> cg(q.get_context(), q.get_device(), {sycl_ext::property::graph::assume_buffer_outlives_graph{}});
    cg.begin_recording(q);

    for (size_t i = 0; i < kernels.size(); i++) {
      auto& kernel = kernels[i];
      task_graph->next_kernel();
      nodes.push_back({i, kernel});
      q.submit(kernel.cgh);  
    }

    cg.end_recording(q);
  }

  void compute_graph_structure() {
    auto task_graph = detail::TaskGraphState::getInstance();

    auto dependencies = task_graph->get_dependencies();

  }

public:
  TaskGraphBuilder(std::vector<belated_kernel>& kernels) : kernels(kernels) {}

  inline void build() {
    compute_dependencies();
    compute_graph_structure();
  }

  inline const std::vector<belated_kernel>& get_kernels() const {
    return kernels;
  }

  inline std::vector<node_t> get_nodes() const {
    return nodes;
  }

  inline std::vector<edge_t> get_edges() const {
    return edges;
  }
};

} // namespace detail
} // namespace synergy