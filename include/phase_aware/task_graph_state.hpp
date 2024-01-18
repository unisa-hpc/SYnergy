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
  /**
   * Identifies the dependencies between tasks in the task graph.
   */
  void identify_dependencies() {
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

  /**
   * @brief Compute the graph structure
   * @details This function computes the graph structure by comparing the dependencies of each kernel
   * @todo This function doesn't consider the access mode of the accessors; it can be improved by considering the access mode
  */
  void compute_graph_structure() {
    auto task_graph = detail::TaskGraphState::getInstance();

    auto dependencies = task_graph->get_dependencies();
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto& kernel = dependencies[i];
      for (auto& dependency : kernel) {
        for (size_t j = i + 1; j < dependencies.size(); j++) {
          auto& other_kernel = dependencies[j];
          for (auto& other_dependency : other_kernel) {
            if (dependency.buffer_id == other_dependency.buffer_id) {
              edges.push_back({nodes[i], nodes[j]});
            }
          }
        }
      }
    }
  }

public:
  TaskGraphBuilder(std::vector<belated_kernel>& kernels) : kernels(kernels) {}

  /**
   * @brief Builds the task graph state.
   */
  inline void build() {
    if (consistent) {
      return;
    }
    identify_dependencies();
    compute_graph_structure();
    consistent = true;
  }

  inline const std::vector<belated_kernel>& get_kernels() const {
    return kernels;
  }

  inline std::vector<node_t> get_nodes() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::TaskGraphBuilder error: you must call build() before getting the nodes");
    }
    return nodes;
  }

  inline std::vector<edge_t> get_edges() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::TaskGraphBuilder error: you must call build() before getting the edges");
    }
    return edges;
  }

  inline std::vector<node_t> get_topological_order() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::TaskGraphBuilder error: you must call build() before getting the topological order");
    }
    std::vector<node_t> topological_order;
    std::vector<bool> visited(nodes.size(), false);
    std::function<void(size_t)> dfs = [&](size_t node_id) {
      visited[node_id] = true;
      for (auto& edge : edges) {
        if (edge.src.id == node_id && !visited[edge.dst.id]) {
          dfs(edge.dst.id);
        }
      }
      topological_order.push_back(nodes[node_id]);
    };
    for (size_t i = 0; i < nodes.size(); i++) {
      if (!visited[i]) {
        dfs(i);
      }
    }
    std::vector<node_t> ret;
    for (int i = topological_order.size() - 1; i >= 0; i--) {
      ret.push_back(topological_order[i]);
    }
    return ret;
  }
};

} // namespace detail
} // namespace synergy