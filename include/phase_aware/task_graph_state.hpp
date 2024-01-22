#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "buffer.hpp"
#include "accessor.hpp"
#include "belated_kernel.hpp"
namespace synergy {
namespace detail {
struct dependency_t {
  uint64_t buffer_id;
  sycl::access::mode mode;
};
struct node_t {
  size_t id;
  belated_kernel& kernel;
};

struct edge_t {
  node_t& src;
  node_t& dst;
};

struct task_graph_t {
  std::vector<node_t> nodes;
  std::vector<edge_t> edges;
};

class task_graph_builder {
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
    namespace sycl_ext = sycl::ext::oneapi::experimental;

    sycl::queue q;
    sycl_ext::command_graph<sycl_ext::graph_state::modifiable> cg(q.get_context(), q.get_device(), {sycl_ext::property::graph::assume_buffer_outlives_graph{}});
    cg.begin_recording(q);

    for (size_t i = 0; i < kernels.size(); i++) {
      auto& kernel = kernels[i];
      nodes.push_back({i, kernel});
      q.submit(kernel.cgh);
    }

    cg.end_recording(q);
    cg.print_graph("./graph.dot");
  }

  /**
   * @brief Compute the graph structure
   * @details This function computes the graph structure by comparing the dependencies of each kernel
  */
  void compute_graph_structure() {
    std::vector<std::pair<int, int>> unparsed_edges;
    std::vector<int> unparsed_nodes;

    // read from file called graph.dot
    std::ifstream graph_file("./graph.dot");
    for (std::string line; std::getline(graph_file, line);) {
      if (line.find("->") != std::string::npos) {
        std::string src = line.substr(0, line.find("->"));
        std::string dst = line.substr(line.find("->") + 2, line.find(";") - line.find("->") - 2);
        src.erase(std::remove(src.begin(), src.end(), ' '), src.end());
        dst.erase(std::remove(dst.begin(), dst.end(), ' '), dst.end());
        src.erase(std::remove(src.begin(), src.end(), '"'), src.end());
        dst.erase(std::remove(dst.begin(), dst.end(), '"'), dst.end());
        int src_id = std::stoi(src, nullptr, 16);
        int dst_id = std::stoi(dst, nullptr, 16);
        unparsed_edges.push_back({src_id, dst_id});
        unparsed_nodes.push_back(src_id);
        unparsed_nodes.push_back(dst_id);
      }
    }

    std::sort(unparsed_nodes.begin(), unparsed_nodes.end());
    unparsed_nodes.erase(std::unique(unparsed_nodes.begin(), unparsed_nodes.end()), unparsed_nodes.end());

    std::map<int, uint64_t> node_map;
    for (size_t i = 0; i < unparsed_nodes.size(); i++) {
      node_map[unparsed_nodes[i]] = i;
    }

    for (auto& edge : unparsed_edges) {
      edges.push_back({nodes[node_map[edge.first]], nodes[node_map[edge.second]]});
    }
  }

public:
  task_graph_builder(std::vector<belated_kernel>& kernels) : kernels(kernels) {}

  /**
   * @brief Builds the task graph state.
   */
  inline task_graph_t build() {
    if (consistent) {
      return {nodes, edges};
    }
    identify_dependencies();
    compute_graph_structure();
    consistent = true;
    return {nodes, edges};
  }

  inline const std::vector<belated_kernel>& get_kernels() const {
    return kernels;
  }

  inline std::vector<node_t> get_nodes() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::task_graph_builder error: you must call build() before getting the nodes");
    }
    return nodes;
  }

  inline std::vector<edge_t> get_edges() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::task_graph_builder error: you must call build() before getting the edges");
    }
    return edges;
  }

  inline std::vector<node_t> get_topological_order() const {
    if (!consistent) {
      throw std::runtime_error("synergy::detail::task_graph_builder error: you must call build() before getting the topological order");
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