#include <synergy.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <vector>

namespace sycl_ext = sycl::ext::oneapi::experimental;

int main() {
  constexpr size_t N = 1024;

  std::vector<int> a(N, 1);
  std::vector<int> b(N, 3);
  std::vector<int> c(N, 0);

  synergy::queue q {sycl::gpu_selector_v};

  synergy::buffer<int, 1> bufA(a.data(), a.size());
  synergy::buffer<int, 1> bufB(b.data(), b.size());
  synergy::buffer<int, 1> bufC(c.data(), c.size());

  synergy::detail::belated_kernel k1 {[&](sycl::handler& cgh) {
    synergy::accessor<int, 1, sycl::access_mode::read_write> accA(bufA, cgh);
    cgh.parallel_for<class increment_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accA[idx] += 1;
    });
  }};

  synergy::detail::belated_kernel k2 {[&](sycl::handler& cgh) {
    synergy::accessor<int, 1, sycl::access_mode::read_write> accB(bufB, cgh);
    cgh.parallel_for<class subtract_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accB[idx] -= 1;
    });
  }};

  synergy::detail::belated_kernel k3 {[&](sycl::handler& cgh) {
    synergy::accessor<int, 1, sycl::access_mode::read> accA(bufA, cgh);
    synergy::accessor<int, 1, sycl::access_mode::read> accB(bufB, cgh);
    synergy::accessor<int, 1, sycl::access_mode::discard_write> accC(bufC, cgh);

    cgh.parallel_for<class vector_add_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accC[idx] = accA[idx] + accB[idx];
    });
  }};

  std::vector<synergy::detail::belated_kernel> kernels {k1, k2, k3};

  synergy::detail::TaskGraphBuilder builder {kernels};
  builder.build();
  auto edges = builder.get_edges();
  for (auto& edge : edges) {
    std::cout << edge.src.id << " -> " << edge.dst.id << std::endl;
  }

  auto topological_order = builder.get_topological_order();
  std::cout << "Topological order: ";
  for (auto& node : topological_order) {
    std::cout << node.id << " ";
  }
  std::cout << std::endl;

  for (auto& kernel : kernels) {
    q.submit(kernel.cgh);
  }
  q.wait();

  sycl::host_accessor<int, 1, sycl::access_mode::read_write> accC(bufC);
  for (size_t i = 0; i < N; ++i) {
    if (accC[i] != 4) {
      std::cout << "Error: accC[" << i << "] = " << accC[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Success!" << std::endl;
}