#include <synergy.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <vector>

namespace sycl_ext = sycl::ext::oneapi::experimental;

int main() {
  constexpr size_t N = 1024;

  std::vector<int> a(N, 1);
  std::vector<int> b(N, 3);
  std::vector<int> c(N, 0);

  synergy::phase_aware::queue q {sycl::gpu_selector_v};

  sycl::buffer<int, 1> bufA(a.data(), a.size());
  bufA.set_write_back(false);
  sycl::buffer<int, 1> bufB(b.data(), b.size());
  bufB.set_write_back(false);
  sycl::buffer<int, 1> bufC(c.data(), c.size());
  bufC.set_write_back(false);

  q.add([&](sycl::handler& cgh) {
    sycl::accessor<int, 1, sycl::access_mode::read_write> accA(bufA, cgh);
    cgh.parallel_for<class increment_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accA[idx] += 1;
    });
  });

  q.add([&](sycl::handler& cgh) {
    sycl::accessor<int, 1, sycl::access_mode::read_write> accB(bufB, cgh);
    cgh.parallel_for<class subtract_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accB[idx] -= 1;
    });
  });

  q.add([&](sycl::handler& cgh) {
    sycl::accessor<int, 1, sycl::access_mode::read> accA(bufA, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read> accB(bufB, cgh);
    sycl::accessor<int, 1, sycl::access_mode::discard_write> accC(bufC, cgh);

    cgh.parallel_for<class vector_add_kernel>(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      accC[idx] = accA[idx] + accB[idx];
    });
  });

  q.phases_selection(synergy::target_metric::UNDEFINED, "task_graph.dot");

  sycl::host_accessor<int, 1, sycl::access_mode::read_write> accC(bufC);
  for (size_t i = 0; i < N; ++i) {
    if (accC[i] != 4) {
      std::cout << "Error: accC[" << i << "] = " << accC[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Success!" << std::endl;
}