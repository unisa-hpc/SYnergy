#include <iostream>
#include <vector>

#include <synergy.hpp>

using namespace sycl;

using value_type = float;

int main() {
  // Create a queue with a default device
  synergy::queue q(gpu_selector_v);

  // Create some buffers
  constexpr size_t n = 2048;
  std::vector<value_type> a(n * n);
  std::vector<value_type> b(n * n);
  std::vector<value_type> c(n * n);

  std::fill(a.begin(), a.end(), 1.0);
  std::fill(b.begin(), b.end(), 1.0);

  buffer<value_type, 2> a_buf(a.data(), {n, n});
  buffer<value_type, 2> b_buf(b.data(), {n, n});
  buffer<value_type, 2> c_buf(c.data(), {n, n});

  // Launch the computation
  sycl::event e = q.submit([&](sycl::handler& h) {
    accessor<value_type, 2, access_mode::read> a_acc{a_buf, h};
    accessor<value_type, 2, access_mode::read> b_acc{b_buf, h};
    accessor<value_type, 2, access_mode::read_write> c_acc{c_buf, h};

    range<2> grid{n, n};
    range<2> block{n < 32 ? n : 32, n < 32 ? n : 32};

    h.parallel_for<class mat_mul>(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> idx) {
      int i = idx.get_global_id(0);
      int j = idx.get_global_id(1);

      c_acc[i][j] = 0.0f;
      for (int i = 0; i < 100; i++)
        for (size_t k = 0; k < n; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
    });
  });

  q.wait();

#ifdef SYNERGY_KERNEL_PROFILING
  std::cout << "Kernel energy consumption: " << q.kernel_energy_consumption(e) << " j\n";
#endif
#ifdef SYNERGY_DEVICE_PROFILING
  std::cout << "Device energy consumption: " << q.device_energy_consumption() << " j\n";
#endif

  host_accessor<value_type, 2> h_acc{c_buf};
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      assert(h_acc[i][j] == n);
}