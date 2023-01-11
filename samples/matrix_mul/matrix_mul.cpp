#include <iostream>
#include <synergy/synergy.hpp>

namespace sy = synergy;

using value_type = double;

void mat_mul(sy::queue &q, size_t n, value_type *a, value_type *b, value_type *c)
{
  sycl::buffer<value_type, 2> a_buf(a, {n, n});
  sycl::buffer<value_type, 2> b_buf(b, {n, n});
  sycl::buffer<value_type, 2> c_buf(c, {n, n});

  q.submit(
    [&](sycl::handler &h) {
      sycl::accessor a_acc{a_buf, h};
      sycl::accessor b_acc{b_buf, h};
      sycl::accessor c_acc{c_buf, h};

      sycl::range<2> grid{n, n};
      sycl::range<2> block{n < 32 ? n : 32, n < 32 ? n : 32};

      h.parallel_for<class mat_mul>(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);

        c_acc[i][j] = 0.0f;
        for (int k = 0; k < n; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }

        c_acc[i][j] = 0.0f;
        for (int k = 0; k < n; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }

        c_acc[i][j] = 0.0f;
        for (int k = 0; k < n; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    }
  )
   .wait_and_throw();
}

int main()
{
  // Create a queue with a default device
  sy::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}); // Enable queue profiling by default

  // Create some buffers
  int n = 4096;
  value_type *a = new value_type[n * n];
  value_type *b = new value_type[n * n];
  value_type *c = new value_type[n * n];

  // Initialize the matrices
  for (int i = 0; i < n * n; i++) {
    a[i] = 1.0f;
    b[i] = 1.0f;
  }

  // Launch the computation
  mat_mul(q, n, a, b, c);

  // Check
  for (int i = 0; i < n * n; i++) {
    if (c[i] != n) {
      std::cerr << "Error: c[" << i << "] = " << c[i] << " != " << n << std::endl;
      return 1;
    }
  }

  // Cleanup
  delete[] a;
  delete[] b;
  delete[] c;
}