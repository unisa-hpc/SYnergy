#include <iostream>
#include <synergy.hpp>

constexpr int N = 4096;
constexpr int VAL = 2;

void print_usage() {
  std::cout << "Usage: ./freq_scale <freq1> <freq2>" << std::endl;
}

template<typename value_type>
sycl::event matmul(synergy::queue& q, std::vector<value_type>& a, std::vector<value_type>& b, std::vector<value_type>& c, size_t n, synergy::frequency freq) {
  sycl::buffer<value_type, 2> a_buf{a.data(), sycl::range{N, N}};
  sycl::buffer<value_type, 2> b_buf{b.data(), sycl::range{N, N}};
  sycl::buffer<value_type, 2> c_buf{c.data(), sycl::range{N, N}};
  return q.submit(0, freq, [&](sycl::handler& h) {
    sycl::accessor a_acc{a_buf, h, sycl::read_only};
    sycl::accessor b_acc{b_buf, h, sycl::read_only};
    sycl::accessor c_acc{c_buf, h, sycl::read_write};

    sycl::range<2> grid{n, n};
    sycl::range<2> block{n < 32 ? n : 32, n < 32 ? n : 32};

    h.parallel_for<class mat_mul>(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> idx) {
      int i = idx.get_global_id(0);
      int j = idx.get_global_id(1);

      c_acc[i][j] = 0.0f;
      for (size_t k = 0; k < n; k++) {
        c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
      }
    });
  });
}

void checkConsume(synergy::queue& q, sycl::event& e) {
  auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Retrived Frequency: " << q.get_synergy_device().get_core_frequency() << " MHz\n";
  std::cout << "Execution time: " << (end - start) * 1e-9 << " s\n";
  #ifdef SYNERGY_KERNEL_PROFILING
    std::cout << "Kernel energy consumption: " << q.kernel_energy_consumption(e) << " j\n";
  #endif
  #ifdef SYNERGY_DEVICE_PROFILING
    std::cout << "Device energy consumption: " << q.device_energy_consumption() << " j\n";
  #ifdef SYNERGY_HOST_PROFILING
    std::cout << "Host energy consumption: " << q.host_energy_consumption() << " j\n";
  #endif
  #endif
}

int main(int argc, char **argv) {

  if (argc < 3) {
    print_usage();
    return 1;
  }

  synergy::frequency freq1, freq2;
  try {
    freq1 = atoi(argv[1]);
    freq2 = atoi(argv[2]);
  } catch (std::exception &e) {
    print_usage();
    return 1;
  }

  synergy::queue q { sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}} };

  std::vector<int> a(N * N, 1);
  std::vector<int> b(N * N, 1);
  std::vector<int> c(N * N, 0);

  auto e1 = matmul(q, a, b, c, N, freq1);
  e1.wait();
  std::cout << "Frequency: " << freq1 << " MHz\n";
  checkConsume(q, e1);

  q = synergy::queue { sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}} };
  auto e2 = matmul(q, a, b, c, N, freq2);
  e2.wait();
  std::cout << "Frequency: " << freq2 << " MHz\n";
  checkConsume(q, e2);
}