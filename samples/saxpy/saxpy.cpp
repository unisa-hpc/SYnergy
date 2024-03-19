#include <synergy.hpp>

using namespace sycl;

#define SIZE 64

int main() {
  std::vector<float> x(SIZE);
  std::vector<float> y(SIZE);
  std::vector<float> z(SIZE);
  float alpha{10};

  synergy::queue q{gpu_selector_v};

  buffer<float, 1> x_buf{x.data(), SIZE};
  buffer<float, 1> y_buf{y.data(), SIZE};
  buffer<float, 1> z_buf{z.data(), SIZE};

  event e = q.submit([&](handler& h) {
    accessor<float, 1, access_mode::read> x_acc{x_buf, h};
    accessor<float, 1, access_mode::read> y_acc{y_buf, h};
    accessor<float, 1, access_mode::read_write> z_acc{z_buf, h};
    float a{alpha};

    h.parallel_for(range<1>{SIZE}, [=](sycl::id<1> id) {
      z_acc[id] = a * x_acc[id] + y_acc[id];
    });
  });

  q.wait();
  const auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

  std::cout << "Time: " << (end - start) / 1e3 << " us" << std::endl;

#ifdef SYNERGY_KERNEL_PROFILING
  std::cout << "Kernel energy consumption: " << q.kernel_energy_consumption(e) << " j\n";
#endif
#ifdef SYNERGY_DEVICE_PROFILING
  std::cout << "Device energy consumption: " << q.device_energy_consumption() << " j\n";
#ifdef SYNERGY_HOST_DEVICE
  std::cout << "Host energy consumption: " << q.host_energy_consumption() << " j\n";
#endif
#endif
}