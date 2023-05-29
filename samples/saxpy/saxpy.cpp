#include <synergy.hpp>

using namespace sycl;

int main() {
  std::vector<float> x(2048);
  std::vector<float> y(2048);
  std::vector<float> z(2048);
  float alpha{10};

  synergy::queue q{gpu_selector_v};

  buffer<float, 1> x_buf{x.data(), 2048};
  buffer<float, 1> y_buf{y.data(), 2048};
  buffer<float, 1> z_buf{z.data(), 2048};

  event e = q.submit([&](handler& h) {
    accessor<float, 1, access_mode::read> x_acc{x_buf, h};
    accessor<float, 1, access_mode::read> y_acc{y_buf, h};
    accessor<float, 1, access_mode::read_write> z_acc{z_buf, h};
    float a{alpha};

    h.parallel_for(range<1>{2048}, [=](sycl::id<1> id) {
      z_acc[id] = a * x_acc[id] + y_acc[id];
    });
  });

  q.wait();

#ifdef SYNERGY_KERNEL_PROFILING
  std::cout << "Kernel energy consumption: " << q.kernel_energy_consumption(e) << " j\n";
#endif
#ifdef SYNERGY_DEVICE_PROFILING
  std::cout << "Device energy consumption: " << q.device_energy_consumption() << " j\n";
#endif
}