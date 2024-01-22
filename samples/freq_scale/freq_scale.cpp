#include <iostream>
#include <synergy.hpp>

constexpr int N = 1024;
constexpr int VAL = 2;

void print_usage() {
  std::cout << "Usage: ./freq_scale <memory_frequency> <core_frequency>" << std::endl;
}

int main(int argc, char **argv) {

  if (argc < 3) {
    print_usage();
    return 1;
  }

  synergy::frequency mem_freq, core_freq;
  try {
    mem_freq = atoi(argv[1]);
    core_freq = atoi(argv[2]);
  } catch (std::exception &e) {
    print_usage();
    return 1;
  }

  synergy::queue q1 { mem_freq, core_freq, sycl::gpu_selector_v };
  synergy::queue q2 { sycl::gpu_selector_v };

  std::vector<int> a(N, 2);
  std::vector<int> b(N, 3);

  sycl::buffer<int, 1> buf1{a.data(), N};
  sycl::buffer<int, 1> buf2{b.data(), N};

  q1.submit([&](sycl::handler &h) {
    sycl::accessor acc {buf1, h, sycl::read_write};
    h.parallel_for(sycl::range<1>{N}, [=](auto id) {
      acc[id] *= acc[id] * VAL;
    });
  }).wait();

  q2.submit(mem_freq, core_freq, [&](sycl::handler &h) {
    sycl::accessor acc {buf2, h, sycl::read_write};
    h.parallel_for(sycl::range<1>{N}, [=](sycl::item<1> id) {
      acc[id[0]] *= acc[id[0]] * VAL;
    });
  }).wait();
}