#include <iostream>
#include <synergy.hpp>
#include <vector>

using namespace sycl;

#define N 1000000
int multi_queue(synergy::queue& q, const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> s1(N), s2(N), s3(N);
  int iter = 1;
  sycl::buffer a_buf(a.data(), sycl::range<1>{N});
  sycl::buffer b_buf(b.data(), sycl::range<1>{N});
  sycl::buffer sum_buf1(s1.data(), sycl::range<1>{N});
  sycl::buffer sum_buf2(s2.data(), sycl::range<1>{N});
  sycl::buffer sum_buf3(s3.data(), sycl::range<1>{N});

  size_t num_groups = 1;
  size_t wg_size = 256;
  auto start = std::chrono::steady_clock::now();

  std::vector<sycl::event> events;

  for (int i = 0; i < iter; i++) {
    event e1 = q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf1, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), [=](sycl::nd_item<1> index) {
        size_t loc_id = index.get_local_id();
        sum_acc[loc_id] = 0;
        for (int j = 0; j < 1000; j++)
          for (size_t i = loc_id; i < N; i += wg_size) {
            sum_acc[loc_id] += a_acc[i] + b_acc[i];
          }
      });
    });

    event e2 = q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf2, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), [=](sycl::nd_item<1> index) {
        size_t loc_id = index.get_local_id();
        sum_acc[loc_id] = 0;
        for (int j = 0; j < 1000; j++)
          for (size_t i = loc_id; i < N; i += wg_size) {
            sum_acc[loc_id] += a_acc[i] + b_acc[i];
          }
      });
    });

    event e3 = q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf3, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), [=](sycl::nd_item<1> index) {
        size_t loc_id = index.get_local_id();
        sum_acc[loc_id] = 0;
        for (int j = 0; j < 1000; j++)
          for (size_t i = loc_id; i < N; i += wg_size) {
            sum_acc[loc_id] += a_acc[i] + b_acc[i];
          }
      });
    });

    events.push_back(e1);
    events.push_back(e2);
    events.push_back(e3);
  }

  q.wait();

  auto end = std::chrono::steady_clock::now();
  std::cout << "multi_queue completed on device - took "
            << (end - start).count() / 1e6 << " u-secs" << std::endl;

#ifdef SYNERGY_KERNEL_PROFILING
  for (auto& e : events) {
    std::cout << std::endl;

    uint64_t start = e.get_profiling_info<info::event_profiling::command_start>();
    uint64_t end = e.get_profiling_info<info::event_profiling::command_end>();

    std::cout << "Energy consumption: " << q.kernel_energy_consumption(e) << " j\n";
    std::cout << "Runtime: " << (end - start) / 1e9 << " s\nStart: " << start / 1e9 << " - End: " << end / 1e9 << std::endl;
  }
#endif

  return ((end - start).count());
}

int main() {
  std::vector<int> a;
  std::vector<int> b;
  a.resize(N);
  b.resize(N);
  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 1);

  synergy::queue q1(sycl::gpu_selector_v);
  synergy::queue q2(sycl::gpu_selector_v);

  multi_queue(q1, a, b);
  multi_queue(q2, a, b);

#ifdef SYNERGY_DEVICE_PROFILING
  std::cout << "Device (q1) Energy consumption: " << q1.device_energy_consumption() << " j\n";
  std::cout << "Device (q2) Energy consumption: " << q2.device_energy_consumption() << " j\n";
#endif
}