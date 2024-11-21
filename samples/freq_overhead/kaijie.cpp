#include <synergy.hpp>

int main(int argc, char** argv) {
  int n_kernels = 256;
  synergy::frequency freq1 = 1000;  
  synergy::frequency freq2 = 600;  
  synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  double overhead_time {0}, no_overhead_time{0};

  for (int it = 0; it < n_kernels; it++) {
    std::chrono::high_resolution_clock::time_point overhead_start_time, overhead_end_time;
    
    overhead_start_time = std::chrono::high_resolution_clock::now();
    auto cfe = q.submit(0, 0, [&](sycl::handler& cgh){
      cgh.single_task([=](){
        // Do nothing
      });
    }); // Set frequency
    cfe.wait();
    overhead_end_time = std::chrono::high_resolution_clock::now();
    no_overhead_time += std::chrono::duration_cast<std::chrono::microseconds>(overhead_end_time - overhead_start_time).count();
  }

  for (int it = 0; it < n_kernels; it++) {
    std::chrono::high_resolution_clock::time_point overhead_start_time, overhead_end_time;
    
    overhead_start_time = std::chrono::high_resolution_clock::now();
    auto to_set = it % 2 ? freq1 : freq2;
    std::cout << to_set << std::endl;
    // q.get_synergy_device().set_core_frequency(to_set);
    auto cfe = q.submit(0, to_set, [&](sycl::handler& cgh){
      cgh.single_task([=](){
        // Do nothing
      });
    }); // Set frequency
    cfe.wait();
    overhead_end_time = std::chrono::high_resolution_clock::now();
    overhead_time += std::chrono::duration_cast<std::chrono::microseconds>(overhead_end_time - overhead_start_time).count();
  }

  std::cout << "With freq change: " << overhead_time << " us" << std::endl;
  std::cout << "Without freq change: " << no_overhead_time << " us" << std::endl;
  std::cout << "Overhead: " << overhead_time - no_overhead_time << " us" << std::endl;
}