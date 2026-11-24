#include <synergy.hpp>


void polling_freq(synergy::queue q, int to_set, int polling_time_us) {
  int clock_mhz = 0;

  do {
    std::this_thread::sleep_for(std::chrono::microseconds(polling_time_us));
    // clock_mhz = q.get_synergy_device().get_core_frequency(false);
    // std::cout<<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
    if (clock_mhz >= to_set - 10 && clock_mhz <= to_set + 10) {
      return ;
    }

  } while (clock_mhz != to_set);

}


int main(int argc, char** argv) {
  int n_kernels = 1;
  if (argc != 6){
    std::cerr << "Usage: " << argv[0] << " <num_runs> <polling_time_us> <n_kernels> <freq1> <freq2>" << std::endl;
    return 0;
  }

  int num_runs = std::stoi(argv[1]);
  int polling_time_us = std::stoi(argv[2]);
  n_kernels = std::stoi(argv[3]);


  synergy::frequency freq1 = std::stoi(argv[4]);;  
  synergy::frequency freq2 = std::stoi(argv[5]);  
  synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  std::vector<double> overhead_times;
  std::vector<double> no_overhead_times;
  std::vector<double> freq_overheads;

  // Warm up run
  std::cout<< "######### Warmp up run #########" <<std::endl;
  for (int it = 0; it < n_kernels; it++) {
    auto cfe = q.submit(0, 0, [&](sycl::handler& cgh){
          cgh.single_task([=](){
            // Do nothing
          });
        }); // Set frequency
    cfe.wait();
  }

  
  for (int r=0 ; r < num_runs ; r++){
    std::cout << "Run " << r <<std::endl;

    double overhead_time {0}, no_overhead_time{0};
    
    std::chrono::high_resolution_clock::time_point overhead_start_time, overhead_end_time;
    overhead_start_time = std::chrono::high_resolution_clock::now();
    
    for (int it = 0; it < n_kernels; it++) {
      auto cfe = q.submit([&](sycl::handler& cgh){
        cgh.single_task([=](){
          // Do nothing
        });
      }); // Set frequency
      cfe.wait();
    }
    overhead_end_time = std::chrono::high_resolution_clock::now();
    no_overhead_time += std::chrono::duration_cast<std::chrono::microseconds>(overhead_end_time - overhead_start_time).count();

    overhead_start_time = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n_kernels; it++) {
      auto to_set = it % 2 ? freq1 : freq2;
      // q.get_synergy_device().set_core_frequency(to_set);
      auto cfe = q.submit(0, to_set, [&](sycl::handler& cgh){
        cgh.single_task([=](){
          // Do nothing
        });
      }); // Set frequency
      cfe.wait();
      polling_freq(q, to_set,  polling_time_us); // wait until the freq is effectively changed
    }
    overhead_end_time = std::chrono::high_resolution_clock::now();
    overhead_time += std::chrono::duration_cast<std::chrono::microseconds>(overhead_end_time - overhead_start_time).count();

    std::cout << "With freq change: " << overhead_time / n_kernels << " us" << std::endl;
    std::cout << "Without freq change: " << no_overhead_time / n_kernels << " us" << std::endl;
    std::cout << "Overhead: " << (overhead_time - no_overhead_time) / n_kernels << " us" << std::endl;
    overhead_times.push_back(overhead_time / n_kernels);
    no_overhead_times.push_back(no_overhead_time  / n_kernels);
    freq_overheads.push_back((overhead_time - no_overhead_time ) / n_kernels);
  }
  
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "######### Freq Overhead Results" << " From " << freq1 << " to "<<  freq2 <<"#########"<< std::endl;
  std::cout << "With freq change: " ;
  for(int i = 0; i < num_runs; i++){
    std::cout << overhead_times[i] << " ";
  }

  std::cout << std::endl;
  std::cout << "Without freq change: ";
  for(int i = 0; i < num_runs; i++){
    std::cout << no_overhead_times[i] << " ";
  }
  std::cout << std::endl;
  std::sort(freq_overheads.begin(), freq_overheads.end());
  std::cout << "Freq change overhead: ";
  for(int i = 0; i < num_runs; i++){
    std::cout << freq_overheads[i] << " ";
  }
  std::cout << std::endl;

  std::cout<< "Median: " << freq_overheads[num_runs/2] << " us" <<std::endl;

}