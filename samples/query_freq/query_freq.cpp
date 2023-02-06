#include <iostream>
#include <synergy.hpp>

int main()
{
  synergy::queue q{sycl::gpu_selector_v};
  auto mem_freq = q.query_supported_memory_frequencies();
  auto core_freq = q.query_supported_core_frequencies();

  std::cout << "mem_freq: ";
  for (auto freq : mem_freq)
    std::cout << freq << " ";
  std::cout << std::endl;

  std::cout << "core_freq: ";
  for (auto freq : core_freq)
    std::cout << freq << " ";
  std::cout << std::endl;
}