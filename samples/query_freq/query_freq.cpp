#include <iostream>
#include <synergy.hpp>

int main()
{
  synergy::queue q{sycl::gpu_selector_v};
  auto device = q.get_synergy_device();
  auto mem_freq = device->supported_uncore_frequencies();
  auto core_freq = device->supported_core_frequencies();

  std::cout << "mem_freq: ";
  for (auto freq : mem_freq)
    std::cout << freq << " ";
  std::cout << std::endl;

  std::cout << "core_freq: ";
  for (auto freq : core_freq)
    std::cout << freq << " ";
  std::cout << std::endl;
}