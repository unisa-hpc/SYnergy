//===----------------------------------------------------------------------===//
// Part of the SYnergy Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2022-2026 University of Salerno
//===----------------------------------------------------------------------===//

#include <iostream>
#include <synergy.hpp>

int gpu_backend_selector(const sycl::device& d) {
  auto vendor_name = d.get_platform().get_info<sycl::info::platform::name>();
  if (vendor_name.find("HIP") != std::string::npos || vendor_name.find("CUDA") != std::string::npos)
    return 100;
  return 0;
}

int main() {
  synergy::queue q{gpu_backend_selector};

  auto device = q.get_synergy_device();
  auto mem_freq = device.supported_uncore_frequencies();
  auto core_freq = device.supported_core_frequencies();

  std::cout << "mem_freq: ";
  for (auto freq : mem_freq)
    std::cout << freq << " ";
  std::cout << std::endl;

  std::cout << "core_freq: ";
  for (auto freq : core_freq)
    std::cout << freq << " ";
  std::cout << std::endl;
}