//===----------------------------------------------------------------------===//
// Part of the SYnergy Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2022-2026 University of Salerno
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

namespace synergy {

namespace detail {
struct kernel {
  kernel(sycl::event event) : event{event} {}

  sycl::event event;
  double energy = 0.0;
};

inline bool operator==(const kernel& lhs, const kernel& rhs) { return lhs.event == rhs.event; }

} // namespace detail

} // namespace synergy
