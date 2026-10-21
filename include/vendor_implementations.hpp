//===----------------------------------------------------------------------===//
// Part of the SYnergy Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2022-2026 University of Salerno
//===----------------------------------------------------------------------===//

#pragma once

#ifdef SYNERGY_CUDA_SUPPORT
#include "vendors/nvml_wrapper.hpp"
#endif

#ifdef SYNERGY_ROCM_SUPPORT
#include "vendors/rsmi_wrapper.hpp"
#endif

#ifdef SYNERGY_LZ_SUPPORT
#include "vendors/lz_wrapper.hpp"
#endif