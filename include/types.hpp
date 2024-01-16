#pragma once

namespace synergy {
using frequency = unsigned;
using power = unsigned long long;
using energy = double;
using time = double; // microseconds

enum target_metric {
  UNDEFINED, // Undefined
  EDP, // Energy Delay Product
  ED2P, // Energy Delay Squared Product
  ES_25, // Energy Saving 25%
  ES_50, // Energy Saving 50%
  ES_75, // Energy Saving 75%
  ES_100, // Energy Saving 100%
  PL_00, // Performance Loss 0%
  PL_25, // Performance Loss 25%
  PL_50, // Performance Loss 50%
  PL_75, // Performance Loss 75%
};

} // namespace synergy
