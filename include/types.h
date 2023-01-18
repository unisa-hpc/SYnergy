#ifndef SYNERGY_TYPES_H
#define SYNERGY_TYPES_H

namespace synergy {
using frequency = unsigned long long;

enum class frequency_preset {
  min_frequency,
  default_frequency,
  max_frequency
};

} // namespace synergy

#endif