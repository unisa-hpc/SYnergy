#ifndef _SYNERGY_INFO_H_
#define _SYNERGY_INFO_H_

namespace synergy {
namespace info {

enum class queue {
  memory_frequencies,
  core_frequencies
};

template <typename T, T Param>
struct param_traits {};

} // namespace info
} // namespace synergy

#endif
