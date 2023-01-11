#ifndef _SYNERGY_SCALING_INTERFACE_H_
#define _SYNERGY_SCALING_INTERFACE_H_

namespace synergy {

enum class frequency {
  min_frequency,
  default_frequency,
  max_frequency
};

class scaling_interface {
public:
  virtual void change_frequency(frequency memory_frequency, frequency core_frequency) = 0;
  virtual void change_frequency(unsigned int memory_frequency, unsigned int core_frequency) = 0;
  virtual ~scaling_interface() = default;
};

} // namespace synergy

#endif