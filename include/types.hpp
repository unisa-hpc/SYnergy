#pragma once

namespace synergy {
using frequency = unsigned;
using power = unsigned long long;
using energy = double;
using timestamp_t = unsigned long long;
using temperature_t = unsigned long long;
using power_trace_t = std::vector<std::tuple<synergy::timestamp_t, synergy::power>>;
using freq_trace_t = std::vector<std::tuple<synergy::timestamp_t, synergy::frequency>>;
using temperature_trace_t = std::vector<std::tuple<synergy::timestamp_t, synergy::temperature_t>>;

/* 
    In composite mode the device can be split into subdevices, while in flat mode all the device 
    are considered as root device that can not be splitted           
*/
enum class gpu_domain { composite, flat }; 

} // namespace synergy
