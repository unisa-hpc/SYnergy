#pragma once 
#include "../synergy.hpp"
#include "../log_msg.hpp"


namespace synergy {
    namespace utils {
        void check_core_freq(synergy::device device, frequency target_freq, size_t polling_time_us){
            int clock_mhz = 0;

            do {
                device.set_core_frequency(target_freq);
                std::this_thread::sleep_for(std::chrono::microseconds(polling_time_us));
                clock_mhz = device.get_core_frequency(false);
                // std::cout<<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
                if (clock_mhz >= target_freq - 20 && clock_mhz <= target_freq + 20) {
                return ;
                }
               std::ostringstream clock_freq_msg;
                clock_freq_msg << "Current freq: " << clock_mhz
                            << " / Target freq: " << target_freq << '\n';

                synergy::log::synergy_log(
                    synergy::log::LogLevel::Debug,
                    clock_freq_msg.str());
            } while (clock_mhz != target_freq);
            std::ostringstream target_freq_msg;
            target_freq_msg << "Current freq: " << clock_mhz
                            << " / Target freq: " << target_freq << '\n';

            synergy::log::synergy_log(synergy::log::LogLevel::Info,
                    target_freq_msg.str());
        }
    }
}