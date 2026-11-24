#pragma once
#include "../types.hpp"
#include "synergy.hpp"

namespace synergy {
    namespace profiler{
            class PowerProfiler {
                public:
                      PowerProfiler(const std::vector<synergy::device>& devs,
                                    int sampling_rate_ms) : 
                                        devs(devs), 
                                        sampling_rate_ms(sampling_rate_ms),
                                        power_trace_data(devs.size()),
                                        freq_trace_data(devs.size()),
                                        temperature_trace_data(devs.size())
                                        {}  
                    ~PowerProfiler(){

                    };

                    void start(){
                        running = true;
                        worker = std::thread([this]() {
                            size_t timestamp = 0; 
                            // auto next = std::chrono::steady_clock::now();
                            while (running) {
                                // next += std::chrono::milliseconds(sampling_rate_ms);
                                for (int i = 0; i < devs.size(); i++) {
                                    synergy::device& dev = devs[i];

                                    power power_uw = dev.get_power_usage(); // read power in uw
                                    frequency freq = dev.get_core_frequency(false);
                                    temperature_t temp = dev.get_temperature();

                                    std::tuple<timestamp_t, power> power_tuple = std::make_tuple(timestamp, power_uw);
                                    std::tuple<timestamp_t, frequency> freq_tuple = std::make_tuple(timestamp, freq);
                                    std::tuple<timestamp_t, temperature_t> temp_tuple = std::make_tuple(timestamp, temp);

                                    power_trace_data[i].push_back(power_tuple); // Create (timestamp, power) tuple
                                    freq_trace_data[i].push_back(freq_tuple); // Create (timestamp, freq) tuple
                                    temperature_trace_data[i].push_back(temp_tuple); // Create (timestamp, temp) tuple
                                }
                                // std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate_ms));
                                // std::this_thread::sleep_until(next);

                                timestamp += sampling_rate_ms;
                            }
                        });
                    }
                    void stop(){
                        running = false;
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }
                    
                    void clean(){
                        power_trace_data.clear();
                        power_trace_data.resize(devs.size());
                        freq_trace_data.clear();
                        freq_trace_data.resize(devs.size());
                    }
                    // return an std::vector containing the tuple (timestamp, power)
                    std::vector<power_trace_t> get_power_execution_data() const {
                        // std::vector<power_trace_t> parsed_traces;
                        // parsed_traces.reserve(power_trace_data.size());

                        // for (const auto& trace : power_trace_data) {
                        //     power_trace_t parsed;

                        //     if (!trace.empty()) {
                        //         std::unique_copy(
                        //             trace.begin(),
                        //             trace.end(),
                        //             std::back_inserter(parsed),
                        //             [](const auto& a, const auto& b) {
                        //             return std::get<1>(a) == std::get<1>(b); // compare power
                        //             }
                        //         );
                        //     }

                        //     parsed_traces.push_back(std::move(parsed));
                        // }

                        // return parsed_traces;
                        return power_trace_data;
                    }
                    std::vector<freq_trace_t> get_freq_execution_data() const {
                        // std::vector<power_trace_t> parsed_traces;
                        // parsed_traces.reserve(power_trace_data.size());

                        // for (const auto& trace : power_trace_data) {
                        //     power_trace_t parsed;

                        //     if (!trace.empty()) {
                        //         std::unique_copy(
                        //             trace.begin(),
                        //             trace.end(),
                        //             std::back_inserter(parsed),
                        //             [](const auto& a, const auto& b) {
                        //             return std::get<1>(a) == std::get<1>(b); // compare power
                        //             }
                        //         );
                        //     }

                        //     parsed_traces.push_back(std::move(parsed));
                        // }

                        // return parsed_traces;
                        return freq_trace_data;
                    }
                    
                    std::vector<temperature_trace_t> get_temperature_execution_data() const {
                        // std::vector<power_trace_t> parsed_traces;
                        // parsed_traces.reserve(power_trace_data.size());

                        // for (const auto& trace : power_trace_data) {
                        //     power_trace_t parsed;

                        //     if (!trace.empty()) {
                        //         std::unique_copy(
                        //             trace.begin(),
                        //             trace.end(),
                        //             std::back_inserter(parsed),
                        //             [](const auto& a, const auto& b) {
                        //             return std::get<1>(a) == std::get<1>(b); // compare power
                        //             }
                        //         );
                        //     }

                        //     parsed_traces.push_back(std::move(parsed));
                        // }

                        // return parsed_traces;
                        return temperature_trace_data;
                    }
                    
                    
                private:
                    std::vector<synergy::device> devs;
                    int sampling_rate_ms;
                    std::vector<power_trace_t> power_trace_data;
                    std::vector<freq_trace_t> freq_trace_data;
                    std::vector<temperature_trace_t> temperature_trace_data;


                    std::atomic<bool> running{false};
                    std::thread worker;
            };
    } // end namespace profiler
} // namespace synergy