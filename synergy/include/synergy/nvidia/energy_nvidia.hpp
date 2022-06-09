#pragma once

#include <functional>
#include <utility>
#include <chrono>
#include <atomic>
#include <fstream>
#include <future>
#include <thread>

#include <synergy/energy_interface.hpp>
#ifdef CUDA_SUPPORT
#include <nvml.h>
#include <synergy/nvidia/utils.hpp>
#endif

namespace synergy
{

	class energy_nvidia : public energy_interface
	{
	public:

		energy_nvidia()
		{
			details::check_nvml_error(nvmlInit());
			details::check_nvml_error(nvmlDeviceGetHandleByIndex(0, &device_handle));
			energy_func = [this](sycl::event e){
				nvmlReturn_t nvml_result;
				unsigned int power;
				double energy = 0.0;
				int i = 0;
				
				//Wait until start
				e.get_profiling_info<sycl::info::event_profiling::command_start>();

				while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
				{
					nvml_result = nvmlDeviceGetPowerUsage(device_handle, &power);
					if (nvml_result != NVML_SUCCESS)
					{
						std::cerr << "NVML power usage failed" << std::endl;
						exit(1);
					}
					energy += power * intervals_length / 1000.0; // Get the integral of the power usage over the interval
					i++;
					std::this_thread::sleep_for(std::chrono::milliseconds(intervals_length));
				}
				std::cout << "Energy: " << energy << std::endl; //should be added to a log file
			};

		}

		~energy_nvidia()
		{
			nvmlShutdown();
		}

		void process(sycl::event& e){
			auto&& res = std::async(std::launch::async, energy_func, e);
		}


	private:
		nvmlDevice_t device_handle;
		std::function<void(sycl::event)> energy_func;
		static constexpr int intervals = 100000;
		static constexpr int intervals_length = 15; // ms
};



}