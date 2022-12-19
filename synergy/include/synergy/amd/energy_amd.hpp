#ifndef _SYNERGY_ENERGY_AMD_H_
#define _SYNERGY_ENERGY_AMD_H_

#include <functional>
#include <utility>
#include <chrono>
#include <atomic>
#include <fstream>
#include <future>
#include <thread>

#include <synergy/energy_interface.hpp>
#include <rocm_smi/rocm_smi.h>
#include <synergy/amd/utils.hpp>

namespace synergy
{

	class energy_amd : public energy_interface
	{
	public:

		energy_amd()
		{
			details::check_rsmi_error(rsmi_init(0));
			energy_func = [this](sycl::event e){
				rsmi_status_t rsmi_result;
				uint64_t power;
				double energy = 0.0;
				int i = 0;

				//Wait until start

#ifdef __HIPSYCL__
				e.get_profiling_info<sycl::info::event_profiling::command_start>(); // not working on DPC++
#else 
				while (e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::submitted)
					;
#endif

				while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
				{
					rsmi_result = rsmi_dev_power_ave_get(device_handle, 0, &power);
					if (rsmi_result != RSMI_STATUS_SUCCESS)
					{
						std::cerr << "ROCm SMI power usage failed" << std::endl;
						exit(1);
					}
					energy += power * intervals_length / 1000000.0; // Get the integral of the power usage over the interval
					i++;
					std::this_thread::sleep_for(std::chrono::milliseconds(intervals_length));
				}
				std::cout << "Energy: " << energy << " j" << std::endl; //should be added to a log file
			};

		}

		~energy_amd()
		{
			rsmi_shut_down();
		}

		void process(sycl::event& e)
		{
			auto&& res = std::async(std::launch::async, energy_func, e);
		}


	private:
		uint32_t device_handle = 0;
		std::function<void(sycl::event)> energy_func;
		static constexpr int intervals = 100000;
		static constexpr int intervals_length = 15; // ms
	};
	
}

#endif