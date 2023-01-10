#ifndef _SYNERGY_QUEUE_H_
#define _SYNERGY_QUEUE_H_

#include <sycl/sycl.hpp>
#include <functional>
#include <utility>
#include <chrono>

#include <synergy/utils.hpp>
#include <synergy/energy_implementations.hpp>

namespace synergy
{

	class queue : public sycl::queue
	{
	public:
		using base = sycl::queue;
		
		template<typename ...Args,
		// std::enable_if_t<((!std::is_same_v<sycl::property_list,Args> && !sycl::is_property_v<Args> ) && ...),bool> = true>
		std::enable_if_t<!::details::is_present_v<sycl::property_list,Args...> && !::details::has_property_v<Args...>,bool> = true>
		queue(Args&&... args) 
			: base(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{}) 
		{
			auto vendor = get_device().get_info<sycl::info::device::vendor>();
			if (vendor.find("nvidia"))
			{
#ifdef SYNERGY_CUDA_SUPPORT
				m_energy = std::make_unique<energy_nvidia>();
#else
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
#endif
			}
			else if (vendor.find("amd"))
			{
#ifdef SYNERGY_ROCM_SUPPORT
				m_energy = std::make_unique<energy_amd>();
#else
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
#endif
			}
			else
			{
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
			}
		}

		template<typename ...Args,
		std::enable_if_t<::details::is_present_v<sycl::property_list,Args...> || ::details::has_property_v<Args...>,bool> = true>
		queue(Args&&... args) 
			: base(std::forward<Args>(args)...) 
		{
			auto&& args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
			if constexpr (::details::is_present_v<sycl::property_list,Args...>) {
				sycl::property_list&& prop = std::get<::details::Index<sycl::property_list,Args...>::value>(args_tuple);
				if(!prop.has_property<sycl::property::queue::enable_profiling>()){
					throw std::runtime_error("synergy::queue: enable_profiling property is required");
				}
			}
			else{
				if constexpr (!::details::is_present_v<sycl::property::queue::enable_profiling,Args...>) {
					throw std::runtime_error("synergy::queue: enable_profiling property is required");
				}
			}

			std::string vendor = get_device().get_info<sycl::info::device::vendor>();
			if (vendor.find("nvidia"))
			{
#ifdef SYNERGY_CUDA_SUPPORT
				m_energy = std::make_unique<energy_nvidia>();
#else
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
#endif
			}
			else if (vendor.find("amd"))
			{
#ifdef SYNERGY_ROCM_SUPPORT
				m_energy = std::make_unique<energy_amd>();
#else
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
#endif
			}
			else
			{
				throw std::runtime_error("synergy::queue: vendor \"" + vendor +  "\" not supported");
			}
		}


		template <typename ...Args>
		sycl::event submit(Args&& ...args)
		{
			auto&& event = sycl::queue::submit(std::forward<Args>(args)...);
			m_energy->process(event);
			return event;
		}

		private:
			std::unique_ptr<energy_interface> m_energy;
	};

}

#endif // _SYNERGY_QUEUE_H_