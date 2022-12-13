#ifndef _SYNERGY_QUEUE_H_
#define _SYNERGY_QUEUE_H_

#include <sycl/sycl.hpp>
#include <functional>
#include <utility>
#include <chrono>

#include <synergy/utils.hpp>
#include <synergy/energy.hpp>
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
			: base(std::forward<Args>(args)..., sycl::property::queue::enable_profiling{}), energy_wrapper_() 
		{
			#ifdef SYNERGY_CUDA_SUPPORT
				energy_wrapper_.create<energy_nvidia>();
			#else
				throw std::runtime_error("No energy implementation available");
			#endif
		}

		template<typename ...Args,
		std::enable_if_t<::details::is_present_v<sycl::property_list,Args...> || ::details::has_property_v<Args...>,bool> = true>
		queue(Args&&... args) 
			: base(std::forward<Args>(args)...), energy_wrapper_() 
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
			#ifdef SYNERGY_CUDA_SUPPORT
				energy_wrapper_.create<energy_nvidia>();
			#else
				throw std::runtime_error("No energy implementation available");
			#endif
		}


		template <typename ...Args>
		sycl::event submit(Args&& ...args)
		{
			auto&& event = sycl::queue::submit(std::forward<Args>(args)...);
			energy_wrapper_.process(event);
			return event;
		}

		private:
			energy energy_wrapper_;

	};

}

#endif // _SYNERGY_QUEUE_H_