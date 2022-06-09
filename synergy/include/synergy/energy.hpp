#ifndef _SYNERGY_ENERGY_H_
#define _SYNERGY_ENERGY_H_

#include <memory>
#include <synergy/energy_interface.hpp>

namespace synergy{

struct energy {
	std::shared_ptr<energy_interface> impl_;

	template<typename T, typename ...Args>
	void create(Args&& ...args) {
		impl_ = std::make_shared<T>(std::forward<Args>(args)...);
	}


	void process(sycl::event& event) {
		impl_->process(event);
	}
};

} // namespace synergy

#endif // _SYNERGY_ENERGY_H_