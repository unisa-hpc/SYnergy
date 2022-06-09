#ifndef _SYNERGY_INTERFACE_H_
#define _SYNERGY_INTERFACE_H_
#include <sycl/sycl.hpp>


namespace synergy{

struct energy_interface
{
	virtual void process(sycl::event& event) = 0;
};

} // namespace synergy

#endif