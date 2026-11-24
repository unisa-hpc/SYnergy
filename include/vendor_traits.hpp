#pragma once

namespace synergy::detail {

    template <typename vendor>
    struct vendor_traits {
        static constexpr bool needs_gpu_domain = false;
    };

} // namespace synergy::detail
