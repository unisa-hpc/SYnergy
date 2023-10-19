#include <synergy.hpp>

int main() {

    synergy::queue q{sycl::gpu_selector_v};

    auto core_freqs = q.get_synergy_device().supported_core_frequencies();
    auto uncore_freqs = q.get_synergy_device().supported_uncore_frequencies();

    auto core = q.get_synergy_device().get_core_frequency();
    auto uncore = q.get_synergy_device().get_uncore_frequency();
    std::cout << "Current core frequency: " << core << " MHz" << std::endl;
    std::cout << "Current uncore frequency: " << uncore << " MHz" << std::endl;

    std::cout << "Supported core frequencies:\n";
    for (auto freq : core_freqs) {
        std::cout << freq << " ";
    }
    std::cout << std::endl;

    std::cout << "Supported uncore frequencies:\n";
    for (auto freq : uncore_freqs) {
        std::cout << freq << " ";
    }
    std::cout << std::endl;

    q.get_synergy_device().set_core_frequency(core_freqs[1]);
    q.submit([&](auto& handler) {
        q.parallel_for(sycl::range{1024}, [&](auto&) {
            for (int i = 0; i < 100000000; i++) {
                volatile int a = 1;
            }
        });
    }).wait();

    std::cout << "Current core frequency: " << q.get_synergy_device().get_core_frequency() << " MHz" << std::endl;
}