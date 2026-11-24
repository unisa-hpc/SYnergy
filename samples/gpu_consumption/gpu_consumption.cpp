#include <synergy.hpp>

void function() {
  
}

int main(int argc, char** argv) {

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <milliseconds> <frequency>" << std::endl;
    return 1;
  }
  auto millis = std::stoi(argv[1]);
  auto freq = std::stoi(argv[2]);

  synergy::queue q {sycl::gpu_selector_v};
  q.get_synergy_device().set_core_frequency(freq);

  auto e1 = q.device_energy_consumption();

  std::this_thread::sleep_for(std::chrono::milliseconds(millis));

  auto e2 = q.device_energy_consumption();

  std::cout << e2 - e1 << " J" << std::endl;

}