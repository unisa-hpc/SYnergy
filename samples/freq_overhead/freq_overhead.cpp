#include <synergy.hpp>
#include <vector>
#include <numeric>

constexpr size_t N = 1024;
constexpr size_t NUM_RUNS = 10;
constexpr size_t NUM_ITERS = 15;

void matrix_mul(synergy::queue& q, std::vector<int>& a, std::vector<int>& b, std::vector<int>& c, synergy::frequency freq = 0) {
  sycl::buffer<int, 2> a_buf(a.data(), sycl::range<2>{N, N});
  sycl::buffer<int, 2> b_buf(b.data(), sycl::range<2>{N, N});
  sycl::buffer<int, 2> c_buf(c.data(), sycl::range<2>{N, N});

  for (int _ = 0; _ < NUM_ITERS; _++) {
    sycl::event e = q.submit(0, freq, [&](sycl::handler& h) {
      sycl::accessor a_acc{a_buf, h, sycl::read_only};
      sycl::accessor b_acc{b_buf, h, sycl::read_only};
      sycl::accessor c_acc{c_buf, h, sycl::read_write};

      sycl::range<2> grid{N, N};
      sycl::range<2> block{N < 32 ? N : 32, N < 32 ? N : 32};

      h.parallel_for<class mat_mul>(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);

        c_acc[i][j] = 0.0f;
        for (size_t k = 0; k < N; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    });
  }
}

template<typename T>
void print_metrics(std::vector<T> values, std::string label, std::string unit = "") {

  T avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  // stdev
  T accum = 0.0;
  std::for_each(std::begin(values), std::end(values), [&](const T d) {
    accum += (d - avg) * (d - avg);
  });
  T stdev = sqrt(accum / (values.size() - 1));
  T max = *std::max_element(values.begin(), values.end());
  T min = *std::min_element(values.begin(), values.end());
  T median = values[values.size() / 2];

  std::cout << label << "[" << unit << "]: [ ";
  for (auto val : values) {
    std::cout << val << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << label << "-avg[" << unit << "]: " << avg << std::endl;
  std::cout << label << "-stdev[" << unit << "]: " << stdev << std::endl;
  std::cout << label << "-max[" << unit << "]: " << max << std::endl;
  std::cout << label << "-min[" << unit << "]: " << min << std::endl;
  std::cout << label << "-median[" << unit << "]: " << median << std::endl;
}

int main(int argc, char** argv) {
  synergy::frequency freq = 0;
  if (argc == 2) {
    freq = std::stoi(argv[1]);
  }

  std::vector<int> a(N * N, 1);
  std::vector<int> b(N * N, 1);
  std::vector<int> c(N * N, 0);

  synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};

  std::vector<double> device_times;
  std::vector<synergy::energy> device_consumptions;
  std::vector<synergy::energy> host_consumptions;
  for (int i = 0; i < NUM_RUNS + 1; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    matrix_mul(q, a, b, c, freq);
    q.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    device_times.push_back(duration);
    
    auto device_consumption = q.device_energy_consumption();
    device_consumptions.push_back(device_consumption);
    auto host_consumption = q.host_energy_consumption();
    host_consumptions.push_back(host_consumption);
  }
  device_times.erase(device_times.begin());
  device_consumptions.erase(device_consumptions.begin());
  host_consumptions.erase(host_consumptions.begin());

  print_metrics(device_times, "device-time", "ms");
  print_metrics(device_consumptions, "device-energy", "j");
  print_metrics(host_consumptions, "host-energy", "j");
}