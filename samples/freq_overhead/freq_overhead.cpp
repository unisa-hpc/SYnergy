#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include "bitmap.hpp"

constexpr size_t N = 1024;
constexpr size_t NUM_RUNS = 10;
constexpr size_t NUM_ITERS = 15;
constexpr size_t SAMPLING_TIME = 2000; // milliseconds

class mat_mul_kernel;
class sobel_kernel;

enum class FreqChangePolicy {
  APP,
  PHASE,
  KERNEL
};

/**
 * @brief Sample the energy consumption of the host
 * @param sampling_time The time in milliseconds to sample the energy consumption
 * @return The energy consumption in joules
*/
synergy::energy sample_energy_consumption(unsigned int sampling_time) {
  auto start = synergy::host_profiler::get_host_energy();
  std::this_thread::sleep_for(std::chrono::milliseconds(sampling_time));
  auto end = synergy::host_profiler::get_host_energy();
  return ((end - start) / 1000000);
}


double matrix_mul(synergy::queue& q, 
              sycl::buffer<int, 2>& a_buf,
              sycl::buffer<int, 2>& b_buf,
              sycl::buffer<int, 2>& c_buf,
              synergy::frequency freq,
              FreqChangePolicy policy,
              size_t num_iters) {
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> duration {0};
  for (int it = 0; it < num_iters; it++) {
    if (it == 0 || policy == FreqChangePolicy::KERNEL) {
      start = std::chrono::high_resolution_clock::now();
      auto e = q.submit(0, freq, [&](sycl::handler& cgh) {});
      e.wait();
      end = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    sycl::event e = q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc{a_buf, h, sycl::read_only};
      sycl::accessor b_acc{b_buf, h, sycl::read_only};
      sycl::accessor c_acc{c_buf, h, sycl::read_write};

      sycl::range<2> grid{N, N};
      sycl::range<2> block{N < 32 ? N : 32, N < 32 ? N : 32};

      h.parallel_for<mat_mul_kernel>(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);

        c_acc[i][j] = 0.0f;
        for (size_t k = 0; k < N; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    });
  }
  return duration.count();
}

double sobel(synergy::queue& q, 
                  sycl::buffer<sycl::float4, 2>& input_buf, 
                  sycl::buffer<sycl::float4, 2>& output_buf, 
                  size_t size,
                  synergy::frequency freq,
                  FreqChangePolicy policy,
                  size_t num_iters) {
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double> duration {0};
  for (int it = 0; it < num_iters; it++) {
    if (policy == FreqChangePolicy::KERNEL || (it == 0 && policy == FreqChangePolicy::PHASE)) {
      start = std::chrono::high_resolution_clock::now();
      auto e = q.submit(0, freq, [&](sycl::handler& cgh) {});
      e.wait();
      end = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    sycl::event e = q.submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<sycl::access::mode::read>(cgh);
      auto out = output_buf.get_access<sycl::access::mode::discard_write>(cgh);
      sycl::range<2> ndrange{size, size};

      // Sobel kernel 3x3
      const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

      cgh.parallel_for<sobel_kernel>(
          ndrange, [in, out, kernel, size_ = size](sycl::id<2> gid) {
        int x = gid[0];
        int y = gid[1];

          sycl::float4 Gx = sycl::float4(0, 0, 0, 0);
          sycl::float4 Gy = sycl::float4(0, 0, 0, 0);
          const int radius = 3;

          // constant-size loops in [0,1,2]
          for(int x_shift = 0; x_shift < 3; x_shift++) {
            for(int y_shift = 0; y_shift < 3; y_shift++) {
              // sample position
              uint xs = x + x_shift - 1; // [x-1,x,x+1]
              uint ys = y + y_shift - 1; // [y-1,y,y+1]
              // for the same pixel, convolution is always 0
              if(x == xs && y == ys)
                continue;
              // boundary check
              if(xs < 0 || xs >= size_ || ys < 0 || ys >= size_)
                continue;

              // sample color
              sycl::float4 sample = in[{xs, ys}];

              // convolution calculation
              int offset_x = x_shift + y_shift * radius;
              int offset_y = y_shift + x_shift * radius;

              float conv_x = kernel[offset_x];
              sycl::float4 conv4_x = sycl::float4(conv_x);
              Gx += conv4_x * sample;

              float conv_y = kernel[offset_y];
              sycl::float4 conv4_y = sycl::float4(conv_y);
              Gy += conv4_y * sample;
            }
          }
          // taking root of sums of squares of Gx and Gy
          sycl::float4 color = hypot(Gx, Gy);
          sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
          sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
          out[gid] = clamp(color, minval, maxval);
      });
    });
  }
  return duration.count();
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
  synergy::frequency freq_matmul = 0;
  synergy::frequency freq_sobel = 0;
  FreqChangePolicy policy;
  size_t num_iters;
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <policy> <num-iters> <freq_matmul> <freq_sobel>" << std::endl;
    exit(1);
  }

  policy = (std::string(argv[1]) == "app") ? FreqChangePolicy::APP : 
           (std::string(argv[1]) == "phase") ? FreqChangePolicy::PHASE : 
           (std::string(argv[1]) == "kernel") ? FreqChangePolicy::KERNEL : 
           throw std::runtime_error("Invalid policy");
  num_iters = std::stoi(argv[2]);
  freq_matmul = std::stoi(argv[3]);
  freq_sobel = std::stoi(argv[4]);

  std::vector<int> a(N * N, 1);
  std::vector<int> b(N * N, 1);
  std::vector<int> c(N * N, 0);

  sycl::buffer<int, 2> a_buf(a.data(), sycl::range<2>{N, N});
  sycl::buffer<int, 2> b_buf(b.data(), sycl::range<2>{N, N});
  sycl::buffer<int, 2> c_buf(c.data(), sycl::range<2>{N, N});

  std::vector<sycl::float4> input;
  input.resize(N * N);
  load_bitmap_mirrored("Brommy.bmp", N, input);
  std::vector<sycl::float4> output;
  output.resize(N * N);

  sycl::buffer<sycl::float4, 2> input_buf(input.data(), sycl::range<2>{N, N});
  sycl::buffer<sycl::float4, 2> output_buf(output.data(), sycl::range<2>{N, N});

  auto starting_energy = sample_energy_consumption(SAMPLING_TIME);

  synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};

  std::vector<double> device_times;
  std::vector<synergy::energy> device_consumptions;
  std::vector<synergy::energy> host_consumptions;
  std::vector<double> freq_change_overheads;

  for (int i = 0; i < NUM_RUNS + 1; i++) {
    double freq_change_overhead = 0;
    auto start = std::chrono::high_resolution_clock::now();
    freq_change_overhead += matrix_mul(q, a_buf, b_buf, c_buf, freq_matmul, policy, num_iters);
    freq_change_overhead += sobel(q, input_buf, output_buf, N, freq_sobel, policy, num_iters);
    q.wait_and_throw();
    freq_change_overheads.push_back(freq_change_overhead);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    device_times.push_back(duration);
    
    auto device_consumption = q.device_energy_consumption();
    device_consumptions.push_back(device_consumption);
    auto host_consumption = q.host_energy_consumption();
    host_consumptions.push_back(host_consumption);
  }

  auto ending_energy = sample_energy_consumption(SAMPLING_TIME);

  device_times.erase(device_times.begin());
  device_consumptions.erase(device_consumptions.begin());
  host_consumptions.erase(host_consumptions.begin());

  std::cout << "energy-sample-before[J]: " << (starting_energy) << std::endl;
  std::cout << "energy-sample-after[J]: "  << (ending_energy) << std::endl;
  std::cout << "energy-sample-delta[J]: "  << std::abs(ending_energy - starting_energy) << std::endl;
  std::cout << "energy-sample-time[ms]: " << SAMPLING_TIME << std::endl;
  print_metrics(device_times, "device-time", "ms");
  print_metrics(freq_change_overheads, "freq-change-overhead", "ms");
  print_metrics(device_consumptions, "device-energy", "J");
  print_metrics(host_consumptions, "host-energy", "J");
}