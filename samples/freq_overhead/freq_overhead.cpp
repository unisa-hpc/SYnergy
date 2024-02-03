#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

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
  size_t matmul_size,
  sycl::buffer<int, 2>& a_buf,
  sycl::buffer<int, 2>& b_buf,
  sycl::buffer<int, 2>& c_buf,
  synergy::frequency freq,
  FreqChangePolicy policy,
  size_t num_iters) {
  std::chrono::high_resolution_clock::time_point start, end;
  double duration {0};
  for (int it = 0; it < num_iters; it++) {
    if (it == 0 || policy == FreqChangePolicy::KERNEL) {
      start = std::chrono::high_resolution_clock::now();
      auto e = q.submit(0, freq, [&](sycl::handler& cgh) {});
      e.wait();
      end = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc{a_buf, h, sycl::read_only};
      sycl::accessor b_acc{b_buf, h, sycl::read_only};
      sycl::accessor c_acc{c_buf, h, sycl::read_write};

      sycl::range<2> grid{matmul_size, matmul_size};
      sycl::range<2> block{matmul_size < 32 ? matmul_size : 32, matmul_size < 32 ? matmul_size : 32};

      h.parallel_for<mat_mul_kernel>(sycl::nd_range<2>(grid, block), [=, matmul_size=matmul_size](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);
        c_acc[i][j] = 0.0f;
        for (size_t k = 0; k < matmul_size; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    }).wait();
  }
  return duration;
}

double mersenne(synergy::queue& q,
  size_t mersenne_size,
  sycl::buffer<uint>& buf_ma,
  sycl::buffer<uint>& buf_b,
  sycl::buffer<uint>& buf_c,
  sycl::buffer<uint>& buf_seed,
  sycl::buffer<sycl::float4>& buf_result,
  synergy::frequency freq,
  FreqChangePolicy policy,
  size_t num_iters) {
  #define MT_RNG_COUNT 4096
  #define MT_MM 9
  #define MT_NN 19
  #define MT_WMASK 0xFFFFFFFFU
  #define MT_UMASK 0xFFFFFFFEU
  #define MT_LMASK 0x1U
  #define MT_SHIFT0 12
  #define MT_SHIFTB 7
  #define MT_SHIFTC 15
  #define MT_SHIFT1 18
  #define PI 3.14159265358979f
  std::chrono::high_resolution_clock::time_point start, end;
  double duration {0};
  for (int it = 0; it < num_iters; it++) {
    if ((it == 0 && policy == FreqChangePolicy::PHASE) || policy == FreqChangePolicy::KERNEL) {
      start = std::chrono::high_resolution_clock::now();
      auto e = q.submit(0, freq, [&](sycl::handler& cgh) {});
      e.wait();
      end = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    q.submit([&](sycl::handler& cgh) {
      auto ma_acc = buf_ma.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = buf_b.get_access<sycl::access::mode::read>(cgh);
      auto c_acc = buf_c.get_access<sycl::access::mode::read>(cgh);
      auto seed_acc = buf_seed.get_access<sycl::access::mode::read>(cgh);
      auto result_acc = buf_result.get_access<sycl::access::mode::write>(cgh);

      sycl::range<1> ndrange{mersenne_size};
      cgh.parallel_for<class MerseTwisterKernel>(ndrange, [=, length = mersenne_size, num_iters = num_iters](sycl::id<1> id) {
        int gid = id[0];

        if(gid >= length)
          return;
        for(size_t i = 0; i < num_iters; i++) {
          int iState, iState1, iStateM;
          unsigned int mti, mti1, mtiM, x;
          unsigned int matrix_a, mask_b, mask_c;

          unsigned int mt[MT_NN]; // FIXME

          matrix_a = ma_acc[gid];
          mask_b = b_acc[gid];
          mask_c = c_acc[gid];

          mt[0] = seed_acc[gid];
          for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

          iState = 0;
          mti1 = mt[0];

          float tmp[5];
          for(int i = 0; i < 4; ++i) {
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN)
              iState1 -= MT_NN;
            if(iStateM >= MT_NN)
              iStateM -= MT_NN;
            mti = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

            mt[iState] = x;
            iState = iState1;

            // Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & mask_b;
            x ^= (x << MT_SHIFTC) & mask_c;
            x ^= (x >> MT_SHIFT1);

            tmp[i] = ((float)x + 1.0f) / 4294967296.0f;
          }

          sycl::float4 val;
          val.s0() = tmp[0];
          val.s1() = tmp[1];
          val.s2() = tmp[2];
          val.s3() = tmp[3];

          result_acc[gid] = val;
        }
      });
    }).wait();
  }
  return duration;
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
  synergy::frequency freq_mersenne = 0;
  FreqChangePolicy policy;
  size_t num_iters, num_runs, matmul_size, mersenne_size;
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] << " <policy> <num-runs> <num-iters> <matmul_size> <mersenne_size> <freq_matmul> <freq_mersenne>" << std::endl;
    exit(1);
  }

  policy = (std::string(argv[1]) == "app") ? FreqChangePolicy::APP : 
           (std::string(argv[1]) == "phase") ? FreqChangePolicy::PHASE : 
           (std::string(argv[1]) == "kernel") ? FreqChangePolicy::KERNEL : 
           throw std::runtime_error("Invalid policy");
  num_runs = std::stoi(argv[2]);
  num_iters = std::stoi(argv[3]);
  matmul_size = std::stoi(argv[4]);
  mersenne_size = std::stoi(argv[5]);
  freq_matmul = std::stoi(argv[6]);
  freq_mersenne = std::stoi(argv[7]);

  std::vector<int> matA(matmul_size * matmul_size, 1);
  std::vector<int> matB(matmul_size * matmul_size, 1);
  std::vector<int> matC(matmul_size * matmul_size, 0);

  sycl::buffer<int, 2> matA_buf(matA.data(), sycl::range<2>{matmul_size, matmul_size});
  sycl::buffer<int, 2> matB_buf(matB.data(), sycl::range<2>{matmul_size, matmul_size});
  sycl::buffer<int, 2> matC_buf(matC.data(), sycl::range<2>{matmul_size, matmul_size});

  std::vector<uint> ma;
  std::vector<uint> b;
  std::vector<uint> c;
  std::vector<uint> seed;
  std::vector<sycl::float4> result;
  ma.resize(mersenne_size);
  b.resize(mersenne_size);
  c.resize(mersenne_size);
  seed.resize(mersenne_size);
  result.resize(mersenne_size);

  sycl::buffer<uint> buf_ma(ma.data(), sycl::range<1>{mersenne_size});
  sycl::buffer<uint> buf_b(b.data(), sycl::range<1>{mersenne_size});
  sycl::buffer<uint> buf_c(c.data(), sycl::range<1>{mersenne_size});
  sycl::buffer<uint> buf_seed(seed.data(), sycl::range<1>{mersenne_size});
  sycl::buffer<sycl::float4> buf_result(result.data(), sycl::range<1>{mersenne_size});

  auto starting_energy = sample_energy_consumption(SAMPLING_TIME);

  synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};

  std::vector<double> total_times;
  std::vector<synergy::energy> device_consumptions;
  std::vector<synergy::energy> host_consumptions;
  std::vector<double> freq_change_overheads;

  for (int i = 0; i < num_runs; i++) {
    double freq_change_overhead = 0;

    auto start = std::chrono::high_resolution_clock::now();
    freq_change_overhead += matrix_mul(q, matmul_size, matA_buf, matB_buf, matC_buf, freq_matmul, policy, num_iters);
    freq_change_overhead += mersenne(q, mersenne_size, buf_ma, buf_b, buf_c, buf_seed, buf_result, freq_mersenne, policy, num_iters);
    q.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();

    freq_change_overheads.push_back(freq_change_overhead);
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    total_times.push_back(duration);
    auto device_consumption = q.device_energy_consumption();
    device_consumptions.push_back(device_consumption);
    auto host_consumption = q.host_energy_consumption();
    host_consumptions.push_back(host_consumption);
  }

  auto ending_energy = sample_energy_consumption(SAMPLING_TIME);

  std::cout << "energy-sample-before[J]: " << (starting_energy) << std::endl;
  std::cout << "energy-sample-after[J]: "  << (ending_energy) << std::endl;
  std::cout << "energy-sample-delta[J]: "  << std::abs(ending_energy - starting_energy) << std::endl;
  std::cout << "energy-sample-time[ms]: " << SAMPLING_TIME << std::endl;
  print_metrics(total_times, "total-time", "ms");
  print_metrics(freq_change_overheads, "freq-change-overhead", "ms");
  print_metrics(device_consumptions, "device-energy", "J");
  print_metrics(host_consumptions, "host-energy", "J");
}