#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>


constexpr size_t NUM_RUNS = 10;
constexpr size_t NUM_ITERS = 15;
constexpr size_t SAMPLING_TIME = 2000; // milliseconds

class MatMul {
public:
  synergy::queue& q;
  size_t size;
  sycl::buffer<int, 2>& a_buf;
  sycl::buffer<int, 2>& b_buf; 
  sycl::buffer<int, 2>& c_buf;

  MatMul(synergy::queue& q, size_t size, sycl::buffer<int, 2>& a_buf, sycl::buffer<int, 2>& b_buf, sycl::buffer<int, 2>& c_buf) : q{q}, size{size}, a_buf{a_buf}, b_buf{b_buf}, c_buf{c_buf} {}

  sycl::event operator() () {
    return q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc{a_buf, h, sycl::read_only};
      sycl::accessor b_acc{b_buf, h, sycl::read_only};
      sycl::accessor c_acc{c_buf, h, sycl::read_write};

      sycl::range<2> grid{size, size};
      sycl::range<2> block{size < 32 ? size : 32, size < 32 ? size : 32};

      h.parallel_for(sycl::nd_range<2>(grid, block), [=, size=size](sycl::nd_item<2> idx) {
        int i = idx.get_global_id(0);
        int j = idx.get_global_id(1);
        c_acc[i][j] = 0.0f;
        for (size_t k = 0; k < size; k++) {
          c_acc[i][j] += a_acc[i][k] * b_acc[k][j];
        }
      });
    });
  }
};
class Mersenne {
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
#define PI 3.14159265358979fq
public:
  synergy::queue& q;
  size_t mersenne_size;
  sycl::buffer<uint>& buf_ma;
  sycl::buffer<uint>& buf_b;
  sycl::buffer<uint>& buf_c;
  sycl::buffer<uint>& buf_seed;
  sycl::buffer<sycl::float4>& buf_result;

  Mersenne(synergy::queue& q, size_t mersenne_size, sycl::buffer<uint>& buf_ma, sycl::buffer<uint>& buf_b, sycl::buffer<uint>& buf_c, sycl::buffer<uint>& buf_seed, sycl::buffer<sycl::float4>& buf_result) : q{q}, mersenne_size{mersenne_size}, buf_ma{buf_ma}, buf_b{buf_b}, buf_c{buf_c}, buf_seed{buf_seed}, buf_result{buf_result} {}

  sycl::event operator() () {
    return q.submit([&](sycl::handler& cgh) {
      auto ma_acc = buf_ma.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = buf_b.get_access<sycl::access::mode::read>(cgh);
      auto c_acc = buf_c.get_access<sycl::access::mode::read>(cgh);
      auto seed_acc = buf_seed.get_access<sycl::access::mode::read>(cgh);
      auto result_acc = buf_result.get_access<sycl::access::mode::write>(cgh);

      sycl::range<1> ndrange{mersenne_size};
      cgh.parallel_for<class MerseTwisterKernel>(ndrange, [=, length = mersenne_size, num_iters = 5000](sycl::id<1> id) {
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
    });
  };
};

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


struct FreqChangeCost {
  double time;
  synergy::energy device_energy;
  synergy::energy host_energy;
  double overhead_time;
  synergy::energy overhead_device_energy;
  synergy::energy overhead_host_energy;
};

template<typename T>
FreqChangeCost launch_kernel(synergy::queue& q, synergy::frequency freq, FreqChangePolicy policy, bool is_first, size_t num_iters, T& kernel) {
  std::chrono::high_resolution_clock::time_point start_time, end_time;
  synergy::energy start_energy_device, end_energy_device, start_energy_host, end_energy_host;
  double overhead_time {0}, time {0};
  synergy::energy overhead_device_energy {0}, device_energy {0}, overhead_host_energy {0}, host_energy {0};
  for (int it = 0; it < num_iters; it++) {
    if ((it == 0 && ((is_first && policy == FreqChangePolicy::APP) || (policy == FreqChangePolicy::PHASE))) || policy == FreqChangePolicy::KERNEL) {
      start_time = std::chrono::high_resolution_clock::now();
      start_energy_device = q.device_energy_consumption();
      start_energy_host = q.host_energy_consumption();
      q.get_synergy_device().set_core_frequency(freq);
      end_time = std::chrono::high_resolution_clock::now();
      end_energy_device = q.device_energy_consumption();
      end_energy_host = q.host_energy_consumption();

      overhead_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      overhead_device_energy += end_energy_device - start_energy_device;
      overhead_host_energy += end_energy_host - start_energy_host;
      q.set_target_frequencies(0, 0);
    }
    
    // start_time = std::chrono::high_resolution_clock::now();
    start_energy_device = q.device_energy_consumption();
    start_energy_host = q.host_energy_consumption();
    auto e = kernel(); // Kernel Launch
    // end_time = std::chrono::high_resolution_clock::now();
    end_energy_device = q.device_energy_consumption();
    end_energy_host = q.host_energy_consumption();
    auto start_kernel = e.template get_profiling_info<sycl::info::event_profiling::command_start>() / 1000000;
    auto end_kernel = e.template get_profiling_info<sycl::info::event_profiling::command_end>() / 1000000;
    
    // time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    time += end_kernel - start_kernel;
    device_energy += end_energy_device - start_energy_device;
    host_energy += end_energy_host - start_energy_host;
  }
  return {time, device_energy, host_energy, overhead_time, overhead_device_energy, overhead_host_energy};
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
  std::vector<double> freq_change_time_overheads;
  std::vector<synergy::energy> freq_change_device_energy_overheads;
  std::vector<synergy::energy> freq_change_host_energy_overheads;

  MatMul matmul_kernel{q, matmul_size, matA_buf, matB_buf, matC_buf};
  Mersenne mersenne_kernel{q, mersenne_size, buf_ma, buf_b, buf_c, buf_seed, buf_result};

  for (int i = 0; i < num_runs; i++) {

    auto ret_mat = launch_kernel(q, freq_matmul, policy, true, num_iters, matmul_kernel);
    auto ret_mer = launch_kernel(q, freq_mersenne, policy, false, num_iters, mersenne_kernel);

    q.wait_and_throw();

    total_times.push_back(ret_mat.time + ret_mer.time);
    device_consumptions.push_back(ret_mat.device_energy + ret_mer.device_energy);
    host_consumptions.push_back(ret_mat.host_energy + ret_mer.host_energy);
    freq_change_time_overheads.push_back(ret_mat.overhead_time + ret_mer.overhead_time);
    freq_change_device_energy_overheads.push_back(ret_mat.overhead_device_energy + ret_mer.overhead_device_energy);
    freq_change_host_energy_overheads.push_back(ret_mat.overhead_host_energy + ret_mer.overhead_host_energy);
  }

  auto ending_energy = sample_energy_consumption(SAMPLING_TIME);

  std::cout << "energy-sample-before[J]: " << (starting_energy) << std::endl;
  std::cout << "energy-sample-after[J]: "  << (ending_energy) << std::endl;
  std::cout << "energy-sample-delta[J]: "  << std::abs(ending_energy - starting_energy) << std::endl;
  std::cout << "energy-sample-time[ms]: " << SAMPLING_TIME << std::endl;
  print_metrics(total_times, "total-time", "ms");
  print_metrics(device_consumptions, "device-energy", "J");
  print_metrics(host_consumptions, "host-energy", "J");
  print_metrics(freq_change_time_overheads, "freq-change-time-overhead", "ms");
  print_metrics(freq_change_device_energy_overheads, "freq-change-device-energy-overhead", "J");
  print_metrics(freq_change_host_energy_overheads, "freq-change-host-energy-overhead", "J");
}