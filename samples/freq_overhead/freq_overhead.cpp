#include <synergy.hpp>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <memory>
#include "bitmap.h"

constexpr size_t SAMPLING_TIME = 2000; // milliseconds

class Sobel {
public:
  synergy::queue& q;
  size_t size;
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;
  std::shared_ptr<sycl::buffer<sycl::float4, 2>> input_buf;
  std::shared_ptr<sycl::buffer<sycl::float4, 2>> output_buf;

  Sobel(synergy::queue& q, size_t size) : q{q}, size{size} {
    input.resize(size * size);
    load_bitmap_mirrored("./Brommy.bmp", size, input);
    output.resize(size * size);
    input_buf = std::make_shared<sycl::buffer<sycl::float4, 2>>(input.data(), sycl::range<2>{size, size});
    output_buf = std::make_shared<sycl::buffer<sycl::float4, 2>>(output.data(), sycl::range<2>{size, size});
  }

  sycl::event operator() () {
    return q.submit([&](sycl::handler& cgh) {
      auto in = input_buf->get_access<sycl::access::mode::read>(cgh);
      auto out = output_buf->get_access<sycl::access::mode::discard_write>(cgh);
      sycl::range<2> ndrange{size, size};

      // Sobel kernel 3x3
      const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

      cgh.parallel_for<class SobelBenchKernel>(
        ndrange, [in, out, kernel, size_ = size, num_iters=2](sycl::id<2> gid) {
          int x = gid[0];
          int y = gid[1];

          for(size_t i = 0; i < num_iters; i++) {
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
          }
        });
    });
  }

};

class MatMul {
public:
  synergy::queue& q;
  size_t size;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<sycl::buffer<int, 2>> a_buf;
  std::shared_ptr<sycl::buffer<int, 2>> b_buf; 
  std::shared_ptr<sycl::buffer<int, 2>> c_buf;

  MatMul(synergy::queue& q, size_t size) : q{q}, size{size} {
    a.resize(size * size);
    b.resize(size * size);
    c.resize(size * size);
    std::fill(a.begin(), a.end(), 1);
    std::fill(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);
    a_buf = std::make_shared<sycl::buffer<int, 2>>(a.data(), sycl::range<2>{size, size});
    b_buf = std::make_shared<sycl::buffer<int, 2>>(b.data(), sycl::range<2>{size, size});
    c_buf = std::make_shared<sycl::buffer<int, 2>>(c.data(), sycl::range<2>{size, size});
  }

  sycl::event operator() () {
    return q.submit([&](sycl::handler& h) {
      sycl::accessor a_acc{*(a_buf.get()), h, sycl::read_only};
      sycl::accessor b_acc{*(b_buf.get()), h, sycl::read_only};
      sycl::accessor c_acc{*(c_buf.get()), h, sycl::read_write};

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
public:
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
  synergy::queue& q;
  size_t size;
  std::vector<uint> ma;
  std::vector<uint> b;
  std::vector<uint> c;
  std::vector<uint> seed;
  std::vector<sycl::float4> result;
  std::shared_ptr<sycl::buffer<uint>> buf_ma;
  std::shared_ptr<sycl::buffer<uint>> buf_b;
  std::shared_ptr<sycl::buffer<uint>> buf_c;
  std::shared_ptr<sycl::buffer<uint>> buf_seed;
  std::shared_ptr<sycl::buffer<sycl::float4>> buf_result;

  Mersenne(synergy::queue& q, size_t size) : q{q}, size{size} {
    ma.resize(size);
    b.resize(size);
    c.resize(size);
    seed.resize(size);
    result.resize(size);
    buf_ma = std::make_shared<sycl::buffer<uint>>(ma.data(), sycl::range<1>{size});
    buf_b = std::make_shared<sycl::buffer<uint>>(b.data(), sycl::range<1>{size});
    buf_c = std::make_shared<sycl::buffer<uint>>(c.data(), sycl::range<1>{size});
    buf_seed = std::make_shared<sycl::buffer<uint>>(seed.data(), sycl::range<1>{size});
    buf_result = std::make_shared<sycl::buffer<sycl::float4>>(result.data(), sycl::range<1>{size});
  }

  sycl::event operator() () {
    return q.submit([&](sycl::handler& cgh) {
      auto ma_acc = buf_ma->get_access<sycl::access::mode::read>(cgh);
      auto b_acc = buf_b->get_access<sycl::access::mode::read>(cgh);
      auto c_acc = buf_c->get_access<sycl::access::mode::read>(cgh);
      auto seed_acc = buf_seed->get_access<sycl::access::mode::read>(cgh);
      auto result_acc = buf_result->get_access<sycl::access::mode::write>(cgh);

      sycl::range<1> ndrange{size};
      cgh.parallel_for<class MerseTwisterKernel>(ndrange, [=, length = size, num_iters = 5000](sycl::id<1> id) {
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

template<typename T>
void print_metrics(std::vector<T> values, std::string label, std::string unit = "") {
  if (values.empty()) {
    std::cout << label << "[" << unit << "]: [  ]" << std::endl;
    std::cout << label << "-avg[" << unit << "]: " << -1 << std::endl;
    std::cout << label << "-stdev[" << unit << "]: " << -1 << std::endl;
    std::cout << label << "-max[" << unit << "]: " << -1 << std::endl;
    std::cout << label << "-min[" << unit << "]: " << -1 << std::endl;
    std::cout << label << "-median[" << unit << "]: " << -1 << std::endl;
    return;
  }
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
  double kernel_time;
  double overhead_time;
  synergy::energy overhead_device_energy;
  synergy::energy overhead_host_energy;
};

template<typename T>
FreqChangeCost launch_kernel(synergy::queue& q, synergy::frequency freq, FreqChangePolicy policy, bool is_first, size_t num_iters, T& kernel) {
  std::chrono::high_resolution_clock::time_point overhead_start_time, overhead_end_time;
  synergy::energy overhead_start_energy_device, overhead_end_energy_device, overhead_start_energy_host, overhead_end_energy_host;
  double overhead_time {0}, kernel_time{0};
  synergy::energy overhead_device_energy {0}, overhead_host_energy {0};
  for (int it = 0; it < num_iters; it++) {
    if ((it == 0 && ((is_first && policy == FreqChangePolicy::APP) || (policy == FreqChangePolicy::PHASE))) || policy == FreqChangePolicy::KERNEL) {
      overhead_start_time = std::chrono::high_resolution_clock::now();
      overhead_start_energy_device = q.device_energy_consumption();
#ifdef SYNERGY_HOST_PROFILING
      overhead_start_energy_host = q.host_energy_consumption();
#endif
      auto cfe = q.submit(0, freq, [&](sycl::handler& cgh){
        cgh.single_task([=](){
          // Do nothing
        });
      }); // Set frequency
      cfe.wait();
      overhead_end_time = std::chrono::high_resolution_clock::now();
      overhead_end_energy_device = q.device_energy_consumption();
#ifdef SYNERGY_HOST_PROFILING
      overhead_end_energy_host = q.host_energy_consumption();
#endif
      overhead_time += std::chrono::duration_cast<std::chrono::milliseconds>(overhead_end_time - overhead_start_time).count();
      overhead_device_energy += overhead_end_energy_device - overhead_start_energy_device;
#ifdef SYNERGY_HOST_PROFILING
      overhead_host_energy += overhead_end_energy_host - overhead_start_energy_host;
#endif
    }
    
    auto e = kernel(); // Kernel Launch
    e.wait();
    kernel_time += (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000; // to milliseconds
  }
  return {kernel_time, overhead_time, overhead_device_energy, overhead_host_energy};
}

int main(int argc, char** argv) {
  synergy::frequency freq_matmul = 0;
  synergy::frequency freq_sobel = 0;
  FreqChangePolicy policy;
  size_t num_iters, num_runs, matmul_size, sobel_size;
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] << " <policy> <num-runs> <num-iters> <matmul_size> <sobel_size> <freq_matmul> <freq_sobel>" << std::endl;
    exit(1);
  }

  policy = (std::string(argv[1]) == "app") ? FreqChangePolicy::APP : 
           (std::string(argv[1]) == "phase") ? FreqChangePolicy::PHASE : 
           (std::string(argv[1]) == "kernel") ? FreqChangePolicy::KERNEL : 
           throw std::runtime_error("Invalid policy");
  num_runs = std::stoi(argv[2]);
  num_iters = std::stoi(argv[3]);
  matmul_size = std::stoi(argv[4]);
  sobel_size = std::stoi(argv[5]);
  freq_matmul = std::stoi(argv[6]);
  freq_sobel = std::stoi(argv[7]);

  std::vector<double> total_times;
  std::vector<double> kernel_times;
  std::vector<synergy::energy> device_consumptions;
  std::vector<synergy::energy> host_consumptions;
  std::vector<double> freq_change_time_overheads;
  std::vector<synergy::energy> freq_change_device_energy_overheads;
  std::vector<synergy::energy> freq_change_host_energy_overheads;

  for (int i = 0; i < num_runs; i++) {
    synergy::queue q {sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}};
    MatMul matmul_kernel{q, matmul_size};
    Sobel sobel_kernel{q, sobel_size};
    auto start_time = std::chrono::high_resolution_clock::now();

    // launch kernels
    auto ret_mat = launch_kernel(q, freq_matmul, policy, true, num_iters, matmul_kernel);
    auto ret_sob = launch_kernel(q, freq_sobel, policy, false, num_iters, sobel_kernel);
    q.wait_and_throw(); // wait for all kernels to finish

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    total_times.push_back(duration);

    kernel_times.push_back(ret_mat.kernel_time + ret_sob.kernel_time);

    auto device_consumption = q.device_energy_consumption();
    device_consumptions.push_back(device_consumption);

#ifdef SYNERGY_HOST_PROFILING
    auto host_consumption = q.host_energy_consumption();
    host_consumptions.push_back(host_consumption);
#endif
    freq_change_time_overheads.push_back(ret_mat.overhead_time + ret_sob.overhead_time);
    freq_change_device_energy_overheads.push_back(ret_mat.overhead_device_energy + ret_sob.overhead_device_energy);
#ifdef SYNERGY_HOST_PROFILING
    freq_change_host_energy_overheads.push_back(ret_mat.overhead_host_energy + ret_sob.overhead_host_energy);
#endif
  }

  print_metrics(total_times, "total-time", "ms");
  print_metrics(total_times, "kernel-time", "ms");
  print_metrics(device_consumptions, "device-energy", "J");
  print_metrics(host_consumptions, "host-energy", "J");
  print_metrics(freq_change_time_overheads, "freq-change-time-overhead", "ms");
  print_metrics(freq_change_device_energy_overheads, "freq-change-device-energy-overhead", "J");
  print_metrics(freq_change_host_energy_overheads, "freq-change-host-energy-overhead", "J");
}