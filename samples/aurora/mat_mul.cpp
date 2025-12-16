#include <synergy.hpp>
#include <fstream>   // <-- this is what you need
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

namespace logs {
    std::unique_ptr<std::ofstream> kernels_log_file = nullptr;
    std::unique_ptr<std::ofstream> device_log_file  = nullptr;

    void init_log_files(int myrank, const std::string& base_dir) {
        namespace fs = std::filesystem;
        fs::path dir(base_dir);
        if (!fs::exists(dir)) fs::create_directories(dir);

        fs::path kernels_path = dir / ("rank_" + std::to_string(myrank) + "_kernels.log");
        fs::path device_path  = dir / ("rank_" + std::to_string(myrank) + "_device.log");

        kernels_log_file = std::make_unique<std::ofstream>(kernels_path, std::ios::app);
        device_log_file  = std::make_unique<std::ofstream>(device_path,  std::ios::app);

        (*kernels_log_file) << "=== Kernel log started for rank "
                            << myrank <<  " ===" << std::endl;
        (*device_log_file) << "=== Device log started for rank "
                           << myrank << " ===" << std::endl;
    }

    void log_kernel(const std::string& msg) {
        if (kernels_log_file && kernels_log_file->is_open()) {
            (*kernels_log_file) << msg << std::endl;
            kernels_log_file->flush();
        }
    }

    void log_device(const std::string& msg) {
        if (device_log_file && device_log_file->is_open()) {
            (*device_log_file) << msg << std::endl;
            device_log_file->flush();
        }
    }

    void close_log_files() {
        if (kernels_log_file && kernels_log_file->is_open()) {
            (*kernels_log_file) << "=== Kernel log closed ===" << std::endl;
            kernels_log_file->close();
        }
        if (device_log_file && device_log_file->is_open()) {
            (*device_log_file) << "=== Device log closed ===" << std::endl;
            device_log_file->close();
        }
    }
}




double get_kernel_execution_time_ms(const sycl::event& e) {
    using namespace sycl;
    // Retrieve timestamps in nanoseconds
    uint64_t start = e.get_profiling_info<info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<info::event_profiling::command_end>();

    // Convert to milliseconds
    double duration_ms = (end - start) * 1e-6;
    return duration_ms;
}


using namespace sycl;

class MatMulKernel {
public:
    // Accessors as mbles
    accessor<float, 2, access::mode::read> a;
    accessor<float, 2, access::mode::read> b;
    accessor<float, 2,  access::mode::write> c;
    size_t N;

    // Constructor
    MatMulKernel(accessor<float, 2, access::mode::read > a_,
                 accessor<float, 2, access::mode::read> b_,
                 accessor<float, 2, access::mode::write> c_,
                 size_t N_)
        : a(a_), b(b_), c(c_), N(N_) {}

    // The kernel operator
    void operator()(id<2> idx) const {
        size_t row = idx[0];
        size_t col = idx[1];
        float sum = 0.0f;
        for (size_t k = 0; k < N; k++) {
            sum += a[row][k] * b[k][col];
        }
        c[row][col] = sum;
    }
};



class DummyKernel {
public:
    // Accessors as mbles
    accessor<float, 2, access::mode::read> a;
    accessor<float, 2, access::mode::read> b;
    accessor<float, 2,  access::mode::write> c;
    size_t N;

    // Constructor
    DummyKernel(accessor<float, 2, access::mode::read > a_,
                 accessor<float, 2, access::mode::read> b_,
                 accessor<float, 2, access::mode::write> c_,
                 size_t N_)
        : a(a_), b(b_), c(c_), N(N_) {};
    // The kernel operator
    void operator()(id<2> idx) const {
        
    }
};

int main(int argc, char*argv[]) {
    std::ostringstream kernel_info;
    std::ostringstream device_info;

    int core_freq=0;
    std::string log_dir;
    if(argc==3){
        core_freq = atoi(argv[1]);
        log_dir = argv[2];
    }
    else{
        std::cout << "Usage: ./freq_scaling_inte <core_freq> <log_dir>" << std::endl;
    }
    
    logs::init_log_files(0,log_dir); // Init log file

    constexpr size_t N = 4096;
    synergy::queue q{gpu_selector_v};

    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f);

    // Initialize input matrices
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    {
        
        event e;
        buffer<float, 2> a_buf(A.data(), range<2>(N, N));
        buffer<float, 2> b_buf(B.data(), range<2>(N, N));
        buffer<float, 2> c_buf(C.data(), range<2>(N, N));
        // Change the device frequency using SYnergy submit command: the first value is uncore frequency, the second is core frequency.
        // 0 represents the default frequency.
        q.submit(0, core_freq, [&](handler& h) {
            // Create accessors
            accessor<float, 2, access::mode::read > a(a_buf, h);
            accessor<float, 2, access::mode::read > b(b_buf, h);
            accessor<float, 2, access::mode::write > c(c_buf, h);

            // Instantiate the functor kernel
            DummyKernel kernel(a, b, c, N);

            // Launch kernel
            h.parallel_for(range<2>(N, N), kernel);
        });
        e.wait();

        double start_energy = q.device_energy_consumption();
        auto start_time=std::chrono::high_resolution_clock::now();
        e = q.submit(0, core_freq, [&](handler& h) {
            // Create accessors
            accessor<float, 2, access::mode::read > a(a_buf, h);
            accessor<float, 2, access::mode::read > b(b_buf, h);
            accessor<float, 2, access::mode::write > c(c_buf, h);

            // Instantiate the functor kernel
            MatMulKernel kernel(a, b, c, N);

            // Launch kernel
            h.parallel_for(range<2>(N, N), kernel);
        });
        e.wait();
        auto end_time=std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double kernel_time_ms = get_kernel_execution_time_ms(e);
        double end_energy = q.device_energy_consumption();

        device_info << "All GPUs Time [ms]: "
        << elapsed_ms << std::endl;


        device_info << "All GPUs Energy [J]: "
        << end_energy - start_energy
        << std::endl;

        kernel_info << "All GPUs Time [ms]: "
        << kernel_time_ms 
        << std::endl;
        kernel_info << "All GPUs Energy [J]: "
        << q.kernel_energy_consumption(e)
        << std::endl;




    }


    
    // Print the result
    logs::log_kernel(kernel_info.str());
    logs::log_device(device_info.str());
    logs::close_log_files();

    return 0;
}
