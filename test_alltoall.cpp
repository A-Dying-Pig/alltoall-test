#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <alltoall_global_scheduler.h>
#include <alltoall_local_scheduler.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <ctime>
#include <nccl.h>



// void test_NetAllGather(std::shared_ptr<mscclpp::Communicator> comm,
//                        std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
//                        const int rank, const int nranks,
//                        const size_t buff_size, const bool inplace, const bool columnwise) {
//     // Intialize host and device buffers
//     std::vector<__fp16> host_buff(buff_size / sizeof(__fp16));
//     for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = __fp16(int((i * rank) % 101));
//     void* input_buff;
//     CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
//     CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
//     void* output_buff;
//     if (inplace) {
//         output_buff = input_buff;
//     } else {
//         CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
//     }

//     // Initialize NetWrapper
//     NetAllGather wrapper;
//     int dim1, input_dim2, output_dim2;
//     if (columnwise) {
//         const size_t alignment = sizeof(int4) / sizeof(__fp16) * nranks;
//         dim1 = ((int) sqrt(buff_size / sizeof(__fp16))) / alignment * alignment;
//         input_dim2 = dim1 / nranks;
//         output_dim2 = dim1;
//     } else {
//         dim1 = buff_size / sizeof(__fp16);
//         input_dim2 = 1;
//         output_dim2 = 1;
//     }
// 	wrapper.init(comm,
// 				 connections,
// 				 rank,
// 				 nranks,
// 				 pllmTensor<half>{
//                     (half*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
// 				 pllmTensor<half>{
//                     (half*)output_buff, dim1, output_dim2, PllmLayout::ROW_MAJOR});

// 	MPI_Barrier(MPI_COMM_WORLD);
//     wrapper.setColumnwise(columnwise);
//     wrapper(0, 32, 1024, true);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Check allgather correctness
//     const size_t nelem_per_shard = host_buff.size() / nranks;
//     CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));
//     if (columnwise) {
//         bool correct = true;
//         for (int i = 0; i < dim1; ++i) {
//             for (int j = 0; j < output_dim2; ++j) {
//                 const int remoteRank = (int) (j / input_dim2);
//                 __fp16 expected = __fp16(int(((i * input_dim2 + j % input_dim2) * remoteRank) % 101));
//                 if (host_buff[i * output_dim2 + j] != expected) {
//                     std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank
//                               << " at index (" << i << "," << j << ")" << std::endl;
//                     correct = false;
//                     break;
//                 }
//             }
//             if (!correct) break;
//         }
//     } else {
//         for (size_t i = 0; i < host_buff.size(); ++i) {
//             const int remoteRank = i / nelem_per_shard;
//             __fp16 expected = __fp16(int((i * remoteRank) % 101));
//             if (host_buff[i] != expected) {
//                 std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank << " at index " << i << std::endl;
//                 break;
//             }
//         }
//     }

//     CUDA_CHECK(cudaFree(input_buff));
//     if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
//     std::cout << "Rank " << rank << " NetAllGather test ("
//               << "buff_size=" << buff_size << ",inplace=" << inplace
//               << ",columnwise=" << columnwise << ") finished" << std::endl;
// }

// void perf_NetAllGather(std::shared_ptr<mscclpp::Communicator> comm,
//                        std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
//                        const int rank, const int nranks,
//                        const size_t buff_size, const bool inplace,
//                        const int nblocks, const int nthreads,
//                        const int warmup_iters, const int perf_iters) {
//     // Intialize host and device buffers
//     std::vector<__fp16> host_buff(buff_size / sizeof(__fp16));
//     for (size_t i = 0; i < host_buff.size(); ++i) host_buff[i] = __fp16(int((i * rank) % 101));
//     void* input_buff;
//     CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
//     CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
//     void* output_buff;
//     if (inplace) {
//         output_buff = input_buff;
//     } else {
//         CUDA_CHECK(cudaMalloc(&output_buff, buff_size));
//     }

//     int dim1 = buff_size / sizeof(__fp16);

//     // Initialize NetWrapper
//     NetAllGather wrapper;
//     wrapper.init(
//         comm, connections, rank, nranks,
//         pllmTensor<half>{(half*)input_buff, dim1, 1, PllmLayout::ROW_MAJOR},
//         pllmTensor<half>{(half*)output_buff, dim1, 1, PllmLayout::ROW_MAJOR});

//     hipEvent_t start_event, end_event;
//     CUDA_CHECK(hipEventCreate(&start_event));
//     CUDA_CHECK(hipEventCreate(&end_event));

//     MPI_Barrier(MPI_COMM_WORLD);

//     for (int i = 0; i < warmup_iters; ++i) wrapper(0, nblocks, nthreads, true);
//     CUDA_CHECK(hipEventRecord(start_event));
//     for (int i = 0; i < perf_iters; ++i) wrapper(0, nblocks, nthreads, true);
//     CUDA_CHECK(hipEventRecord(end_event));
//     CUDA_CHECK(cudaDeviceSynchronize());

//     float elapsed_time;
//     CUDA_CHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
//     double avg_time = (double) elapsed_time / perf_iters;
//     double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9;

//     CUDA_CHECK(cudaFree(input_buff));
//     if (input_buff != output_buff) CUDA_CHECK(cudaFree(output_buff));
//     CUDA_CHECK(hipEventDestroy(start_event));
//     CUDA_CHECK(hipEventDestroy(end_event));
//     if (rank == 0) {
//         std::cout << "Rank " << rank << " NetAllGather perf ("
//                   << "nblocks=" << nblocks << ",nthreads=" << nthreads
//                   << ",buff_size=" << buff_size << ",inplace=" << inplace
//                   << ") finished: time=" << avg_time * 1e3 << "us "
//                   << "algbw=" << algbw << "GBps" << std::endl;
//     }
// }

void uniform_distribution(uint * workload, uint nrank, uint mean){
    uint bound = MIN(MAX_BUFFER_SIZE_PER_RANK, mean * 2);
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            workload[i * nrank + j] = rand() % bound;
        }
    }
}

void allocate_device_memory(
    void ** sendbuf, void ** recvbuff, void ** tempbuff, void ** syncbuff, void ** verifybuff,
    size_t ** sendcount, size_t ** sendpos, size_t ** recvcount, size_t ** recvpos
){

}

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    return -1           ;                           \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
    srand((unsigned)time(0));
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    hipSetDevice(rank);

    // Print off a hello world message
    std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;

    // Initialize Communicator
    ncclComm_t comm;
    ncclUniqueId ncclId;
    if (rank ==0) NCCLCHECK(ncclGetUniqueId(&ncclId));
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCLCHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));

    //let rank 0 process generate a random workload
    uint * workload = new uint[nranks * nranks];
    if (rank == 0) uniform_distribution(workload, nranks, 1024);
    MPI_Bcast(&workload, nranks * nranks * sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "receive demand matrix from rank " << rank << std::endl;
    print_matrix(workload, nranks, nranks);

    delete[] workload;

    // // Tests
    // constexpr size_t buff_size = 16 * 1024 * 1024 + 4 * 8;
    // test_NetAllGather(comm, connections, rank, nranks, buff_size, true, false);

    // // Performance
    // constexpr int warmup_iters = 10;
    // constexpr int perf_iters = 100;
    // constexpr size_t perf_buff_size = 1024 * 1024 * 1024;
    // for (int nblocks = 7; nblocks < 108; nblocks *= 2) {
    //     perf_NetAllGather(comm, connections, rank, nranks, perf_buff_size, true,
    //                       nblocks, 256, warmup_iters, perf_iters);
    // }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }

    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}