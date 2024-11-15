#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fast_alltoall/alltoall_global_scheduler.h>
#include <fast_alltoall/alltoall_local_scheduler.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <ctime>
#include <nccl.h>

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

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("error when executing cmd:%u",cmd);      \
    exit(1);                                      \
  }                                                 \
} while(0)


#define RCCLCHECK(cmd) do {                         \
  hipError_t res = cmd;                           \
  if (res != hipSuccess) {                         \
    printf("error when executing cmd:%u",cmd);      \
    exit(1);                                      \
  }                                                 \
} while(0)

void uniform_distribution(uint * workload, uint nrank, uint mean){
    uint bound = MIN(MAX_BUFFER_SIZE_PER_RANK, mean * 2);
    for (uint i = 0; i < nrank; i++){
        for (uint j = 0; j < nrank; j++){
            workload[i * nrank + j] = rand() % bound;
        }
    }
    // clean the diagnal
    for (uint i = 0; i < nrank; i++){
         workload[i * nrank + i] = 0;
    }
}


struct alltoall_parameters{
    void * sendbuff;
    void * recvbuff;
    void * tempbuff;
    void * syncbuff;
    void * verifybuff;
    size_t * sendcount;
    size_t * sendpos;
    size_t * recvcount;
    size_t * recvpos;
    struct scheduling_result_t * sched;
};

void allocate_device_memory(struct alltoall_parameters * param, uint server_n, uint gpu_n){
    uint dim = server_n * gpu_n;
    // test performance with int32 data type
    uint data_type_size = sizeof(int32_t);
    RCCLCHECK(hipMalloc(&param->sendbuff, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMalloc(&param->recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMalloc(&param->tempbuff, 2 * gpu_n * gpu_n * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMalloc(&param->syncbuff, 2 * dim * data_type_size));
    param->verifybuff = malloc(dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    RCCLCHECK(hipMalloc(&param->sendcount, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMalloc(&param->sendpos, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMalloc(&param->recvcount, dim * sizeof(size_t)));
    RCCLCHECK(hipMalloc(&param->recvpos, dim * sizeof(size_t)));
    // RCCLCHECK(hipMalloc(&param->sched, sizeof(struct scheduling_result_t)));
}

void initialize_buffer(struct alltoall_parameters * param, uint* workload, uint rank, uint server_n, uint gpu_n){
    uint dim = gpu_n * server_n;
    uint local_gpu_id = rank % gpu_n;
    size_t * host_sendcount = new size_t[ gpu_n * dim ];
    size_t * host_recvcount = new size_t[ dim ];
    uint data_type_size = sizeof(int32_t);
    memset(host_sendcount, 0, sizeof(size_t) * gpu_n * dim);
    memset(host_recvcount, 0, sizeof(size_t) * dim);
    for (uint i = 0; i < dim; i++){
        host_sendcount[i * gpu_n + local_gpu_id] = workload[rank * dim + i];
        host_recvcount[i] = workload[i * dim + rank];
    }
    RCCLCHECK(hipMemcpy(param->sendcount, host_sendcount, sizeof(size_t) * gpu_n * dim, hipMemcpyHostToDevice));
    RCCLCHECK(hipMemcpy(param->recvcount, host_recvcount, sizeof(size_t) * dim, hipMemcpyHostToDevice));
    RCCLCHECK(hipMemset(param->sendpos, 0, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMemset(param->recvpos, 0, dim * sizeof(size_t)));
    RCCLCHECK(hipMemset(param->recvbuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    int32_t * host_sendbuff = new int32_t[gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK];
    memset((void *)host_sendbuff, 0, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    for (uint i = 0; i < dim; i ++){
        for (uint sz = 0; sz < host_sendcount[i * gpu_n + local_gpu_id]; sz++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + (sz & 0xffff);
            host_sendbuff[i * gpu_n * MAX_BUFFER_SIZE_PER_RANK + local_gpu_id * MAX_BUFFER_SIZE_PER_RANK + sz] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(param->sendbuff, (void*)host_sendbuff, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size,hipMemcpyHostToDevice));
    memset(param->verifybuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    for (uint i = 0; i < dim; i++){
        for (uint sz = 0; sz < workload[i * dim + rank]; sz++){
            int32_t unique_data = ((i & 0xff) << 24) + ((rank & 0xff) << 16) + (sz & 0xffff);
            int32_t * vb = (int32_t *)param->verifybuff;
            vb[i * MAX_BUFFER_SIZE_PER_RANK + sz] = unique_data;
        }
    }
    delete[] host_sendcount;
    delete[] host_recvcount;
    delete[] host_sendbuff;
}

void free_buffer(struct alltoall_parameters * param){
    RCCLCHECK(hipFree(param->sendbuff));
    RCCLCHECK(hipFree(param->recvbuff));
    RCCLCHECK(hipFree(param->tempbuff));
    RCCLCHECK(hipFree(param->syncbuff));
    RCCLCHECK(hipFree(param->verifybuff));
    RCCLCHECK(hipFree(param->sendcount));
    RCCLCHECK(hipFree(param->sendpos));
    RCCLCHECK(hipFree(param->recvcount));
    RCCLCHECK(hipFree(param->recvpos));
    // RCCLCHECK(hipFree(param->sched));
}


void verify_correctness(struct alltoall_parameters * param, uint rank, uint dim, ncclComm_t comm, hipStream_t stream){
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(ncclAllToAllv2(rank,
        param->sendbuff,
        param->sendcount,
        param->sendpos,
        param->recvbuff,
        param->recvcount,
        param->recvpos,
        param->tempbuff,
        param->syncbuff,
        param->sched,
        ncclInt32,
        comm,
        stream));
    RCCLCHECK(hipDeviceSynchronize());
    uint data_type_size = sizeof(int32_t);
    void * host_recvbuff = malloc(dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    RCCLCHECK(hipMemcpy(host_recvbuff, param->recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size, hipMemcpyDeviceToHost));
    std::cout << "verification: " << (0 == memcmp(host_recvbuff, param->verifybuff,  dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size)) << std::endl;
}

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
    hipStream_t stream;
    RCCLCHECK(hipStreamCreate(&stream));

    //let rank 0 process generate a random workload
    uint * workload = new uint[nranks * nranks];
    if (rank == 0) uniform_distribution(workload, nranks, 1024);
    MPI_Bcast(workload, nranks * nranks * sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "receive demand matrix from rank " << rank << std::endl;
    print_matrix(workload, nranks, nranks);
    MPI_Barrier(MPI_COMM_WORLD);


    // scheduler is deterministic given a workload
    struct GlobalScheduler scheduler;
    uint server_n = 2, gpu_n = 8;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload);
    // run_scheduler(&scheduler);

    // struct alltoall_parameters param;
    // allocate_device_memory(&param, server_n, gpu_n);
    // initialize_buffer(&param, workload, rank, server_n, gpu_n);
    // param.sched = scheduler.sched;
    // // RCCLCHECK(hipMemcpy(param->sched, scheduler.sched, sizeof(struct scheduling_result_t), hipMemcpyHostToDevice));

    // // verify correctness
    // verify_correctness(&param, rank, gpu_n * server_n, comm, stream);

    // free_buffer(&param);

    delete[] workload;
    free_global_scheduler(&scheduler);


    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    RCCLCHECK(hipStreamDestroy(stream));
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}