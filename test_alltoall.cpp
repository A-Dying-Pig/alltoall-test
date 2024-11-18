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
    RCCLCHECK(hipMallocManaged(&param->verifybuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMallocManaged(&param->sendcount, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged(&param->sendpos, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged(&param->recvcount, dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged(&param->recvpos, dim * sizeof(size_t)));
    // RCCLCHECK(hipMalloc(&param->sched, sizeof(struct scheduling_result_t)));
}

void initialize_buffer(struct alltoall_parameters * param, uint* workload, uint rank, uint server_n, uint gpu_n){
    uint dim = gpu_n * server_n;
    uint local_gpu_id = rank % gpu_n;
    uint data_type_size = sizeof(int32_t);
    // std:: cout << "Rank " << rank <<" local id: " << local_gpu_id << " dim: " << dim << std::endl;
    RCCLCHECK(hipMemset(param->sendcount, 0, sizeof(size_t) * gpu_n * dim));
    RCCLCHECK(hipMemset(param->recvcount, 0, sizeof(size_t) * dim));
    for (uint i = 0; i < dim; i++){
        param->sendcount[i * gpu_n + local_gpu_id] = workload[rank * dim + i];
        param->recvcount[i] = workload[i * dim + rank];
    }
    RCCLCHECK(hipMemset(param->sendpos, 0, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMemset(param->recvpos, 0, dim * sizeof(size_t)));
    RCCLCHECK(hipMemset(param->recvbuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    int32_t * host_sendbuff = new int32_t[gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK];
    memset((void *)host_sendbuff, 0, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    for (uint i = 0; i < dim; i ++){
        for (uint sz = 0; sz < param->sendcount[i * gpu_n + local_gpu_id]; sz++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + (sz & 0xffff);
            host_sendbuff[i * gpu_n * MAX_BUFFER_SIZE_PER_RANK + local_gpu_id * MAX_BUFFER_SIZE_PER_RANK + sz] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(param->sendbuff, (void*)host_sendbuff, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size, hipMemcpyHostToDevice));
    memset(param->verifybuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    for (uint i = 0; i < dim; i++){
        for (uint sz = 0; sz < workload[i * dim + rank]; sz++){
            int32_t unique_data = ((i & 0xff) << 24) + ((rank & 0xff) << 16) + (sz & 0xffff);
            int32_t * vb = (int32_t *)param->verifybuff;
            vb[i * MAX_BUFFER_SIZE_PER_RANK + sz] = unique_data;
        }
    }
}


void reset_buffer_counter(struct alltoall_parameters * param, uint server_n, uint gpu_n){
    uint dim = server_n * gpu_n;
    uint data_type_size = sizeof(int32_t);
    RCCLCHECK(hipMemset(param->recvbuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMemset(param->sendpos, 0, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMemset(param->recvpos, 0, dim * sizeof(size_t)));
}

void free_buffer(struct alltoall_parameters * param){
    RCCLCHECK(hipFree(param->sendbuff));
    RCCLCHECK(hipFree(param->recvbuff));
    RCCLCHECK(hipFree(param->tempbuff));
    RCCLCHECK(hipFree(param->verifybuff));
    RCCLCHECK(hipFree(param->sendcount));
    RCCLCHECK(hipFree(param->sendpos));
    RCCLCHECK(hipFree(param->recvcount));
    RCCLCHECK(hipFree(param->recvpos));
    // RCCLCHECK(hipFree(param->sched));
}


void verify_correctness_v2(struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream){
    uint dim = param->sched->gpu_n * param->sched->server_n;
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(ncclAllToAllv2(
        param->sendbuff,
        param->sendcount,
        param->sendpos,
        param->recvbuff,
        param->recvcount,
        param->recvpos,
        param->tempbuff,
        param->sched,
        ncclInt32,
        comm,
        stream));
    RCCLCHECK(hipDeviceSynchronize());
    uint data_type_size = sizeof(int32_t);
    void * host_recvbuff;
    hipMallocManaged(&host_recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    RCCLCHECK(hipMemcpy(host_recvbuff, param->recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size, hipMemcpyDeviceToHost));
    std::cout << "Rank " << param->sched->rankid << " V2 correctness: " << (0 == memcmp(host_recvbuff, param->verifybuff,  dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size) ? "True" : "False") << std::endl;
}

void perf_v2(struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);

    uint warmup_iters = 20, perf_iters = 100;
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(ncclAllToAllv0(
            param->sched->rankid,
            param->sched->gpu_n,
            param->sendbuff,
            param->sendcount,
            param->sendpos,
            param->recvbuff,
            param->recvcount,
            param->recvpos,
            ncclInt32,
            comm,
            stream));
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(start_event));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(ncclAllToAllv0(
            param->sched->rankid,
            param->sched->gpu_n,
            param->sendbuff,
            param->sendcount,
            param->sendpos,
            param->recvbuff,
            param->recvcount,
            param->recvpos,
            ncclInt32,
            comm,
            stream));
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(end_event));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9;

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    if (param->sched->rankid == 0) {
        std::cout << "AlltoAllv2 perf ("
                  << ",buff_size=" << buff_size
                  << ") finished: time=" << avg_time * 1e3 << "us "
                  << "algbw=" << algbw << "GBps" << std::endl;
    }


}

void verify_correctness_v0(struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream){
    uint dim = param->sched->gpu_n * param->sched->server_n;
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(ncclAllToAllv0(
        param->sched->rankid,
        param->sched->gpu_n,
        param->sendbuff,
        param->sendcount,
        param->sendpos,
        param->recvbuff,
        param->recvcount,
        param->recvpos,
        ncclInt32,
        comm,
        stream));
    RCCLCHECK(hipDeviceSynchronize());
    uint data_type_size = sizeof(int32_t);
    void * host_recvbuff;
    hipMallocManaged(&host_recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    RCCLCHECK(hipMemcpy(host_recvbuff, param->recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size, hipMemcpyDeviceToHost));
    std::cout << "V0 correctness: " << (0 == memcmp(host_recvbuff, param->verifybuff,  dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size) ? "True" : "False") << std::endl;
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
    // Print off a hello world message
    int dev_n = 0;
    RCCLCHECK(hipGetDeviceCount(&dev_n));
    RCCLCHECK(hipSetDevice(rank % dev_n));
    std::cout << "Rank " << rank << " out of " << nranks << " successfully set device" << std::endl;

    // Initialize Communicator
    ncclComm_t comm;
    ncclUniqueId ncclId;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&ncclId));
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCLCHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));
    hipStream_t stream;
    RCCLCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));


    //let rank 0 process generate a random workload
    uint * workload = new uint[nranks * nranks];
    if (rank == 0) uniform_distribution(workload, nranks, 1024);
    MPI_Bcast(workload, nranks * nranks * sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " receive demand matrix" << std::endl;
    uint buff_size = 0;
    for (uint i = 0; i < nranks; i++){
        for (uint j = 0; j < nranks; j++){
            buff_size += workload[i * nranks + j];
        }
    }
    buff_size *= sizeof(int32_t);
    // print_matrix(workload, nranks, nranks);

    // scheduler is deterministic given a workload
    uint server_n = 2, gpu_n = 8;
    struct GlobalScheduler scheduler;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload);
    run_scheduler(&scheduler);
    scheduler.sched->rankid = rank;
    std::cout << "Rank " << rank << " finishes scheduling!" << std::endl;

    struct alltoall_parameters param;
    allocate_device_memory(&param, server_n, gpu_n);
    initialize_buffer(&param, workload, rank, server_n, gpu_n);
    param.sched = scheduler.sched;
    std::cout << "Rank " << rank << " initializes buffers!" << std::endl;


    // verify correctness
    std::cout << "TESTING CORRECTNESS, rank: " << rank << std::endl;
    // verify_correctness_v2(&param, comm, stream);
    // reset_buffer_counter(&param, server_n, gpu_n);
    verify_correctness_v0(&param, comm, stream);

    std::cout << "TESTING PERFORMANCE, rank: " << rank << std::endl;
    // perf_v2(&param, comm, stream, buff_size);

    free_buffer(&param);

    delete[] workload;
    free_global_scheduler(&scheduler);

    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    RCCLCHECK(hipStreamDestroy(stream));
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}