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
#include <iomanip>


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
    printf("error when executing cmd: %u, %s\n",cmd,      \
            hipGetErrorString(res));                   \
    exit(1);                                      \
  }                                                 \
} while(0)

void uniform_distribution(uint * workload, uint nrank, uint mean){
    uint bound = mean * 2;
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


void print_sendbuffs(void * send_buff, uint MAX_BUFFER_SIZE_PER_RANK, uint dim, uint gpu_n, uint rank){
    std::cout << "send buffs: BUFFER_SIZE_PER_RANK " << MAX_BUFFER_SIZE_PER_RANK << ", rank " << rank << ", dim: " << dim << std::endl;
    for (uint j = 0; j < dim; j++){
        for (uint k = 0; k < gpu_n; k++){
            for (uint z = 0; z < MAX_BUFFER_SIZE_PER_RANK; z++){
                int32_t * ptr = (int32_t *) send_buff + j * gpu_n * MAX_BUFFER_SIZE_PER_RANK + k * MAX_BUFFER_SIZE_PER_RANK + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << "|";
        }
        std::cout << "  ";
    }
    std::cout << dec << std::endl;

}

void print_recvbuffs(void * recv_buff, uint MAX_BUFFER_SIZE_PER_RANK, uint dim, uint rank){
    std::cout << "recv buffs: BUFFER_SIZE_PER_RANK " << MAX_BUFFER_SIZE_PER_RANK << ", rank " << rank << ", dim: " << dim << std::endl;
    for (uint j = 0; j < dim; j++){
        for (uint z = 0; z < MAX_BUFFER_SIZE_PER_RANK; z++){
            int32_t * ptr = (int32_t *) recv_buff + j * MAX_BUFFER_SIZE_PER_RANK + z;
            std::cout << hex << std::setfill('0') << std::setw(8)<< *ptr;
        }
        std::cout << " ";
    }
    std::cout << dec << std::endl;
}

struct alltoall_parameters allocate_device_memory(uint MAX_BUFFER_SIZE_PER_RANK, uint server_n, uint gpu_n){
    uint dim = server_n * gpu_n;
    // test performance with int32 data type
    uint data_type_size = sizeof(int32_t);
    void * verifybuff, * sendbuff, * recvbuff, * tempbuff;
    size_t * sendcount, * sendpos, * recvcount, *recvpos;
    RCCLCHECK(hipMalloc((void **)&sendbuff, gpu_n * dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMalloc((void **)&recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMalloc((void **)&tempbuff, 2 * gpu_n * gpu_n * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMallocManaged((void **)&verifybuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    RCCLCHECK(hipMallocManaged((void **)&sendcount, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged((void **)&sendpos, gpu_n * dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged((void **)&recvcount, dim * sizeof(size_t)));
    RCCLCHECK(hipMallocManaged((void **)&recvpos, dim * sizeof(size_t)));
    RCCLCHECK(hipDeviceSynchronize());
    struct alltoall_parameters ret = {
        .sendbuff = sendbuff,
        .recvbuff = recvbuff,
        .tempbuff = tempbuff,
        .verifybuff = verifybuff,
        .sendcount = sendcount,
        .sendpos = sendpos,
        .recvcount = recvcount,
        .recvpos = recvpos,
        .sched = NULL
    };
    return ret;
}

__global__ void identity_kernel(int32_t *sendbuff, size_t count) {
    int i0 = blockIdx.x*(count/gridDim.x);
    i0 += blockIdx.x < count%gridDim.x ? blockIdx.x : count%gridDim.x;
    int i1 = (blockIdx.x+1)*(count/gridDim.x);
    i1 += blockIdx.x+1 < count%gridDim.x ? blockIdx.x+1 : count%gridDim.x;
    int i = i0 + threadIdx.x;
    while(i < i1) {
        sendbuff[i] = sendbuff[i] * 1;
        i += blockDim.x;
    }
}

void initialize_buffer(struct alltoall_parameters * param, uint* workload, uint rank, uint server_n, uint gpu_n){
    uint dim = gpu_n * server_n;
    uint local_gpu_id = rank % gpu_n;
    uint data_type_size = sizeof(int32_t),
        MAX_BUFFER_SIZE_PER_RANK = param->sched->MAX_BUFFER_SIZE_PER_RANK;
    RCCLCHECK(hipSetDevice(rank % gpu_n));
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
    // print_sendbuffs(host_sendbuff, MAX_BUFFER_SIZE_PER_RANK, dim, gpu_n, rank);
    // print_recvbuffs(param->verifybuff, MAX_BUFFER_SIZE_PER_RANK, dim, rank);
    RCCLCHECK(hipDeviceSynchronize());
    // int sendbuff_sz = dim * gpu_n * MAX_BUFFER_SIZE_PER_RANK;
    // int block_n = std::min<int>(32, (sendbuff_sz + 4*512-1)/(4*512));
    // identity_kernel<<<block_n, 512, 0, hipStreamDefault>>>((int32_t*)param->sendbuff, sendbuff_sz);
    // std:: cout << "rank " << rank << ": sendcount:";
    // for (uint i = 0; i < dim; i ++){
    //     std::cout << param->sendcount[i * gpu_n + local_gpu_id] << " ";
    // }
    // std::cout << endl;
    // std:: cout << "rank " << rank << ": recvcount:";
    // for (uint i = 0; i < dim; i ++){
    //     std::cout << param->recvcount[i] << " ";
    // }
    // std::cout << endl;

}


void reset_buffer_counter(struct alltoall_parameters * param, uint server_n, uint gpu_n){
    uint dim = server_n * gpu_n;
    uint data_type_size = sizeof(int32_t),
    MAX_BUFFER_SIZE_PER_RANK = param->sched->MAX_BUFFER_SIZE_PER_RANK;
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



bool verify_correctness_v2(struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream){
    uint dim = param->sched->gpu_n * param->sched->server_n,
        MAX_BUFFER_SIZE_PER_RANK = param->sched->MAX_BUFFER_SIZE_PER_RANK;
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
    // print_recvbuffs(host_recvbuff, MAX_BUFFER_SIZE_PER_RANK, dim, param->sched->rankid);
    return (0 == memcmp(host_recvbuff, param->verifybuff,  dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
}

struct perf_test_ret_t{
    double algbw;
    double time;
};

struct perf_test_ret_t perf_v0(uint warmup_iters, uint perf_iters, struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(ncclAllToAllv0(
            param->sched->rankid,
            param->sched->gpu_n,
            param->sched->MAX_BUFFER_SIZE_PER_RANK,
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
            param->sched->MAX_BUFFER_SIZE_PER_RANK,
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
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9 / (param->sched->gpu_n * param->sched->server_n);

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
    return r;
}


struct perf_test_ret_t perf_v2(uint warmup_iters, uint perf_iters, struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
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
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(start_event));
    for (int i = 0; i < perf_iters; ++i) {
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
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(end_event));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9 / (param->sched->gpu_n * param->sched->server_n);

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
    return r;
}


bool verify_correctness_v0(struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream){
    uint dim = param->sched->gpu_n * param->sched->server_n,
    MAX_BUFFER_SIZE_PER_RANK = param->sched->MAX_BUFFER_SIZE_PER_RANK;
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(ncclAllToAllv0(
        param->sched->rankid,
        param->sched->gpu_n,
        param->sched->MAX_BUFFER_SIZE_PER_RANK,
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
    hipError_t err;
    bool finished = false;
    while(!finished){
        err = hipStreamQuery(stream);
        if (err == hipSuccess) {
            finished = true;
            break;
        }
    }
    uint data_type_size = sizeof(int32_t);
    void * host_recvbuff;
    hipMallocManaged(&host_recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    RCCLCHECK(hipMemcpy(host_recvbuff, param->recvbuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size, hipMemcpyDeviceToHost));
    // print_recvbuffs(host_recvbuff, MAX_BUFFER_SIZE_PER_RANK, dim, param->sched->rankid);
    return (0 == memcmp(host_recvbuff, param->verifybuff,  dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
}

struct max_sum_ret_t{
    uint max;
    uint sum;
};

struct max_sum_ret_t max_sum_matrix(uint * workload, uint dim){
    uint max = 0, sum = 0;
    for (uint i = 0; i < dim; i ++){
        for (uint j = 0; j < dim; j++){
            max = MAX(max, workload[i *dim + j]);
            sum += workload[i *dim + j];
        }
    }
    struct max_sum_ret_t r = {.max = max, .sum = sum};
    return r;
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
    if (rank == 0) uniform_distribution(workload, nranks, 1024000);
    MPI_Bcast(workload, nranks * nranks * sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
    // std::cout << "Rank " << rank << " receive demand matrix" << std::endl;
    struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
    uint buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
    if (rank == 0) print_matrix(workload, nranks, nranks);

    // scheduler is deterministic given a workload
    uint server_n = 2, gpu_n = 8;
    struct GlobalScheduler scheduler;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload);
    run_scheduler(&scheduler);
    scheduler.sched->rankid = rank;
    scheduler.sched->MAX_BUFFER_SIZE_PER_RANK = workload_max_sum.max;
    // std::cout << "Rank " << rank << " scheduling succeeds" << std::endl;

    RCCLCHECK(hipSetDevice(rank % gpu_n));
    struct alltoall_parameters param = allocate_device_memory(scheduler.sched->MAX_BUFFER_SIZE_PER_RANK, server_n, gpu_n);
    param.sched = scheduler.sched;
    initialize_buffer(&param, workload, rank, server_n, gpu_n);
    // std::cout << "Rank " << rank << " buffers initialization succeeds" << std::endl;


    // verify correctness
    if (rank == 0) std::cout << "TESTING CORRECTNESS" << std::endl;
    int correctness_this_rank = verify_correctness_v2(&param, comm, stream);
    int * correctness;
    if (rank == 0) correctness = new int[nranks];
    MPI_Gather(&correctness_this_rank, 1, MPI_INT, correctness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int correctness_n = 0;
        for (uint i = 0; i < nranks; i++) correctness_n += correctness[i];
        std::cout << "CORRECTNESS: " << ((correctness_n == nranks) ? "succeed" : "fail") << std::endl;
    }


    uint warmup_iters = 1, perf_iters = 1;
    if (rank == 0) std::cout << "TESTING PERFORMANCE: warmup iters: " << warmup_iters <<", test iters: " << perf_iters << std::endl;
    struct perf_test_ret_t v0_ret = perf_v0(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    struct perf_test_ret_t v2_ret = perf_v2(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    if (rank == 0){
        std::cout << "Buff size: " << buff_size << " B, v0 algbw: " << v0_ret.algbw << " GBps, v0 time: " << v0_ret.time << " ms, v2 algbw: " << v2_ret.algbw << " GBps, v2 time: " << v2_ret.time <<  " ms, Speed up: " << v2_ret.algbw / v0_ret.algbw << std::endl;
    }

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