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
#include <chrono>
using namespace std::chrono;


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
            // workload[i * nrank + j] = bound;
            // workload[i * nrank + j] = (rand() % bound + 0x1ff) & 0xfffffe00;
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


struct buffer_ptr_t{
    void * sendbuff;
    void * recvbuff;
    void * lbsend;
    void * lbrecv;
    void * crosbuff;
    void * rstrbuff;
    void * verifybuff;
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

void print_buffs(struct buffer_ptr_t * bufs, struct buffer_parameter_t * buff_parameter, uint rank, uint server_n, uint gpu_n){

    std::cout << "----------rank " << rank << "-----------" << std::endl;

    std::cout << "sendbuff: " << std::endl;
    uint dim = server_n * gpu_n;
    for (uint j = 0; j < dim; j++){
        for (uint k = 0; k < gpu_n; k++){
            uint sz = buff_parameter -> sendbuff_region[j].src_gpu_sz[k];
            for (uint z = 0; z < sz; z++){
                int32_t * ptr = (int32_t *) bufs->sendbuff + buff_parameter -> sendbuff_region[j].src_gpu_disp[k] + z;
                std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
            }
            std::cout << "|";
        }
        std::cout << "  ";
    }

    std::cout << "lbsend: " << std::endl;
    for (uint i = 0; i < gpu_n; i ++){
        for (uint j = 0; j < gpu_n; j++){
            for (uint s = 0; s < server_n; s++){
                uint sz = buff_parameter -> lbsend_area[i].dst_gpu_region[j].server_sz[s];
                for (uint z = 0; z < sz; z ++){
                    int32_t * ptr = (int32_t *) bufs->lbsend + buff_parameter -> lbsend_area[i].dst_gpu_region[j].server_disp[s] + z;
                    std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
                }
                std::cout << "+";
            }
            std::cout << "|";
        }
        std::cout << " ";
    }

    std::cout << "recvbuff: " << std::endl;
    for (uint j = 0; j < dim; j++){
        uint sz = buff_parameter -> recvbuff_sz[j];
        for (uint z = 0; z < sz; z++){
            int32_t * ptr = (int32_t *) bufs->recvbuff +  buff_parameter -> recvbuff_disp[j] + z;
            std::cout << hex << std::setfill('0') << std::setw(8) << * ptr;
        }
        std::cout << "  ";
    }

    std::cout << dec << std::endl;
}


struct buffer_ptr_t init_buffers(struct buffer_parameter_t * buff_parameter, uint server_n, uint gpu_n, uint rank){
    // allocate memory
    uint data_type_size = sizeof(int32_t);
    void * sendbuff = NULL, * recvbuff = NULL, * lbsend = NULL, * lbrecv = NULL, * crosbuff = NULL, * rstrbuff = NULL, * verifybuff = NULL;
    RCCLCHECK(hipMalloc((void **)&sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size));
    RCCLCHECK(hipMalloc((void **)&recvbuff, buff_parameter -> recvbuff_total_sz * data_type_size));
    if (buff_parameter -> lbsend_total_sz) RCCLCHECK(hipMalloc((void **)&lbsend, buff_parameter -> lbsend_total_sz * data_type_size));
    if (buff_parameter -> lbrecv_total_sz) RCCLCHECK(hipMalloc((void **)&lbrecv, buff_parameter -> lbrecv_total_sz * data_type_size));
    if (buff_parameter -> crosbuff_total_sz) RCCLCHECK(hipMalloc((void **)&crosbuff, buff_parameter -> crosbuff_total_sz * data_type_size));
    if (buff_parameter -> rstrbuff_total_sz) RCCLCHECK(hipMalloc((void **)&rstrbuff, buff_parameter -> rstrbuff_total_sz * data_type_size));
    verifybuff = malloc(buff_parameter -> recvbuff_total_sz * data_type_size);
    // initialize memory
    RCCLCHECK(hipMemset(recvbuff, 0, buff_parameter -> recvbuff_total_sz));


    uint local_rank_id = rank % gpu_n;
    uint dim = server_n * gpu_n;
    if (buff_parameter -> lbsend_total_sz){
        int32_t * host_lbsend = new int32_t[buff_parameter -> lbsend_total_sz];
        memset(host_lbsend, 0, buff_parameter -> lbsend_total_sz * data_type_size);
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu ++){
            for (uint dst_gpu = 0; dst_gpu < gpu_n; dst_gpu ++){
                for (uint s = 0; s < server_n; s++){
                    uint disp = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_disp[s];
                    uint sz = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_sz[s];
                    uint offset = buff_parameter -> lbsend_area[local_gpu].dst_gpu_region[dst_gpu].server_offset[s];
                    uint dst_gpu_global_id = s * gpu_n + dst_gpu;
                    for (uint z = 0; z < sz; z ++){
                        uint unique_data =  ((rank & 0xff) << 24) + ((dst_gpu_global_id & 0xff) << 16) + ((z + offset) & 0xffff);
                        host_lbsend[disp + z] = unique_data;
                    }
                }
            }
        }
        RCCLCHECK(hipMemcpy(lbsend, (void *) host_lbsend, buff_parameter -> lbsend_total_sz * data_type_size, hipMemcpyHostToDevice));
    }

    int32_t * host_sendbuff = new int32_t[buff_parameter -> sendbuff_total_sz];
    memset(host_sendbuff, 0, buff_parameter -> sendbuff_total_sz * data_type_size);
    for (uint i = 0; i < dim; i++){
        uint disp = buff_parameter -> sendbuff_region[i].src_gpu_disp[local_rank_id];
        uint sz = buff_parameter -> sendbuff_region[i].src_gpu_sz[local_rank_id];
        uint offset = buff_parameter -> sendbuff_region[i].src_gpu_offset[local_rank_id];
        for (uint j = 0; j < sz; j++){
            int32_t unique_data = ((rank & 0xff) << 24) + ((i & 0xff) << 16) + ((j + offset) & 0xffff);
            host_sendbuff[disp + j] = unique_data;
        }
    }
    RCCLCHECK(hipMemcpy(sendbuff, (void *) host_sendbuff, buff_parameter -> sendbuff_total_sz * data_type_size, hipMemcpyHostToDevice));
    memset(verifybuff, 0,  buff_parameter -> recvbuff_total_sz);
    for (uint i = 0; i < dim; i++){
        uint disp = buff_parameter -> recvbuff_disp[i];
        uint sz = buff_parameter -> recvbuff_sz[i];
        for (uint j = 0; j < sz; j++){
            int32_t unique_data = ((i & 0xff) << 24) + ((rank & 0xff) << 16) + (j & 0xffff);
            int32_t * vb = (int32_t *) verifybuff;
            vb [disp + j] = unique_data;
        }
    }

    RCCLCHECK(hipDeviceSynchronize());

    struct buffer_ptr_t bufs = {
        .sendbuff = sendbuff,
        .recvbuff = recvbuff,
        .lbsend = lbsend,
        .lbrecv = lbrecv,
        .crosbuff = crosbuff,
        .rstrbuff = rstrbuff,
        .verifybuff = verifybuff
    };
    return bufs;
}



void free_buffers( struct buffer_ptr_t * bufs){
    RCCLCHECK(hipFree(bufs -> sendbuff));
    RCCLCHECK(hipFree(bufs -> recvbuff));
    if (bufs -> lbsend) RCCLCHECK(hipFree(bufs -> lbsend));
    if (bufs -> lbrecv) RCCLCHECK(hipFree(bufs -> lbrecv));
    if (bufs -> crosbuff) RCCLCHECK(hipFree(bufs -> crosbuff));
    if (bufs -> rstrbuff) RCCLCHECK(hipFree(bufs -> rstrbuff));
    free (bufs -> verifybuff);

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
    verifybuff = malloc(dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    sendcount =  (size_t *) malloc(gpu_n * dim * sizeof(size_t));
    sendpos = (size_t *) malloc(gpu_n * dim * sizeof(size_t));
    recvcount = (size_t *) malloc( dim * sizeof(size_t) );
    recvpos = (size_t *) malloc( dim * sizeof(size_t));
    // RCCLCHECK(hipMallocManaged((void **)&verifybuff, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    // RCCLCHECK(hipMallocManaged((void **)&sendcount, gpu_n * dim * sizeof(size_t)));
    // RCCLCHECK(hipMallocManaged((void **)&sendpos, gpu_n * dim * sizeof(size_t)));
    // RCCLCHECK(hipMallocManaged((void **)&recvcount, dim * sizeof(size_t)));
    // RCCLCHECK(hipMallocManaged((void **)&recvpos, dim * sizeof(size_t)));
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
    memset(param->sendcount, 0, sizeof(size_t) * gpu_n * dim);
    memset(param->recvcount, 0, sizeof(size_t) * dim);
    // RCCLCHECK(hipMemset(param->sendcount, 0, sizeof(size_t) * gpu_n * dim));
    // RCCLCHECK(hipMemset(param->recvcount, 0, sizeof(size_t) * dim));
    for (uint i = 0; i < dim; i++){
        param->sendcount[i * gpu_n + local_gpu_id] = workload[rank * dim + i];
        param->recvcount[i] = workload[i * dim + rank];
    }
    memset(param->sendpos, 0, gpu_n * dim * sizeof(size_t));
    memset(param->recvpos, 0, dim * sizeof(size_t));
    // RCCLCHECK(hipMemset(param->sendpos, 0, gpu_n * dim * sizeof(size_t)));
    // RCCLCHECK(hipMemset(param->recvpos, 0, dim * sizeof(size_t)));
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
    int sendbuff_sz = dim * gpu_n * MAX_BUFFER_SIZE_PER_RANK;
    int block_n = std::min<int>(32, (sendbuff_sz + 4*512-1)/(4*512));
    identity_kernel<<<block_n, 512, 0, hipStreamDefault>>>((int32_t*)param->sendbuff, sendbuff_sz);

    int recvbuff_sz = dim * MAX_BUFFER_SIZE_PER_RANK;
    block_n = std::min<int>(32, (recvbuff_sz + 4*512-1)/(4*512));
    identity_kernel<<<block_n, 512, 0, hipStreamDefault>>>((int32_t*)param->recvbuff, recvbuff_sz);



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
    memset(param->recvbuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size);
    memset(param->sendpos, 0, gpu_n * dim * sizeof(size_t));
    memset(param->recvpos, 0, dim * sizeof(size_t));
    // RCCLCHECK(hipMemset(param->recvbuff, 0, dim * MAX_BUFFER_SIZE_PER_RANK * data_type_size));
    // RCCLCHECK(hipMemset(param->sendpos, 0, gpu_n * dim * sizeof(size_t)));
    // RCCLCHECK(hipMemset(param->recvpos, 0, dim * sizeof(size_t)));
}

void free_buffer(struct alltoall_parameters * param){
    RCCLCHECK(hipFree(param->sendbuff));
    RCCLCHECK(hipFree(param->recvbuff));
    RCCLCHECK(hipFree(param->tempbuff));
    free(param->verifybuff);
    free(param->sendcount);
    free(param->sendpos);
    free(param->recvcount);
    free(param->recvpos);
    // RCCLCHECK(hipFree(param->verifybuff));
    // RCCLCHECK(hipFree(param->sendcount));
    // RCCLCHECK(hipFree(param->sendpos));
    // RCCLCHECK(hipFree(param->recvcount));
    // RCCLCHECK(hipFree(param->recvpos));
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



ncclResult_t
ncclAllToAllv4(void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream[]){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = sched->rankid,
        local_rank_id = sched->rankid % sched->gpu_n,
        server_id = sched->rankid / sched->gpu_n,
        server_n = sched->server_n,
        gpu_n = sched->gpu_n,
        rankid = sched->rankid,
        MAX_BUFFER_SIZE_PER_RANK = sched->MAX_BUFFER_SIZE_PER_RANK;
    ncclResult_t ret, state;

    NCCLCHECK(ncclGroupStart());
    for (uint r = 0; r < gpu_n; r++){
        uint global_comm_gpu = server_id * gpu_n + r;
        size_t send_data_sz = ((sched -> intrinsic_ata)[server_id][local_rank_id * gpu_n + r] + 0x1ff) % 0xfffffe00;
        uint sendbuf_offset = global_comm_gpu * gpu_n + local_rank_id;
        void * src_ptr = (char *)sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
        NCCLCHECK(ncclSend(
            src_ptr,
            MAX_BUFFER_SIZE_PER_RANK,
            datatype,
            global_comm_gpu,
            comm,
            stream[0]
        ));
        size_t recv_data_sz = ((sched -> intrinsic_ata)[server_id][r * gpu_n + local_rank_id] + 0x1ff) % 0xfffffe00;
        void * dst_ptr = (char *)recvbuff + global_comm_gpu * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
        NCCLCHECK(ncclRecv(
            dst_ptr,
            MAX_BUFFER_SIZE_PER_RANK,
            datatype,
            global_comm_gpu,
            comm,
            stream[0]
        ));
        // printf("rankid: %u, server_id: %u, other local gpu: %u, send data sz: %lu, recv data sz: %lu\n", rankid, server_id, r, send_data_sz, recv_data_sz);
    }

    NCCLCHECK(ncclGroupEnd());

    // NCCLCHECK(ncclGroupStart());

    // for (uint dis = 1; dis < gpu_n; dis++){
    //     uint local_dst_gpu = (local_rank_id + dis) % gpu_n;
    //     uint global_dst_gpu = server_id * gpu_n  + local_dst_gpu;
    //     uint local_src_gpu = (local_rank_id + gpu_n - dis) % gpu_n;
    //     uint global_src_gpu =  server_id * gpu_n  + local_src_gpu;

    //     // std::cout << "rank : " << sched->rankid << ", dst gpu: " << local_dst_gpu << ", src gpu: " << local_src_gpu << std::endl;
    //     size_t send_data_sz = (sched -> intrinsic_ata)[server_id][local_rank_id * gpu_n + local_dst_gpu];
    //     size_t recv_data_sz = (sched -> intrinsic_ata)[server_id][local_src_gpu * gpu_n + local_rank_id];
    //     uint sendbuf_offset = global_dst_gpu * gpu_n + local_rank_id;
    //     void * src_ptr = (char *)sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
    //     NCCLCHECK(ncclSend(
    //         src_ptr,
    //         send_data_sz,
    //         datatype,
    //         global_dst_gpu,
    //         comm,
    //         stream
    //     ));
    //     void * dst_ptr = (char *)recvbuff + global_src_gpu * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
    //     NCCLCHECK(ncclRecv(
    //         dst_ptr,
    //         recv_data_sz,
    //         datatype,
    //         global_src_gpu,
    //         comm,
    //         stream
    //     ));
    // }
    // NCCLCHECK(ncclGroupEnd());

    return ncclSuccess;
}

ncclResult_t
ncclAllToAllv5(void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = sched->rankid,
        local_rank_id = sched->rankid % sched->gpu_n,
        server_id = sched->rankid / sched->gpu_n,
        server_n = sched->server_n,
        gpu_n = sched->gpu_n,
        rankid = sched->rankid,
        MAX_BUFFER_SIZE_PER_RANK = sched->MAX_BUFFER_SIZE_PER_RANK;
    ncclResult_t ret, state;

    NCCLCHECK(ncclGroupStart());
    uint global_comm_gpu = (server_id == 0 ? 1 : 0) * gpu_n + local_rank_id;
    // uint sendbuf_offset = global_comm_gpu * gpu_n + local_rank_id;
    NCCLCHECK(ncclSend(
        sendbuff,
        MAX_BUFFER_SIZE_PER_RANK,
        datatype,
        global_comm_gpu,
        comm,
        stream
    ));
    NCCLCHECK(ncclRecv(
        recvbuff,
        MAX_BUFFER_SIZE_PER_RANK,
        datatype,
        global_comm_gpu,
        comm,
        stream
    ));
    NCCLCHECK(ncclGroupEnd());

    return ncclSuccess;
}



ncclResult_t
ncclAllToAllv6(void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = sched->rankid,
        local_rank_id = sched->rankid % sched->gpu_n,
        server_id = sched->rankid / sched->gpu_n,
        server_n = sched->server_n,
        gpu_n = sched->gpu_n,
        rankid = sched->rankid,
        MAX_BUFFER_SIZE_PER_RANK = sched->MAX_BUFFER_SIZE_PER_RANK;
    ncclResult_t ret, state;

    if (global_rank_id == 0){
        void * src_ptr = (char *)sendbuff;
        NCCLCHECK(ncclSend(
            src_ptr,
            MAX_BUFFER_SIZE_PER_RANK * gpu_n,
            datatype,
            8,
            comm,
            stream
        ));
    }else if (global_rank_id == 8){
        void * dst_ptr = (char *)recvbuff;
        NCCLCHECK(ncclSend(
            dst_ptr,
            MAX_BUFFER_SIZE_PER_RANK * gpu_n,
            datatype,
            0,
            comm,
            stream
        ));
    }
    return ncclSuccess;
}



ncclResult_t
ncclAllToAllv3(void* sendbuff, size_t sendcounts[], size_t sendpos[],
    void* recvbuff, const size_t recvcounts[], size_t recvpos[],
    void* tempbuff, struct scheduling_result_t * sched,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream, hipEvent_t* ts, uint * dcpy_sz){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = sched->rankid,
        local_rank_id = sched->rankid % sched->gpu_n,
        server_id = sched->rankid / sched->gpu_n,
        server_n = sched->server_n,
        gpu_n = sched->gpu_n,
        rankid = sched->rankid,
        MAX_BUFFER_SIZE_PER_RANK = sched->MAX_BUFFER_SIZE_PER_RANK;
    ncclResult_t ret, state;
    // printf("rankid: %u, gpu_n: %u, server_id: %u, server_n: %u\n", rankid, gpu_n, server_id, server_n);

    /* ------------------------------------------------------
        Preparation Stage: Instrinsic AllToAll and Balance
     ----------------------------------------------------- */

    // Instrinsic AllToAll
    if (ts) hipEventRecord(ts[0], stream);
    NCCLCHECK(ncclGroupStart());
    for (uint r = 0; r < gpu_n; r++){
        uint global_comm_gpu = server_id * gpu_n + r;
        size_t send_data_sz = (sched -> intrinsic_ata)[server_id][local_rank_id * gpu_n + r];
        // printf("rankid: %u, server_id: %u, send to local gpu: %u, data sz: %lu\n", rankid, server_id, r, send_data_sz);
            uint sendbuf_offset = global_comm_gpu * gpu_n + local_rank_id;
            void * src_ptr = (char *)sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
            NCCLCHECK(ncclSend(
                src_ptr,
                send_data_sz,
                datatype,
                global_comm_gpu,
                comm,
                stream
            ));
            sendpos[sendbuf_offset] += send_data_sz;
        size_t recv_data_sz = (sched -> intrinsic_ata)[server_id][r * gpu_n + local_rank_id];
            void * dst_ptr = (char *)recvbuff + global_comm_gpu * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
            NCCLCHECK(ncclRecv(
                dst_ptr,
                recv_data_sz,
                datatype,
                global_comm_gpu,
                comm,
                stream
            ));
            recvpos[global_comm_gpu] += recv_data_sz;
    }
    NCCLCHECK(ncclGroupEnd());

    if (ts) hipEventRecord(ts[1], stream);

    // Load balance
    NCCLCHECK(ncclGroupStart());
    for (uint s = 0; s != server_n; s++){
        if (s == server_id){
            continue;
        }
        for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
            for (uint channel_id = 0; channel_id < gpu_n; channel_id ++){
                size_t send_data_sz = (sched -> balance)[server_id][s][local_rank_id * gpu_n + local_gpu].sz[channel_id];
                if (send_data_sz > 0){
                    uint global_dst_gpu = s *  gpu_n + channel_id;
                    uint sendbuf_offset = global_dst_gpu * gpu_n + local_rank_id;
                    uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
                    void * src_ptr = (char *) sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t) + sendpos[sendbuf_offset] * sizeof(int32_t);
                    NCCLCHECK(ncclSend(
                        src_ptr,
                        send_data_sz,
                        datatype,
                        intermediate_gpu_global_id,
                        comm,
                        stream
                    ));
                    sendpos[sendbuf_offset] += send_data_sz;
                }
                size_t recv_data_sz = (sched -> balance)[server_id][s][local_gpu * gpu_n + local_rank_id].sz[channel_id];
                if (recv_data_sz > 0){
                    uint global_dst_gpu = s *  gpu_n + channel_id;
                    uint sendbuf_offset = global_dst_gpu * gpu_n + local_gpu;
                    uint src_gpu_global_id = server_id * gpu_n + local_gpu;
                    void * dst_ptr = (char *) sendbuff + sendbuf_offset * MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t);
                    NCCLCHECK(ncclRecv(
                        dst_ptr,
                        recv_data_sz,
                        datatype,
                        src_gpu_global_id,
                        comm,
                        stream
                    ));
                    sendcounts [sendbuf_offset] += recv_data_sz;
                }
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());

    if (ts) hipEventRecord(ts[2], stream);


    /* ------------------------------------------------------
        Pipeline Stage
     ----------------------------------------------------- */

    uint TEMPBUFF_OFFSET = sizeof(int32_t)* gpu_n * gpu_n * MAX_BUFFER_SIZE_PER_RANK;
    uint cur_tempbuff_offset = 0, prev_tempbuff_offset = 0;

    // First step
    struct scheduling_step_t cur_step = (sched -> steps)[0];
    uint dst_server = cur_step.to_server[server_id];
    uint src_server = cur_step.from_server[server_id];
    uint dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    uint src_gpu_global_id = src_server * gpu_n + local_rank_id;
    uint * channel_send_sched = cur_step.channel[server_id][local_rank_id];
    uint * channel_recv_sched = cur_step.channel[src_server][local_rank_id];

    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
        for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu++){
            uint send_data_sz = channel_send_sched[local_gpu * gpu_n + from_gpu];
            if (send_data_sz > 0){
                uint global_dst_gpu = dst_server * gpu_n + local_gpu;
                uint sendbuff_offset = global_dst_gpu * gpu_n + from_gpu;
                void * src_ptr = (char *) sendbuff + sendbuff_offset * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + sendpos[sendbuff_offset] * sizeof(int32_t);
                NCCLCHECK(ncclSend(
                    src_ptr,
                    send_data_sz,
                    datatype,
                    dst_gpu_global_id,
                    comm,
                    stream
                ));
                sendpos[sendbuff_offset] += send_data_sz;
            }
            uint recv_data_sz =  channel_recv_sched[local_gpu * gpu_n + from_gpu];
            if (recv_data_sz > 0){
                uint tempbuff_id = local_gpu * gpu_n + from_gpu;
                void * dst_ptr = (char *) tempbuff + cur_tempbuff_offset + tempbuff_id *  sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    recv_data_sz,
                    datatype,
                    src_gpu_global_id,
                    comm,
                    stream
                ));
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());

    if (ts) hipEventRecord(ts[3], stream);
    // else if (ret == ncclSuccess) {
    // /* Successfully issued */
    // printf("Rank %u: step 0 - issue succeeded\n", rankid);
    // }

    // middle steps
    uint prev_dst_server = dst_server,
        prev_src_server = src_server;
    struct scheduling_step_t prev_step = cur_step;
    struct recv_data_t * restore_send_sched, * restore_recv_sched, * dcopy_sched;
    uint step_n = sched -> step_n;

    // for (uint step_id = 1; step_id < step_n - 1; step_id ++){
    //     cur_step = (sched -> steps)[step_id];
    //     dst_server = cur_step.to_server[server_id];
    //     src_server = cur_step.from_server[server_id];

    //     dst_gpu_global_id = dst_server * gpu_n + local_rank_id;
    //     src_gpu_global_id = src_server * gpu_n + local_rank_id;

    //     channel_send_sched = cur_step.channel[server_id][local_rank_id];
    //     channel_recv_sched = cur_step.channel[src_server][local_rank_id];

    //     restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
    //     dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

    //     cur_tempbuff_offset = (step_id % 2 == 1) ? TEMPBUFF_OFFSET : 0;
    //     prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;

    //     NCCLCHECK(ncclGroupStart());
    //     for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
    //         for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu++){
    //             uint send_data_sz = channel_send_sched[local_gpu * gpu_n + from_gpu];
    //             if (send_data_sz > 0){
    //                 uint global_dst_gpu = dst_server * gpu_n + local_gpu;
    //                 uint sendbuff_offset = global_dst_gpu * gpu_n + from_gpu;
    //                 void * src_ptr = (char *) sendbuff + sendbuff_offset * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + sendpos[sendbuff_offset] * sizeof(int32_t);
    //                 NCCLCHECK(ncclSend(
    //                     src_ptr,
    //                     send_data_sz,
    //                     datatype,
    //                     dst_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //                 sendpos[sendbuff_offset] += send_data_sz;
    //             }
    //             uint recv_data_sz =  channel_recv_sched[local_gpu * gpu_n + from_gpu];
    //             if (recv_data_sz > 0){
    //                 uint tempbuff_id = local_gpu * gpu_n + from_gpu;
    //                 void * dst_ptr = (char *) tempbuff + cur_tempbuff_offset + tempbuff_id *  sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
    //                 NCCLCHECK(ncclRecv(
    //                     dst_ptr,
    //                     recv_data_sz,
    //                     datatype,
    //                     src_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //             }
    //         }
    //     }


    //     // restore
    //     for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
    //         if (local_gpu == local_rank_id){
    //             continue;
    //         }
    //         for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
    //             uint send_data_sz = restore_send_sched[local_gpu * gpu_n + from_gpu].sz;
    //             if (send_data_sz > 0){
    //                 dst_gpu_global_id = server_id * gpu_n + local_gpu;
    //                 uint src_gpu_tempbuff_id = local_gpu * gpu_n + from_gpu;
    //                 void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
    //                 NCCLCHECK(ncclSend(
    //                     src_ptr,
    //                     send_data_sz,
    //                     datatype,
    //                     dst_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //             }
    //             restore_recv_sched = cur_step.restore[prev_src_server][local_gpu];
    //             uint recv_data_sz = restore_recv_sched[local_rank_id * gpu_n + from_gpu].sz;
    //             if (recv_data_sz > 0){
    //                 src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
    //                 uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
    //                 void * dst_ptr = (char *) recvbuff + src_gpu_global_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * gpu_n + from_gpu].offset * sizeof(int32_t);
    //                 NCCLCHECK(ncclRecv(
    //                     dst_ptr,
    //                     recv_data_sz,
    //                     datatype,
    //                     intermediate_gpu_global_id,
    //                     comm,
    //                     stream
    //                 ));
    //                 recvpos[src_gpu_global_id] += recv_data_sz;
    //             }
    //         }
    //     }


    //     //direct cpy
    //     for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
    //         if (dcopy_sched[from_gpu].sz > 0){
    //             src_gpu_global_id = prev_src_server  * gpu_n + from_gpu;
    //             uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
    //             void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
    //             void * dst_ptr = (char *) recvbuff + src_gpu_global_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * sizeof(int32_t);
    //             NCCLCHECK(ncclSend(
    //                 src_ptr,
    //                 dcopy_sched[from_gpu].sz,
    //                 datatype,
    //                 rankid,
    //                 comm,
    //                 stream
    //             ));
    //             NCCLCHECK(ncclRecv(
    //                 dst_ptr,
    //                 dcopy_sched[from_gpu].sz,
    //                 datatype,
    //                 rankid,
    //                 comm,
    //                 stream
    //             ));
    //             recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
    //         }
    //     }

    //     NCCLCHECK(ncclGroupEnd());

    //     prev_src_server = src_server;
    //     prev_dst_server = dst_server;
    // }

    // last restore

    prev_tempbuff_offset = ((step_n - 1) % 2 == 1) ? 0 : TEMPBUFF_OFFSET;
    cur_step = (sched -> steps)[step_n - 1];
    restore_send_sched = cur_step.restore[prev_src_server][local_rank_id];
    dcopy_sched = cur_step.direct_cpy[prev_src_server][local_rank_id];

    // // direct cpy
    // for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
    //     if (dcopy_sched[from_gpu].sz > 0){
    //         src_gpu_global_id = prev_src_server  * gpu_n + from_gpu;
    //         uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
    //         void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
    //         void * dst_ptr = (char *) recvbuff + src_gpu_global_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * sizeof(int32_t);
    //         hipMemcpy(dst_ptr, src_ptr, sizeof(int32_t) * dcopy_sched[from_gpu].sz, hipMemcpyDeviceToDevice);
    //         recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
    //     }
    // }

    // restore
    NCCLCHECK(ncclGroupStart());
    for (uint local_gpu = 0; local_gpu < gpu_n; local_gpu++){
        if (local_gpu == local_rank_id){
            continue;
        }
        for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
            uint send_data_sz = (restore_send_sched[local_gpu * gpu_n + from_gpu].sz + 0x1ff) & 0xfffffe00;
            if (send_data_sz > 0){
                dst_gpu_global_id = server_id * gpu_n + local_gpu;
                uint src_gpu_tempbuff_id = local_gpu * gpu_n + from_gpu;
                void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
                NCCLCHECK(ncclSend(
                    src_ptr,
                    send_data_sz,
                    datatype,
                    dst_gpu_global_id,
                    comm,
                    stream
                ));
            }
            restore_recv_sched = cur_step.restore[prev_src_server][local_gpu];
            uint recv_data_sz = (restore_recv_sched[local_rank_id * gpu_n + from_gpu].sz + 0x1ff) & 0xfffffe00;
            if (recv_data_sz > 0){
                src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
                uint intermediate_gpu_global_id = server_id * gpu_n + local_gpu;
                void * dst_ptr = (char *) recvbuff + src_gpu_global_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + restore_recv_sched[local_rank_id * gpu_n + from_gpu].offset * sizeof(int32_t);
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    recv_data_sz,
                    datatype,
                    intermediate_gpu_global_id,
                    comm,
                    stream
                ));
                recvpos[src_gpu_global_id] += recv_data_sz;
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());

    if (ts) hipEventRecord(ts[4], stream);
    NCCLCHECK(ncclGroupStart());
    // //direct cpy
    for (uint from_gpu = 0; from_gpu < gpu_n; from_gpu ++){
        if (dcopy_sched[from_gpu].sz > 0){
            *dcpy_sz += dcopy_sched[from_gpu].sz;
            src_gpu_global_id = prev_src_server * gpu_n + from_gpu;
            uint src_gpu_tempbuff_id = local_rank_id * gpu_n + from_gpu;
            void * src_ptr = (char *) tempbuff + prev_tempbuff_offset + src_gpu_tempbuff_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK;
            void * dst_ptr = (char *) recvbuff + src_gpu_global_id * sizeof(int32_t) * MAX_BUFFER_SIZE_PER_RANK + dcopy_sched[from_gpu].offset * sizeof(int32_t);
            NCCLCHECK(ncclSend(
                    src_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));
                NCCLCHECK(ncclRecv(
                    dst_ptr,
                    dcopy_sched[from_gpu].sz,
                    datatype,
                    rankid,
                    comm,
                    stream
                ));

            // hipMemcpyWithStream(dst_ptr, src_ptr, sizeof(int32_t) * dcopy_sched[from_gpu].sz, hipMemcpyDeviceToDevice, stream);
            recvpos[src_gpu_global_id] += dcopy_sched[from_gpu].sz;
        }
    }
    NCCLCHECK(ncclGroupEnd());
    if (ts) hipEventRecord(ts[5], stream);

    return ncclSuccess;
}


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
    RCCLCHECK(hipEventRecord(start_event, stream));
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
    RCCLCHECK(hipEventRecord(end_event, stream));
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

    hipEvent_t start_event, end_event, ts[8];
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventCreate(&ts[i]));
    }

    uint dcpy_sz = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        ncclAllToAllv3(
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
            stream,
            NULL,
            &dcpy_sz);
        // NCCLCHECK(ncclAllToAllv2(
        //     param->sendbuff,
        //     param->sendcount,
        //     param->sendpos,
        //     param->recvbuff,
        //     param->recvcount,
        //     param->recvpos,
        //     param->tempbuff,
        //     param->sched,
        //     ncclInt32,
        //     comm,
        //     stream));
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        ncclAllToAllv3(
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
            stream,
            ts,
            &dcpy_sz);
        // NCCLCHECK(ncclAllToAllv2(
        //     param->sendbuff,
        //     param->sendcount,
        //     param->sendpos,
        //     param->recvbuff,
        //     param->recvcount,
        //     param->recvpos,
        //     param->tempbuff,
        //     param->sched,
        //     ncclInt32,
        //     comm,
        //     stream));
        reset_buffer_counter(param, param->sched->server_n, param->sched->gpu_n);
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9 / (param->sched->gpu_n * param->sched->server_n);


    float et[8];
    for (uint i = 0; i < 8; i++){
        et[i] = 0;
    }

    for (uint i = 0; i < 5; i++){
        hipEventElapsedTime(&et[i], ts[i], ts[i+1]);
    }
    if (param->sched->rankid == 0){
        std::cout << "total: "<< avg_time << " ms, t0-t1: " << et[0] << " ms, t1-t2: " << et[1] << " ms, t2-t3: " << et[2] << " ms, t3-t4: " << et[3] << " ms, t4-t5: " << et[4] << " ms" << std::endl;
        std::cout << "dcpy throughput: " << dcpy_sz / perf_iters * sizeof(int32_t) / (et[4] * 1e-3) / 1e9 << " GBps" << std::endl;
    }

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventDestroy(ts[i]));
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
    return r;
}


struct perf_test_ret_t perf_v3(uint warmup_iters, uint perf_iters, struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event, ts[8];
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventCreate(&ts[i]));
    }

    hipStream_t streams[8];
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
    }


    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        ncclAllToAllv4(
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
            streams);
    }
    RCCLCHECK(hipDeviceSynchronize());
    RCCLCHECK(hipEventRecord(start_event, streams[0]));
    for (int i = 0; i < perf_iters; ++i) {
        ncclAllToAllv4(
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
            streams);
    }
    RCCLCHECK(hipEventRecord(end_event, streams[0]));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9 / (param->sched->gpu_n * param->sched->server_n);

    if (param->sched->rankid == 0){
        std::cout << "total v4: "<< avg_time << " ms" << std::endl;
    }

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventDestroy(ts[i]));
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};

    if (param->sched->rankid == 0){
        std::cout << "Buff size per GPU: " << buff_size / (param->sched->gpu_n * param->sched->server_n) / (param->sched->gpu_n * param->sched->server_n - 1)* (param->sched->gpu_n - 1) << " B, algbw: " << algbw << " GBps" << std::endl;
    }
    return r;
}



struct perf_test_ret_t perf_v4(uint warmup_iters, uint perf_iters, struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    void *sendbuff, *recvbuff;
    std::cout <<  param->sched->MAX_BUFFER_SIZE_PER_RANK<< std::endl;
    RCCLCHECK(hipSetDevice(param->sched->rankid % param->sched->gpu_n));
    RCCLCHECK(hipMalloc(&sendbuff, param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t)));
    RCCLCHECK(hipMalloc(&recvbuff, param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t)));
    RCCLCHECK(hipMemset(sendbuff, 1, param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t)));
    RCCLCHECK(hipMemset(recvbuff, 0, param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t)));
    int sendbuff_sz =  param->sched->MAX_BUFFER_SIZE_PER_RANK;
    int block_n = std::min<int>(32, (sendbuff_sz + 4*512-1)/(4*512));
    identity_kernel<<<block_n, 512, 0, hipStreamDefault>>>((int32_t*)sendbuff, sendbuff_sz);

    RCCLCHECK(hipDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        ncclAllToAllv5(
            sendbuff,
            param->sendcount,
            param->sendpos,
            recvbuff,
            param->recvcount,
            param->recvpos,
            param->tempbuff,
            param->sched,
            ncclInt32,
            comm,
            stream);
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        ncclAllToAllv5(
            sendbuff,
            param->sendcount,
            param->sendpos,
            recvbuff,
            param->recvcount,
            param->recvpos,
            param->tempbuff,
            param->sched,
            ncclInt32,
            comm,
            stream);
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double)  param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t) / (avg_time * 1e-3) / 1e9;

    if (param->sched->rankid == 0){
        std::cout << "total v5: "<< avg_time << " ms, algbw: " <<  algbw << " GBps" << std::endl;
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
    return r;
}





void perf_v5(uint warmup_iters, uint perf_iters, struct alltoall_parameters * param, ncclComm_t comm, hipStream_t stream, uint buff_size){

    auto t1 = high_resolution_clock::now();
        hipEvent_t start_event, end_event;
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));


    RCCLCHECK(hipEventRecord(start_event, stream));
    for (uint z = 0; z < perf_iters; z++){
        for (uint i = 0 ; i < param->sched->gpu_n; i++){
            RCCLCHECK(hipMemcpyWithStream((char*)param->recvbuff + i * param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t)  , (char*)param->tempbuff + param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t) , param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t), hipMemcpyDeviceToDevice, stream));
        }
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());
    auto t2 = high_resolution_clock::now();

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));

    auto  cpu_et = duration_cast<microseconds>(t2 - t1).count();
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) param->sched->gpu_n * param->sched->MAX_BUFFER_SIZE_PER_RANK * sizeof(int32_t) / (avg_time * 1e-3) / 1e9;
    if (param->sched->rankid == 0){
        std::cout << "memcpy: "<< avg_time << " us, algbw: " <<  algbw << " GBps" << std::endl;
    }
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





ncclResult_t
fastAllToAll(void* sendbuff, void* recvbuff, void * lbsend, void * lbrecv, void * crosbuff, void * rstrbuff,
    struct scheduling_result_gpu_t * gpu_sched,
    ncclDataType_t data_type, ncclComm_t comm, hipStream_t stream){

    // Get parameters
    int rank_n;
    NCCLCHECK(ncclCommCount(comm, &rank_n));

    uint global_rank_id = gpu_sched->rankid,
        local_rank_id = gpu_sched->rankid % gpu_sched->gpu_n,
        erver_id = gpu_sched->rankid / gpu_sched->gpu_n,
        server_n = gpu_sched->server_n,
        gpu_n = gpu_sched->gpu_n,
        rankid = gpu_sched->rankid;

    // load balance
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < gpu_sched -> balance_send_n; i++){
        NCCLCHECK(ncclSend((char *)lbsend + gpu_sched -> balance_send[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_send[i].sz,
                        data_type,
                        gpu_sched -> balance_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched -> balance_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)lbrecv + gpu_sched -> balance_recv[i].disp * sizeof(int32_t),
                        gpu_sched -> balance_recv[i].sz,
                        data_type,
                        gpu_sched -> balance_recv[i].gpu,
                        comm,
                        stream));
    }
    NCCLCHECK(ncclGroupEnd());

    for (uint i = 0; i < gpu_sched -> balance_memcpy_n; i ++){
        RCCLCHECK(hipMemcpy((char*)sendbuff + gpu_sched -> balance_memcpy[i].dst_disp * sizeof(int32_t),
                         (char*) lbrecv + gpu_sched -> balance_memcpy[i].src_disp * sizeof(int32_t),
                          gpu_sched -> balance_memcpy[i].sz * sizeof(int32_t),
                          hipMemcpyDeviceToDevice));
    }

    // intrinsic alltoall and first cross node
    struct scheduling_step_gpu_t * cur_step = &gpu_sched -> steps[0];
    uint cross_send_sz = cur_step -> crossnode_send.sz;
    uint cross_recv_sz = cur_step -> crossnode_recv.sz;
    NCCLCHECK(ncclGroupStart());
    // intrindic alltoall
    for (uint i = 0; i < gpu_sched -> intrinsic_send_n; i ++){
        NCCLCHECK(ncclSend((char *)sendbuff + gpu_sched -> intrinsic_send[i].disp * sizeof(int32_t),
                        gpu_sched -> intrinsic_send[i].sz,
                        data_type,
                        gpu_sched -> intrinsic_send[i].gpu,
                        comm,
                        stream));
    }

    for (uint i = 0; i < gpu_sched ->intrinsic_recv_n; i ++){
        NCCLCHECK(ncclRecv((char *)recvbuff + gpu_sched -> intrinsic_recv[i].disp * sizeof(int32_t),
                        gpu_sched -> intrinsic_recv[i].sz,
                        data_type,
                        gpu_sched -> intrinsic_recv[i].gpu,
                        comm,
                        stream));
    }

    // first cross-node send
    if (cross_send_sz > 0){
        NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                        cross_send_sz,
                        data_type,
                        cur_step -> crossnode_send.gpu,
                        comm,
                        stream));
    }
    if (cross_recv_sz > 0){
        NCCLCHECK(ncclRecv((char *)crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
                cross_recv_sz,
                data_type,
                cur_step -> crossnode_recv.gpu,
                comm,
                stream));
    }
    NCCLCHECK(ncclGroupEnd());

    // middle steps
    for (uint step_id = 1; step_id < gpu_sched -> step_n - 1; step_id ++){
        cur_step = &gpu_sched -> steps[step_id];
        cross_send_sz = cur_step -> crossnode_send.sz;
        cross_recv_sz = cur_step -> crossnode_recv.sz;
        NCCLCHECK(ncclGroupStart());
        // cross node transfer
        if (cross_send_sz > 0){
            NCCLCHECK(ncclSend((char *)sendbuff + cur_step -> crossnode_send.disp * sizeof(int32_t),
                            cross_send_sz,
                            data_type,
                            cur_step -> crossnode_send.gpu,
                            comm,
                            stream));
        }
        if (cross_recv_sz > 0){
            NCCLCHECK(ncclRecv((char *)crosbuff + cur_step -> crossnode_recv.disp * sizeof(int32_t),
                    cross_recv_sz,
                    data_type,
                    cur_step -> crossnode_recv.gpu,
                    comm,
                    stream));
        }
        // data restore of previous step
        for (uint i = 0; i < cur_step -> restore_send_n; i ++){
            NCCLCHECK(ncclSend((char *)crosbuff + cur_step -> restore_send[i].disp * sizeof(int32_t),
                cur_step -> restore_send[i].sz,
                data_type,
                cur_step -> restore_send[i].gpu,
                comm,
                stream));
        }

        for (uint i = 0; i < cur_step -> restore_recv_n; i++){
            NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * sizeof(int32_t),
                cur_step -> restore_recv[i].sz,
                data_type,
                cur_step -> restore_recv[i].gpu,
                comm,
                stream));
        }
        NCCLCHECK(ncclGroupEnd());

        for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
            RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice));
        }

        for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
            RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                        (char*) crosbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                        cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                        hipMemcpyDeviceToDevice));
        }
    }

    // final data restore
    cur_step = &gpu_sched -> steps[gpu_sched -> step_n - 1];
    NCCLCHECK(ncclGroupStart());
    for (uint i = 0; i < cur_step -> restore_send_n; i ++){
        NCCLCHECK(ncclSend((char *)crosbuff + cur_step -> restore_send[i].disp * sizeof(int32_t),
            cur_step -> restore_send[i].sz,
            data_type,
            cur_step -> restore_send[i].gpu,
            comm,
            stream));
    }

    for (uint i = 0; i < cur_step -> restore_recv_n; i++){
        NCCLCHECK(ncclRecv((char *)rstrbuff + cur_step -> restore_recv[i].disp * sizeof(int32_t),
            cur_step -> restore_recv[i].sz,
            data_type,
            cur_step -> restore_recv[i].gpu,
            comm,
            stream));
    }

    NCCLCHECK(ncclGroupEnd());
    for (uint i = 0; i < cur_step -> direct_memcpy_n; i++){
        RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> direct_memcpy[i].dst_disp * sizeof(int32_t),
                (char*) crosbuff + cur_step -> direct_memcpy[i].src_disp * sizeof(int32_t),
                cur_step -> direct_memcpy[i].sz * sizeof(int32_t),
                hipMemcpyDeviceToDevice));
    }

    for (uint i = 0; i < cur_step -> restore_memcpy_n; i++){
        RCCLCHECK(hipMemcpy((char*)recvbuff + cur_step -> restore_memcpy[i].dst_disp * sizeof(int32_t),
                    (char*) crosbuff + cur_step -> restore_memcpy[i].src_disp * sizeof(int32_t),
                    cur_step -> restore_memcpy[i].sz * sizeof(int32_t),
                    hipMemcpyDeviceToDevice));
    }

    return ncclSuccess;
}


bool verify_correctness_fastalltoall(struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched, uint recvbuff_sz, ncclComm_t comm, hipStream_t stream){
    uint recvbuff_sz_in_bytes = recvbuff_sz * sizeof(int32_t);
    MPI_Barrier(MPI_COMM_WORLD);
    NCCLCHECK(fastAllToAll(
        bufs->sendbuff,
        bufs->recvbuff,
        bufs->lbsend,
        bufs->lbrecv,
        bufs->crosbuff,
        bufs->rstrbuff,
        gpu_sched,
        ncclInt32,
        comm,
        stream));
    RCCLCHECK(hipDeviceSynchronize());
    uint data_type_size = sizeof(int32_t);
    void * host_recvbuff;
    hipMallocManaged(&host_recvbuff, recvbuff_sz_in_bytes);
    RCCLCHECK(hipMemcpy(host_recvbuff, bufs->recvbuff, recvbuff_sz_in_bytes, hipMemcpyDeviceToHost));
    // print_recvbuffs(host_recvbuff, MAX_BUFFER_SIZE_PER_RANK, dim, param->sched->rankid);
    return (0 == memcmp(host_recvbuff, bufs->verifybuff,  recvbuff_sz_in_bytes));
}


struct perf_test_ret_t perf_fastalltoall(uint warmup_iters, uint perf_iters, struct buffer_ptr_t * bufs, struct scheduling_result_gpu_t * gpu_sched, ncclComm_t comm, hipStream_t stream, uint buff_size){

    hipEvent_t start_event, end_event, ts[8];
    RCCLCHECK(hipEventCreate(&start_event));
    RCCLCHECK(hipEventCreate(&end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventCreate(&ts[i]));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmup_iters; ++i) {
        NCCLCHECK(fastAllToAll(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(start_event, stream));
    for (int i = 0; i < perf_iters; ++i) {
        NCCLCHECK(fastAllToAll(
            bufs->sendbuff,
            bufs->recvbuff,
            bufs->lbsend,
            bufs->lbrecv,
            bufs->crosbuff,
            bufs->rstrbuff,
            gpu_sched,
            ncclInt32,
            comm,
            stream));
    }
    RCCLCHECK(hipEventRecord(end_event, stream));
    RCCLCHECK(hipDeviceSynchronize());

    float elapsed_time;
    RCCLCHECK(hipEventElapsedTime(&elapsed_time, start_event, end_event));
    double avg_time = (double) elapsed_time / perf_iters;
    double algbw = (double) buff_size / (avg_time * 1e-3) / 1e9 / (gpu_sched->gpu_n * gpu_sched->server_n);


    float et[8];
    for (uint i = 0; i < 8; i++){
        et[i] = 0;
    }

    for (uint i = 0; i < 5; i++){
        hipEventElapsedTime(&et[i], ts[i], ts[i+1]);
    }
    if (gpu_sched->rankid == 0){
        std::cout << "total: "<< avg_time << " ms, t0-t1: " << et[0] << " ms, t1-t2: " << et[1] << " ms, t2-t3: " << et[2] << " ms, t3-t4: " << et[3] << " ms, t4-t5: " << et[4] << " ms" << std::endl;
    }

    RCCLCHECK(hipEventDestroy(start_event));
    RCCLCHECK(hipEventDestroy(end_event));
    for (uint i = 0; i < 8; i++){
        RCCLCHECK(hipEventDestroy(ts[i]));
    }
    struct perf_test_ret_t r = {.algbw = algbw, .time = avg_time};
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
    if (rank == 0) uniform_distribution(workload, nranks, 2048000);
    MPI_Bcast(workload, nranks * nranks * sizeof(uint), MPI_BYTE, 0, MPI_COMM_WORLD);
    // std::cout << "Rank " << rank << " receive demand matrix" << std::endl;
    struct max_sum_ret_t workload_max_sum = max_sum_matrix(workload, nranks);
    uint buff_size = workload_max_sum.sum * sizeof(int32_t);    // bytes
    if (rank == 0) print_matrix(workload, nranks, nranks);


    // scheduler is deterministic given a workload
    auto sched_t1 = high_resolution_clock::now();
    uint server_n = 2, gpu_n = 8;
    struct GlobalScheduler scheduler;
    init_global_scheduler(&scheduler, server_n, gpu_n, workload, rank);
    auto sched_t2 = high_resolution_clock::now();
    run_scheduler(&scheduler);
    scheduler.sched->rankid = rank;
    uint cross_node_sz = 0;
    for (uint i = 0; i < server_n; i++){
        uint s2s_sum = 0;
        for (uint j = 0; j < server_n; j++){
            s2s_sum += scheduler.locals[i]->server2server_data[j];
        }
        cross_node_sz = MAX(cross_node_sz, s2s_sum);
    }
    if (rank == 0) std:: cout << "cross node bound: " << cross_node_sz * sizeof(int32_t) / 1e9 / 11 * 1000 << " ms" << std::endl;
    // make it 512-byte aligned
    //  scheduler.sched->MAX_BUFFER_SIZE_PER_RANK = workload_max_sum.max;
    scheduler.sched->MAX_BUFFER_SIZE_PER_RANK = (workload_max_sum.max + 0x7f) & 0xffffff80;
    auto sched_t3 = high_resolution_clock::now();
    if (rank == 0) std::cout << "Scheduler Time: " << duration_cast<microseconds>(sched_t3 - sched_t1).count() << " us, init: " << duration_cast<microseconds>(sched_t2 - sched_t1).count() << " us, runtime: " << duration_cast<microseconds>(sched_t3 - sched_t2).count() << "us" << std::endl;
    // std::cout << "Rank " << rank << " scheduling succeeds" << std::endl;

    auto mem_t1 = high_resolution_clock::now();
    RCCLCHECK(hipSetDevice(rank % gpu_n));
    struct alltoall_parameters param = allocate_device_memory(scheduler.sched->MAX_BUFFER_SIZE_PER_RANK, server_n, gpu_n);
    auto mem_t2 = high_resolution_clock::now();
    param.sched = scheduler.sched;
    initialize_buffer(&param, workload, rank, server_n, gpu_n);
    auto mem_t3 = high_resolution_clock::now();
    struct buffer_ptr_t buffer_ptrs = init_buffers(scheduler.buff_parameter, server_n, gpu_n, rank);
    auto mem_t4 = high_resolution_clock::now();
    if (rank == 0) std:: cout << "buffer allocation: " << duration_cast<microseconds>(mem_t2 - mem_t1).count() << " us, init buffer: " << duration_cast<microseconds>(mem_t3 - mem_t2).count() << " us, buffer initialization v2: " <<  duration_cast<microseconds>(mem_t4 - mem_t3).count() << std::endl;

    // std::cout << "Rank " << rank << " buffers initialization succeeds" << std::endl;


    // verify correctness
    if (rank == 0) std::cout << "TESTING CORRECTNESS" << std::endl;
    // int correctness_this_rank = verify_correctness_v2(&param, comm, stream);
    int correctness_this_rank = verify_correctness_fastalltoall(&buffer_ptrs, scheduler.gpu_sched, scheduler.buff_parameter->recvbuff_total_sz, comm, stream);
    int * correctness;
    if (rank == 0) correctness = new int[nranks];
    MPI_Gather(&correctness_this_rank, 1, MPI_INT, correctness, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int correctness_n = 0;
        for (uint i = 0; i < nranks; i++) correctness_n += correctness[i];
        std::cout << "CORRECTNESS: " << ((correctness_n == nranks) ? "succeed" : "fail") << std::endl;
    }

    uint warmup_iters = 10, perf_iters = 20;
    if (rank == 0) std::cout << "TESTING PERFORMANCE: warmup iters: " << warmup_iters <<", test iters: " << perf_iters << std::endl;
    perf_fastalltoall(warmup_iters, perf_iters, &buffer_ptrs, scheduler.gpu_sched, comm, stream, buff_size);
    // struct perf_test_ret_t v0_ret = perf_v0(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    // struct perf_test_ret_t v2_ret = perf_v2(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    // if (rank == 0){
    //     std::cout << "Buff size: " << buff_size << " B, v0 algbw: " << v0_ret.algbw << " GBps, v0 time: " << v0_ret.time << " ms, v2 algbw: " << v2_ret.algbw << " GBps, v2 time: " << v2_ret.time <<  " ms, Speed up: " << v2_ret.algbw / v0_ret.algbw << std::endl;
    // }
    // perf_v3(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    // perf_v4(warmup_iters, perf_iters, &param, comm, stream, buff_size);
    // perf_v5(warmup_iters, perf_iters, &param, comm, stream, buff_size);


    free_buffer(&param);
    free_buffers(&buffer_ptrs);

    delete[] workload;
    free_global_scheduler(&scheduler);

    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    RCCLCHECK(hipStreamDestroy(stream));
    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}