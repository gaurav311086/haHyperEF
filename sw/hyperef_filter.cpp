#include <assert.h>

#include <iomanip>
#include <iostream>
#include <bitset>
#include "common.h"
#include "data_formatter.h"
#include "data_loader.h"
#include "hyperef.h"

// device memory channels
#define MAX_HBM_CHANNEL_COUNT 32
#define CHANNEL_NAME(n) n | XCL_MEM_TOPOLOGY
const int HBM[MAX_HBM_CHANNEL_COUNT] = {
    CHANNEL_NAME(0),  CHANNEL_NAME(1),  CHANNEL_NAME(2),  CHANNEL_NAME(3),
    CHANNEL_NAME(4),  CHANNEL_NAME(5),  CHANNEL_NAME(6),  CHANNEL_NAME(7),
    CHANNEL_NAME(8),  CHANNEL_NAME(9),  CHANNEL_NAME(10), CHANNEL_NAME(11),
    CHANNEL_NAME(12), CHANNEL_NAME(13), CHANNEL_NAME(14), CHANNEL_NAME(15),
    CHANNEL_NAME(16), CHANNEL_NAME(17), CHANNEL_NAME(18), CHANNEL_NAME(19),
    CHANNEL_NAME(20), CHANNEL_NAME(21), CHANNEL_NAME(22), CHANNEL_NAME(23),
    CHANNEL_NAME(24), CHANNEL_NAME(25), CHANNEL_NAME(26), CHANNEL_NAME(27),
    CHANNEL_NAME(28), CHANNEL_NAME(29), CHANNEL_NAME(30), CHANNEL_NAME(31)};

const int DDR[2] = {CHANNEL_NAME(32), CHANNEL_NAME(33)};

//--------------------------------------------------------------------------------------------------
// reference and verify utils
//--------------------------------------------------------------------------------------------------

void compute_ref(spmv::io::CSRMatrix<float> &mat, std::vector<float> &vector,
                 std::vector<float> &ref_result) {
  ref_result.resize(mat.num_rows);
  std::fill(ref_result.begin(), ref_result.end(), 0);
  for (size_t row_idx = 0; row_idx < mat.num_rows; row_idx++) {
    IDX_T start = mat.adj_indptr[row_idx];
    IDX_T end = mat.adj_indptr[row_idx + 1];
    for (size_t i = start; i < end; i++) {
      IDX_T idx = mat.adj_indices[i];
      ref_result[row_idx] += mat.adj_data[i] * vector[idx];
    }
  }
}

bool verify(std::vector<float> reference_results,
            std::vector<VAL_T> kernel_results) {
  float epsilon = 0.0001;
  if (reference_results.size() != kernel_results.size()) {
    std::cout << "Error: Size mismatch" << std::endl;
    std::cout << "  Reference result size: " << reference_results.size()
              << "  Kernel result size: " << kernel_results.size() << std::endl;
    return false;
  }
  for (size_t i = 0; i < reference_results.size(); i++) {
    bool match = abs(float(kernel_results[i]) - reference_results[i]) < epsilon;
    if (!match) {
      std::cout << "Error: Result mismatch" << std::endl;
      std::cout << "  i = " << i
                << "  Reference result = " << reference_results[i]
                << "  Kernel result = " << kernel_results[i] << std::endl;
      return false;
    }
  }
  return true;
}

void unpack_vector(aligned_vector<PACKED_VAL_T> &pdv, std::vector<VAL_T> &dv) {
  dv.resize(pdv.size() * PACK_SIZE);
  for (size_t i = 0; i < pdv.size(); i++) {
    for (size_t k = 0; k < PACK_SIZE; k++) {
      dv[i * PACK_SIZE + k] = pdv[i].data[k];
    }
  }
}

void unpack_vector_flt(aligned_vector<PACKED_VAL_T> &pdv,
                       std::vector<float> &dv) {
	uint32_t dv_size = dv.size();
  dv.resize(pdv.size() * PACK_SIZE);
  for (size_t i = 0; i < pdv.size(); i++) {
    for (size_t k = 0; k < PACK_SIZE; k++) {
      dv[i * PACK_SIZE + k] = float(pdv[i].data[k]);
    }
  }
  dv.resize(dv_size);
}

void transferH2H(aligned_vector<PACKED_VAL_T> &pdv,
                       aligned_vector<PACKED_VAL_T> &dv) {
  for (size_t i = 0; i < pdv.size(); i++) {
      dv[i] = pdv[i];
	  
	for(size_t k=0; k< PACK_SIZE; k++) {
		// std::cout << float(pdv[i].data[k]) << "," << i*PACK_SIZE+k << std::endl;
	}
    
  }
}

//---------------------------------------------------------------
// test harness utils
//---------------------------------------------------------------

#define CL_CREATE_EXT_PTR(name, data, channel) \
  cl_mem_ext_ptr_t name;                       \
  name.obj = data;                             \
  name.param = 0;                              \
  name.flags = channel;

#define CL_BUFFER_RDONLY(context, size, ext, err)                            \
  cl::Buffer(context,                                                        \
             CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, \
             size, &ext, &err);

#define CL_BUFFER_WRONLY(context, size, ext, err)                             \
  cl::Buffer(context,                                                         \
             CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, \
             size, &ext, &err);

#define CL_BUFFER(context, size, ext, err)                                    \
  cl::Buffer(context,                                                         \
             CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, \
             size, &ext, &err);

#define CHECK_ERR(err)                                                    \
  if (err != CL_SUCCESS) {                                                \
    printf("OCL Error at %s:%d, error code is: %d\n", __FILE__, __LINE__, \
           err);                                                          \
    exit(EXIT_FAILURE);                                                   \
  }

struct cl_runtime {
  cl::Context context;
  cl::CommandQueue command_queue;
  cl::Kernel spmv_sk0;
  cl::Kernel spmv_sk1;
  cl::Kernel spmv_sk2;
  cl::Kernel vector_loader;
  cl::Kernel result_drain;
};

//---------------------------------------------------------------
// test harness
//---------------------------------------------------------------


bool hyperef_filter_kernel_accel(cl_runtime &runtime,
		       uint32_t mx, uint32_t sm_size,
                       spmv::io::CSRMatrix<float> &AD,
                       spmv::io::CSRMatrix<float> &D, std::vector<std::vector<float>> &sm,
                       std::vector<std::vector<float>> &sm_vec, uint32_t loop_count,
                       bool skip_empty_rows = false) {
  using namespace spmv::io;
  using namespace std;

  //--------------------------------------------------------------------
  // load and format the matrix
  //--------------------------------------------------------------------
  util_round_csr_matrix_dim<float>(
      AD, PACK_SIZE * NUM_HBM_CHANNELS * INTERLEAVE_FACTOR, PACK_SIZE);
  util_round_csr_matrix_dim<float>(
      D, PACK_SIZE * NUM_HBM_CHANNELS * INTERLEAVE_FACTOR, PACK_SIZE);
  CSRMatrix<VAL_T> AD_mat = csr_matrix_convert_from_float<VAL_T>(AD);
  CSRMatrix<VAL_T> D_mat = csr_matrix_convert_from_float<VAL_T>(D);

  size_t num_row_partitions =
      (AD.num_rows + LOGICAL_OB_SIZE - 1) / LOGICAL_OB_SIZE;
  size_t num_col_partitions =
      (AD.num_cols + LOGICAL_VB_SIZE - 1) / LOGICAL_VB_SIZE;
  size_t num_partitions = num_row_partitions * num_col_partitions;
  size_t num_virtual_hbm_channels = NUM_HBM_CHANNELS * INTERLEAVE_FACTOR;
  CPSRMatrix<PACKED_VAL_T, PACKED_IDX_T, PACK_SIZE> cpsr_matrix_AD =
      csr2cpsr<PACKED_VAL_T, PACKED_IDX_T, VAL_T, IDX_T, PACK_SIZE>(
          AD_mat, IDX_MARKER, LOGICAL_OB_SIZE, LOGICAL_VB_SIZE,
          num_virtual_hbm_channels, skip_empty_rows);
  CPSRMatrix<PACKED_VAL_T, PACKED_IDX_T, PACK_SIZE> cpsr_matrix_D =
      csr2cpsr<PACKED_VAL_T, PACKED_IDX_T, VAL_T, IDX_T, PACK_SIZE>(
          D_mat, IDX_MARKER, LOGICAL_OB_SIZE, LOGICAL_VB_SIZE,
          num_virtual_hbm_channels, skip_empty_rows);
  using partition_indptr_t = struct {
    IDX_T start;
    PACKED_IDX_T nnz;
  };
  using ch_partition_indptr_t = std::vector<partition_indptr_t>;
  using ch_packed_idx_t = std::vector<PACKED_IDX_T>;
  using ch_packed_val_t = std::vector<PACKED_VAL_T>;
  using ch_mat_pkt_t = aligned_vector<SPMV_MAT_PKT_T>;
  std::vector<ch_partition_indptr_t> ADchannel_partition_indptr(
      num_virtual_hbm_channels);
  std::vector<ch_partition_indptr_t> Dchannel_partition_indptr(
      num_virtual_hbm_channels);
  for (size_t c = 0; c < num_virtual_hbm_channels; c++) {
    ADchannel_partition_indptr[c].resize(num_partitions);
    Dchannel_partition_indptr[c].resize(num_partitions);
    ADchannel_partition_indptr[c][0].start = 0;
    Dchannel_partition_indptr[c][0].start = 0;
  }
  std::vector<ch_packed_idx_t> ADchannel_indices(num_virtual_hbm_channels);
  std::vector<ch_packed_val_t> ADchannel_vals(num_virtual_hbm_channels);
  std::vector<ch_mat_pkt_t> ADchannel_packets(NUM_HBM_CHANNELS);
  std::vector<ch_packed_idx_t> Dchannel_indices(num_virtual_hbm_channels);
  std::vector<ch_packed_val_t> Dchannel_vals(num_virtual_hbm_channels);
  std::vector<ch_mat_pkt_t> Dchannel_packets(NUM_HBM_CHANNELS);
  // Iterate virtual channels and map virtual channels (vc) to physical channels
  // (pc)
  for (size_t pc = 0; pc < NUM_HBM_CHANNELS; pc++) {
    for (size_t j = 0; j < num_row_partitions; j++) {
      for (size_t i = 0; i < num_col_partitions; i++) {
        size_t num_packets_each_virtual_channel[INTERLEAVE_FACTOR];
        for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
          size_t vc = pc + f * NUM_HBM_CHANNELS;
          auto indptr_partition = cpsr_matrix_AD.get_packed_indptr(j, i, vc);
          uint32_t num_packets =
              *std::max_element(indptr_partition.back().data,
                                indptr_partition.back().data + PACK_SIZE);
          num_packets_each_virtual_channel[f] = num_packets;
        }
        uint32_t max_num_packets = *std::max_element(
            num_packets_each_virtual_channel,
            num_packets_each_virtual_channel + INTERLEAVE_FACTOR);
        for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
          size_t vc = pc + f * NUM_HBM_CHANNELS;
          auto indices_partition = cpsr_matrix_AD.get_packed_indices(j, i, vc);
          ADchannel_indices[vc].insert(ADchannel_indices[vc].end(),
                                       indices_partition.begin(),
                                       indices_partition.end());
          auto vals_partition = cpsr_matrix_AD.get_packed_data(j, i, vc);
          ADchannel_vals[vc].insert(ADchannel_vals[vc].end(),
                                    vals_partition.begin(),
                                    vals_partition.end());
          ADchannel_indices[vc].resize(
              ADchannel_partition_indptr[vc][j * num_col_partitions + i].start +
              max_num_packets);
          ADchannel_vals[vc].resize(
              ADchannel_partition_indptr[vc][j * num_col_partitions + i].start +
              max_num_packets);
          assert(ADchannel_indices[vc].size() == ADchannel_vals[vc].size());
          auto indptr_partition = cpsr_matrix_AD.get_packed_indptr(j, i, vc);
          ADchannel_partition_indptr[vc][j * num_col_partitions + i].nnz =
              indptr_partition.back();
          if (!((j == (num_row_partitions - 1)) &&
                (i == (num_col_partitions - 1)))) {
            ADchannel_partition_indptr[vc][j * num_col_partitions + i + 1]
                .start =
                ADchannel_partition_indptr[vc][j * num_col_partitions + i]
                    .start +
                max_num_packets;
          }
        }
      }
    }

    ADchannel_packets[pc].resize(num_partitions * (1 + INTERLEAVE_FACTOR) +
                                 ADchannel_indices[pc].size() *
                                     INTERLEAVE_FACTOR);
    // partition indptr
    for (size_t ij = 0; ij < num_partitions; ij++) {
      ADchannel_packets[pc][ij * (1 + INTERLEAVE_FACTOR)].indices.data[0] =
          ADchannel_partition_indptr[pc][ij].start * INTERLEAVE_FACTOR;
      for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
        size_t vc = pc + f * NUM_HBM_CHANNELS;
        ADchannel_packets[pc][ij * (1 + INTERLEAVE_FACTOR) + 1 + f].indices =
            ADchannel_partition_indptr[vc][ij].nnz;
      }
    }
    // matrix indices and vals
    uint32_t offset = num_partitions * (1 + INTERLEAVE_FACTOR);
    for (size_t i = 0; i < ADchannel_indices[pc].size(); i++) {
      for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
        size_t vc = pc + f * NUM_HBM_CHANNELS;
        size_t ii = i * INTERLEAVE_FACTOR + f;
        ADchannel_packets[pc][offset + ii].indices = ADchannel_indices[vc][i];
        ADchannel_packets[pc][offset + ii].vals = ADchannel_vals[vc][i];
      }
    }
  }
  // Iterate virtual channels and map virtual channels (vc) to physical channels
  // (pc)
  for (size_t pc = 0; pc < NUM_HBM_CHANNELS; pc++) {
    for (size_t j = 0; j < num_row_partitions; j++) {
      for (size_t i = 0; i < num_col_partitions; i++) {
        size_t num_packets_each_virtual_channel[INTERLEAVE_FACTOR];
        for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
          size_t vc = pc + f * NUM_HBM_CHANNELS;
          auto indptr_partition = cpsr_matrix_D.get_packed_indptr(j, i, vc);
          uint32_t num_packets =
              *std::max_element(indptr_partition.back().data,
                                indptr_partition.back().data + PACK_SIZE);
          num_packets_each_virtual_channel[f] = num_packets;
        }
        uint32_t max_num_packets = *std::max_element(
            num_packets_each_virtual_channel,
            num_packets_each_virtual_channel + INTERLEAVE_FACTOR);
        for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
          size_t vc = pc + f * NUM_HBM_CHANNELS;
          auto indices_partition = cpsr_matrix_D.get_packed_indices(j, i, vc);
          Dchannel_indices[vc].insert(Dchannel_indices[vc].end(),
                                      indices_partition.begin(),
                                      indices_partition.end());
          auto vals_partition = cpsr_matrix_D.get_packed_data(j, i, vc);
          Dchannel_vals[vc].insert(Dchannel_vals[vc].end(),
                                   vals_partition.begin(),
                                   vals_partition.end());
          Dchannel_indices[vc].resize(
              Dchannel_partition_indptr[vc][j * num_col_partitions + i].start +
              max_num_packets);
          Dchannel_vals[vc].resize(
              Dchannel_partition_indptr[vc][j * num_col_partitions + i].start +
              max_num_packets);
          assert(Dchannel_indices[vc].size() == Dchannel_vals[vc].size());
          auto indptr_partition = cpsr_matrix_D.get_packed_indptr(j, i, vc);
          Dchannel_partition_indptr[vc][j * num_col_partitions + i].nnz =
              indptr_partition.back();
          if (!((j == (num_row_partitions - 1)) &&
                (i == (num_col_partitions - 1)))) {
            Dchannel_partition_indptr[vc][j * num_col_partitions + i + 1]
                .start =
                Dchannel_partition_indptr[vc][j * num_col_partitions + i]
                    .start +
                max_num_packets;
          }
        }
      }
    }

    Dchannel_packets[pc].resize(num_partitions * (1 + INTERLEAVE_FACTOR) +
                                Dchannel_indices[pc].size() *
                                    INTERLEAVE_FACTOR);
    // partition indptr
    for (size_t ij = 0; ij < num_partitions; ij++) {
      Dchannel_packets[pc][ij * (1 + INTERLEAVE_FACTOR)].indices.data[0] =
          Dchannel_partition_indptr[pc][ij].start * INTERLEAVE_FACTOR;
      for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
        size_t vc = pc + f * NUM_HBM_CHANNELS;
        Dchannel_packets[pc][ij * (1 + INTERLEAVE_FACTOR) + 1 + f].indices =
            Dchannel_partition_indptr[vc][ij].nnz;
      }
    }
    // matrix indices and vals
    uint32_t offset = num_partitions * (1 + INTERLEAVE_FACTOR);
    for (size_t i = 0; i < Dchannel_indices[pc].size(); i++) {
      for (size_t f = 0; f < INTERLEAVE_FACTOR; f++) {
        size_t vc = pc + f * NUM_HBM_CHANNELS;
        size_t ii = i * INTERLEAVE_FACTOR + f;
        Dchannel_packets[pc][offset + ii].indices = Dchannel_indices[vc][i];
        Dchannel_packets[pc][offset + ii].vals = Dchannel_vals[vc][i];
      }
    }
  }
  // std::cout << "INFO : Matrix loading/preprocessing complete!" << std::endl;

  //--------------------------------------------------------------------
  // ready input vector
  //--------------------------------------------------------------------  
  std::vector<aligned_vector<PACKED_VAL_T>> sm_fpga((loop_count+1),aligned_vector<PACKED_VAL_T> (AD.num_cols / PACK_SIZE));
  aligned_vector<PACKED_VAL_T> sm_ping(AD.num_cols / PACK_SIZE);
  aligned_vector<PACKED_VAL_T> sm_pong(AD.num_rows / PACK_SIZE);
  //we do the rest later while invoking the kernel

  //--------------------------------------------------------------------
  // allocate space for buffers
  //--------------------------------------------------------------------
  // we dont do that 
  // std::cout << "INFO : Input/result initialization complete!" << std::endl;

  //--------------------------------------------------------------------
  // allocate memory on FPGA and move data
  //--------------------------------------------------------------------
  cl_int err;

  // handle matrix
  std::vector<cl::Buffer> ADchannel_packets_buf(NUM_HBM_CHANNELS);
  cl_mem_ext_ptr_t ADchannel_packets_ext[NUM_HBM_CHANNELS];
  for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
    ADchannel_packets_ext[c].obj = ADchannel_packets[c].data();
    ADchannel_packets_ext[c].param = 0;
    ADchannel_packets_ext[c].flags = HBM[c];
    size_t channel_packets_size =
        sizeof(SPMV_MAT_PKT_T) * ADchannel_packets[c].size();
    if ((2*channel_packets_size) >= 256 * 1000 * 1000) {
      std::cout << "Error: Trying to allocate "
                << channel_packets_size / 1000 / 1000 << " MB on HBM channel "
                << c << std::endl
                << ", but the capcity of one HBM channel is 256 MB."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    ADchannel_packets_buf[c] = CL_BUFFER_RDONLY(
        runtime.context, channel_packets_size, ADchannel_packets_ext[c], err);
    CHECK_ERR(err);
  }
  std::vector<cl::Buffer> Dchannel_packets_buf(NUM_HBM_CHANNELS);
  cl_mem_ext_ptr_t Dchannel_packets_ext[NUM_HBM_CHANNELS];
  for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
    Dchannel_packets_ext[c].obj = Dchannel_packets[c].data();
    Dchannel_packets_ext[c].param = 0;
    Dchannel_packets_ext[c].flags = HBM[c];
    size_t channel_packets_size =
        sizeof(SPMV_MAT_PKT_T) * Dchannel_packets[c].size();
    if (channel_packets_size >= 256 * 1000 * 1000) {
      std::cout << "Error: Trying to allocate "
                << channel_packets_size / 1000 / 1000 << " MB on HBM channel "
                << c << std::endl
                << ", but the capcity of one HBM channel is 256 MB."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    Dchannel_packets_buf[c] = CL_BUFFER_RDONLY(
        runtime.context, channel_packets_size, Dchannel_packets_ext[c], err);
    CHECK_ERR(err);
  }

  // transfer data
  for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
                       {ADchannel_packets_buf[c]}, 0 /* 0 means from host*/));
  }
  for (size_t c = 0; c < NUM_HBM_CHANNELS; c++) {
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
                       {Dchannel_packets_buf[c]}, 0 /* 0 means from host*/));
  }
  CHECK_ERR(err);
  // std::cout << "INFO : Host -> Device data transfer complete!" << std::endl;

  //--------------------------------------------------------------------
  // invoke kernel
  //--------------------------------------------------------------------
  // set kernel arguments that won't change across row iterations
  
  
  
  uint32_t sz = sm_size;

  
  //--------------------------------------------------------------------
  // 					ready input vector --start
  //--------------------------------------------------------------------
	size_t ping_size = sizeof(VAL_T) * AD.num_cols;
	size_t pong_size = sizeof(VAL_T) * AD.num_rows;
	for (size_t i = 0; i < sm_ping.size(); i++) {
		for (size_t k = 0; k < PACK_SIZE; k++) {
			sm_fpga[0][i].data[k] = VAL_T(sm[0][i * PACK_SIZE + k]);
		}
	}
  std::vector<cl::Buffer> sm_bufs(loop_count+1);
  cl_mem_ext_ptr_t sm_packets_ext[loop_count+1];
  for(uint32_t i=0;i<=loop_count;i++){     
    sm_packets_ext[i].obj = sm_fpga[i].data();
    sm_packets_ext[i].param = 0;
    sm_packets_ext[i].flags = HBM[20+uint32_t(i/((loop_count+1)/2 +1))];
    //std::cout << "INFO : Allocated in " << 20+uint32_t(i/((loop_count+1)/2 +1)) << "," << i << "of size " << ping_size << " !" << std::endl;
    sm_bufs[i] = CL_BUFFER(runtime.context, ping_size , sm_packets_ext[i], err);
	  CHECK_ERR(err);
  }
	CHECK_ERR(err);
	// Handle vector and result
	CL_CREATE_EXT_PTR(ping, sm_ping.data(), HBM[20]);
	CL_CREATE_EXT_PTR(pong, sm_pong.data(), HBM[21]);
	cl::Buffer ping_buf = CL_BUFFER(runtime.context, ping_size, ping, err);
	cl::Buffer pong_buf = CL_BUFFER(runtime.context, pong_size, pong, err);
	CHECK_ERR(err);
  
	//transfers all zeroes to output
  OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
                       {pong_buf}, 0 /* 0 means from host*/));
  OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
                       {ping_buf}, 0 /* 0 means from host*/));
  runtime.command_queue.finish();
  for(uint32_t j=0;j<=loop_count;j++){
    OCL_CHECK(err, err = runtime.command_queue.enqueueMigrateMemObjects(
                         {sm_bufs[j]}, 0 /* 0 means from host*/));
    runtime.command_queue.finish();
  }
  // std::cout << "INFO : Multiple buffers Host -> Device data transfer complete!" << std::endl;
  
  
  for (uint32_t loop = 0; loop < loop_count; loop++) {
    
    

    //--------------------------------------------------------------------
    // 					ready input vector --done!
    //--------------------------------------------------------------------
	
    // first
    // pong=D*sm_bufs[loop]
    for (size_t c = 0; c < SK0_CLUSTER; c++) {
      OCL_CHECK(err,
                err = runtime.spmv_sk0.setArg(c, Dchannel_packets_buf[c]));
    }
    for (size_t c = 0; c < SK1_CLUSTER; c++) {
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(
                         c, Dchannel_packets_buf[c + SK0_CLUSTER]));
    }
    for (size_t c = 0; c < SK2_CLUSTER; c++) {
      OCL_CHECK(err,
                err = runtime.spmv_sk2.setArg(
                    c, Dchannel_packets_buf[c + SK0_CLUSTER + SK1_CLUSTER]));
    }
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(0, sm_bufs[loop]));
    OCL_CHECK(err,
              err = runtime.vector_loader.setArg(1, (unsigned)AD.num_cols));
    OCL_CHECK(err, err = runtime.result_drain.setArg(0, pong_buf));

    size_t rows_per_ch_in_last_row_part;
    if (AD.num_rows % LOGICAL_OB_SIZE == 0) {
      rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
      rows_per_ch_in_last_row_part =
          AD.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }
    for (size_t row_part_id = 0; row_part_id < num_row_partitions;
         row_part_id++) {
      unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
      if (row_part_id == num_row_partitions - 1) {
        part_len = rows_per_ch_in_last_row_part;
      }
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err,
                err = runtime.result_drain.setArg(1, (unsigned)row_part_id));

      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.vector_loader));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk0));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk1));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk2));
      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.result_drain));
      runtime.command_queue.finish();
    }
	
    // ping=AD*pong
    for (size_t c = 0; c < SK0_CLUSTER; c++) {
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(c, ADchannel_packets_buf[c]));
    }
    for (size_t c = 0; c < SK1_CLUSTER; c++) {
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(
                         c, ADchannel_packets_buf[c + SK0_CLUSTER]));
    }
    for (size_t c = 0; c < SK2_CLUSTER; c++) {
      OCL_CHECK(err,
                err = runtime.spmv_sk2.setArg(
                    c, ADchannel_packets_buf[c + SK0_CLUSTER + SK1_CLUSTER]));
    }
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(0, pong_buf));
    OCL_CHECK(err,
              err = runtime.vector_loader.setArg(1, (unsigned)AD.num_cols));
    OCL_CHECK(err, err = runtime.result_drain.setArg(0, ping_buf));

    if (AD.num_rows % LOGICAL_OB_SIZE == 0) {
      rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
      rows_per_ch_in_last_row_part =
          AD.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }
    for (size_t row_part_id = 0; row_part_id < num_row_partitions;
         row_part_id++) {
      unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
      if (row_part_id == num_row_partitions - 1) {
        part_len = rows_per_ch_in_last_row_part;
      }
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err,
                err = runtime.result_drain.setArg(1, (unsigned)row_part_id));

      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.vector_loader));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk0));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk1));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk2));
      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.result_drain));
      runtime.command_queue.finish();
    }
    // sm_bufs[loop+1]=D*ping
    for (size_t c = 0; c < SK0_CLUSTER; c++) {
      OCL_CHECK(err,
                err = runtime.spmv_sk0.setArg(c, Dchannel_packets_buf[c]));
    }
    for (size_t c = 0; c < SK1_CLUSTER; c++) {
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(
                         c, Dchannel_packets_buf[c + SK0_CLUSTER]));
    }
    for (size_t c = 0; c < SK2_CLUSTER; c++) {
      OCL_CHECK(err,
                err = runtime.spmv_sk2.setArg(
                    c, Dchannel_packets_buf[c + SK0_CLUSTER + SK1_CLUSTER]));
    }
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 4,
                                                 (unsigned)num_col_partitions));
    OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 5,
                                                 (unsigned)num_partitions));
    OCL_CHECK(err, err = runtime.vector_loader.setArg(0, ping_buf));
    OCL_CHECK(err,
              err = runtime.vector_loader.setArg(1, (unsigned)AD.num_cols));
    OCL_CHECK(err, err = runtime.result_drain.setArg(0, sm_bufs[loop+1]));

    if (AD.num_rows % LOGICAL_OB_SIZE == 0) {
      rows_per_ch_in_last_row_part = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    } else {
      rows_per_ch_in_last_row_part =
          AD.num_rows % LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
    }
    for (size_t row_part_id = 0; row_part_id < num_row_partitions;
         row_part_id++) {
      unsigned part_len = LOGICAL_OB_SIZE / NUM_HBM_CHANNELS;
      if (row_part_id == num_row_partitions - 1) {
        part_len = rows_per_ch_in_last_row_part;
      }
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk0.setArg(SK0_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk1.setArg(SK1_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 2,
                                                   (unsigned)row_part_id));
      OCL_CHECK(err, err = runtime.spmv_sk2.setArg(SK2_CLUSTER + 3,
                                                   (unsigned)part_len));
      OCL_CHECK(err,
                err = runtime.result_drain.setArg(1, (unsigned)row_part_id));

      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.vector_loader));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk0));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk1));
      OCL_CHECK(err, err = runtime.command_queue.enqueueTask(runtime.spmv_sk2));
      OCL_CHECK(err,
                err = runtime.command_queue.enqueueTask(runtime.result_drain));
      runtime.command_queue.finish();
    }
  }
  std::cout << "INFO : Run complete!" << std::endl;
  
  for (uint32_t loop = 0; loop < loop_count; loop++) {
    runtime.command_queue.enqueueMigrateMemObjects({sm_bufs[loop+1]}, CL_MIGRATE_MEM_OBJECT_HOST);
    runtime.command_queue.finish();
  }
  for (uint32_t loop = 0; loop < loop_count; loop++) {
    std::vector<float> sm_ot;
    std::vector<float> sm_norm(mx);
	
    sm_ot = SubRoutine002Filter_opt(sm_fpga[loop+1],sz);
    sm_norm = SubRoutine003Filter(sm_ot,sz);
    for (uint32_t j = 0; j < mx; j++) {
      sm_vec[j][loop] = sm_norm[j];
    }
  }
  return true;
}



std::vector<std::vector<float>>Filter_hw(cl_runtime &runtime, std::vector<float> rv, uint32_t k, spmv::io::CSCMatrix <float> &AD, uint32_t mx, uint32_t initial, uint32_t interval, uint32_t Ntot){
  std::vector<std::vector<float>> V(mx,std::vector<float>(Ntot));
  spmv::io::CSCMatrix<float> AD_diagnal;
  spmv::io::CSCMatrix<float> D;
  std::vector<float> AD_diagnal_val = {0.1};
  uint32_t sz;
  std::vector<std::vector<float>> sm_vec( mx , std::vector<float> (k));
  std::vector<float> sm_ot;
  std::vector<std::vector<float>> sm( k+1 , std::vector<float> (((rv.size()+PACK_SIZE-1)/PACK_SIZE)*PACK_SIZE));
  std::vector<float> sm_tmp;
  std::vector<float> sm_norm;
  uint32_t count;

  sz = AD.num_rows;
  AD_diagnal = create_diagonal_matrix(AD_diagnal_val,sz);
	
  AD = sum(AD,AD_diagnal);
	
  D=SubRoutine001Filter(AD);
	
  sm_ot = SubRoutine002Filter(rv,AD.num_rows);
	
  sm_tmp = SubRoutine003Filter(sm_ot,AD.num_rows);

  for(uint32_t i=0;i<sm_tmp.size();i++)
    sm[0][i]=sm_tmp[i];

  count = 1;

  spmv::io::CSRMatrix<float> AD_dash;
  spmv::io::CSRMatrix<float> D_dash;
  D_dash = csc2csr(D);
  AD_dash = csc2csr(AD);
  
  bool status =  hyperef_filter_kernel_accel(runtime,mx,sz,
                     AD_dash,
                     D_dash, sm,
                     sm_vec, k);
  
  for(uint32_t j=0;j<mx;j++){
      for(uint32_t i=0;i<Ntot;i++){
          V[j][i]=sm_vec[j][(i*interval)+interval-1];
      }
  }
  
  return V;
}

//---------------------------------------------------------------
// test cases
//---------------------------------------------------------------
std::string GRAPH_DATASET_DIR = "../datasets/graph/";
std::string NN_DATASET_DIR = "../datasets/pruned_nn/";
std::string HYPEREF_DATASET_DIR = "../datasets/hmetis/";

bool	test_hyperef(cl_runtime & runtime, std::string dataset) {
	std::cout << "------ Running test: on hmetis ("<<dataset<<") " << std::endl;
	spmv::io::CSCMatrix<float> ar = load_csr_matrix_from_float_hmetis_unweighted(    HYPEREF_DATASET_DIR + dataset);

  uint32_t L = 1;

  std::vector<float> W(ar.num_cols);
  std::fill(W.begin(), W.end(), 1.0);

  for (uint32_t loop = L; loop <= L; loop++) {
    uint32_t mx;
    uint32_t initial;
    uint32_t SmS;
    uint32_t interval;
    uint32_t Nrv;
    uint32_t Nsm;
    uint32_t Ntot;

    mx = ar.num_rows;

    spmv::io::CSCMatrix<float> A;
    // star expansion
    A = StarW(ar, W);
    // computing the smoothed vectors
    initial = 0;

    SmS = 300;

    interval = 20;

    Nrv = 1;

    Nsm = int(float(SmS - initial) / float(interval));

    Ntot = Nrv * Nsm;

    // Qvec = zeros(Float64, 0);

    // Eratio = zeros(Float64, length(ar), Ntot)

    // SV = zeros(Float64, mx, Ntot)

    std::vector<std::vector<float>> SV(mx, std::vector<float>(Ntot));
    for (uint32_t ii = 0; ii < Nrv; ii++) {
      // std::vector<std::vector<float>>   sm(mx,std::vector<float>(Nsm));  // vector<vector<int>> vec( n , vector<int> (m, 0));
      std::vector<std::vector<float>>   sm;  // vector<vector<int>> vec( n , vector<int> (m, 0));
      // sm = zeros(mx, Nsm)
      std::vector<float> rv(A.num_rows);

      std::generate(rv.begin(), rv.end(), [&](){return
       ((float(get_random(0,INT32_MAX) / float(INT32_MAX)))- 0.5)*2.0;});

      /*std::string line;
      std::ifstream myFile;                         // creates stream myFile
      myFile.open(HYPEREF_DATASET_DIR + "RVdata");  // opens .txt file
      uint32_t lines_read;
      lines_read = 0;
      while (myFile.good()) {
        line.clear();
        getline(myFile, line);
        rv[lines_read++] = std::strtof(line.c_str(), nullptr);
        if (lines_read == A.num_rows) break;
      }
      myFile.close();*/

      sm = Filter_hw(runtime, rv, SmS, A, mx, initial, interval, Nsm);
		
    // #ifdef 0  
	  // std::cout << "Now I am here   " << sm.size() << std::endl;
      /*std::ofstream outfile("Debug");
      for (uint32_t j = 0; j < mx; j++) {
		  // std::cout << sm[j][Nsm-1] << j << "," << Nsm-1 << std::endl;
        outfile << sm[j][Nsm - 1] << std::endl;
      }
      outfile.close();*/

      // SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm
      for (uint32_t j = 0; j < mx; j++) {
        for (uint32_t i = 0; i < Nsm; i++) {
          SV[j][ii * Nsm + i] = sm[j][i];
        }
      }
    }
  }

  return true;
}

//---------------------------------------------------------------
// main
//---------------------------------------------------------------

int main(int argc, char **argv) {
	
  // parse command-line arguments
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <hw_emu/hw> <xclbin>" << std::endl;
    return 0;
  }
  std::string target = argv[1];
  std::string xclbin = argv[2];
  std::string dataset = argv[3];
  
  if (target != "hw_emu" && target != "hw") {
    std::cout << "This host program only support hw_emu and hw!" << std::endl;
    return 1;
  }

  // setup Xilinx openCL runtime
  cl_runtime runtime;
  cl_int err;
  if (target == "sw_emu" || target == "hw_emu") {
    setenv("XCL_EMULATION_MODE", target.c_str(), true);
  }
  cl::Device device;
  bool found_device = false;
  auto devices = xcl::get_xil_devices();
  for (size_t i = 0; i < devices.size(); i++) {
    if (devices[i].getInfo<CL_DEVICE_NAME>() == "xilinx_u280_xdma_201920_3") {
      device = devices[i];
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    std::cout << "ERROR : Failed to find "
              << "xilinx_u280_xdma_201920_3"
              << ", exit!\n";
    exit(EXIT_FAILURE);
  }
  runtime.context = cl::Context(device, NULL, NULL, NULL);
  auto file_buf = xcl::read_binary_file(xclbin);
  cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
  cl::Program program(runtime.context, {device}, binaries, NULL, &err);
  if (err != CL_SUCCESS) {
    std::cout << "ERROR : Failed to program device with xclbin file"
              << std::endl;
    return 1;
  } else {
    std::cout << "INFO : Successfully programmed device with xclbin file"
              << std::endl;
  }
  OCL_CHECK(err, runtime.spmv_sk0 = cl::Kernel(program, "spmv_sk0", &err));
  OCL_CHECK(err, runtime.spmv_sk1 = cl::Kernel(program, "spmv_sk1", &err));
  OCL_CHECK(err, runtime.spmv_sk2 = cl::Kernel(program, "spmv_sk2", &err));
  OCL_CHECK(err, runtime.vector_loader =
                     cl::Kernel(program, "spmv_vector_loader", &err));
  OCL_CHECK(err, runtime.result_drain =
                     cl::Kernel(program, "spmv_result_drain", &err));

  OCL_CHECK(err, runtime.command_queue =
                     cl::CommandQueue(runtime.context, device,
                                      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                          CL_QUEUE_PROFILING_ENABLE,
                                      &err));

  // run tests
  bool passed = true;
  std::string testm[18] = {"01","02","03","04","05","06","07","08","09",
					"10","11","12","13","14","15","16","17","18"};
  std::string fileh = "ibm";
  std::string filet = ".hgr";
  std::string  runall_str = "all";
  std::string  run1butrepeat_str = "repeat";
  if(dataset.compare(runall_str) == 0 ){
	for(int i =0; i<18; i++){
		std::string datasetname = fileh+testm[i]+filet;
		passed = passed && test_hyperef(runtime,datasetname);
	}	  
  } else {
	  if(dataset.compare(run1butrepeat_str)==0){
			srand(time(NULL));
			int i=rand() % 18 + 1;
			std::string datasetname = fileh+testm[i]+filet;
			for(int i =0; i<16; i++){
				passed = passed && test_hyperef(runtime,datasetname);
			}
	  } else {
			passed = passed && test_hyperef(runtime,dataset);
	  }
  }

  std::cout << (passed ? "===== All Test Passed! ====="
                       : "===== Test FAILED! =====")
            << std::endl;
  return passed ? 0 : 1;
}
