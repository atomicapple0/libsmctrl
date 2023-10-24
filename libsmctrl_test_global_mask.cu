#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"
#include "testbench.h"

__global__ void read_and_store_smid(uint8_t* smid_arr) {
  if (threadIdx.x != 1)
    return;
  int smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  smid_arr[blockIdx.x] = smid;
}

// Assuming SMs continue to support a maximum of 2048 resident threads, six
// blocks of 1024 threads should span at least three SMs without partitioning
#define NUM_BLOCKS 6

int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

// Warning: Mutates input array via qsort
int count_unique(uint8_t* arr, int len) {
  qsort(arr, len, 1, sort_asc);
  int num_uniq = 1;
  for (int i = 0; i < len - 1; i++)
    num_uniq += (arr[i] != arr[i + 1]);
  return num_uniq;
}

int main() {
  cudaError_t err; // Needed by SAFE() macro
  int res;
  uint8_t *smids_native_d, *smids_native_h;
  uint8_t *smids_partitioned_d, *smids_partitioned_h;
  int uniq_native, uniq_partitioned;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc;

  // Determine number of SMs per TPC
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_test_global: Unable to get TPC configuration for test");
  sms_per_tpc = num_sms/num_tpcs;

  // Test baseline (native) behavior without partitioning
  SAFE(cudaMalloc(&smids_native_d, NUM_BLOCKS));
  if (!(smids_native_h = (uint8_t*)malloc(NUM_BLOCKS)))
    error(1, errno, "libsmctrl_test_global: Unable to allocate memory for test");
  read_and_store_smid<<<NUM_BLOCKS, 1024>>>(smids_native_d);
  SAFE(cudaMemcpy(smids_native_h, smids_native_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

  uniq_native = count_unique(smids_native_h, NUM_BLOCKS);
  if (uniq_native < sms_per_tpc) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: In baseline test, %d blocks of 1024 "
           "threads were launched on the GPU, but only %d SMs were utilized, "
           "when it was expected that at least %d would be used.\n", NUM_BLOCKS,
           uniq_native, sms_per_tpc);
    return 1;
  }

  // Verify that partitioning changes the SMID distribution
  libsmctrl_set_global_mask(~0x1); // Enable only one TPC
  SAFE(cudaMalloc(&smids_partitioned_d, NUM_BLOCKS));
  if (!(smids_partitioned_h = (uint8_t*)malloc(NUM_BLOCKS)))
    error(1, errno, "libsmctrl_test_global: Unable to allocate memory for test");
  read_and_store_smid<<<NUM_BLOCKS, 1024>>>(smids_partitioned_d);
  SAFE(cudaMemcpy(smids_partitioned_h, smids_partitioned_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

  // Make sure it only ran on the number of TPCs provided
  // May run on up to two SMs, as up to two per TPC
  uniq_partitioned = count_unique(smids_partitioned_h, NUM_BLOCKS);
  if (uniq_partitioned > sms_per_tpc) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: With global TPC mask set to "
           "constrain all kernels to a single TPC, a kernel of %d blocks of "
           "1024 threads was launched and found to run on %d SMs (at most %d---"
           "one TPC---expected).\n", NUM_BLOCKS, uniq_partitioned, sms_per_tpc);
    return 1;
  }

  // Make sure it ran on the right TPC
  if (smids_partitioned_h[NUM_BLOCKS - 1] > sms_per_tpc - 1) {
    printf("libsmctrl_test_global: ***Test failure.***\n"
           "libsmctrl_test_global: Reason: With global TPC mask set to"
           "constrain all kernels to the first TPC, a kernel was run and found "
           "to run on an SM ID as high as %d (max of %d expected).\n",
           smids_partitioned_h[NUM_BLOCKS - 1], sms_per_tpc - 1);
    return 1;
  }

  printf("libsmctrl_test_global: Test passed!\n"
         "libsmctrl_test_global: Reason: With a global partition enabled which "
         "contained only TPC ID 0, the test kernel was found to use only %d "
         "SMs (%d without), and all SMs in-use had IDs below %d (were contained"
         " in the first TPC).\n", uniq_partitioned, uniq_native, sms_per_tpc);
  return 0;
}
