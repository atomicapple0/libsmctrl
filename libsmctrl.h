/**
 * Copyright 2022 Joshua Bakita
 * Library to control TPC masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* PARTITIONING FUNCTIONS */

// Set global default TPC mask for all kernels, incl. CUDA-internal ones
// @param mask   A bitmask of enabled/disabled TPCs (see Notes on Bitmasks)
// Supported: CUDA 10.2, and CUDA 11.0 - CUDA 11.8
extern void libsmctrl_set_global_mask(uint64_t mask);
// Set default TPC mask for all kernels launched via `stream`
// (overrides global mask)
// @param stream A cudaStream_t (aka CUstream_st*) to apply the mask on
// @param mask   A bitmask of enabled/disabled TPCs (see Notes on Bitmasks)
// Supported: CUDA 8.0 - CUDA 11.8
extern void libsmctrl_set_stream_mask(void* stream, uint64_t mask);
// Set TPC mask for the next kernel launch from the caller's CPU thread
// (overrides global and per-stream masks, applies only to next launch).
// @param mask   A bitmask of enabled/disabled TPCs (see Notes on Bitmasks)
// Supported: CUDA 11.0 - CUDA 11.8
extern void libsmctrl_set_next_mask(uint64_t mask);

// **DEPRECATED**: Old name for libsmctrl_set_global_mask()
extern void set_sm_mask(uint64_t mask) __attribute__((deprecated("Use libsmctrl_set_global_mask()")));

/**
 * Notes on Bitmasks
 *
 * All of the core partitioning functions take a `uint64_t mask` parameter. A
 * set bit in the mask indicates that the respective Thread Processing Cluster
 * (TPC) is to be __disabled__.
 *
 * Examples
 * To prohibit the next kernel from using TPC 0:
 *     libsmctrl_set_next_mask(0x1);
 * Allow kernels to only use TPC 0 by default:
 *     libsmctrl_set_global_mask(~0x1ull);
 * Allow kernels in a stream to only use TPCs 2, 3, and 4:
 *     libsmctrl_set_stream_mask(stream, ~0b00111100ull);
 *
 * Note that the bitwise inversion operator (~, as used above) is very useful,
 * just be sure to apply it to 64-bit integer literals only! (~0x1 != ~0x1ull)
 */

/* INFORMATIONAL FUNCTIONS */

// Get number of GPCs for devices number `dev`, and a GPC-indexed array
// containing masks of which TPCs are associated with each GPC.
// Note that the `nvdebug` module must be loaded to use this function.
// @param  num_enabled_gpcs (out) Location to store number of GPCs in
// @param  tpcs_for_gpc     (out) Pointer to store pointer to output buffer at
// @param  dev               (in) `nvdebug` device ID
// @return 0 on success, `errno`-compatible error code on failure
extern int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev);
// Get total number of TPCs on device number `dev`. Requires `nvdebug`.
// @param  num_tpcs        (out) Location to store number of TPCs at
// @param  dev              (in) `nvdebug` device ID
// @return 0 on success, `errno`-compatible error code on failure
extern int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev);
// Identical to above, but for a CUDA device ID. Does not require `nvdebug`.
extern int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev);

#ifdef __cplusplus
}
#endif
