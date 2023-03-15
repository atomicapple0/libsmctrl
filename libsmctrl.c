/**
 * Copyright 2022 Joshua Bakita
 * Library to control SM masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */

//#include "/playpen/playpen/cuda-11.8/include/cuda.h"
#include <cuda.h>
//#include <cuda_runtime.h>
//#ifndef CUDA_VERSION
//#warning libsmctrl: CUDA driver library must be included before libsmctrl.h.
//#endif

#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

// Layout of mask control fields in CUDA's `globals` struct
struct global_sm_control {
	uint32_t enabled;
	uint64_t mask;
} __attribute__((packed));

/*** CUDA Globals Manipulation. CUDA 10.2 only ***/

// Ends up being 0x7fb7fa3408 in some binaries
static struct global_sm_control* g_sm_control = NULL;

/* Find the location of CUDA's `globals` struct and the SM mask control fields
 * No symbols are exported from within `globals`, so this has to do a very
 * messy lookup, following the pattern of the assembly of `cuDeviceGetCount()`.
 * Don't call this before the cuda library has been initialized.
 */
static void setup_sm_control_10() {
	if (g_sm_control)
		return;
	// Defeat relocation. cudbgReportDriverApiErrorFlags is relocated by
	// the loader, but not subject to ASLR (it's always at a constant
	// offset in the loaded instance of libcuda.so). Our target is also at
	// a constant offset, so we can use the address of
	// cudbgReportDriverApiErrorFlags as a reference point.
	// Note: cudbgReportDriverApiErrorFlags is currently the closest known
	// symbol to **the table**. cudbgDebuggerInitialized is the closest to
	// globals itself (+7424 == SM mask control), but we perfer the table
	// lookup approach for now, as that's what cuDeviceGetCount() does.
	extern uint32_t cudbgReportDriverApiErrorFlags;
	uint32_t* sym = 0;//&cudbgReportDriverApiErrorFlags;
	// In some binaries, the following works out to 0x7fb7ea6000, and
	// that's what shows up in the adrp instruction in cuDeviceGetCount()
	// in the lead-up to get globals.numDevices. Find this offset by
	// calling cuDeviceGetCount(0xdeadbeef), catching the segfault in GDB,
	// disassembling the prior instructions, taking the adrp constant, and
	// subtracting the address of cudbgReportDriverApiErrorFlags from it.
	uintptr_t* tbl_base = (uintptr_t*)((uintptr_t)sym - 1029868);
	// Address of `globals` is at offset 3672 (entry 459?)
	uintptr_t globals_addr = *(tbl_base + 459); // Offset 3672 on aarch64
	// SM mask control is at offset 4888 in the `globals` struct
	g_sm_control = (struct global_sm_control*)(globals_addr + 4888);
	// SM mask should be empty by default
	if (g_sm_control->enabled || g_sm_control->mask)
		fprintf(stderr, "Warning: Found non-NULL SM disable mask during setup! g_sm_control is likely invalid---use at own risk.\n");
}

/*** QMD/TMD-based SM Mask Control via Debug Callback. CUDA 11+ ***/

// Tested working on CUDA x86_64 11.0-11.8.
// Tested not working on aarch64 or x86_64 10.2
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
static uint64_t g_sm_mask = 0;
static __thread uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;
static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
	if (*(uint32_t*)in_params < 0x50) {
		fprintf(stderr, "Unsupported CUDA version for callback-based SM masking. Aborting...\n");
		return;
	}
	if (!**((uintptr_t***)in_params+8)) {
		fprintf(stderr, "Called with NULL halLaunchDataAllocation\n");
		return;
	}
	//fprintf(stderr, "cta: %lx\n", *(uint64_t*)(**((char***)in_params + 8) + 74));
	// TODO: Check for supported QMD version (>XXX, <4.00)
	// TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 72 bytes (rather than 8 bytes) wide
	uint32_t *lower_ptr = (uint32_t*)(**((char***)in_params + 8) + 84);
	uint32_t *upper_ptr = (uint32_t*)(**((char***)in_params + 8) + 88);
	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}
	//fprintf(stderr, "lower mask: %x\n", *lower_ptr);
	//fprintf(stderr, "upper mask: %x\n", *upper_ptr);
}

static void setup_sm_control_11() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup can only be called once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, launchCallback, NULL);
	if (res) {
		fprintf(stderr, "libsmctrl: Error subscribing to launch callback. Error %d\n", res);
		return;
	}
	res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
	if (res)
		fprintf(stderr, "libsmctrl: Error enabling launch callback. Error %d\n", res);
}

// Common masking control
void libsmctrl_set_global_mask(uint64_t mask) {
	int ver;
	cuDriverGetVersion(&ver);
	if (ver <= 10020) {
		if (!g_sm_control)
			setup_sm_control_10();
		g_sm_control->mask = mask;
		g_sm_control->enabled = 1;
	} else {
		if (!sm_control_setup_called)
			setup_sm_control_11();
		g_sm_mask = mask;
	}
}

void set_sm_mask(uint64_t mask) {
	libsmctrl_set_global_mask(mask);
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask(uint64_t mask) {
	if (!sm_control_setup_called)
		setup_sm_control_11();
	g_next_sm_mask = mask;
}


/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
// CUDA 9.0 and 9.1 use the same offset
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x24c
// CUDA 10.0, 10.1 and 10.2 use the same offset
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset

// Layout in CUDA's `stream` struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
} __attribute__((packed));

// Should work for CUDA 9.1, 10.0-11.8, 12.0-12.1
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
		break;
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	default: {
		// For experimenting to determine the right mask offset, set MASK_OFF (positive and negative numbers supported)
		char* mask_off_str = getenv("MASK_OFF");
		fprintf(stderr, "libsmctrl: Stream masking unsupported on this CUDA version (%d)!\n", ver);
		if (mask_off_str) {
			int off = atoi(mask_off_str);
			fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 11.8 base %#x (total off: %#x)\n", off, CU_11_8_MASK_OFF, CU_11_8_MASK_OFF+off);
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF + off);
		} else {
			return;
		}}
	}

	hw_mask->upper = mask >> 32;
	hw_mask->lower = mask;
}

int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	int num_sms;
	int major;
	int minor;
	// TODO: Use nvdebug instead of this hardcoded hack
	cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	// SM masking only works on sm_35+
	if (major < 3 || (major == 3 && minor < 5))
		return -ENOTSUP;
	// Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
	// as the P100, which is uniquely sm_60
	int sms_per_tpc;
	if (major > 6 || (major == 6 && minor == 0))
		sms_per_tpc = 2;
	else
		sms_per_tpc = 1;
	// It looks like there may be some upcoming weirdness (TPCs with only one SM?)
	// with Hopper
	if (major >= 9)
		fprintf(stderr, "libsmctrl: WARNING, SM masking is untested on Hopper, and will likely yield incorrect results! Proceed with caution.\n");
	*num_tpcs = num_sms/sms_per_tpc;
	return 0;
}

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return -errno;
	read(fd, f_data, 18);
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

static uint64_t tpc_mask_per_gpc_per_dev[16][12];
// Output mask is vtpc-indexed (virtual TPC)
int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, vtpc_idx = 0;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before using libsmctrl_get_gpc_info()\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return -ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		uint64_t* tpc_mask = &tpc_mask_per_gpc_per_dev[dev][*num_enabled_gpcs - 1];
		*tpc_mask = 0;
		for (j = 0; j < num_tpc_per_gpc; j++) {
				// Skip disabled TPCs
				if ((1 << j) & gpc_tpc_mask)
					continue;
				*tpc_mask |= (1ull << vtpc_idx);
				vtpc_idx++;
		}
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

