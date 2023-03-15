#include <stdio.h>
#include <stdint.h>
#include "libsmctrl.h"

int main() {
	uint32_t num_gpcs;
	uint64_t* masks;
	libsmctrl_get_gpc_info(&num_gpcs, &masks, 1);
	printf("Num GPCs: %d\n", num_gpcs);
	for (int i = 0; i < num_gpcs; i++) {
		printf("Mask of TPCs associated with GPC %d: %#018lx\n", i, masks[i]);
	}
	return 0;
}
