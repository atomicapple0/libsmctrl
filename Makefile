CC = gcc
# -fPIC is needed in all cases, as we may be linked into another shared library
CFLAGS = -fPIC
LDFLAGS = -lcuda -I/usr/local/cuda/include

.PHONY: clean tests

libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ $(CFLAGS) $(LDFLAGS)

libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c
	$(CC) $< -o $@ -L. -lsmctrl $(LDFLAGS)

tests: libsmctrl_test_gpc_info

clean:
	rm -f libsmctrl.so libsmctrl.a
