

# fork of libsmctrl (you probably want to use the original repo which has new features)
Artifact from [Hardware Compute Partitioning on NVIDIA GPUs*
](https://www.cs.unc.edu/~jbakita/rtas23.pdf) paper. Original repo is [here](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/).

## Build
```
make libsmctrl.a
```
Note: Requires the CUDA toolkit headers to compile, and must be linked with `-lcuda`.
