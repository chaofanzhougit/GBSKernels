# results/perf/ — static resource usage and profiler captures

`ptxas_*.txt` records each kernel's static resource usage (stack frame,
registers, spills) from a profiling compile (`nvcc -Xptxas -v`) — the valid
profiler evidence in this directory, quoted by the design notes.

`ncu_*_PROFILER_FAILED.csv` are Nsight Compute captures that ran without GPU
performance-counter permission (`ERR_NVGPUCTRPERM`; "No kernels were profiled").
They are retained because `results/` is append-only, but they contain **no
hardware-counter data**, and no occupancy or DRAM-throughput claim may cite them.
A successful counter capture requires a host that grants profiling permission
(`NVreg_RestrictProfilingToAdminUsers=0` or `CAP_PERFMON`).
