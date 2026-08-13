# Hardware and Efficiency Measurement Scope

## Artifact-confirmed profile

| Item | Value |
| --- | --- |
| Profiler architecture | Isolated CPU FLOPs/storage worker and fresh-process GPU runtime worker |
| Runtime device recorded | `/GPU:0` |
| Input | Two float32 tensors, each `[1, 2048, 1]` |
| Batch size | 1 |
| Warm-up / timed runs | 100 / 1000 |
| Mean / median latency | 4.4232663 / 4.5292 ms |
| p25 / p75 / p95 latency | 3.82405 / 4.91835 / 5.12766 ms |
| Throughput | 226.077276875688 samples/s |
| TF allocator current / peak | 28.169857 / 41.084896 MiB |
| Trainable parameters | 7,380,173 |
| TensorFlow graph FLOPs | 243,171,883 |
| Estimated MACs | 121,585,941 |
| Weight file / SavedModel size | 28.226341 / 28.758881 MiB |

Latency excludes data loading and uses synchronized batch-1 forward passes.
TensorFlow allocator memory is not total board VRAM. FLOPs are TensorFlow
graph float operations and MACs are estimated as FLOPs divided by two.
Cross-model runtime comparisons require the same script, hardware and
software stack.

The primary evidence is `efficiency_profile.json` (SHA-256
`094bdd0bf45d6b7642e027d95209ffc0974b5050ced7dda826c3264eafbdea0f`) in
`MHFL-MCA Supplementary Experiments/Results/Revision Experiments/full_20260807_r1/03_Efficiency_Profile/`.

## Hardware identity evidence boundary

The formal 2026-08-07 artifact records only `/GPU:0`. It does **not**
serialize the exact GPU model, CPU, RAM, NVIDIA driver, CUDA runtime or exact
cuDNN version. Those fields are **NOT RECORDED** in the immutable runtime
artifact and must not be inferred from the latency result.

## Current audit host

The following values were observed on 2026-08-13. They describe the machine
used to audit the repository and do not prove the identity of the historical
profiling host.

| Component | Observed value |
| --- | --- |
| Computer | Lenovo 83DG |
| CPU | Intel Core i7-14650HX, 16 cores / 24 logical processors |
| RAM | 16,873,545,728 bytes (about 15.71 GiB) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8188 MiB |
| Compute capability | 8.9 |
| NVIDIA driver | 566.24 |
| `nvidia-smi` driver-supported CUDA | 12.7 |
| `nvcc` | CUDA 11.5, V11.5.119 |
| Conda records | `cudatoolkit` 11.5.2; `cudnn` 8.4.1.50 |
| TensorFlow build metadata | CUDA 11.2 (`cudart64_112.dll`); cuDNN 8 (`cudnn64_8.dll`) |
| OS | Windows 11 Home Chinese, 10.0.26200, 64-bit |

The CUDA value printed by `nvidia-smi` is the driver compatibility level, not
the TensorFlow runtime. The `nvcc`, Conda and TensorFlow-build values describe
different software layers and are intentionally reported separately.

## Resource planning

| Scope | Approximate storage |
| --- | ---: |
| Unrestricted repository LFS pull | 8.858 GiB |
| All datasets | 3.937 GiB |
| KAIST vibration + current data | 2.027 GiB |
| UO MATLAB subset used by revision scripts | 0.282 GiB |

The README recommendation of at least 8 GB VRAM is an operational
recommendation, not a measured minimum. Peak training RAM/VRAM, total training
time, energy use and temporary-disk peak were not recorded. See
[`DATA.md`](DATA.md) for selective downloads.
