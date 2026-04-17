# GEMM-full-AIE-PL

This repository contains the hardware and software components for GEMM acceleration on Versal VCK190 platform using the AIE array, programmable logic (FPGA), and host CPU for control.

## Repository structure

- `aie/`
  Contains the AIE codes for GEMM mapping on the Versal AIE array.

- `pl_kernels/`
  Contains the HLS code for GEMM tiling and accumulation in the PL of the Versal device.

- `host_cpu_sw/`
  Contains the host CPU code that allocates buffers and enables accelerator execution.
