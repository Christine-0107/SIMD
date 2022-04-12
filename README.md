# SIMD
SIMD编程作业
进行了普通高斯和特殊高斯消去算法的SIMD优化。
普通高斯在ARM平台、Windowsx86平台、Intel DevCloud x86平台进行了实验，运用了Neon、SSE、AVX、AVX512指令集，讨论了对齐与不对齐、优化不同位置等编程策略带来的影响。
