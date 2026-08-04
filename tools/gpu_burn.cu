// Heavy multi-engine GPU load for Jetson Orin (sm_87) thermal soak.
// Runs three concurrent streams: tensor-core HMMA, DRAM streaming, FP32 FMA.
// Usage: gpu_burn2 <seconds>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// --- Tensor cores: back-to-back HMMA on 16x16x16 fragments -------------------
__global__ void tc_burn(half *a, half *b, float *c, int iters) {
#if __CUDA_ARCH__ >= 700
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

	wmma::fill_fragment(acc, 0.0f);
	wmma::load_matrix_sync(fa, a, 16);
	wmma::load_matrix_sync(fb, b, 16);
	for (int i = 0; i < iters; ++i) {
		wmma::mma_sync(acc, fa, fb, acc);
		wmma::mma_sync(acc, fa, fb, acc);
		wmma::mma_sync(acc, fa, fb, acc);
		wmma::mma_sync(acc, fa, fb, acc);
	}
	// Guarded store: keeps the work live without polluting DRAM traffic.
	if (acc.x[0] == 1.2345e30f) wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);
#else
	// No tensor cores on this architecture; fall back to FP32 so the stream
	// still contributes load rather than returning immediately.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = idx * 1.0001f;
	for (int i = 0; i < iters; ++i) { x = fmaf(x, 1.0000017f, 0.9999983f); }
	if (x == 1.2345e30f) c[0] = x;
#endif
}

// --- DRAM: wide streaming read-modify-write to load EMC ----------------------
__global__ void bw_burn(float4 *buf, size_t n, int iters) {
	size_t stride = (size_t)gridDim.x * blockDim.x;
	for (int it = 0; it < iters; ++it) {
		for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
			float4 v = buf[i];
			v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
			buf[i] = v;
		}
	}
}

// --- FP32 pipes --------------------------------------------------------------
__global__ void fp_burn(float *out, int iters) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float a = idx * 1.0001f, b = 1.0000017f, c = 0.9999983f;
	for (int i = 0; i < iters; ++i) {
		a = fmaf(a, b, c); a = fmaf(a, c, b);
		a = fmaf(a, b, c); a = fmaf(a, c, b);
	}
	if (a == 12345.6789f) out[idx] = a;
}

int main(int argc, char **argv) {
	int secs = (argc > 1) ? atoi(argv[1]) : 60;

	cudaDeviceProp p;
	if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
	int sms = p.multiProcessorCount;

	// ~512 MB working set: far larger than L2, so every pass goes to DRAM.
	size_t bytes = 512ull << 20;
	size_t n4 = bytes / sizeof(float4);

	half *d_a = nullptr, *d_b = nullptr;
	float *d_c = nullptr, *d_out = nullptr;
	float4 *d_buf = nullptr;
	bool ok = cudaMalloc(&d_a, 16*16*sizeof(half)) == cudaSuccess
	       && cudaMalloc(&d_b, 16*16*sizeof(half)) == cudaSuccess
	       && cudaMalloc(&d_c, 16*16*sizeof(float)) == cudaSuccess
	       && cudaMalloc(&d_out, sms*32*256*sizeof(float)) == cudaSuccess
	       && cudaMalloc(&d_buf, bytes) == cudaSuccess;
	if (!ok) { fprintf(stderr, "alloc failed\n"); return 1; }
	cudaMemset(d_a, 0x3c, 16*16*sizeof(half));
	cudaMemset(d_b, 0x3c, 16*16*sizeof(half));
	cudaMemset(d_buf, 0, bytes);

	cudaStream_t s_tc, s_bw, s_fp;
	cudaStreamCreate(&s_tc); cudaStreamCreate(&s_bw); cudaStreamCreate(&s_fp);

	printf("GPU %s  SMs=%d  CC=%d.%d  working set=%zuMB  duration=%ds\n",
	       p.name, sms, p.major, p.minor, bytes >> 20, secs);
	printf("streams: tensor-core HMMA | DRAM streaming | FP32 FMA\n");
	fflush(stdout);

	time_t end = time(nullptr) + secs;
	unsigned long long rounds = 0;
	while (time(nullptr) < end) {
		tc_burn<<<sms * 16, 32, 0, s_tc>>>(d_a, d_b, d_c, 20000);
		bw_burn<<<sms * 8, 256, 0, s_bw>>>(d_buf, n4, 6);
		fp_burn<<<sms * 8, 256, 0, s_fp>>>(d_out, 20000);
		cudaError_t e = cudaDeviceSynchronize();
		if (e != cudaSuccess) { fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(e)); return 1; }
		++rounds;
	}
	printf("completed %llu rounds\n", rounds);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out); cudaFree(d_buf);
	return 0;
}
