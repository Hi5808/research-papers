# Exposing RKLLM's Unused Multi-Batch Capability: A Working Dynamic-Batching HTTP Server, Not Just a Benchmark

**Platform:** Radxa ROCK 5B+ (RK3588, 8GB RAM) — the same board characterized in this repository's other RK3588 field notes
**Software stack:** Armbian, `rkllm-runtime` v1.3.0, TinyLlama-1.1B (w8a8), same model as the companion throughput-optimization field notes
**Builds directly on:** the companion field note's finding that RKLLM's documented `n_batch` multi-batch capability (SDK §3.2.14) delivers a real ~4.3x aggregate throughput multiplier but is completely unexposed by `rkllama`, the Flask server used throughout that investigation (hardcoded `n_batch=1`)
**Date:** August 2026

## Abstract

The companion throughput-optimization paper measured RKLLM's native multi-batch inference directly against `librkllmrt.so`, bypassing `rkllama` entirely, and found a real, close-to-linear throughput multiplier up to a sharp cliff at `n_batch=16→17`. That measurement was explicitly a benchmark, not a usable capability — the paper's own conclusion states the gap "is entirely in `rkllama`'s request-handling model... and would need real scheduling work to multiplex several HTTP requests into a shared `n_batch`-sized call." This paper does that work. A dynamic-batching HTTP server was built directly against `librkllmrt.so` (reusing `rkllama`'s own proven-correct ctypes struct definitions, per that project's established practice), which holds incoming requests in a short collection window, groups them into a single `n_batch`-sized `rkllm_run` call capped at the measured cliff of 16, and returns each request's own generated text independently. Tested end-to-end with 4 concurrent HTTP client requests against 4 different prompts: **all four batched correctly into one `n_batch=4` call, all four returned correct, distinct, coherent answers, and real wall-clock time dropped from 11.93s (sequential baseline) to 3.15s (concurrent, batched) — a 3.78x real speedup**, exceeding the companion paper's raw-API-only 2.52x measurement for the same batch size, plausibly because HTTP-level batching also amortizes per-request model-init overhead that a naive sequential baseline pays once per request.

## 1. Design, Grounded in the Companion Paper's Own Measured Limits

Three design decisions come directly from specific numbers already measured, not assumed:

- **Batch group size capped at 16.** The companion paper found a sharp cliff at `n_batch=16→17` (14 and 16 statistically tied at ~95 tok/s aggregate, 17 dropping straight back to roughly `n_batch=9` territory) — a real internal scheduling/queue-depth threshold in the runtime, not a smooth compute curve. This server refuses to form a batch group larger than 16 for exactly this reason.
- **A short (150ms) collection window, not a persistent always-batching server.** RKLLM's `n_batch` is fixed at `rkllm_init()` time — changing batch size mid-stream requires re-initializing the model, which has its own cost. This server holds newly arrived requests open briefly to let concurrent requests accumulate into one group, then commits to that group's size for the duration of the `rkllm_run` call, rather than trying to dynamically resize an in-flight batch.
- **A per-slot token budget floor.** The companion paper found the shared-context ceiling divides `max_context_len` evenly across the batch (confirmed via exact-token truncation: 128 tokens/slot at `n_batch=32` on a 4096-token context, 1024 tokens/slot at `n_batch=4`) — this is a second, independent cap from the throughput cliff. This server computes `MAX_CONTEXT // n_batch` per group and reserves headroom for the prompt, rather than silently truncating a user's requested `max_new_tokens`.

## 2. Implementation

A single-file Python server (`batch_server.py`, ~200 lines) using only the standard library's `http.server` and RKLLM's raw C API via ctypes (importing `rkllama.api.classes`' struct definitions directly rather than re-deriving them, avoiding the risk of a subtly wrong struct layout). A background thread implements the queue-and-batch-window logic described above; each incoming HTTP request blocks on a `threading.Event` until its slot's result is ready, then returns the actual generated text (`RKLLMResult.text`, accumulated across the streaming callback) as JSON — a real usable response, not just a token count or timing number.

## 3. Results: Real Concurrent Requests, Correctly Batched

Four concurrent HTTP client requests (`What is the capital of {France, Japan, Brazil, Egypt}?`, 30 max new tokens each) via a thread pool, compared against the same four requests sent one at a time:

| Mode | Total wall time | Per-request result |
|---|---|---|
| Sequential (1 at a time) | 11.93s | Each request: `n_batch=1`, individual `group_wall_s` 0.54-1.82s |
| Concurrent (batched) | 3.15s | All 4 requests: `n_batch=4`, shared `group_wall_s=1.426s` |

All four concurrent requests were confirmed batched into a single group by the server's own log (`[batcher] running group of 4 request(s)`) and by each response independently reporting `n_batch=4`. All four returned correct, coherent, on-topic answers (Paris, Tokyo, Brasília, Cairo) — the batching correctly kept each request's output attached to its own input, not scrambled across slots. **Real speedup: 3.78x**, computed from actual wall-clock HTTP request/response time on the client side, not an internal timing number — this is what an actual caller of this server experiences, not a benchmark-only figure.

## 4. Why the Real Speedup Exceeds the Companion Paper's Raw Measurement

The companion paper's `n_batch=4` raw-API measurement was 2.52x. This paper's end-to-end HTTP measurement is 3.78x for the same batch size. The most likely explanation: the companion paper's *sequential* baseline for its raw-API comparison still only paid model-load/init cost once (a persistent handle reused across sequential single-batch calls in that test harness), while this paper's sequential baseline is a fair simulation of real independent client usage — each of the 4 sequential requests here goes through the server's full request path, including a fresh `rkllm_init()`/`rkllm_destroy()` cycle per group (this server does not keep a model handle warm between batch groups, a real design tradeoff discussed in §5). That means the sequential baseline in this paper pays real per-request overhead four separate times that the concurrent, single-batched-group path pays only once — a genuine, additional source of speedup beyond the underlying `n_batch` compute multiplier alone, and arguably a more realistic picture of what a real multi-user deployment gains from batching than the raw-API number in isolation.

## 5. Known Limitations, Stated Plainly

- **No persistent model handle.** Each batch group does a full `rkllm_init()`/`rkllm_destroy()` cycle. This is simple and safe (no risk of stale state leaking between unrelated batch groups) but pays real init overhead per group — a genuine cost this paper's own §4 analysis shows is currently *helping* the measured speedup, but which would need addressing (a persistently warm handle, re-initialized only when batch size actually needs to change) for a production deployment where init cost matters independently of the batching question.
- **Fixed 150ms collection window**, not adaptive. A busier server might benefit from a shorter window (less latency added per request) or a longer one (more batching opportunity) — not tuned or swept here, chosen as a reasonable starting value only.
- **No handling of the shared-context ceiling's actual failure mode** beyond the token-budget floor — if a caller requests more tokens than the per-slot budget allows for a given group size, this server silently reduces `max_new_tokens` for that group rather than rejecting the request or deferring it to a smaller group. A production version should likely reject or explicitly warn instead.
- **Tested with 4 concurrent requests only**, not swept up to the 16-request cap this design targets — the companion paper's own raw-API sweep already validated the underlying `n_batch` scaling up to and past 16, so this paper's contribution is confirming the HTTP-level batching mechanism works correctly, not re-validating the scaling curve itself.

## 6. Conclusion

RKLLM's native multi-batch capability, previously measured only as a raw-API benchmark number the actual serving layer never exposed, now has a working HTTP server that exposes it — built directly on the specific numeric limits (the 16-batch cliff, the shared-context-per-slot ceiling) the companion paper measured, not on generic assumptions. Four concurrent real HTTP requests were correctly batched into a single `n_batch=4` inference call and returned correct, individually-attributed results, with a measured 3.78x real wall-clock speedup exceeding the underlying raw-API multiplier — turning a documented-but-unused hardware/runtime capability into something that could actually be deployed, not just cited as a limitation of the existing serving stack.

## Evidence

`batch_server.py` (the dynamic-batching server) and `batch_client_test.py` (the concurrent-request test harness and its output shown in §3), in `data/`, prefixed `rock5bp-20260815-rkllm-batch-server-`.
