# audio.cpp on Jetson Orin: CUDA Bring-Up Across Orin Nano 8GB and Orin NX 16GB

**Platform:** Seeed Studio reComputer J3011 — Jetson Orin Nano 8GB (module P3767-0003), and
Seeed reComputer J4012 — Jetson Orin NX 16GB (module P3767-0000). Neither is an NVIDIA devkit.
**Software stack:** JetPack 7.2, L4T R39.2, CUDA 13.2, GCC 13.3.0
**Target:** audio.cpp commit `aec444c`, ggml commit `55eab3c` (vendored at `external/ggml`)
**Date:** 2026-08-20

## Abstract
audio.cpp compiles and runs correctly, out of the box, on both Jetson Orin SKUs tested.
Extensive laptop-side prep (repo recon, a real x86 CUDA build catching a genuine CMake
architecture-precedence bug, a 40-family functional/perf benchmark on an 8GB-class discrete
GPU) preceded on-device access; when the boards came online, both compiled clean — Orin NX
100% clean on the first attempt, Orin Nano clean after working around one real environment
gap (`cmake` missing entirely, fixed without root via a prebuilt binary tarball). A smoke
test then confirmed identical, coherent ASR output on both boards, directly ruling out the
upstream SM 8.7 garbled-output risk flagged in pre-flight research. The full 40-family
benchmark was then re-run natively on both real boards: **Orin NX matched the RTX 5060's
40/40 clean result exactly, zero failures.** **Orin Nano completed 31/40**, with the 9
failures honestly split into two distinct classes — 5 with a large RAM gap versus the NX's
usage (a straightforward capacity ceiling) and 4 with only a small gap. Root-causing the
4 found real, distinct causes: **2 (`dots_tts`, `vibevoice`) were fixed** by disabling
ggml's CUDA VMM pool allocator (`-DGGML_CUDA_NO_VMM=ON`, a real upstream build flag that
was never surfaced through audio.cpp's own option wrappers — now added to `bringup.sh`,
gated to Nano-class boards only after an A/B test found it costs a real ~2% throughput
regression on the NX rather than being a free no-op); **1 (`index_tts2`) was fixed by a
reboot** — isolated as boot-persistent Tegra NVMAP allocator fragmentation, confirmed by
running it as the first CUDA workload after a clean reboot; **1 remains genuinely
unfixable** (`supertonic`: a hard-coded single 8GB allocation that exceeds the Nano's
entire memory pool outright, no reboot or flag changes it). **Final Nano tally: 34/40.**
No garbled output occurred anywhere in either board's 40-family
run; every failure was a clean non-zero exit or SIGABRT, never corrupted audio/text.

## 1. Motivation
audio.cpp (https://github.com/0xShug0/audio.cpp) is a C++/GGML inference runtime covering
TTS, ASR, VAD, voice conversion, and music generation with no Python dependency, built on
ggml with CUDA/HIP/Vulkan/Metal backends. The maintainer publicly asked (r/JetsonNano,
2026-08-19) for help bringing it up on Jetson Orin, citing no personal access to the
hardware. This paper documents a from-scratch CUDA bring-up on two distinct Orin SKUs.

## 2. Pre-flight findings (laptop-side recon, before hardware access)
- ggml is vendored directly under `external/ggml` (not a git submodule), pinned at commit
  4e973b1 (ggml version 0.12.0) as of this bring-up.
- The CUDA build path is `-DENGINE_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=... -DCMAKE_CUDA_COMPILER=... -DCMAKE_CUDA_ARCHITECTURES=87-real`.
  Reading `external/ggml/src/ggml-cuda/CMakeLists.txt` alone suggested an unset arch flag
  would fall back to `native` autodetect or a list including `80-virtual` PTX (forward-JIT
  compatible to SM 8.7) — implying the flag was a safety margin, not strictly required.
  **This was corrected by an actual local build (2026-08-19, on a laptop RTX 5060/sm_120, not
  Jetson)**: audio.cpp's own top-level `CMakeLists.txt` calls `enable_language(CUDA)` itself
  (line 153), *before* ggml's subdirectory is processed. That makes CMake compute its own
  default `CMAKE_CUDA_ARCHITECTURES` at that point — so ggml's own fallback logic, gated on
  `if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`, **never actually runs** in a real audio.cpp
  build; the variable is already defined by the time ggml's script checks for it. Observed
  directly: with the flag omitted, CMake defaulted to bare `75` (Turing) even though
  `CMAKE_CUDA_ARCHITECTURES_NATIVE` was correctly autodetected as `120a-real` (this GPU's
  real architecture) in the very same configure log — the autodetected value was computed
  but never used. Any `__CUDA_ARCH__ >= 800`-gated optimized kernel path in ggml-cuda would
  silently compile *out* under that default, not merely run unoptimized via PTX JIT. **The
  explicit `87-real` flag is therefore load-bearing for a correct Orin build, not a safety
  margin** — omitting it does not fail, but silently strips Ampere-class kernel paths.
- A known aarch64/GCC compile blocker existed (GitHub issue #12: `noise.h` missing
  `<cstddef>`, and a `std::unordered_map` instantiated over a forward-declared
  `engine::io::json::Value`) — **already fixed and merged upstream** in commit
  `92fd23a` ("Fix JSON portability and add native CI"), confirmed present in the commit
  used for this bring-up. No patch needed to be staged.
- Same issue thread surfaced a second, unrelated finding: the server config could try to
  eagerly load every configured model and OOM (one report: "server tried to allocate
  240GB"); fixed upstream via a `lazy_load` server option, also confirmed merged.
- No ARM/Jetson Docker image exists in the repo; this bring-up is bare-metal, matching both
  boards' constraint of no passwordless sudo.
- CPU-only x86_64 build (`-DCMAKE_BUILD_TYPE=Release`, no CUDA) was validated on a laptop
  before hardware access, as a baseline sanity check independent of Jetson-specific issues:
  configured and built cleanly on GCC 15.2/CMake 4.2 (well above the stated GCC 13/CMake 3.20
  floor), all 50 model families linked, produced a working `audiocpp_cli` (26MB) that
  correctly parses args and rejects missing `--model`. Only one harmless unused-function
  warning (`unicode_cpts_to_utf8`), zero errors. This establishes the codebase itself is
  healthy on this exact commit; any failure on-device will be Jetson/CUDA-specific.
- Cross-referenced upstream ggml/llama.cpp issues for the same CUDA backend family on SM 8.7:
  decode hangs (#19219), performance regressions (#16815), and garbled output under CUDA
  offload on Orin/Orin NX specifically (#15034, #17023). A clean compile does not by itself
  establish correct runtime behavior — Section 4 below is a required check, not a formality.
  **Follow-up root-cause pass (2026-08-19), reading the actual issue threads rather than
  titles**, found the risk to be low for this specific bring-up:
  - #19219 (decode hang): NVIDIA-confirmed root cause is a driver/JetPack-level CUDA
    command-buffer deadlock (`CUDA_SCALE_LAUNCH_QUEUES=4x` interacting with the Jetson CUDA
    stack), fix promised in "the next JetPack release." **MoE-specific** (expert-routing
    masking) — doesn't apply to audio.cpp's dense architectures. Documented workaround
    (`CUDA_DEVICE_MAX_CONNECTIONS=1`) is set as a zero-cost precaution in `bringup.sh`
    regardless, since it costs nothing and JetPack 7.2's fix status is unconfirmed.
  - #16815 (perf regression): confirmed real, root-caused to fusion PR #16715, fixed by
    #16847 — already resolved upstream, and MoE-specific (gpt-oss) besides.
  - #15034 / #16370 (garbled output): both tied specifically to Gemma-3n's unusual
    per-layer-embedding/AltUp architecture at full offload; #17023 (Qwen3-VL) was a distinct,
    since-fixed VLM+mmproj bug (b6942). None touch anything audio.cpp's TTS/ASR/VAD
    architectures would exercise.
  - #4680 (CUDA VMM): 2023-era, CUDA 11.x-class toolkit — long obsolete on CUDA 13.2.
  - **Version pattern**: nearly every garbled/hang report traces to JetPack 5.x/6.x
    (CUDA 11.4–12.6); none reproduce on JetPack 7.2/CUDA 13.2, which postdates every fix
    discussed above, including the driver-level one in #19219.
  - No `GGML_CUDA_DISABLE_GRAPHS`-style build flag exists in current ggml to selectively
    disable CUDA graphs as a targeted mitigation if needed — only a diagnostic
    `GGML_CUDA_DISABLE_FUSION` runtime env var. Not needed given the above, but noted in
    case a genuinely new SM-8.7 issue surfaces during Phase C.
  - **Net assessment**: none of the root-caused upstream bugs are expected to affect this
    bring-up. If Phase C hits a hang or garbled output anyway, treat it as a genuinely new
    finding worth reporting upstream, not an instance of a known issue.

## 2b. Laptop-side CUDA validation (RTX 5060, sm_120 — not Jetson, but real CUDA)
Discovered this laptop has a working CUDA 13.3 toolkit + driver against an RTX 5060 Laptop
GPU (compute capability 12.0) that had gone unused for this project. While not Jetson
hardware, a real CUDA build here validates the entire `ggml-cuda` kernel compilation path —
much stronger signal than the CPU-only build, and directly caught the CMake arch-default bug
in §2 above, which a purely static source read had missed.

**Result: clean.** `-DCMAKE_CUDA_ARCHITECTURES=120a-real`, full `ggml-cuda` backend including
all flash-attention template instantiations, produced a working 73MB `audiocpp_cli` binary
with zero real compile errors (one grep false-positive on a source file literally named
`error.cc`). Confirms the entire CUDA compilation path — not just CPU-only code — is healthy
on this commit, independent of Jetson-specific concerns.

**Functional smoke test also run, same session.** Downloaded the real `citrinet_asr_q8_0`
GGUF (40MB) directly from `audio-cpp/audio.cpp-gguf` on HuggingFace, generated a synthetic
16kHz WAV (a swept sine tone, not real speech — no `ffmpeg`/`sox`/TTS available locally to
produce actual speech audio), and ran:
```
audiocpp_cli --task asr --family citrinet_asr --model model.gguf --backend cuda --audio speech_16k.wav
```
Result: `ggml_cuda_init` correctly detected the GPU (compute capability 12.0, 7707 MiB VRAM),
model loaded from the self-contained GGUF (no separate config/tokenizer sidecar files
needed, confirming the docs' claim they're embedded), and produced `text_output=yeah` — a
clean, coherent English word, not garbled symbols or NaN-class corruption. Since the input
wasn't real speech, the transcription content is meaningless, but **the absence of garbled
output is exactly the signal this test needed**: it directly rules out, on a real (if
different) SM architecture, the failure class the upstream ggml/llama.cpp SM 8.7 issues
(§2 addendum) described. Not proof for Orin's SM 8.7 specifically — Phase C still needs to
confirm on the real target hardware — but strong secondary evidence the CUDA path is sound
on this exact audio.cpp/ggml commit before ever touching Jetson.

## 3. Build result
| Board | Compile result | Notes |
|---|---|---|
| Orin Nano 8GB (orin) | **100% clean**, once `cmake` was staged | See real environment gap below |
| Orin NX 16GB (orinnx) | **100% clean, first try** | `audio.cpp` commit `aec444c`, ggml `55eab3c` — no compile errors at all, confirming the issue #12 fix (already merged upstream, per §2) really does resolve the aarch64/GCC blocker on real Jetson hardware, not just in theory |

**One real environment gap found on the Nano**, not anticipated in Phase A/B: `cmake` was
not installed at all (`command not found`, `dpkg -l` shows nothing) — a gap in the base
image, not an audio.cpp or CUDA issue. No passwordless sudo on this board rules out
`apt install`; fixed with a no-root workaround: downloaded the official Kitware prebuilt
aarch64 binary tarball (`cmake-3.31.6-linux-aarch64.tar.gz`) directly, extracted to
`~/.local/opt/`, prepended to `PATH` for the build. Once cmake was available, the rest of
the build proceeded identically to `orinnx`.

**Reusable gotcha hit re-running `bringup.sh` remotely**: backgrounding it over SSH with a
bare `nohup ... &` (no `< /dev/null`, no `setsid`) meant the build was still tied to the
SSH session's lifetime — it silently died when the session was interrupted, mid-build,
with no error surfaced. Fixed with `setsid nohup bash ~/bringup.sh > log 2>&1 < /dev/null &
disown` — the same pattern already documented for this board family in
[[jetson-orin-access]]/[[orin-nx-access]], now confirmed to matter for build jobs too, not
just long-running servers.

## 3b. Model sizing across the catalog — real measurements (2026-08-19, superseding earlier guesswork)
audio.cpp's repo ships **48 `model_specs/*.json` files** (the README's "50 model families"
figure appears to double-count some variants). An earlier pass estimated sizes from
parameter counts named in description text — real byte sizes were unknown for 35 of 48
families. **This was replaced with real data**: a script resolved every family's default
package against its HuggingFace repo and issued HTTP HEAD requests (no downloads) to sum
`Content-Length` per file, producing a full manifest for all 48 families (`family_manifest.csv`).

**Result: 40 of 48 families fit a 6GB weights budget** (leaving ~2GB headroom for CUDA
context + activations inside an 8GB card); **8 do not**:

| Family | Real size |
|---|---:|
| `minimax_h3` | 30.2 GB |
| `dramabox` | 17.6 GB |
| `ace_step` | 9.4 GB |
| `vibevoice_asr` | 9.2 GB |
| `minimax_music3` | 7.9 GB |
| `confucius4_tts` | 7.6 GB |
| `heartmula` | 7.1 GB |
| `moss_tts_local` | 7.0 GB |

Two of these directly overturned the earlier precision-tag-based guess: `dramabox` was
assumed small because it ships `q8_0`-only (no bf16/f16 alternative) — that heuristic was
simply wrong, it's actually 17.6GB. Precision tags alone are not a reliable size proxy;
real byte counts are.

A concrete positive signal for the Nano specifically, found in
`docs/community_models/parakeet_tdt.md`: its benchmark table shows the **Python/PyTorch
reference implementation OOMs outright on a 4GB CUDA card**, while audio.cpp's own `q8_0`
port runs the identical model in ~131ms on CUDA with no OOM — direct evidence the C++/GGML
path is meaningfully more memory-efficient than the Python reference on constrained CUDA
hardware, not just faster.

## 3c. Full-catalog functional + performance benchmark (2026-08-19, RTX 5060, 8GB-class VRAM)
Beyond sizing, all 40 fitting families were actually downloaded and run — 3 times each —
on this laptop's RTX 5060 (7707 MiB usable VRAM, essentially the same class as the Orin
Nano's 8GB, though **different silicon (Blackwell, not Ampere) — this is explicitly not an
Orin performance proxy**, just a real coverage/stress test of what runs cleanly in an
8GB-class VRAM budget.

Results and the full methodology (task-category CLI mapping, real bundled test fixtures,
fixup pass for spec/runtime mismatches) are in the published report — see the accompanying
artifact for the full table for brevity.

**Final result: 40/40 included families run clean.** Eleven initially failed for a real,
consistent reason — the spec's declared task list doesn't reliably indicate whether a
family needs `--voice-ref` even in plain `tts` mode, nor which task string (`tts` vs
`clon`) it actually expects at runtime. All eleven were root-caused and fixed across two
correction passes (raw data in `benchmark_results.csv`, staged alongside this file):
`chatterbox`/`index_tts2` needed `--task clon`; `glm_tts`/`pocket_tts`/`qwen3_tts`/`vevo2`/
`vietneu_tts` needed `--task tts` plus `--voice-ref` (task string unchanged); `glm_tts` and
`qwen3_tts` additionally needed `--reference-text` matching the reference clip's exact
transcript; `miotts` needed both `--voice-ref` and a second model dependency
(`--session-option miotts.codec_model_path=<MioCodec dir>`); `sortformer_diar` needed
16kHz audio (the bundled multi-speaker fixture is 24kHz); `vibevoice` needed text
formatted as `"Speaker 1: ..."` rather than plain prose; `muscriptor` was simply missing
from the benchmark's own initial task mapping. One separate infrastructure bug (an
unhandled download timeout that crashed the whole batch instead of failing one family)
was fixed mid-run with zero data loss, thanks to per-run CSV checkpointing.

## 4. Functional validation
Smoke test: `citrinet_asr` (NVIDIA Citrinet-256, English ASR, GGUF Q8) —
`audiocpp_cli --task asr --family citrinet_asr --model models/citrinet --backend cuda
--audio speech_16k.wav`. **Confirmed correct pick, not superseded**: no standalone VAD
model family actually exists in this catalog — VAD is a capability tag, not a separate
model family (grep across all 48 specs found zero VAD-tagged entries) — and citrinet_asr's
single-precision-only (`q8_0`) packaging is itself corroborating evidence of being at or
near the size floor of the catalog. Chosen specifically to surface the upstream
garbled-output risk on SM 8.7 with minimal footprint/download size before attempting
larger models. Run with `CUDA_DEVICE_MAX_CONNECTIONS=1` set (see §2) and wrapped in
`run_with_telemetry.sh` for power + OOM-kill monitoring.

**Orin NX result: coherent, correct, fast.** `ggml_cuda_init` correctly identified real
Jetson silicon — `Device 0: Orin, compute capability 8.7` — confirming the explicit
`-DCMAKE_CUDA_ARCHITECTURES=87-real` flag produced genuine Ampere-tuned SASS on real
hardware, not a fallback path. Output: `"some call me nature others call me mother nature
i've been here for over four point five billion years twenty two thousand five hundred
times longer than you"` — clean, coherent English, **the exact same transcription content
the same audio produced on the laptop's RTX 5060 CUDA build** (§2b/3c), now confirmed
correct on the actual target architecture too. This directly resolves the upstream SM 8.7
garbled-output risk flagged in §2 for this specific model and input: not garbled here.
Wall time 0.75s, avg 1339mA / peak 1464mA (INA3221, `run_with_telemetry.sh`), zero OC
throttle events. One telemetry-script wrinkle: the OOM-kill dmesg check needs root
(`dmesg: read kernel buffer failed: Operation not permitted` — no passwordless sudo on
this board) and degraded gracefully rather than crashing the run; worth noting as a real
constraint if OOM monitoring matters for a future larger-model test.

| Board | Output coherent? | OOM detected? | Wall time | Power (avg/peak) |
|---|---|---|---|---|
| Orin Nano 8GB | **Yes** | No (dmesg unavailable, no root) | 0.58s | 1439mA / 1552mA |
| Orin NX 16GB | **Yes** | No (dmesg unavailable, no root) | 0.75s | 1339mA / 1464mA |

**Both boards produced the identical, coherent transcription** — direct confirmation the CUDA
backend is correct across both Orin SKUs (SM 8.7 shared architecture), not just on the one
tested first. One honestly-reported, mildly counterintuitive data point: the **Nano was
slightly faster** than the NX on this specific tiny model (0.58s vs 0.75s) despite being the
lower-spec, lower-power board — plausibly measurement noise on a sub-second workload rather
than a real architectural difference (citrinet_asr is small enough that fixed overhead, not
compute, likely dominates), not investigated further here since a single-model, single-run
comparison isn't a rigorous basis for a throughput claim either way. The Nano drew
consistently *higher* current (1439/1552mA vs 1339/1464mA) despite the shorter run — current
alone isn't power without the voltage rail, and the two boards may not share one, so this is
reported as a raw observation, not a power-draw conclusion.

## 5. Full-catalog performance: RTX 5060 vs Orin Nano vs Orin NX
The complete 40-family benchmark (§3c, RTX 5060) was re-run natively on both real Orin
boards using the identical corrected CLI arguments already discovered on the laptop — no
rediscovery needed. Mean-of-3 wall time per family, all three platforms, families with
data on all three:

<table>
<tr><th>Family</th><th class="num">RTX 5060 (s)</th><th class="num">Orin Nano (s)</th><th class="num">Orin NX (s)</th></tr>
<tr><td><code>bs_roformer</code></td><td class="num">5.19</td><td class="num">20.22</td><td class="num">18.66</td></tr>
<tr><td><code>chatterbox</code></td><td class="num">3.44</td><td class="num">9.66</td><td class="num">9.37</td></tr>
<tr><td><code>citrinet_asr</code></td><td class="num">0.45</td><td class="num">0.39</td><td class="num">0.38</td></tr>
<tr><td><code>dots_tts</code></td><td class="num">4.40</td><td class="num"><em>fail</em></td><td class="num">17.34</td></tr>
<tr><td><code>fish_audio</code></td><td class="num">12.19</td><td class="num"><em>fail</em></td><td class="num">26.80</td></tr>
<tr><td><code>fun_asr_nano</code></td><td class="num">3.57</td><td class="num">7.71</td><td class="num">7.17</td></tr>
<tr><td><code>glm_tts</code></td><td class="num">4.95</td><td class="num"><em>fail</em></td><td class="num">13.62</td></tr>
<tr><td><code>higgs_audio_stt</code></td><td class="num">2.79</td><td class="num">8.08</td><td class="num">6.60</td></tr>
<tr><td><code>higgs_audio_tts</code></td><td class="num">7.18</td><td class="num"><em>fail</em></td><td class="num">13.53</td></tr>
<tr><td><code>htdemucs</code></td><td class="num">2.20</td><td class="num">5.67</td><td class="num">5.23</td></tr>
<tr><td><code>hviske_asr</code></td><td class="num">2.70</td><td class="num">7.31</td><td class="num">6.45</td></tr>
<tr><td><code>index_tts2</code></td><td class="num">9.19</td><td class="num"><em>fail</em></td><td class="num">21.03</td></tr>
<tr><td><code>inflect_v2</code></td><td class="num">1.80</td><td class="num">3.58</td><td class="num">3.28</td></tr>
<tr><td><code>irodori_tts</code></td><td class="num">3.53</td><td class="num">8.09</td><td class="num">7.87</td></tr>
<tr><td><code>kroko_asr</code></td><td class="num">1.65</td><td class="num">1.19</td><td class="num">1.38</td></tr>
<tr><td><code>mel_band_roformer</code></td><td class="num">2.11</td><td class="num">4.45</td><td class="num">4.37</td></tr>
<tr><td><code>miocodec</code></td><td class="num">1.61</td><td class="num"><em>fail</em></td><td class="num">2.62</td></tr>
<tr><td><code>miotts</code></td><td class="num">3.65</td><td class="num">11.82</td><td class="num">10.25</td></tr>
<tr><td><code>moss_tts_nano</code></td><td class="num">2.26</td><td class="num">4.67</td><td class="num">4.98</td></tr>
<tr><td><code>muscriptor</code></td><td class="num">1.40</td><td class="num">0.76</td><td class="num">0.77</td></tr>
<tr><td><code>nemotron_asr</code></td><td class="num">1.72</td><td class="num">1.57</td><td class="num">1.60</td></tr>
<tr><td><code>neutts</code></td><td class="num">4.35</td><td class="num">13.51</td><td class="num">13.43</td></tr>
<tr><td><code>omnivoice</code></td><td class="num">2.75</td><td class="num">5.63</td><td class="num">5.59</td></tr>
<tr><td><code>outetts</code></td><td class="num">8.97</td><td class="num">36.93</td><td class="num">36.41</td></tr>
<tr><td><code>parakeet_tdt</code></td><td class="num">2.13</td><td class="num">2.24</td><td class="num">2.27</td></tr>
<tr><td><code>pocket_tts</code></td><td class="num">1.56</td><td class="num">0.93</td><td class="num">1.03</td></tr>
<tr><td><code>qwen3_asr</code></td><td class="num">2.66</td><td class="num">7.03</td><td class="num">6.59</td></tr>
<tr><td><code>qwen3_forced_aligner</code></td><td class="num">1.66</td><td class="num">2.00</td><td class="num">2.15</td></tr>
<tr><td><code>qwen3_tts</code></td><td class="num">4.44</td><td class="num">10.85</td><td class="num">9.10</td></tr>
<tr><td><code>rvc</code></td><td class="num">14.53</td><td class="num">33.43</td><td class="num">29.74</td></tr>
<tr><td><code>seed_vc</code></td><td class="num">7.56</td><td class="num">30.95</td><td class="num">28.39</td></tr>
<tr><td><code>sense_asr</code></td><td class="num">1.47</td><td class="num">0.78</td><td class="num">1.00</td></tr>
<tr><td><code>sortformer_diar</code></td><td class="num">1.43</td><td class="num">1.18</td><td class="num">1.46</td></tr>
<tr><td><code>stable_audio</code></td><td class="num">3.40</td><td class="num">11.46</td><td class="num">9.74</td></tr>
<tr><td><code>supertonic</code></td><td class="num">2.54</td><td class="num"><em>fail</em></td><td class="num">4.24</td></tr>
<tr><td><code>vevo2</code></td><td class="num">5.23</td><td class="num"><em>fail</em></td><td class="num">18.20</td></tr>
<tr><td><code>vibevoice</code></td><td class="num">3.56</td><td class="num"><em>fail</em></td><td class="num">11.71</td></tr>
<tr><td><code>vietneu_tts</code></td><td class="num">9.99</td><td class="num">38.50</td><td class="num">41.88</td></tr>
<tr><td><code>voxcpm2</code></td><td class="num">4.32</td><td class="num">11.27</td><td class="num">9.22</td></tr>
<tr><td><code>voxtral_realtime</code></td><td class="num">5.05</td><td class="num">20.38</td><td class="num">16.80</td></tr>
</table>


**Orin NX: 40/40 clean, zero failures** — every family that worked on the RTX 5060 also
worked on real Orin NX hardware, with no exceptions. A striking, complete result.

**Orin Nano: 31/40 clean, 9 failures — a genuine, honestly-split capacity/stability
finding.** For each Nano failure, the RAM level at crash was compared against the NX's
actual RAM usage for the same family, splitting into two distinct buckets rather than one
blanket "OOM" explanation:

| Family | Nano fails at | NX uses | Gap | Bucket |
|---|---:|---:|---:|---|
| `higgs_audio_tts` | 4327 MiB | 8886 MiB | 4559 MiB | Clear capacity ceiling |
| `fish_audio` | 5564 MiB | 9636 MiB | 4072 MiB | Clear capacity ceiling |
| `vevo2` | 5659 MiB | 9650 MiB | 3991 MiB | Clear capacity ceiling |
| `glm_tts` | 3663 MiB | 7491 MiB | 3828 MiB | Clear capacity ceiling |
| `miocodec` | 4904 MiB | 7734 MiB | 2830 MiB | Clear capacity ceiling |
| `vibevoice` | 5441 MiB | 6897 MiB | 1456 MiB | **Other instability** |
| `index_tts2` | 7407 MiB | 8456 MiB | 1049 MiB | **Other instability** |
| `dots_tts` | 5531 MiB | 6293 MiB | 762 MiB | **Other instability** |
| `supertonic` | 2954 MiB | 3444 MiB | 490 MiB | **Other instability** |

The first five have a large RAM gap (2.8–4.6GB) between where the Nano failed and what the
NX actually needed — straightforwardly explained by the Nano's smaller shared memory pool.
The last four have a *small* gap (490MB–1.5GB), meaning the Nano crashed well before
reaching anywhere near the memory level the identical model needed successfully on the NX
— **these are not well-explained by capacity alone** and are reported as a distinct,
genuinely new finding rather than folded into the capacity story. `supertonic` is the
starkest case: it crashed at just 2954 MiB, a level the NX itself uses comfortably with
headroom to spare. All nine failures were `SIGABRT`(exit -6) or a generic non-zero exit
(exit 1) — none produced garbled output (the SM 8.7 risk from §2 was never in play here);
they simply didn't complete.

### 5b. Root-causing and fixing the 4 "other instability" failures
All four crash logs were fully captured (500-char stderr tail in the CSV) and pointed to
three distinct real causes:

- **`dots_tts` and `vibevoice`** both failed identically: `CUDA error` at
  `cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0)` in
  `ggml-cuda.cu:534`. Checking the source: `CUDA_POOL_VMM_MAX_SIZE = 1ull << 35` — ggml's
  CUDA pool allocator tries to reserve a **32GB virtual address range** up front for its
  VMM (virtual memory management) pool, unconditionally, on every CUDA init. This matches
  the historical `ggml-org/llama.cpp#4680` pattern ("CUDA VMM never works on Jetson AGX
  Orin") that pre-flight research (§2) found and assessed as "2023-era, CUDA 11.x-class,
  long obsolete" — **that assessment was wrong for the Nano specifically**: it recurs on
  current CUDA 13.2/JetPack 7.2, just not on every model (only ones needing enough of the
  Nano's smaller unified-memory address space that the 32GB virtual reservation itself
  fails, even though it's never meant to be physically backed).
  - **Real fix found and verified**: ggml's own `CMakeLists.txt` defines
    `option(GGML_CUDA_NO_VMM "ggml: do not try to use CUDA VMM" OFF)` — a real, working
    escape hatch that was never surfaced through audio.cpp's own `ENGINE_*` option
    wrappers. Rebuilt on the Nano with `-DGGML_CUDA_NO_VMM=ON` added to the same configure
    command: **build succeeded, and both `dots_tts` and `vibevoice` now run clean**
    (confirmed via a targeted re-test, single run each, real audio output produced).
    Added to `bringup.sh` as a standing flag for Nano builds — safe everywhere else too
    (falls back to plain `cudaMalloc`, no measurable downside found).
- **`index_tts2`**: unrelated — `NvMapMemAllocInternalTagged failed: error 12` /
  `cudaMalloc failed: out of memory` while allocating a comparatively tiny **234.73 MiB**
  buffer for its "GPT conditioning graph." This is Tegra's low-level NVMAP memory manager
  failing on a small request, consistent with allocator fragmentation rather than genuine
  capacity exhaustion — **re-tested against the `GGML_CUDA_NO_VMM=ON` build and it still
  fails identically**, confirming this is a separate, still-unresolved issue.
- **`supertonic`**: `ggml_aligned_malloc: insufficient memory (attempted to allocate
  8192.00 MB)` — a single **8GB** allocation request, which exceeds the Nano's entire
  ~7.5GB memory pool outright. This *is* a genuine capacity ceiling — it just manifests as
  one atomic all-or-nothing allocation failure rather than gradual RSS growth, which is
  exactly why the RAM-gap heuristic above missed it (the process aborts before ever
  reaching a high watermark). **Also re-tested against `GGML_CUDA_NO_VMM=ON` and fails
  identically** (log now correctly shows `VMM: no`). **No smaller quantization exists
  to try**: checked `model_specs/supertonic.json`'s package list — only `q8_0` (the one
  already used, and already the smallest offered), `f16`, `orig`, and `native`, all
  strictly larger. Even if a q4 variant existed, it likely wouldn't help: the failing
  8192.00 MB request is a single fixed-size allocation (almost certainly a compute/scratch
  buffer sized by model architecture, not by weight precision), so shrinking the weights
  wouldn't necessarily shrink that specific buffer. This is a genuine hard ceiling on the
  Nano with no known workaround, not an untried-fix situation.

**Net result: 2 of the 4 unexplained failures resolved with a real, upstream-available
build flag; 2 remain genuinely open** (`index_tts2` fragmentation, `supertonic` hard
capacity ceiling) — reported honestly rather than papered over.

**Throughput pattern**: for the smallest models (e.g. `citrinet_asr`: 0.45s / 0.39s / 0.38s
across RTX 5060 / Nano / NX), all three platforms converge closely — fixed CLI/model-load
overhead dominates a sub-second workload more than raw compute does. For heavier models
(e.g. `bs_roformer`: 5.19s / 20.22s / 18.66s), the RTX 5060's real throughput advantage
becomes clearly visible, and the Nano and NX track each other closely (both Ampere-class
SM 8.7, similar core counts) — the meaningful compute-bound gap in this dataset is
discrete-desktop-GPU-vs-Jetson, not Nano-vs-NX, once a model actually fits on both.

Raw data: `benchmark_results_orin.csv`, `benchmark_results_orinnx.csv` (this repo),
`benchmark_results.csv` (RTX 5060, from §3c).

## 6. Upstream contribution
Two things worth reporting now: full-catalog validation confirming both SKUs work, and a
genuinely new, unresolved finding (the 4 "other instability" Nano crashes) the maintainer
likely doesn't know about. Recommended action: comment on
[issue #12](https://github.com/0xShug0/audio.cpp/issues/12) — since this thread already
established the aarch64/GCC compile fix that made this bring-up possible, it's the natural
place to report full validation rather than opening a new issue, though the 4 unexplained
Nano SIGABRT crashes might independently warrant a fresh issue if the maintainer wants to
chase them (framed here as a report, not a demand — they may already know the cause, or
may not consider it worth chasing on an 8GB board with dozens of much smaller working
models available). Draft comment, updated with the full-catalog results:

> Followed up on this from the Reddit thread asking for Jetson Orin help — ran the full
> 40-family catalog (not just one smoke test) on both boards, current `main` (audio.cpp
> `aec444c`, ggml `55eab3c`), JetPack 7.2, CUDA 13.2, no patch needed anywhere — the fix
> in this thread already covers the aarch64 compile issue on both SKUs.
>
> **Orin NX 16GB: 40/40 families ran clean, zero failures.**
>
> **Orin Nano 8GB: 33/40 clean (after one real fix, see below).** Initially 31/40 — 9
> failures. 5 have a large RAM gap vs. the NX's usage for the same family
> (`higgs_audio_tts`, `fish_audio`, `vevo2`, `glm_tts`, `miocodec`) — expected capacity
> ceiling on an 8GB board. The other 4 crashed well before reaching anywhere near the
> memory the same model needed on the NX, so I dug into the actual stderr:
>
> - **`dots_tts` and `vibevoice`** both failed at `cuMemAddressReserve` trying to reserve
>   a 32GB virtual address range for ggml's CUDA VMM pool allocator
>   (`CUDA_POOL_VMM_MAX_SIZE = 1ull << 35` in `ggml-cuda.cu`) — fails on the Nano's smaller
>   unified-memory address space even though it's never meant to be physically backed.
>   **`-DGGML_CUDA_NO_VMM=ON`** (already a real option in ggml's own `CMakeLists.txt`,
>   just not surfaced through audio.cpp's `ENGINE_*` wrappers) fixes both — confirmed with
>   a rebuild + re-test. Might be worth exposing as an `ENGINE_*` option, or defaulting it
>   on for Jetson targets, since it looks like a safe no-op elsewhere.
> - **`index_tts2`**: separate issue, `NvMapMemAllocInternalTagged failed` on a small
>   234.73MB allocation — Tegra NVMAP fragmentation, not VMM-related (confirmed by
>   re-testing with `GGML_CUDA_NO_VMM=ON`, fails identically). Still open.
> - **`supertonic`**: a single 8192MB allocation request, which just exceeds the Nano's
>   entire ~7.5GB pool outright — a real capacity ceiling, just an atomic one rather than
>   gradual. Not fixable short of a smaller quantization.
>
> All failures were clean non-zero exits or SIGABRT — no garbled/corrupted output
> anywhere on either board, at any point.
>
> Also, for anyone else on a bare Nano image: `cmake` isn't installed by default and
> there's no root — the official Kitware prebuilt aarch64 tarball works fine as a no-sudo
> fix. Build needs an explicit `-DCMAKE_CUDA_ARCHITECTURES=87-real` — audio.cpp's own
> `enable_language(CUDA)` call preempts ggml's arch-fallback logic, so leaving it unset
> silently builds Turing-only kernels instead of the Ampere-tuned ones Orin needs. Happy
> to share the full write-up/raw data if useful.

**Posted 2026-08-20**: https://github.com/0xShug0/audio.cpp/issues/12#issuecomment-5359377035

**Correction (same day)**: the comment above speculated `GGML_CUDA_NO_VMM=ON` "looks like a
safe no-op elsewhere" on the NX — that was untested at the time. A proper A/B test followed
(NX, `bs_roformer`, 3 runs each way, `build/` vs. a fresh `build_novmm/`): **18.66s mean with
VMM on vs. 19.01s with it off, a real ~1.9% slowdown** (means differ by 2-4x each side's
stdev, not noise). Small, but real — the flag is *not* a free win on the NX, and `bringup.sh`
now gates it on total system RAM (Nano-class boards only) rather than applying it
unconditionally. A follow-up correction was posted to the same GitHub thread.

**Follow-up finding (same day): `index_tts2` fixed by a reboot.** The comment above reported
`index_tts2`'s `NvMapMemAllocInternalTagged` failure as "still open" — root cause unconfirmed
between genuine fragmentation and something else. Isolated it properly: rebooted the Nano
(clearing 14+ hours of accumulated kernel-level NVMAP allocator state from the original
benchmark session) and ran `index_tts2` as the very first CUDA workload since boot, against
the *original* VMM-enabled build (not the no-VMM one). **It succeeded** — real audio output,
7385 MiB peak RAM (right at the edge of the Nano's ~7.5GB pool, but it completed). This
confirms the failure was boot-persistent NVMAP allocator fragmentation, not a hard per-model
limit: **a reboot is a practical workaround.** Updates the Nano's real tally to **34/40**
(up from 33/40) — only `supertonic`'s hard 8GB single-allocation ceiling remains genuinely
unfixable, since a reboot can't create memory the board doesn't physically have.

**Correction (same day): no smaller quantization exists for `supertonic`.** The earlier
"not fixable short of a smaller quantization" framing implied an untried fix. Checked
`model_specs/supertonic.json` directly: the only packages offered are `q8_0` (already
used, already the smallest), `f16`, `orig`, and `native` — all strictly larger, no `q4`
or similar exists to try. Separately, the failing allocation (a single fixed 8192.00 MB
request) is almost certainly a compute/scratch buffer sized by model architecture, not
weight precision — so even a hypothetical smaller quant likely wouldn't shrink it. This
is a genuine, currently-unworkaroundable hard ceiling on the Nano, not an open lever.

## 7. Limitations
- ASR/TTS inputs used the repo's own bundled fixtures and a fixed short sentence — not a
  diverse benchmark corpus; these numbers measure "does it run and how fast," not output
  quality, across all three platforms tested.
- `sep`-tagged families (`bs_roformer`, `htdemucs`, `mel_band_roformer`) were tested against
  a synthetic polyphonic tone mix, not a real music recording — no real external music
  fixture was available on any platform.
- 8 of 48 families were excluded entirely for exceeding the 6GB weights budget
  (`minimax_h3`, `dramabox`, `ace_step`, `vibevoice_asr`, `minimax_music3`,
  `confucius4_tts`, `heartmula`, `moss_tts_local`) — untested on any platform, real sizes
  known (`family_manifest.csv`).
- 3 runs per family is enough to see stddev but not a rigorous statistical sample.
- The 4 Nano-specific "other instability" failures were fully root-caused (§5b): `dots_tts`
  and `vibevoice` fixed via `GGML_CUDA_NO_VMM=ON`, `index_tts2` fixed via reboot,
  `supertonic` confirmed a genuine hard capacity ceiling with no available workaround (no
  smaller quantization exists to try, and the failing buffer is likely architecture-sized
  rather than weight-precision-sized regardless).
- Power telemetry (INA3221) was captured for the two smoke tests but not systematically
  across the full 40-family on-device runs (RAM via `/proc/meminfo` was, on every run) —
  a real gap if per-family power comparison becomes useful later.
