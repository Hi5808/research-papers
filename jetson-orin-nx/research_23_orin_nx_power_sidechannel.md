# A Trivial Power Side-Channel on Jetson Orin NX: Identifying Running Workloads Blind, With No Access But the Power Rail

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-300) on a J401 carrier board
**Software stack:** JetPack 7.2, L4T R39.2.0, MAXN_SUPER power mode
**Sensor:** onboard INA3221 VDD_IN current/voltage monitor (`/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4`) — the same sensor this project's power/thermal papers already characterize in depth
**Date:** August 2026

## Abstract

This project's existing power telemetry work (papers 9, 11, 12) reads this board's INA3221 current sensor to measure *known* workloads for benchmarking. This paper asks the inverse question: with only that same current-sensor stream and nothing else — no logs, no process list, no network access — can an observer identify *which* of several distinct workloads is running, blind? Four workload classes (idle, sustained CPU stress, disk I/O, and TensorRT-Edge-LLM decode inference) were sampled at ~20Hz for 3 independent trials each, reduced to four simple statistical features (mean current, standard deviation, range, and fraction of high-current spikes), and classified with a bare nearest-neighbor comparison against the other 11 trials — a leave-one-trial-out blind test with no training beyond the other real samples. **Result: 12/12 trials correctly identified, 100% accuracy**, with the four workload classes occupying visually and numerically distinct regions of feature space: idle sits at a nearly flat ~1224mA (stdev 3.6-7.5), disk I/O at ~1280mA with moderate spikiness, sustained CPU stress at the highest mean (~2470mA) but with the *lowest* variance of any active workload (stdev ~20 — a genuinely steady load), and LLM decode inference at a lower mean (~1830mA) than CPU stress but with by far the largest variance of the four (stdev 700+, current range approaching 2900mA) — a clear power-domain signature of decode's inherently bursty, step-by-step token generation timing. This is a small-scale proof of concept, not a hardened attack, but the separation is clean enough with almost no signal processing that it's worth documenting as a real, measured finding specific to this board rather than treating power-side-channel risk as purely theoretical.

## 1. Why This Question, on This Board Specifically

Papers 9, 11, and 12 already established that this board's power draw is workload-shape-dependent in ways rich enough to root-cause a hardware throttling mechanism (`soctherm_oc`) from register-level current readings alone. That same richness is, from a different angle, exactly the property a power side-channel needs: if a workload's compute pattern leaves a detectable signature in its power draw, an observer with only power-rail access — no software access to the device at all — can potentially infer what's running. This is a well-established general class of hardware-security research (it underlies real attacks against smart cards and some IoT devices), but it had not been tested on this specific board's specific sensor/software stack, and this project already has the measurement infrastructure in place to test it cheaply.

## 2. Method

Four workload classes, three independent trials each, ~8-10 seconds of sampling per trial at 50ms polling intervals (the sensor's own internal update rate is closer to ~40-50ms regardless of polling frequency, per direct measurement — polling faster does not yield more real information, consistent with typical INA3221 conversion-time behavior):

- **idle**: no deliberate load, board otherwise quiescent
- **cpu_stress**: `stress --cpu 8` (all 8 cores)
- **disk_io**: `dd if=/dev/zero of=... bs=1M count=2000 oflag=direct` (sustained sequential write, O_DIRECT to bypass page cache)
- **llm_decode**: `llm_inference` against the working Qwen3-1.7B-Instruct TensorRT-Edge-LLM engine (same engine validated in this project's other recent papers)

Each trace was reduced to four features: mean current, standard deviation, min-max range, and the fraction of samples exceeding one standard deviation above the mean ("spikiness"). Classification used the simplest possible method deliberately — a Euclidean nearest-neighbor match against all other trials' feature vectors, scaled by each feature's rough typical magnitude — specifically to test whether *any* signal is present at all before reaching for anything more sophisticated. Each of the 12 trials was classified using only the other 11 as reference (leave-one-out), so no trial's own label ever leaked into its own classification.

## 3. Results

| Workload | Mean current (mA) | Stdev | Range | Spikiness |
|---|---|---|---|---|
| idle | ~1224 | 3.6-7.5 | 16-56 | 0.18-0.24 |
| disk_io | ~1280 | 57-65 | 160-184 | 0.23-0.24 |
| cpu_stress | ~2465 | 19-21 | 136-144 | 0.000 |
| llm_decode | ~1830 | 700-752 | 2864-2888 | 0.15 |

All 12 leave-one-out classifications were correct (12/12, 100%), with the nearest match always coming from the same workload class and by a clear margin in every case — the closest cross-class confusion distance was still substantially larger than the largest within-class distance. Two properties of the data make this an easy separation rather than a marginal one: **CPU stress has the highest mean but is essentially noise-free** (spikiness exactly 0.000 across all three trials — a sustained, deterministic load with no spikes above one standard deviation at all, mechanically true almost by definition for a saturating busy-loop), while **LLM decode inference has a much lower mean but roughly 35-100x the variance of any other workload** (stdev 700+ vs. cpu_stress's ~20) — decode's autoregressive, one-token-at-a-time execution pattern produces large power swings between compute bursts that no other tested workload comes close to.

## 4. What This Does and Doesn't Show

**Does not show:** that this is a practical, hardened real-world attack. The experiment used clean, isolated, single-workload conditions — no concurrent processes, no attacker adapting to noisy or mixed real-world traffic, a tiny sample size (3 trials per class, 4 classes), and a toy nearest-neighbor classifier chosen specifically to test whether *any* signal exists rather than to build something robust. A real adversarial setting (multiple simultaneous processes, an attacker with no knowledge of the reference classes in advance, noise from a genuinely uncontrolled environment) would be a substantially harder problem, not attempted here.

**Does show:** that under clean single-workload conditions on this specific board, the four tested workload classes are trivially, cleanly separable using nothing but coarse statistics of the power draw — no frequency-domain analysis, no machine learning, no sophisticated signal processing, and it took a handful of lines of Python to get perfect classification. This is a meaningful lower bound on the actual amount of side-channel information this board's power draw carries: if a bare four-feature nearest-neighbor classifier achieves 100% separation this easily, an adversary willing to invest more effort (frequency analysis, more workload classes, handling of concurrent/mixed conditions) very likely has considerably more room to work with than this proof of concept demonstrates, not less.

## 5. Conclusion

This board's INA3221 power telemetry — the same sensor this project has used throughout for legitimate benchmarking — carries enough workload-dependent signal to trivially distinguish idle, disk I/O, sustained CPU load, and LLM decode inference from each other with perfect accuracy on a small clean dataset, using nothing more sophisticated than mean and variance. This should be read as a real, if preliminary, hardware-security data point specific to this board and software stack, not a general claim about all Jetson hardware or a demonstrated practical attack — the natural next step, not attempted here, would be testing under realistic adversarial conditions (mixed/concurrent workloads, more workload classes including different model sizes and quantization schemes, and an actual unknown-to-the-classifier held-out workload) to establish whether this separates the ease of the toy setup here from a genuine practical risk.

## Evidence

Raw power traces for all 12 trials (4 workload classes × 3 trials, ~50ms polling), the collection harness (`power_sidechannel_collect.sh`), and the classification script (`sidechannel_classify.py`), in `data/`, prefixed `orinnx-20260815-power-sidechannel-`.
