# Characterizing an Unbranded NVMe SSD: The Dead Sensor and the Span-Dependent Cache

**Platform:** Seeed Studio reComputer J4012 — NVIDIA Jetson Orin NX 16GB (module P3767-0000, SKU 699-13767-0000-300 rev H.2) on a J401 carrier, not an NVIDIA developer kit
**Software stack:** JetPack 7.2, L4T R39.2.0, Ubuntu 24.04.4, kernel 6.8.12-1021-tegra (oot variant), fio 3.36, nvme-cli, smartmontools 7.4
**Date:** August 2026

> **Platform identification note.** The device tree `model` string on this unit
> reads `NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super`. The
> "Developer Kit" part is wrong, and the trap is the same one documented in
> [paper 7](research_7_jetson_fan_curve_thermal.md): the DT `model` string is
> free text set by the BSP and can name hardware that is not present.
>
> Four independent identifiers place the hardware, listed weakest to strongest:
>
> ```
> DT compatible   nvidia,p3768-0000+p3767-0000-super / nvidia,p3767-0000
> DT chosen/sku   699-13767-0000-300 H.2      (3767-0000 = Orin NX 16GB)
> TNSPEC          3767-300-0000-H.2-1-0-recomputer-orin-j401-
> flashed image   mfi_recomputer-orin-nx-16g-j401-7.2.0-39.2.0-2026-06-18.tar.gz
> ```
>
> Note that `compatible` names the **p3768-0000 NVIDIA reference carrier**,
> which is not what this board is — Seeed derives its BSP from NVIDIA's and the
> string is inherited. TNSPEC is better but is written at flash time and
> describes the flashing config, not necessarily the silicon. The decisive
> source is the image name recorded in `/etc/nv_tegra_release`, which names the
> module capacity and the carrier together: `orin-nx-16g-j401`. A J401 carrier
> carrying an Orin NX 16GB is sold as the **reComputer J4012**.
>
> Corroborating: 15 GiB usable RAM and 8x Cortex-A78AE (CPU part `0xd42`) match
> Orin NX 16GB and rule out the 8 GB P3767-0001 and the 6-core Orin Nano SKUs.
> PCIe enumerates three root ports — M.2 M-key (the drive under test), M.2
> E-key (an Intel 7260), and a Realtek RTL8111 GbE.
>
> This matters for every thermal statement below. The M.2 slot sits on a J401
> carrier with its own airflow path, not in an NVIDIA reference enclosure.

## Abstract

A 1 TB NVMe SSD of unknown provenance was characterized in place, as the live boot device of a Jetson Orin NX, without wiping or reflashing it. The drive is unbranded — model string `SSD 1TB`, placeholder serial, and a PCI subsystem ID identical to the device ID — but its controller is positively identifiable as a Silicon Motion SM2263XT, DRAM-less, PCIe Gen3 x4. Health telemetry reports a drive that is effectively new: 0% used, 100% spare, one power-on hour, 67 GB written, zero media errors. A 690 GiB test campaign left every one of those counters unchanged. Sequential read reaches 2271 MB/s against the raw device; sustained sequential write holds 573 MB/s for the first 107.8 GiB and then falls 4.0x to 143 MB/s as the SLC cache exhausts. 4K random read is span-dependent, as a DRAM-less controller with a 64 MiB host-memory-buffer request predicts. Three findings are negative and are reported as such: the drive's temperature sensor is frozen at a single value and its thermal telemetry is therefore worthless; a compressibility control test was confounded by ordering and proves nothing; and this paper's own sampling harness was found to overstate throughput by 32%, which is why every headline number here comes from fio rather than from the harness.

## Key Finding #1: The drive is untraceable, but the controller is not

Three independent identifiers agree that no OEM ever branded this drive.

```
model            SSD 1TB
serial           AA000000000000000073
firmware         T1103N0L
subsysnqn        nqn.2026-07.com.siliconmotion:nvm-subsystem-sn-AA000000000000000073
lspci            Silicon Motion SM2263EN/SM2263XT (DRAM-less) [126f:2263] (rev 03)
Identify vid     0x126f
Identify ssvid   0x126f
```

The serial is a placeholder — `AA` followed by fourteen zeros and a two-digit unit counter. More telling, **the PCI subsystem ID equals the device ID**. A vendor that buys controllers programs its own SSVID/SSID; this one never did. Two sources say so independently: PCI configuration space via `lspci`, and NVMe Identify Controller via `nvme id-ctrl`.

`EN` and `XT` share a device ID, so pci.ids cannot separate them. Identify does:

```
hmpre    16384    (x 4 KiB = 64 MiB preferred host memory buffer)
hmmin     8192    (x 4 KiB = 32 MiB minimum)
```

A controller with onboard DRAM does not request a host memory buffer. Non-zero HMPRE settles it: this is the **DRAM-less SM2263XT**, borrowing 64 MiB of host RAM to cache its flash translation layer. Finding #4 measures what that costs.

**Implication:** there is no manufacturer warranty, no published TBW rating, and no endurance specification against which the drive's own wear estimate can be validated. Any buyer can confirm all of this in ten seconds with `lspci`, which is why it is stated here first rather than buried.

## Key Finding #2: The drive is effectively new, and 690 GiB of testing did not age it

Pre-test SMART, and the same counters after the full campaign:

| Counter | Before | After | Delta |
|---|---|---|---|
| percentage used | **0%** | 0% | 0 |
| available spare | **100%** (threshold 10%) | 100% | 0 |
| media errors | **0** | 0 | 0 |
| error log entries | **0** | 0 | 0 |
| critical warning | 0 | 0 | 0 |
| data units written | 131,279 | 1,577,168 | +1,445,889 |
| data units read | 4,679 | 5,682,704 | +5,678,025 |
| power-on hours | **1** | 4 | +3 |
| power cycles | 3 | 3 | 0 |
| unsafe shutdowns | 2 | 2 | 0 |
| controller busy (min) | 9 | 147 | +138 |

One NVMe data unit is 1000 x 512 B = **512,000 bytes**, not 512 or 524,288. The conversion is worth stating explicitly because getting it wrong by 1000x is a common error:

```
TBW  = data_units_written x 512000 / 1e12
TiBW = data_units_written x 512000 / 2^40
```

So the drive had written **0.067 TB (0.061 TiB)** before testing — consistent with a single JetPack flash — and the campaign added **0.740 TB (689.5 GiB)** of writes and 2707.5 GiB of reads.

Two numbers need context rather than concealment:

- **2 of 3 power cycles were unsafe shutdowns (67%).** As a ratio that looks alarming. In absolute terms it is two events on a board that was power-cycled during flashing, and `media_errors` remains 0. The ratio is an artifact of a tiny denominator.
- **`percentage_used = 0%` is a vendor-defined estimate**, not a measurement, produced by unbranded firmware with no published endurance spec. Here it is corroborated independently: 1 power-on hour and 0.067 TB written are physically consistent with zero wear. The estimate and the raw counters agree.

**Implication:** the test campaign is itself evidence. 690 GiB of writes moved no health counter at all, and the fio-reported write total agrees with the SMART delta — which validates the firmware's write accounting as well as the drive's condition.

## Key Finding #3: The SLC cache holds 107.8 GiB, then write speed falls 4.0x

Sustained sequential write, 128 KiB blocks, QD32, single writer, 400 GiB in one pass:

| Region | Bandwidth |
|---|---|
| Burst (SLC cache) | **573.2 MB/s** |
| Peak 1 s sample | 599.7 MB/s |
| **Cliff at** | **107.8 GiB written** |
| Post-cache steady state | **142.9 MB/s** |
| Degradation | **4.01x** (25% of burst) |
| Whole-run mean (400 GiB) | 271.5 MB/s |

The cliff was detected on fio's own 1 Hz bandwidth log: a 10-sample rolling median, then the first point after which the median stays below 60% of the plateau for at least 60 consecutive samples. The sustain requirement is what separates a real cache exhaustion from a transient garbage-collection pause.

**The same analysis run on a completely independent data source agrees.** The sampling harness derives throughput from `/proc/diskstats` deltas, which is measured separately from fio's own accounting:

| Source | Cliff position | Degradation |
|---|---|---|
| fio 1 Hz bandwidth log | 107.8 GiB | 4.01x |
| `/proc/diskstats` sampler | 107.3 GiB | 4.0x |

Two independent measurements placing the cliff 0.5% apart is much harder to dismiss than either alone.

**Interpretation:** a post-cache sustained write of ~143 MB/s is characteristic of TLC NAND written direct-to-array. QLC would typically be substantially lower. This is offered as an inference from behaviour, clearly labelled as such — the vendor log pages that would confirm NAND type did not decode (see Limitations).

**Practical pattern:** for writes under ~100 GiB in a single burst, expect ~570 MB/s. For anything larger, budget 143 MB/s.

## Key Finding #4: 4K random read is span-dependent — the DRAM-less signature

A DRAM-less controller cannot hold its full flash-translation-layer map on board. At roughly 4 bytes of mapping per 4 KiB page, a 1 TB drive needs ~1 GB of table; the 64 MiB host memory buffer covers only a fraction of it. Reads inside the cached window are fast, reads outside it pay an extra NAND fetch for the mapping.

Measuring one span would mislead in one direction or the other, so the curve was measured (filesystem, O_DIRECT, 4 KiB):

| Span | QD1 | QD4 | QD32 | 4 jobs x QD32 |
|---|---|---|---|---|
| 4 GiB | 8,382 | 26,299 | 102,992 | 119,367 |
| 64 GiB | 8,409 | 24,555 | 105,310 | **136,930** |
| 300 GiB | 7,119 | 21,813 | 84,859 | 109,133 |

IOPS. The 4 GiB span is smaller than the machine's 15 GiB of RAM; `direct=1` (O_DIRECT) is what makes that result valid — the page cache is bypassed entirely and no buffered result appears anywhere in this paper.

The effect is real but moderate: **300 GiB is ~20% slower than 64 GiB** at QD32. This is a mild version of the DRAM-less penalty, not a severe one.

**Practical pattern:** working sets up to ~64 GiB get the drive's best random-read behaviour. Beyond that, expect roughly a fifth less.

## Key Finding #5: ext4 costs 3% on sequential reads and 2.5x on 4K random reads

Because raw-device *reads* are non-destructive, every read test could be run against `/dev/nvme0n1` directly as well as through the filesystem. The difference is the filesystem's cost, measured rather than asserted:

| Test | Through ext4 | Raw device | Delta |
|---|---|---|---|
| Sequential read, 128 KiB, QD32 | 2197.7 MB/s | **2271.5 MB/s** | ext4 costs **3.2%** |
| 4K random read, 4 jobs x QD32 | 109,133 IOPS (300 GiB span) | **269,512 IOPS** (full 931 GiB span) | raw is **2.47x** faster |

The sequential result is unremarkable — 3.2% is ordinary filesystem overhead. The random result is not, and it deserves care: the raw test covered the **entire 931 GiB device**, a *larger* span than the 300 GiB filesystem test, and was still 2.5x faster. Span cannot explain it; per Finding #4 a larger span should have been slower.

The remaining difference is the per-I/O cost of ext4's extent mapping on 4 KiB requests, which is charged to every single operation and is proportionally enormous at that block size. The honest framing: **the filesystem numbers throughout this paper understate the drive**, and for small random reads they understate it by a large factor.

## Key Finding #6: The temperature sensor is dead

This is the most important negative result in the paper, and it invalidates a section that was planned.

The kernel nvme driver exports the drive's own composite sensor through hwmon. Across the entire 2.8-hour campaign — 9,191 samples at 1 Hz, including 400 GiB of sustained writes — it returned:

```
distinct values of /sys/class/hwmon/hwmon2/temp1_input over 9191 samples:
   9191  39850
```

**One value. Never once did it move.** For contrast, the SoC junction sensor logged into the same CSV at the same timestamps produced 191 distinct values spanning 56.3–62.4 °C, so the sampler is demonstrably working.

A second, independent source confirms it. SMART log page 02h reports `temperature: 313` (40 °C) — byte-identical before the campaign, after it, and after the final cooldown.

Consequently **every thermal counter this drive reports is meaningless**:

```
warning_temp_time      0 min      thm_temp1_total_time    0 s   (0 transitions)
critical_comp_time     0 min      thm_temp2_total_time    0 s   (0 transitions)
```

Those zeros are not evidence of good thermal behaviour. They are derived from a sensor that does not work, and **they would read zero on a drive that was on fire.** Reporting "no thermal throttling observed" on this basis would be false.

What can honestly be said is behavioural, not thermal: sustained write throughput after the SLC cliff was flat at ~143 MB/s with a converged least-squares drift verdict, showing no progressive degradation of the kind thermal throttling produces. That is weak evidence of no throttling, and it is the only evidence available.

**Implication for a buyer:** the drive appears to function correctly, but it cannot report its own temperature. In an enclosed or passively cooled host, where thermal headroom actually matters, this drive gives no warning whatsoever. That is a genuine defect and it is disclosed here rather than omitted.

## Key Finding #7: The 128 KiB I/O cap is the host's, not the drive's

The block layer reports `max_hw_sectors_kb=128`, so no single I/O larger than 128 KiB reaches the device. It would be easy — and wrong — to attribute this to the drive. Identify Controller says:

```
mdts     6      => 2^6 x 4 KiB = 256 KiB maximum data transfer
```

The controller supports **256 KiB** transfers. The 128 KiB ceiling is imposed by the host's PCIe/DMA configuration on this Tegra platform.

This has a measurable consequence that any benchmark of this drive must account for: a job specifying `bs=1M, iodepth=8` does not present the device with 8 outstanding 1 MiB commands. The block layer splits each into 8 x 128 KiB, so the device sees 64 outstanding 128 KiB commands. Both were therefore measured and reported:

| Configuration | Bandwidth |
|---|---|
| `bs=128k` (device-native, honest hardware number) | 2197.7 MB/s |
| `bs=1M` (application view, split by the block layer) | 2183.8 MB/s |
| `bs=128k`, 4 jobs | 1671.4 MB/s |

The four-job result being *lower* is worth noting: this workload is not thread-limited, and splitting a sequential stream across threads destroys its sequentiality.

**Implication:** on a host without this cap the drive may perform better than measured here. The ceiling is not a property of the drive.

## Key Finding #8: libaio was not the bottleneck — the drive was

The obvious objection to any benchmark run on an 8-core Arm SoC is that the host CPU, not the drive, set the limit. Running the peak random-read configuration under a second I/O engine bounds it:

| Engine | 4K random read, 4 jobs x QD32 |
|---|---|
| libaio | 109,133 IOPS |
| io_uring | 115,719 IOPS |

Six percent apart. Had io_uring been dramatically faster, the libaio figures would have been host-limited and this paper would have had to say so. It was not, so the drive was the limit — which is the stronger claim, and it is now evidenced rather than assumed.

## Key Finding #9: Random write is this drive's weakness

| Configuration | IOPS | Bandwidth | Mean latency |
|---|---|---|---|
| 4K randwrite QD1 | 8,434 | 34.5 MB/s | 106 µs |
| 4K randwrite QD4 | 10,580 | 43.3 MB/s | 361 µs |
| 4K randwrite QD32 | 11,735 | 48.1 MB/s | 2,718 µs |
| 4K randwrite 4 jobs x QD32 | 11,942 | 48.9 MB/s | 10,727 µs |
| After 30 min preconditioning | 12,269 | 50.3 MB/s | 10,417 µs |

Throughput saturates near **12,000 IOPS** and additional queue depth buys latency, not performance. Against ~110,000 random *read* IOPS, this is an approximately 9:1 read/write asymmetry — expected for a DRAM-less consumer controller, but a real constraint.

QD1 latency, measured with `ioengine=psync` rather than libaio (at depth 1 the libaio submit/reap path is host overhead, not drive service time — the engine change is disclosed because an unexplained switch reads as cherry-picking):

| Test | Mean | p50 | p99 | p99.9 | p99.99 | Max |
|---|---|---|---|---|---|---|
| 4K random read | 128.7 µs | 118.3 | 193.5 | 444.4 | 4,177.9 | 45,236 µs |
| 4K random write | 116.3 µs | **39.2** | 317.4 | 618.5 | 126,353 | **457,509 µs** |
| 4K random write + fdatasync | 316.6 µs | 264.2 | 452.6 | 3,915.8 | 5,668.9 | 5,808 µs |

The random-write tail is the number a buyer should see: a p50 of 39 µs absorbed by the write cache, but a **worst case of 457 ms** — a garbage-collection stall four orders of magnitude above the median. The durable variant (`fdatasync=1`, forcing an NVMe FLUSH per write) is slower on average but far better behaved in the tail, with a maximum of 5.8 ms.

**Practical pattern:** fine for boot volumes, model storage and sequential media work. Not a database drive.

## Key Finding #10: Full-surface integrity is clean

Two non-destructive integrity tests, both passed:

- **Full-LBA surface scan.** All **1,953,525,168 LBAs** (931.0 GiB) read against the raw device with `continue_on_error=none`, at 2271.5 MB/s. fio error code 0. `media_errors` was 0 immediately before and immediately after.
- **crc32c write-verify** over 64 GiB with `verify_fatal=1`. fio error code 0 — every block read back matched what was written.

After the campaign the test file was removed and `fstrim` released **890.2 GiB (955,889,848,320 bytes)** back to the controller, with `df` confirming the filesystem returned to 853 G free.

## Methodology Note: Measure Your Own Instrument

The sampling harness written for this campaign reports throughput from `/proc/diskstats` deltas divided by the nominal sample interval. Cross-checking it against fio on the same workload:

| Source | Surface scan bandwidth |
|---|---|
| fio | 2271.5 MB/s |
| harness (`/proc/diskstats`) | 2995.4 MB/s |

The harness **overstates by 31.9%**. The cause is that it divides the counter delta by the *nominal* 1.000 s interval while the real loop period is longer — each sample forks a dozen subshells to read sysfs, and under heavy I/O that overhead grows. Measured across the whole run the mean period was 1.103 s, and during I/O-heavy phases it reached ~1.32 s.

Two consequences, both acted on:

1. **Every performance number in this paper comes from fio**, never from the harness. The harness's absolute values are unreliable.
2. The harness remains valid for *relative* shape within a phase — which is why it independently reproduced the SLC cliff position to within 0.5% (Finding #3). A consistent multiplicative bias does not move the location of a 4x discontinuity.

The general lesson matches [paper 7](research_7_jetson_fan_curve_thermal.md)'s insistence on logging millidegrees: an instrument that is never checked against a second instrument will silently report whatever it reports. The 32% error would have been invisible without fio to compare against, and it would have produced a paper claiming this drive exceeds its own PCIe Gen3 x4 link — the harness's peak samples reached 7165 MB/s against a theoretical ceiling of 3940 MB/s. **A number that violates a hardware limit is the instrument confessing.**

## Limitations

- **Write tests ran through mounted ext4 on the live root partition**, not the raw device — writing to the raw device would have destroyed the system. Per Finding #5 the filesystem costs 3.2% on sequential reads and far more on 4K random, so **all write figures understate the drive**.
- **The operating system ran on the drive throughout.** The idle noise floor was measured rather than assumed: 0.00–0.12 MB/s of background writes over a 10-minute baseline. Competing I/O was minimized, not eliminated.
- **Random-write results are optimistic.** True SNIA steady state requires two full-device sequential passes plus a full-device random pass, which is impossible non-destructively on a live root filesystem. Steady state was reached only within a 300 GiB span on a drive with ~450 GiB of free, TRIMmed spare area, so garbage collection was far easier than it would be on a full drive. **Real-world random write on a full drive will be worse than reported here.** This is the one caveat that flatters the drive, which is why it is stated in bold.
- **The compressibility control test is inconclusive.** Paired 60 s sequential writes with and without `refill_buffers=1` returned 227.4 MB/s and 380.7 MB/s respectively. The intended reading — that a controller cheating on compressible data would be *faster* without refill — is contradicted: the fresh-random run was faster. The two ran back-to-back immediately after the 400 GiB fill, when garbage collection was still draining, so the ordering confounds the comparison entirely. **This test proves nothing about compression** and would need re-running on a rested drive.
- **Thermal behaviour is uncharacterized.** Per Finding #6 the drive's sensor is frozen. No thermal claim, positive or negative, can be made from this drive's telemetry. Ambient was not instrumented.
- **Vendor log pages did not decode.** Silicon Motion log IDs were probed for NAND-level data (raw P/E cycles, bad-block counts). Nothing interpretable was returned. No claim is made about undecoded bytes; inventing field meanings for a vendor log is how a report loses credibility when checked.
- **Single specimen, N=1** for the long tests. Results describe this drive, not the model — and "the model" is not a meaningful category for an unbranded product.
- **Host-imposed 128 KiB I/O cap** (Finding #7) and **PCIe Gen3 x4** (~3940 MB/s theoretical). The drive's endpoint reports `max_link_speed` of 8 GT/s, so Gen3 is the controller's native ceiling and the link is *not* artificially limiting it.
- **The board clock was at epoch** (no RTC battery, no NTP without internet). All absolute timestamps in the raw logs read 1969; the sampler's relative-second timeline and every measurement are unaffected. Host-derived offset: 20,669 days.

## Reproduction

```bash
# 1. Stage packages on an offline board (no internet on the Jetson).
#    Resolve URIs from the BOARD's apt lists, not the x86 host's - they differ.
ssh orinnx 'apt-get install --print-uris -qq fio smartmontools' \
  | sed "s/^.\(http[^ ]*\).*/\1/;s/'$//" > uris.txt
wget -P debs -i uris.txt && scp debs/*.deb orinnx:~/nvme-char/debs/

# 2. The ONLY step needing a password. Installs packages, quiesces fstrim.timer
#    and unattended upgrades, installs an allowlisting root helper scoped to
#    read-only nvme verbs, and captures pre-test identity + health.
ssh -t orinnx 'sudo bash ~/nvme-char/00-root-setup.sh'

# 3. Everything else is unprivileged. ~2.8 h. Survives SSH drops.
ssh orinnx 'cd ~/nvme-char && ./run-suite.sh --dry-run'   # validate first
ssh orinnx 'cd ~/nvme-char && setsid nohup ./run-suite.sh &'

# 4. Analyse.
awk -f tools/nvme-char-analyze.awk -v warn_c=82 -v cliff_phase=A-slc-fill \
    results/*/nvme-soak-*.csv

# 5. Revoke the temporary privilege grant.
ssh orinnx 'sudo -n /usr/local/sbin/nvme-char-priv teardown'
```

The privilege grant used during testing was exactly:

```
orinnx ALL=(root) NOPASSWD: /usr/local/sbin/nvme-char-priv
```

That helper allowlists read-only NVMe verbs only. `format`, `sanitize`, `set-feature`, `fw-download`, `fw-commit`, `write*`, `dsm` and all namespace-management commands are unreachable through it, and its `fio-ro` verb refuses to invoke fio against a raw device without `--readonly`. The refusals were exercised and recorded — see `nvme-20260804-safety-layer-proof.txt`. The grant was revoked at teardown.

## Tooling

- [`jetson-tools/nvme-soak.sh`](https://github.com/Hi5808/jetson-tools) — 1 Hz sampler; sources `jetson_thermal_lib.sh` unchanged (`jt_hwmon_by_name nvme` already locates the composite sensor). Reads the current phase label from a marker file on tmpfs each tick, so one CSV spans an arbitrary external suite without the sampler generating I/O to the drive it is measuring. **See the Methodology Note before trusting its absolute values.**
- `jetson-tools/nvme-char-analyze.awk` — per-phase envelope, SLC cliff detection, and the same least-squares final-third drift regression `jetson-soak.sh` uses, applied to drive temperature.

## Raw Data

| File | Contents |
|---|---|
| `nvme-20260804-slc-write-profile.csv` | The headline figure's data: 1,582 rows of `t_s, cumulative_gib, write_mbs` from fio's own 1 Hz log. Cliff at 107.8 GiB. |
| `nvme-20260804-sampler-full-run.csv` | All 9,191 harness samples, 18 columns, whole campaign. `nvme_mc` is the frozen sensor — retained as the evidence for Finding #6, not as data. |
| `nvme-20260804-smart-{pre,post,final}.json` | SMART before, immediately after, and after cooldown + TRIM. |
| `nvme-20260804-identity-health-pre.txt` | Full `nvme id-ctrl`/`id-ns`/feature dump, human-readable. |
| `nvme-20260804-lspci.txt` | `lspci -nnvv` for the controller — the independent provenance source. |
| `nvme-20260804-safety-layer-proof.txt` | Recorded refusals of `nvme format`, `sanitize`, `set-feature`, `write`, `fw-commit`, and unguarded raw-device fio. |
| `nvme-20260804-fio-json.tar.gz` | Every fio run as `json+`, including full latency percentile distributions. |

The two CSVs differ deliberately in what they measure: the profile is fio's self-reported bandwidth, the sampler is `/proc/diskstats`. Keeping both is what made the 32% instrument bias detectable.

## Conclusion

The drive is healthy, genuinely near-new, and performs as a Gen3 x4 DRAM-less TLC SSD should: ~2.3 GB/s sequential read, ~570 MB/s write into a 107.8 GiB SLC cache falling to 143 MB/s beyond it, ~110k random read IOPS, and a weak ~12k random write IOPS. Full-surface and crc32c verification passed with zero media errors.

Two things a datasheet would not tell you emerged only from measuring: the drive's temperature telemetry is **non-functional**, and the 128 KiB I/O ceiling belongs to the **host**, not the drive.

The broader lesson is the Methodology Note. This campaign nearly published throughput figures 32% too high — numbers that would have exceeded the drive's own PCIe link and been trivially falsifiable by any reader who checked. What caught it was measuring the same quantity two independent ways and noticing they disagreed. For a report whose entire purpose is to be believed by a skeptical reader, the redundant measurement was worth more than any single result it produced.
