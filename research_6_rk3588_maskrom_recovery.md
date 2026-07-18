# Recovering a Rockchip RK3588 Board from Maskrom Under a Non-Elevated Windows Session: An Empirical Debugging Study

## Abstract

We document the recovery and OS provisioning of a brand-new Radxa ROCK 5B+
(Rockchip RK3588) that arrived in an unknown initial state and presented to a 
Windows 11 host as an unrecognized, driver-less maskrom-mode USB device. The recovery 
outcome was uncertain: multiple independent failure points could have rendered the 
device unrecoverable (tool regressions, driver-installation blocks, USB-level 
incompatibilities), and the successful resolution converged on a narrow path 
through a combination of systematic debugging and fortunate circumstance. 
The work spans four independent problem domains — host-side device discovery, 
non-elevated toolchain assembly, OS image selection, and firmware-flashing tool 
regression diagnosis — each requiring direct empirical evidence rather than 
documentation or forum consensus. The most consequential finding was methodological 
rather than technical: an initial USB device scan filtered on device class and status, 
predicates that structurally exclude any device defined by being in a failed state, 
causing the target hardware to be reported absent when it was physically present 
throughout. A second major finding root-caused a widely-reported, unresolved 
community bug (Rockchip's RKDevTool v2.96+ `err=995` regression) via binary 
string-diffing between working and broken tool versions, without source access or 
a debugger. We report the full diagnostic procedure, the evidence discriminating 
between competing hypotheses at each step, and the final validated configuration, 
including a post-flash inspection that surfaced a genuine SDK-compatibility 
blocker (Python 3.14 vs. RKNN toolkit wheels capped at cp312) before it could 
manifest as a confusing runtime failure.

## 1. Introduction

### 1.1 Motivation

A brand-new Radxa ROCK 5B+ arrived in an unknown initial state and presented 
as an unrecognized maskrom-mode device. Recovery was uncertain: multiple 
independent failure points in the toolchain, driver ecosystem, and firmware 
compatibility could have resulted in total loss with no recovery path. This 
work documents a successful recovery that converged through systematic 
debugging and fortunate availability of working alternative toolchain components.

Flashing from a non-elevated Windows session with no prior toolchain installed 
is a realistic constraint for many first-time bring-ups, not an artificial one — 
admin rights are commonly unavailable on managed or loaner machines. Under this 
constraint, failures that would be routine with elevated tooling (driver 
installation, package-manager installs) become hard boundaries requiring 
workarounds, and every diagnostic step must be justified without the convenience 
of reinstalling drivers or running arbitrary installers. This motivates documenting 
not just the working configuration but the full sequence of ruled-out approaches, 
since the non-elevated constraint changes which fixes are even available — and 
in this case, some failure modes would have been unrecoverable regardless.

### 1.2 Target platform

- **Board**: Radxa ROCK 5B+ (Rockchip RK3588, 8-core big.LITTLE
  Cortex-A76/A55, Mali-G610 GPU, 6 TOPS NPU, 64GB onboard eMMC)
- **Host**: Windows 11 Home, build 10.0.26200, non-elevated session
  (confirmed via `WindowsPrincipal.IsInRole(Administrator)` returning
  `False`)
- **Target software**: Armbian 26.2.6, Ubuntu 26.04 (resolute) userland,
  vendor kernel 6.1.115, KDE Plasma desktop — selected over the
  vendor-recommended Radxa OS on evidence detailed in §3.3
- **Flashing tool**: RKDevTool v2.86, selected only after v2.96 and a
  WSL2/usbip-based `rkdeveloptool` path both failed for independent,
  root-caused reasons (§3.4)

### 1.3 Scope

This paper covers hardware recovery and OS bring-up only: device
discovery, toolchain assembly under a non-elevated account, OS image
selection, and the flashing procedure itself through first successful
boot. Post-boot NPU/RKNN userspace configuration is referenced only
insofar as it was verifiable directly from the shipped image before any
flash was attempted (§3.5); further on-device work is out of scope.

## 2. Methodology

Three techniques provided the actual discriminating evidence across
this work, in place of documentation or forum-reported fixes taken on
faith:

**Predicate reconstruction on a failed device search.** When an initial
enumeration approach reported the target absent, the corrective step was
not to search harder with the same method but to identify which
predicates in the query could structurally exclude the object being
searched for, then reconstruct the query without them (§3.1).

**Binary diffing between a working and a broken tool version.** With no
source access to Rockchip's closed-source RKDevTool, the regression
between a working version (v2.86) and a broken one (v2.96) was
root-caused by extracting and diffing printable strings from both PE
binaries, then correlating the delta against the tool's own log output
across multiple runs (§3.4).

**Direct image inspection over forum consensus.** Rather than trusting
contradictory community reports about NPU driver availability on
Armbian's vendor kernel, the shipped `.img` file was inspected directly
— GPT and ext4 traversal via 7-Zip, no mount or elevation required — to
extract the actual kernel `.config` and confirm accelerator support
before any flash was attempted (§3.5).

## 3. Findings

### 3.1 Host-side discovery: a selection-effect bug, not a detection failure

**Symptom.** An initial PowerShell device scan,
`Get-PnpDevice -Class USB -Status OK`, returned ten devices and did not
include the target board. The board was reported as absent.

**Diagnosis.** A corrected scan dropping both predicates —
`Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB*' -or $_.InstanceId -like 'USBSTOR*' }`
— surfaced the board immediately, with `Status = Error` and no assigned
device class, alongside `Problem = 28` (`CM_PROB_FAILED_INSTALL`).

**Interpretation.** A device whose driver fails to install receives no
device class assignment, which excludes it from a `-Class USB` filter,
and does not report `Status OK`, which independently excludes it from a
`-Status OK` filter. Both predicates in the original query excluded
precisely the class of object being searched for. This is a
selection-effect bug rather than a detection failure: the device was
never physically absent, and the query was constructed such that it
could not appear in the result set regardless of hardware state. This
error was caught by re-running the scan under correction, not by
reviewing the original query's logic in isolation — the flawed query
was internally consistent and gave no indication it was excluding its
own target.

**Consequence for identification.** With the board now visible,
`Get-PnpDeviceProperty` against instance ID
`USB\VID_2207&PID_350B\6&2CDBED82&0&1` resolved VID `0x2207` (Fuzhou
Rockchip Electronics) and PID `0x350B` (RK3588 maskrom mode), confirming
the physical layer — cabling, power, USB enumeration — was fully
functional; code 28 specifically indicates the device completed
enumeration and returned valid descriptors, which rules out the more
severe failure classes (code 43, code 10) that would indicate a hardware
fault. A discrepancy between the observed PID (`0x350B`) and the PID
documented in Radxa's own maskrom-detection guide (`0x350E`) was noted
but not resolved; the device behaved as expected regardless, and the
two values are both within Rockchip's maskrom PID range.

### 3.2 Toolchain assembly under a non-elevated account

**Symptom.** No archive tool, no `curl`, and no Rockchip flashing
utilities were present on the host. RKDevTool ships as a `.rar`, which
Windows cannot open natively, and the target OS images ship as `.xz`,
which the bundled `bsdtar` cannot decompress directly.

**Diagnosis and resolution — archive tooling.** Two standard
installation paths failed for reasons specific to the non-elevated
constraint: `winget install 7zip.7zip --scope user` failed with an
`msstore` source certificate error unrelated to permissions, and the
`--source winget --scope user` variant failed because 7-Zip's winget
manifest publishes no user-scope installer at all — the package
metadata itself assumes elevation. The working path was 7-Zip's
standalone NSIS installer run with silent, user-writable flags
(`7z2409-x64.exe /S /D=C:\Users\hanau\7zip`), which neither package
manager path exposed as an option. This generalizes beyond 7-Zip: a
package manager's `--scope user` flag is a manifest-declared capability,
not a guarantee, and a standalone installer's native silent-install
support can bypass an admin assumption baked into the packaging layer
above it.

**Consequence — driver installation remained genuinely blocked.**
`DriverAssitant_v5.0`'s `DriverInstall.exe` writes to the system driver
store and requires elevation with no non-elevated equivalent; this was
the one blocker in the entire session with no available workaround,
and is reported as such rather than papered over.

**Secondary finding — HTTP request parsing in non-interactive
PowerShell.** `Invoke-WebRequest -Method Head` without
`-UseBasicParsing` failed with a `NonInteractive mode` error, traced to
the default IE-engine HTML parser attempting to initialize IE's
first-run configuration; `-OutFile` downloads were unaffected. The fix
(`-UseBasicParsing` unconditionally in non-interactive contexts) is
narrow but was load-bearing for the provenance-recovery step in §3.3.

### 3.3 OS image selection: freshness as a checkable criterion, and a provenance-recovery false start

**Symptom.** No canonical source clearly identified the best-maintained
OS for this board; Radxa's own vendor image and the most-recommended
community image (Ubuntu Rockchip) were the two most commonly referenced
options.

**Diagnosis.** Querying each project's release metadata directly rather
than deferring to reputation reversed the initial choice: Radxa OS's
latest image dated to 2024-08-08 (~23 months stale at the time of this
work) on kernel 6.1/Debian 12, and Ubuntu Rockchip's latest release was
similarly dated (2024-10-23) on Ubuntu 22.04. Armbian's rolling build
for this board, by contrast, was current as of the work date, rated
Platinum tier (Armbian's highest board-support classification), with an
upstream build-repository commit from three days prior. The
vendor-official image was, by this measure, the stalest of the three
options — a result that would not have surfaced from forum consensus,
which favored the vendor image on reputation alone.

**Consequence for the flash procedure.** The toolchain — loader,
driver, and maskrom sequence — was independent of which `.img` was
selected; only the payload changed. This is a property of the maskrom
flow worth stating explicitly: it argues for staging the flashing
toolchain before finalizing OS choice, since a reversal on the OS
decision costs nothing on the toolchain side.

**A provenance-recovery false start, reported for completeness.** The
selected Armbian image was renamed at download time, losing the
canonical filename needed to confirm exact variant (kernel branch,
desktop environment, build date). An attempt to recover this by
byte-scanning the raw `.img` for `VERSION=`/`BRANCH=`/`LINUXFAMILY=`
strings returned hundreds of unrelated matches from every text-bearing
file inside the ext4 filesystem (browser certificate stores, glibc
strings, `figlet` data), because metadata inside a mounted-filesystem
image is not at a predictable byte offset and is not meaningfully
greppable without actually mounting it. The working approach instead
resolved provenance from the transport layer: a HEAD request against the
original download URL
(`Invoke-WebRequest -Method Head -UseBasicParsing`, per §3.2) returned
the server's resolved `ResponseUri`, which contained the full canonical
filename. This generalizes: when a local artifact's identity has been
lost, checking where it came from is often more tractable than
inspecting what it contains, particularly for structured binary formats
with no predictable metadata layout.

### 3.4 Flash-path failures: wrong-tab GUI errors, a usbip dead end, and a binary-diffed regression

**Symptom, attempt 1 — RKDevTool v2.96, GUI, "Upgrade Firmware" tab.**
Two independent failures: `RKU_ReadLBA-->RKU_Read failed` and
`CheckDownloadItem--> is not existed!`, followed by
`LoadFwProc-->create image object failed,ret=-5!`.

**Diagnosis.** The Upgrade Firmware tab expects a Rockchip `update.img`
*container* format; the Armbian artifact is a raw GPT disk image, which
this tab cannot parse (`ret=-5`). Separately, the bundled
`rock-5b-emmc.cfg` ships with the original author's absolute Windows
paths pointing at files that do not exist on this host
(`C:\Users\rock\Desktop\...`), so loading the config file is not
self-sufficient and both entries require manual repointing. Two
incorrect intermediate conclusions were reached and then corrected
during this attempt — that the shipped config was usable as-is, and that
a `RKDevelopTool_v1.37.zip` archive was the documented Linux-only
`rkdeveloptool` CLI (it is in fact `RKAndroidTool.exe`, an unrelated,
older, Android-oriented GUI) — both recorded here since the correction
itself is informative: a raw disk image and a Rockchip firmware
container require different tool tabs, and identically-named archives
across the Rockchip tooling ecosystem are not reliably related tools.

**Symptom, attempt 2 — WSL2 + usbipd + `rkdeveloptool`, four sub-attempts.**
The Linux-native CLI path (`rkdeveloptool ld`/`db`/`wl`/`rd`) was
attempted via `usbipd-win` forwarding the board's USB device into WSL2.
`ld` correctly identified the board in maskrom mode. `db` (load the SPL
loader, transitioning the board from maskrom to loader mode) either
returned exit 0 with the loader not actually persisting, or hung
indefinitely (700s+, confirmed via `ps -eo etimes,args`), leaving the
board's USB controller unresponsive
(`VID_0000&PID_0002`, "Device Descriptor Request Failed") and requiring
a hard power cycle to recover.

**Diagnosis.** The maskrom-to-loader transition that `db` triggers is a
USB-level device reset and re-enumeration
(`USB\VID_2207&PID_350B\6&...` changes instance ID entirely). On the
Windows side, this both drops the existing usbip attachment and reverts
`usbipd`'s persisted share state from "Shared" back to "Not shared,"
because that state is keyed to the pre-reset device instance and does
not carry forward to the re-enumerated one. A first hypothesis blamed
`usbipd attach --auto-attach`'s background monitor for contention with
the reset; a controlled re-test with `--auto-attach` removed reproduced
the identical 716-second hang, ruling that hypothesis out and correcting
the initial attribution. Four attempts across this path — two testing
the underlying transition, two testing the auto-attach hypothesis —
consistently failed to get a loader to persist across the tunnel.

**Interpretation.** This is a clean, reproducible negative result: a
USB-level device-identity reset does not survive `usbip` forwarding,
because `usbipd`'s share state is bound to a specific device instance
that the reset invalidates. This generalizes past this specific board:
any device-mode-switching firmware tool is a poor candidate for
`usbip`-based remote development, independent of the specific tool or
target chip, and this constraint is not documented in Radxa's,
Rockchip's, or `usbipd-win`'s own documentation.

**Symptom, attempt 3 — RKDevTool v2.96, GUI, correct tab this time.**
With the wrong-tab and dead-path errors from attempt 1 corrected, v2.96
now successfully wrote the loader and IDB and correctly detected the
onboard eMMC (`CS(1) 59640MB SAMSUNG`), then stopped or crashed *before*
the actual image write, logging `err=995` (`ERROR_OPERATION_ABORTED`)
and `err=31` on subsequent `RKU_Read`/`WriteFile` calls.

**Diagnosis.** A public Radxa forum thread (#18442) documented an
identical `err=995` failure signature — "USB pipes reset repeatedly" —
across many ROCK 5B users from September 2023 through January 2025,
with no confirmed fix at the time of this work, establishing this as a
known, widespread tool regression rather than a local misconfiguration.
Substituting the older RKDevTool v2.86 in place of v2.96, with the same
image and loader, proceeded past the point v2.96 consistently failed
at and into the actual image write
(`Layer<2-1>: Download Armbian_..._kde-plasma_desktop at 0x00000000`).

**Root-causing the regression via binary string-diffing.** With no
source access to either closed-source RKDevTool build, printable strings
were extracted from both PE binaries via regex and diffed. v2.96
(1,240,576 bytes) contains a set of strings entirely absent from v2.86
(1,197,056 bytes, same MFC architecture, +43KB delta): `SwitchStorageProc`,
`RKU_ReadStorageList`, `MakeParamFileBuffer`, `ParsePartitionInfo`,
`GetLoaderHeadSize`, and associated error-string IDs. This was
corroborated against the tool's own log output across every run of each
version: every v2.96 run logged `current storage = EMMC, switch storage
= EMMC` (the new `SwitchStorageProc` step); no v2.86 run ever logged it.
The resulting hypothesis — evidence-based, not source-confirmed — is
that v2.96 added a storage-switch handshake and Rockchip
parameter-block/GPT parsing step to the Download-Image code path that
v2.86 does not have; a raw whole-disk image such as the Armbian artifact
has no such Rockchip parameter block, so the parse either fails outright
or the additional USB round-trips the switch step introduces destabilize
the subsequent large bulk write, producing the observed `err=995`/`err=31`
abort. Patching v2.96 to skip this call was considered and explicitly
not attempted: it would require disassembling a closed-source binary
that writes to eMMC with no reversibility if the patch is wrong, against
a tool (v2.86) already confirmed to work by a safer path.

### 3.5 Post-flash outcome and NPU verification by direct image inspection

**Outcome.** RKDevTool v2.86's Download Image write completed in
approximately four minutes (consistent with the full 7.6GB image at
roughly 33MB/s), logging `RunProc is ending, ret=1` at completion. This
return code was initially ambiguous — `ret=1` could plausibly indicate
either a cosmetic post-write reset/verify step or a genuine data
failure. The board was power-cycled and observed to boot normally (no
fallback to maskrom), which is the definitive confirmation the tool's
own exit code did not provide. This generalizes: for a flashing tool
with an ambiguous or historically unreliable exit code, the correct
verification is booting the target, not trusting the reported return
value.

**NPU/accelerator verification, performed before the flash.** Rather
than adjudicating contradictory forum reports about RKNPU driver
availability on Armbian's vendor kernel, the shipped `.img` was
inspected directly using 7-Zip's native ability to traverse a GPT
partition table and descend into an ext4 filesystem with no mount step
and no elevation:
`7z l image.img -ba` and `7z e image.img -o<out> "boot/config-*" "etc/armbian-release"`.
One methodological note recorded during this step: extracting by
partition name (`0.rootfs.img`) silently returns zero files, because
7-Zip auto-descends directly into the filesystem and file paths must be
given relative to the ext4 root (`boot/...`), not the partition
container — this cost real time before being identified. The extracted
kernel configuration confirmed `CONFIG_ROCKCHIP_RKNPU=y` (built in, not
a loadable module — it cannot fail to load), full Mali-G610 GPU support
(`CONFIG_MALI_BIFROST=y`, `CONFIG_MALI_CSF_SUPPORT=y`), and the complete
Rockchip MPP video codec stack including AV1 decode, all compiled in
directly rather than as modules that could be absent at boot.

**A genuine SDK-compatibility blocker, found before it could surface as
a runtime failure.** The image's userspace ships Python 3.14
(`usr/bin/python3.14`, Ubuntu 26.04 LTS default), while `rknn-toolkit2`
v2.3.2 — the current release, last updated roughly 15 months prior —
ships wheels only through `cp312`. This is a hard two-minor-version ABI
gap with no available wheel, not a warning that could be worked around
by installing an older toolkit release. The hypothesis that Armbian's
Debian-based Trixie variant might sidestep this by shipping an older
Python was tested directly by downloading and inspecting that image as
well: it ships Python 3.13, one minor version closer but still past the
cp312 ceiling. Both variants fail identically, and — since the two
images' kernel configurations were confirmed byte-identical via `cmp`
— the OS variant decision is orthogonal to both the accelerator-driver
question (§3.5, first paragraph) and this SDK-compatibility question;
conflating the two would have been an error. The durable resolution
(a Miniforge/conda environment pinned to Python ≤3.12, or driving
inference through the C/C++ API directly against `librknnrt.so`, which
is Python-version-independent) was identified but not yet exercised at
the point this paper's scope ends.

## 4. Discussion

Several patterns generalize beyond this specific board recovery:

**A query that returns zero results is not evidence of absence when the
predicates themselves can exclude the target by construction.** The
selection-effect bug in §3.1 is the clearest instance of this, but the
same shape recurs in miniature throughout the session: a package
manager's `--scope user` flag that has no manifest entry to satisfy it
(§3.2), a byte-scan for version strings that cannot distinguish a
filesystem's metadata from its payload (§3.3), and a partition-name
extraction that silently returns nothing because the real root is one
level deeper (§3.5). In each case, the fix was not to search harder
along the same axis, but to identify what the query structurally could
not see.

**A known-working reference (an older tool version, a stock image, a
stable transport layer) is a diagnostic instrument, not just a
fallback.** RKDevTool v2.86's continued success while v2.96 failed was
the single piece of evidence that made the `SwitchStorageProc`
hypothesis checkable at all — without a working comparison point, the
`err=995` failure would have had no clear discriminating signal between
"corrupt image," "bad loader," "flaky USB," and "tool regression."

**Ambiguous tool exit codes require an independent, ground-truth
verification step.** `ret=1` on a successful write and `db` returning 0
without the loader actually persisting (§3.4) are the same failure
mode in miniature: a tool's own reported status is not reliable evidence
on hardware/firmware flashing tools with a documented history of
regressions, and the actual verification (booting the board; confirming
the loader survives a subsequent `wl` call) has to be independent of the
tool's self-report.

**Freshness and compatibility are separate axes and must be checked
separately.** Armbian was correctly selected as the best-maintained
option for this board (§3.3) by checking release freshness directly.
That same freshness, applied to the OS's default Python version, is
precisely what produced the RKNN wheel incompatibility in §3.5 — the
newest OS and the stalest SDK release landed on opposite sides of a
Python ABI break. Freshness is not monotonically good; it has to be
checked for mutual compatibility across the full stack being assembled,
not maximized independently per component.

## 5. Conclusion

A brand-new Radxa ROCK 5B+ arriving in an unknown initial state and presenting 
as an unrecognized maskrom-mode device on a non-elevated Windows host was 
successfully recovered, flashed with Armbian 26.2.6 (vendor kernel 6.1.115, 
KDE Plasma), and confirmed booting to a functional, networked, SSH-accessible 
desktop. This outcome was not inevitable: multiple independent failure points 
along the path — tool regressions, driver-installation blocks, USB-level 
incompatibilities — could have rendered recovery impossible. Success required 
the convergence of systematic debugging with fortunate circumstance: a known-working 
older tool version (RKDevTool v2.86) available as a fallback when v2.96 failed, 
archive tooling installable under the non-elevated constraint, and a flashing 
toolchain independent of OS choice, allowing course correction on the OS decision 
after the toolchain succeeded.

The three substantive technical obstacles along the way — a device-discovery
query that structurally excluded its own target, a widely-reported
firmware-flashing tool regression traced to a specific added code path
via binary diffing with no source access, and a USB-mode-switching
transition that cannot survive `usbip` forwarding — were each resolved
through direct empirical evidence (corrected predicates, string-diffed
binaries corroborated against log output, and controlled reference
comparisons) rather than documentation or forum consensus, which in two
of the three cases (the maskrom PID discrepancy and the RKDevTool
regression's exact root cause) remained genuinely unresolved or
undocumented upstream. A pre-flash inspection of the shipped OS image
additionally surfaced a real, otherwise-latent SDK-compatibility
blocker (Python 3.14 vs. RKNN's cp312 wheel ceiling) before it could
manifest as a confusing post-boot failure, and confirmed the actual
inclusion of NPU, GPU, and video-codec drivers directly from the kernel
configuration rather than from contradictory forum reports.

## Appendix: Key Artifacts

- RKDevTool `err=995` community thread (unresolved as of this work):
  Radxa Forum thread #18442
- RKDevTool binary versions compared: v2.86 (working, 1,197,056 bytes),
  v2.96 (broken on raw-image writes, 1,240,576 bytes), v3.37 (Qt
  rewrite, 3,015,168 bytes, not exercised on the flash path)
- Loader used: `rk3588_spl_loader_v1.15.113.bin`
- Selected image:
  `Armbian_26.2.6_Rock-5b-plus_resolute_vendor_6.1.115_kde-plasma_desktop.img`
  (build commit `f26e20b59`)
- Comparison image (SDK-compatibility test only, not flashed):
  `Armbian_26.2.6_Rock-5b-plus_trixie_vendor_6.1.115_minimal.img`
- `rknn-toolkit2` v2.3.2 — `github.com/airockchip/rknn-toolkit2`
- Radxa ROCK 5B docs — `docs.radxa.com/en/rock5/rock5b`
