#!/usr/bin/env python3
"""Phase 2: benchmark driver. Downloads each included family once, runs 3x under
nvidia-smi VRAM polling, appends one CSV row per run immediately (resumable)."""
import json, csv, subprocess, time, os, sys, signal
from pathlib import Path
from urllib.parse import quote

REPO = Path("/home/kino/.claude/jobs/58540cad/tmp/audio-cpp-work/audio.cpp")
CLI = REPO / "build_cuda/bin/audiocpp_cli"
MODELS_DIR = Path("/home/kino/.claude/jobs/58540cad/tmp/bench_models")
RESULTS_CSV = Path("/home/kino/audio-cpp-orin-prep/benchmark_results.csv")
MANIFEST = Path("/home/kino/.claude/jobs/58540cad/tmp/family_manifest.csv")
RUNS_PER_FAMILY = 3

ASSETS = REPO / "assets/resources"
DEMO = REPO / "webui/native/demo_voices"
MUSIC_TEST = Path("/home/kino/.claude/jobs/58540cad/tmp/music_test.wav")
OUT_DIR = Path("/home/kino/.claude/jobs/58540cad/tmp/bench_out")
OUT_DIR.mkdir(exist_ok=True)

TEXT_SHORT = "The quick brown fox jumps over the lazy dog near the riverbank."

TASK_MAP = {
    # asr families: --task asr --audio sample_16k.wav
    "citrinet_asr": "asr", "fun_asr_nano": "asr", "higgs_audio_stt": "asr",
    "hviske_asr": "asr", "kroko_asr": "asr", "nemotron_asr": "asr",
    "parakeet_tdt": "asr", "qwen3_asr": "asr", "sense_asr": "asr", "voxtral_realtime": "asr",
    # plain tts
    "chatterbox": "tts", "dots_tts": "tts", "fish_audio": "tts", "glm_tts": "tts",
    "higgs_audio_tts": "tts", "index_tts2": "tts", "inflect_v2": "tts", "irodori_tts": "tts",
    "miotts": "tts", "moss_tts_nano": "tts", "neutts": "tts", "omnivoice": "tts",
    "outetts": "tts", "pocket_tts": "tts", "qwen3_tts": "tts", "supertonic": "tts",
    "vevo2": "tts", "vibevoice": "tts", "vietneu_tts": "tts", "voxcpm2": "tts",
    # sep
    "bs_roformer": "sep", "htdemucs": "sep", "mel_band_roformer": "sep",
    # vc
    "miocodec": "vc", "rvc": "vc", "seed_vc": "vc",
    # speech_analysis
    "qwen3_forced_aligner": "align", "sortformer_diar": "diar",
    # gen
    "stable_audio": "gen",
}

def resolve_default_package(spec):
    pkgs = spec.get("packages", [])
    default_pkg = next((p for p in pkgs if p.get("default")), pkgs[0] if pkgs else None)
    dl = default_pkg.get("download") or (spec.get("package_defaults") or {}).get("download")
    return default_pkg, dl

def download_family(fam, spec, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    pkg, dl = resolve_default_package(spec)
    repo, revision = dl["repo"], dl.get("revision", "main")
    files = pkg["files"]
    local_paths = []
    for fp in files:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(fp)}"
        local = dest_dir / Path(fp).name
        # Verify against the real remote size (from a HEAD request) before trusting
        # any existing local file as complete -- a partial/interrupted download can
        # exist with nonzero size and must not be silently accepted as done.
        expected_size = None
        try:
            head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, timeout=20)
            for line in head.stdout.splitlines():
                if line.strip().lower().startswith("content-length:"):
                    expected_size = int(line.split(":", 1)[1].strip())
        except Exception:
            pass
        need_download = True
        if local.exists() and local.stat().st_size > 0:
            if expected_size is None or local.stat().st_size == expected_size:
                need_download = False
        if need_download:
            try:
                r = subprocess.run(["curl", "-sL", "-o", str(local), url], timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"  DOWNLOAD TIMEOUT: {url}", file=sys.stderr)
                return None
            except Exception as e:
                print(f"  DOWNLOAD ERROR: {url}: {e}", file=sys.stderr)
                return None
            if r.returncode != 0 or not local.exists() or local.stat().st_size == 0:
                return None
        local_paths.append(local)
    # if single gguf file, model path is that file; else the dir
    if len(local_paths) == 1 and local_paths[0].suffix == ".gguf":
        return local_paths[0]
    return dest_dir

def build_cmd(fam, task, model_path, run_idx):
    base = [str(CLI), "--task", task, "--family", fam, "--model", str(model_path), "--backend", "cuda"]
    out_wav = OUT_DIR / f"{fam}_{run_idx}.wav"
    out_txt = OUT_DIR / f"{fam}_{run_idx}.txt"
    out_dir = OUT_DIR / f"{fam}_{run_idx}_stems"
    if task == "asr":
        base += ["--audio", str(ASSETS / "sample_16k.wav"), "--text-out", str(out_txt)]
    elif task == "tts":
        base += ["--text", TEXT_SHORT, "--out", str(out_wav)]
    elif task == "sep":
        base += ["--audio", str(MUSIC_TEST), "--out-dir", str(out_dir)]
    elif task == "vc":
        if fam == "rvc":
            base += ["--audio", str(ASSETS / "a.wav"), "--request-option", "voice_id=default", "--out", str(out_wav)]
        else:
            base += ["--audio", str(ASSETS / "a.wav"), "--voice-ref", str(ASSETS / "b.wav"), "--out", str(out_wav)]
    elif task == "align":
        base += ["--audio", str(ASSETS / "sample_16k.wav"), "--text", TEXT_SHORT, "--language", "English",
                  "--words-out", str(OUT_DIR / f"{fam}_{run_idx}_words.json")]
    elif task == "diar":
        base += ["--audio", str(ASSETS / "four_speaker_short.wav"), "--turns-out", str(OUT_DIR / f"{fam}_{run_idx}_turns.json")]
    elif task == "gen":
        base += ["--text", "ambient synth pad, calm and slow", "--duration-seconds", "8", "--out", str(out_wav)]
    return base

def poll_vram_peak(stop_event_path):
    peak = 0
    while not stop_event_path.exists():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5)
        try:
            v = int(r.stdout.strip().splitlines()[0])
            peak = max(peak, v)
        except Exception:
            pass
        time.sleep(0.2)
    return peak

def run_once(fam, task, model_path, run_idx):
    cmd = build_cmd(fam, task, model_path, run_idx)
    stop_flag = OUT_DIR / f".stop_{fam}_{run_idx}"
    if stop_flag.exists():
        stop_flag.unlink()
    import threading
    peak_holder = {"v": 0}
    def poller():
        peak_holder["v"] = poll_vram_peak(stop_flag)
    t = threading.Thread(target=poller)
    t.start()
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        exit_code = r.returncode
        stderr_tail = (r.stderr or "")[-500:]
        stdout_tail = (r.stdout or "")[-500:]
    except subprocess.TimeoutExpired:
        exit_code = -1
        stderr_tail = "TIMEOUT"
        stdout_tail = ""
    wall = time.time() - start
    stop_flag.touch()
    t.join(timeout=5)
    output_ok = exit_code == 0 and "error" not in stderr_tail.lower()
    return {
        "family": fam, "task": task, "run": run_idx,
        "wall_s": round(wall, 2), "peak_vram_mib": peak_holder["v"],
        "exit_code": exit_code, "output_ok": output_ok,
        "stdout_tail": stdout_tail.replace("\n", " | "),
        "stderr_tail": stderr_tail.replace("\n", " | "),
    }

def main():
    manifest = list(csv.DictReader(open(MANIFEST)))
    included = [r["family"] for r in manifest if r["include"] == "yes"]

    already_done = set()
    if RESULTS_CSV.exists():
        for row in csv.DictReader(open(RESULTS_CSV)):
            already_done.add((row["family"], row["run"]))

    fieldnames = ["family", "task", "run", "wall_s", "peak_vram_mib", "exit_code",
                  "output_ok", "stdout_tail", "stderr_tail"]
    write_header = not RESULTS_CSV.exists()
    csv_f = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    for fam in included:
        task = TASK_MAP.get(fam)
        if not task:
            print(f"SKIP {fam}: no task mapping", file=sys.stderr)
            continue
        if all((fam, str(i)) in already_done for i in range(1, RUNS_PER_FAMILY + 1)):
            print(f"SKIP {fam}: already fully done", file=sys.stderr)
            continue

        try:
            spec = json.load(open(REPO / f"model_specs/{fam}.json"))
            dest_dir = MODELS_DIR / fam
            print(f"=== {fam} ({task}): downloading ===", file=sys.stderr)
            model_path = download_family(fam, spec, dest_dir)
            if model_path is None:
                print(f"FAIL {fam}: download failed", file=sys.stderr)
                writer.writerow({"family": fam, "task": task, "run": 0, "wall_s": 0,
                                  "peak_vram_mib": 0, "exit_code": -2, "output_ok": False,
                                  "stdout_tail": "", "stderr_tail": "DOWNLOAD_FAILED"})
                csv_f.flush()
                continue

            for run_idx in range(1, RUNS_PER_FAMILY + 1):
                if (fam, str(run_idx)) in already_done:
                    continue
                print(f"--- {fam} run {run_idx}/{RUNS_PER_FAMILY} ---", file=sys.stderr)
                result = run_once(fam, task, model_path, run_idx)
                writer.writerow(result)
                csv_f.flush()
                print(f"    wall={result['wall_s']}s peak_vram={result['peak_vram_mib']}MiB "
                      f"exit={result['exit_code']} ok={result['output_ok']}", file=sys.stderr)
        except Exception as e:
            print(f"UNEXPECTED ERROR on {fam}: {e}", file=sys.stderr)
            writer.writerow({"family": fam, "task": task, "run": 0, "wall_s": 0,
                              "peak_vram_mib": 0, "exit_code": -3, "output_ok": False,
                              "stdout_tail": "", "stderr_tail": f"UNEXPECTED: {e}"})
            csv_f.flush()

        # disk hygiene: remove weights after this family's runs to manage space
        try:
            import shutil
            shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception:
            pass

    csv_f.close()
    print("BENCHMARK COMPLETE", file=sys.stderr)

if __name__ == "__main__":
    main()
