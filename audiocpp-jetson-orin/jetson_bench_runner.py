#!/usr/bin/env python3
"""Full 40-family audio.cpp benchmark, self-contained for a single Jetson board.
Runs locally on-device (via ssh + nohup), not orchestrated remotely per-command,
to avoid SSH round-trip latency per model. All CLI args below are the CORRECTED
ones already discovered on the laptop's two fixup passes -- no rediscovery needed.
"""
import json, csv, subprocess, time, sys, threading, shutil
from pathlib import Path
from urllib.parse import quote

REPO = Path.home() / "audio-cpp-jetson" / "audio.cpp"
CLI = REPO / "build" / "bin" / "audiocpp_cli"
MODELS_DIR = Path.home() / "bench_models"
OUT_DIR = Path.home() / "bench_out"
RESULTS_CSV = Path.home() / "benchmark_results_jetson.csv"
RUNS_PER_FAMILY = 3

ASSETS = REPO / "assets/resources"
DEMO = REPO / "webui/native/demo_voices"
VOICE_REF = DEMO / "demo_1_man.wav"
REF_TEXT = "okay,I'm Cemo and what you just heard wasn't a human voice."
TEXT_SHORT = "The quick brown fox jumps over the lazy dog near the riverbank."

MODELS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

MANIFEST_INCLUDE = [
    "bs_roformer","chatterbox","citrinet_asr","dots_tts","fish_audio","fun_asr_nano",
    "glm_tts","higgs_audio_stt","higgs_audio_tts","htdemucs","hviske_asr","index_tts2",
    "inflect_v2","irodori_tts","kroko_asr","mel_band_roformer","miocodec","miotts",
    "moss_tts_nano","muscriptor","nemotron_asr","neutts","omnivoice","outetts",
    "parakeet_tdt","pocket_tts","qwen3_asr","qwen3_forced_aligner","qwen3_tts","rvc",
    "seed_vc","sense_asr","sortformer_diar","stable_audio","supertonic","vevo2",
    "vibevoice","vietneu_tts","voxcpm2","voxtral_realtime",
]

# family -> (task, extra_args_builder(model_path, run_idx) -> list[str])
def asr_args(mp, ri, fam):
    return ["--audio", str(ASSETS / "sample_16k.wav"), "--text-out", str(OUT_DIR / f"{fam}_{ri}.txt")]

def tts_args(mp, ri, fam):
    return ["--text", TEXT_SHORT, "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def tts_voiceref_args(mp, ri, fam):
    return ["--text", TEXT_SHORT, "--voice-ref", str(VOICE_REF), "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def tts_voiceref_reftext_args(mp, ri, fam):
    return ["--text", TEXT_SHORT, "--voice-ref", str(VOICE_REF), "--reference-text", REF_TEXT,
            "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def clon_voiceref_args(mp, ri, fam):
    return ["--text", TEXT_SHORT, "--voice-ref", str(VOICE_REF), "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def sep_args(mp, ri, fam):
    return ["--audio", str(Path.home() / "music_test.wav"), "--out-dir", str(OUT_DIR / f"{fam}_{ri}_stems")]

def vc_args(mp, ri, fam):
    if fam == "rvc":
        return ["--audio", str(ASSETS / "a.wav"), "--request-option", "voice_id=default",
                 "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]
    return ["--audio", str(ASSETS / "a.wav"), "--voice-ref", str(ASSETS / "b.wav"),
             "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def align_args(mp, ri, fam):
    return ["--audio", str(ASSETS / "sample_16k.wav"), "--text", TEXT_SHORT, "--language", "English",
             "--words-out", str(OUT_DIR / f"{fam}_{ri}_words.json")]

def diar_args(mp, ri, fam):
    return ["--audio", str(ASSETS / "sample_16k.wav"), "--turns-out", str(OUT_DIR / f"{fam}_{ri}_turns.json")]

def gen_args(mp, ri, fam):
    return ["--text", "ambient synth pad, calm and slow", "--duration-seconds", "8",
             "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def vibevoice_args(mp, ri, fam):
    text = "Speaker 1: The quick brown fox jumps over the lazy dog near the riverbank."
    return ["--text", text, "--out", str(OUT_DIR / f"{fam}_{ri}.wav")]

def midi_args(mp, ri, fam):
    return ["--audio", str(Path.home() / "music_test.wav"), "--out", str(OUT_DIR / f"{fam}_{ri}.mid")]

FAMILY_CONFIG = {
    "citrinet_asr": ("asr", asr_args), "fun_asr_nano": ("asr", asr_args),
    "higgs_audio_stt": ("asr", asr_args), "hviske_asr": ("asr", asr_args),
    "kroko_asr": ("asr", asr_args), "nemotron_asr": ("asr", asr_args),
    "parakeet_tdt": ("asr", asr_args), "qwen3_asr": ("asr", asr_args),
    "sense_asr": ("asr", asr_args), "voxtral_realtime": ("asr", asr_args),

    "dots_tts": ("tts", tts_args), "fish_audio": ("tts", tts_args),
    "higgs_audio_tts": ("tts", tts_args), "inflect_v2": ("tts", tts_args),
    "irodori_tts": ("tts", tts_args), "moss_tts_nano": ("tts", tts_args),
    "neutts": ("tts", tts_args), "omnivoice": ("tts", tts_args),
    "outetts": ("tts", tts_args), "supertonic": ("tts", tts_args),
    "voxcpm2": ("tts", tts_args),

    "chatterbox": ("clon", clon_voiceref_args), "index_tts2": ("clon", clon_voiceref_args),

    "glm_tts": ("tts", tts_voiceref_reftext_args), "qwen3_tts": ("tts", tts_voiceref_reftext_args),
    "pocket_tts": ("tts", tts_voiceref_args), "vevo2": ("tts", tts_voiceref_args),
    "vietneu_tts": ("tts", tts_voiceref_args),

    "vibevoice": ("tts", vibevoice_args),

    "bs_roformer": ("sep", sep_args), "htdemucs": ("sep", sep_args),
    "mel_band_roformer": ("sep", sep_args),

    "miocodec": ("vc", vc_args), "rvc": ("vc", vc_args), "seed_vc": ("vc", vc_args),

    "qwen3_forced_aligner": ("align", align_args),
    "sortformer_diar": ("diar", diar_args),
    "stable_audio": ("gen", gen_args),
    "muscriptor": ("midi", midi_args),
    # miotts handled specially below (needs codec dependency)
}

def resolve_default_package(spec):
    pkgs = spec.get("packages", [])
    default_pkg = next((p for p in pkgs if p.get("default")), pkgs[0] if pkgs else None)
    dl = default_pkg.get("download") or (spec.get("package_defaults") or {}).get("download")
    return default_pkg, dl

def download_family(fam, dest_dir):
    spec = json.load(open(REPO / f"model_specs/{fam}.json"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    pkg, dl = resolve_default_package(spec)
    repo, revision = dl["repo"], dl.get("revision", "main")
    files = pkg["files"]
    local_paths = []
    for fp in files:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(fp)}"
        local = dest_dir / Path(fp).name
        expected_size = None
        try:
            head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, timeout=20)
            for line in head.stdout.splitlines():
                if line.strip().lower().startswith("content-length:"):
                    expected_size = int(line.split(":", 1)[1].strip())
        except Exception:
            pass
        need_dl = True
        if local.exists() and local.stat().st_size > 0:
            if expected_size is None or local.stat().st_size == expected_size:
                need_dl = False
        if need_dl:
            try:
                r = subprocess.run(["curl", "-sL", "-o", str(local), url], timeout=3600)
            except subprocess.TimeoutExpired:
                return None
            if r.returncode != 0 or not local.exists() or local.stat().st_size == 0:
                return None
        local_paths.append(local)
    if len(local_paths) == 1 and local_paths[0].suffix == ".gguf":
        return local_paths[0]
    return dest_dir

def poll_ram_peak(stop_flag):
    peak = 0
    meminfo = Path("/proc/meminfo")
    while not stop_flag.exists():
        try:
            text = meminfo.read_text()
            total = int([l for l in text.splitlines() if l.startswith("MemTotal:")][0].split()[1])
            avail = int([l for l in text.splitlines() if l.startswith("MemAvailable:")][0].split()[1])
            used_mb = (total - avail) // 1024
            peak = max(peak, used_mb)
        except Exception:
            pass
        time.sleep(0.2)
    return peak

def poll_power(stop_flag, curr_node, volt_node):
    samples = []
    while not stop_flag.exists():
        try:
            c = int(curr_node.read_text().strip())
            samples.append(c)
        except Exception:
            pass
        time.sleep(0.1)
    return samples

def find_power_nodes():
    import glob
    curr = glob.glob("/sys/devices/platform/bus@0/*/i2c-*/*-0040/hwmon/hwmon*/curr1_input")
    return (Path(curr[0]) if curr else None,)

def run_once(fam, task, args_fn, model_path, run_idx):
    cmd = [str(CLI), "--task", task, "--family", fam, "--model", str(model_path), "--backend", "cuda"] + args_fn(model_path, run_idx, fam)
    stop_flag = OUT_DIR / f".stop_{fam}_{run_idx}"
    if stop_flag.exists():
        stop_flag.unlink()
    ram_holder = {"v": 0}
    curr_nodes = find_power_nodes()
    power_samples = []
    def ram_poller():
        ram_holder["v"] = poll_ram_peak(stop_flag)
    def power_poller():
        if curr_nodes[0]:
            power_samples.extend(poll_power(stop_flag, curr_nodes[0], None))
    t1 = threading.Thread(target=ram_poller)
    t2 = threading.Thread(target=power_poller)
    t1.start(); t2.start()
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO),
                            env={**__import__("os").environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        exit_code = r.returncode
        stderr_tail = (r.stderr or "")[-500:]
        stdout_tail = (r.stdout or "")[-500:]
    except subprocess.TimeoutExpired:
        exit_code = -1; stderr_tail = "TIMEOUT"; stdout_tail = ""
    wall = time.time() - start
    stop_flag.touch()
    t1.join(timeout=5); t2.join(timeout=5)
    output_ok = exit_code == 0 and "error" not in stderr_tail.lower() and "failed" not in stderr_tail.lower()
    avg_ma = sum(power_samples) / len(power_samples) if power_samples else 0
    peak_ma = max(power_samples) if power_samples else 0
    return {
        "family": fam, "task": task, "run": run_idx, "wall_s": round(wall, 2),
        "peak_ram_mib": ram_holder["v"], "avg_curr_ma": round(avg_ma, 1), "peak_curr_ma": peak_ma,
        "exit_code": exit_code, "output_ok": output_ok,
        "stdout_tail": stdout_tail.replace("\n", " | "), "stderr_tail": stderr_tail.replace("\n", " | "),
    }

def main():
    already_done = set()
    if RESULTS_CSV.exists():
        for row in csv.DictReader(open(RESULTS_CSV)):
            already_done.add((row["family"], row["run"]))

    fieldnames = ["family","task","run","wall_s","peak_ram_mib","avg_curr_ma","peak_curr_ma",
                  "exit_code","output_ok","stdout_tail","stderr_tail"]
    write_header = not RESULTS_CSV.exists()
    csv_f = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader(); csv_f.flush()

    for fam in MANIFEST_INCLUDE:
        if all((fam, str(i)) in already_done for i in range(1, RUNS_PER_FAMILY + 1)):
            print(f"SKIP {fam}: already done", file=sys.stderr)
            continue

        if fam == "miotts":
            dest_dir = MODELS_DIR / "miotts"
            print(f"=== {fam}: downloading (+ codec dependency) ===", file=sys.stderr)
            model_path = download_family("miotts", dest_dir)
            codec_path = download_family("miocodec", MODELS_DIR / "miocodec_dep")
            if model_path is None or codec_path is None:
                writer.writerow({"family": fam, "task": "tts", "run": 0, "wall_s": 0, "peak_ram_mib": 0,
                                  "avg_curr_ma": 0, "peak_curr_ma": 0, "exit_code": -2, "output_ok": False,
                                  "stdout_tail": "", "stderr_tail": "DOWNLOAD_FAILED"})
                csv_f.flush(); continue
            for run_idx in range(1, RUNS_PER_FAMILY + 1):
                if (fam, str(run_idx)) in already_done:
                    continue
                cmd_extra = ["--text", TEXT_SHORT, "--voice-ref", str(VOICE_REF),
                             "--session-option", f"miotts.codec_model_path={codec_path}",
                             "--out", str(OUT_DIR / f"miotts_{run_idx}.wav")]
                cmd = [str(CLI), "--task", "tts", "--family", "miotts", "--model", str(model_path),
                       "--backend", "cuda"] + cmd_extra
                stop_flag = OUT_DIR / f".stop_miotts_{run_idx}"
                if stop_flag.exists(): stop_flag.unlink()
                start = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO))
                wall = time.time() - start
                stop_flag.touch()
                ok = r.returncode == 0 and "error" not in (r.stderr or "").lower() and "failed" not in (r.stderr or "").lower()
                writer.writerow({"family": "miotts", "task": "tts", "run": run_idx, "wall_s": round(wall,2),
                                  "peak_ram_mib": 0, "avg_curr_ma": 0, "peak_curr_ma": 0,
                                  "exit_code": r.returncode, "output_ok": ok,
                                  "stdout_tail": (r.stdout or "")[-500:].replace("\n"," | "),
                                  "stderr_tail": (r.stderr or "")[-500:].replace("\n"," | ")})
                csv_f.flush()
                print(f"  miotts run {run_idx}: wall={wall:.2f}s ok={ok}", file=sys.stderr)
            shutil.rmtree(dest_dir, ignore_errors=True)
            shutil.rmtree(MODELS_DIR / "miocodec_dep", ignore_errors=True)
            continue

        if fam not in FAMILY_CONFIG:
            print(f"SKIP {fam}: no config", file=sys.stderr)
            continue
        task, args_fn = FAMILY_CONFIG[fam]
        dest_dir = MODELS_DIR / fam
        print(f"=== {fam} ({task}): downloading ===", file=sys.stderr)
        model_path = download_family(fam, dest_dir)
        if model_path is None:
            print(f"FAIL {fam}: download failed", file=sys.stderr)
            writer.writerow({"family": fam, "task": task, "run": 0, "wall_s": 0, "peak_ram_mib": 0,
                              "avg_curr_ma": 0, "peak_curr_ma": 0, "exit_code": -2, "output_ok": False,
                              "stdout_tail": "", "stderr_tail": "DOWNLOAD_FAILED"})
            csv_f.flush(); continue

        try:
            for run_idx in range(1, RUNS_PER_FAMILY + 1):
                if (fam, str(run_idx)) in already_done:
                    continue
                print(f"--- {fam} run {run_idx}/{RUNS_PER_FAMILY} ---", file=sys.stderr)
                result = run_once(fam, task, args_fn, model_path, run_idx)
                writer.writerow(result); csv_f.flush()
                print(f"    wall={result['wall_s']}s ram={result['peak_ram_mib']}MiB "
                      f"exit={result['exit_code']} ok={result['output_ok']}", file=sys.stderr)
        except Exception as e:
            print(f"UNEXPECTED ERROR on {fam}: {e}", file=sys.stderr)
            writer.writerow({"family": fam, "task": task, "run": 0, "wall_s": 0, "peak_ram_mib": 0,
                              "avg_curr_ma": 0, "peak_curr_ma": 0, "exit_code": -3, "output_ok": False,
                              "stdout_tail": "", "stderr_tail": f"UNEXPECTED: {e}"})
            csv_f.flush()

        shutil.rmtree(dest_dir, ignore_errors=True)

    csv_f.close()
    print("JETSON BENCHMARK COMPLETE", file=sys.stderr)

if __name__ == "__main__":
    main()
