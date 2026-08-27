#!/usr/bin/env python3
"""
scripts/collect_bench.py

Сборщик данных STREAM — ТОЛЬКО режим cache_sweep (standard и scaling
убраны по запросу; если понадобятся снова, брать из истории коммитов).
Только измерение и сохранение, ни строчки про графики.

  ./collect_bench.py --variant novec --dtype fp64 \
      --cpuset 0 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536 --ppo 12

СТРУКТУРА ВЫВОДА — три подпапки внутри results/raw/cache_sweep/<кластер>/:

  results/raw/cache_sweep/<x100|a100|mixed>/csv/<run_id>.csv    — общая
      схема, НЕИЗМЕНЁННАЯ (точка синхронизации с другой командой)
  results/raw/cache_sweep/<x100|a100|mixed>/json/<run_id>.json  — наш
      полный формат (compiler_flags, efficiency_pct, freq/temp по точкам)
  results/raw/cache_sweep/<x100|a100|mixed>/logs/<run_id>.log   — сырой
      вывод STREAM за все повторы всех точек
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

X100_CORES = set(range(0, 8))
A100_CORES = set(range(8, 16))
FUNCTIONS = ["Copy", "Scale", "Add", "Triad"]
DTYPE_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2}

THEORETICAL_PEAK_MB_S = 51200.0
MEMORY_FREQUENCY_HZ = None

# Константы из bash-скрипта для расчета NTIMES
TARGET_DATA_PER_RUN = 1024 * 1024 * 1024  # 1 GB
NTIMES_MIN = 5
NTIMES_MAX = 50000


def parse_cpuset(cpuset_str):
    cores = []
    for part in cpuset_str.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return sorted(set(cores))


def cluster_label(cores):
    core_set = set(cores)
    if core_set <= X100_CORES:
        return "x100"
    if core_set <= A100_CORES:
        return "a100"
    return "mixed"


def read_cpu_freqs(cores):
    freqs = {}
    for c in cores:
        p = Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
        try:
            freqs[str(c)] = int(p.read_text().strip())
        except Exception:
            freqs[str(c)] = None
    return freqs


def read_temps():
    temps = {}
    base = Path("/sys/class/thermal")
    if not base.exists():
        return temps
    for zone in sorted(base.glob("thermal_zone*")):
        try:
            name = (zone / "type").read_text().strip()
            val = int((zone / "temp").read_text().strip())
            temps[name] = val / 1000.0
        except Exception:
            continue
    return temps


def parse_stream_output(text):
    result = {"rates_mb_s": {}, "validates": None}
    for fn in FUNCTIONS:
        m = re.search(rf"{fn}:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
        result["rates_mb_s"][fn] = float(m.group(1)) if m else None
    if "Solution Validates" in text:
        result["validates"] = True
    elif "Failed Validation" in text:
        result["validates"] = False
    return result


def run_once(binary, cpuset_str, threads, array_elements, ntimes):
    cmd = ["taskset", "-c", cpuset_str, str(binary)]
    env = os.environ.copy()
    env["STREAM_ARRAY_SIZE"] = str(array_elements)
    env["STREAM_NTIMES"] = str(ntimes)
    if threads > 1:
        env["OMP_NUM_THREADS"] = str(threads)
        # Привязка OpenMP к ядрам (как в bash-скрипте)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"
        # Дополнительно для лучшей производительности:
        env["OMP_SCHEDULE"] = "static"  # статическое распределение итераций
    else:
        env["OMP_NUM_THREADS"] = "1"
        env["OMP_PROG_BIND"] = "false"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.stdout, proc.stderr


def get_binary(variant, dtype):
    b = ROOT / "builds_dynamic" / variant / dtype / f"stream_{variant}_{dtype}_dynamic"
    if not b.exists():
        print(f"ОШИБКА: бинарник не найден: {b}", file=sys.stderr)
        print("Сначала выполните scripts/build_dynamic.sh", file=sys.stderr)
        sys.exit(1)
    return b


def get_build_info(variant, dtype):
    info_path = ROOT / "builds_dynamic" / variant / dtype / "build_info.json"
    try:
        return json.loads(info_path.read_text())
    except Exception:
        return {"compiler": "unknown", "compiler_flags": "unknown"}


def format_label(ws_bytes):
    if ws_bytes >= 1024 ** 3:
        return f"{ws_bytes / 1024**3:.2f}G"
    if ws_bytes >= 1024 ** 2:
        return f"{ws_bytes / 1024**2:.2f}M"
    return f"{ws_bytes / 1024:.0f}K"


def calculate_ntimes(working_set_bytes):
    """
    Расчет NTIMES по той же логике, что и в bash-скрипте:
    ntimes = int(target_bytes / working_set_bytes)
    с ограничениями [5, 50000]
    """
    ntimes = int(TARGET_DATA_PER_RUN / working_set_bytes)
    if ntimes < NTIMES_MIN:
        ntimes = NTIMES_MIN
    if ntimes > NTIMES_MAX:
        ntimes = NTIMES_MAX
    return ntimes


def generate_ws_list(ws_min_kb, ws_max_kb, ppo):
    """
    Генерация точек измерений по той же логике, что и в bash-скрипте:
    n = int(log2(ws_max / ws_min) * ppo) + 1
    kb = int(ws_min * (2 ^ (i / ppo)) + 0.5)  # с математическим округлением
    """
    if ws_min_kb <= 0 or ws_max_kb <= ws_min_kb or ppo <= 0:
        return []
    
    # Вычисляем количество точек
    n = int(math.log2(ws_max_kb / ws_min_kb) * ppo) + 1
    
    ws_list = []
    seen = set()
    
    for i in range(n + 1):
        # Вычисляем размер в КБ с математическим округлением (как в Python)
        kb = int(ws_min_kb * (2 ** (i / ppo)) + 0.5)
        
        # Проверяем границы
        if kb > ws_max_kb:
            continue
        if kb < ws_min_kb:
            continue
        
        # Защита от дубликатов
        if kb not in seen:
            seen.add(kb)
            ws_list.append(kb)
    
    return sorted(ws_list)


def measure_point(binary, cpuset, threads, cores, array_elements, ntimes, repeats, warmup):
    for _ in range(warmup):
        run_once(binary, cpuset, threads, array_elements, ntimes)

    freq_before = read_cpu_freqs(cores)
    temp_before = read_temps()

    runs_rates = {fn: [] for fn in FUNCTIONS}
    validates_all = True
    logs = []
    for i in range(repeats):
        stdout, stderr = run_once(binary, cpuset, threads, array_elements, ntimes)
        parsed = parse_stream_output(stdout)
        if parsed["validates"] is False:
            validates_all = False
        for fn in FUNCTIONS:
            if parsed["rates_mb_s"].get(fn) is not None:
                runs_rates[fn].append(parsed["rates_mb_s"][fn])
        logs.append(f"--- run {i + 1} ---\n{stdout}\n{stderr}\n")

    aggregate = {}
    for fn in FUNCTIONS:
        vals = runs_rates[fn]
        if vals:
            median_mb_s = statistics.median(vals)
            aggregate[fn] = {
                "median_mb_s": median_mb_s, "min_mb_s": min(vals), "max_mb_s": max(vals),
                "stdev_mb_s": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "efficiency_pct": median_mb_s / THEORETICAL_PEAK_MB_S * 100,
            }
        else:
            aggregate[fn] = None

    return {
        "aggregate": aggregate, "validates_all": validates_all,
        "cpu_freq_before_hz": freq_before, "temp_before_c": temp_before, "logs": logs,
    }


CSV_COLUMNS = [
    "label", "array_size", "working_set_bytes", "ntimes",
    "copy_min", "copy_med", "copy_max",
    "scale_min", "scale_med", "scale_max",
    "add_min", "add_med", "add_max",
    "triad_min", "triad_med", "triad_max",
    "cpu_freq_mhz", "temp_c",
]


def save_all(record, all_logs, cluster):
    """Раскладывает вывод по трём подпапкам внутри results/raw/cache_sweep/<cluster>/:
    csv/, json/, logs/ — вместо трёх параллельных деревьев results/raw и
    results/logs, как было раньше."""
    base = ROOT / "results" / "raw" / "cache_sweep" / cluster
    json_dir = base / "json"
    csv_dir = base / "csv"
    log_dir = base / "logs"
    for d in (json_dir, csv_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_id = record["run_id"]

    # --- json ---
    json_path = json_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # --- logs ---
    log_path = log_dir / f"{run_id}.log"
    log_path.write_text("\n".join(all_logs))

    # --- csv: НЕИЗМЕНЁННАЯ схема, точка синхронизации с другой командой ---
    csv_path = csv_dir / f"{run_id}.csv"
    bpe = DTYPE_BYTES[record["data_type"]]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in record["points"]:
            ws_bytes = int(p["x"] * 1024)  # x — это ws_kb
            agg = p["aggregate"]

            def get(fn, key):
                return "" if agg.get(fn) is None else f"{agg[fn][key]:.1f}"

            freq_mhz = ""
            for v in (p.get("cpu_freq_before_hz") or {}).values():
                if v is not None:
                    freq_mhz = f"{v / 1000:.0f}"  # scaling_cur_freq в кГц, не Гц
                    break
            temp_c = ""
            for v in (p.get("temp_before_c") or {}).values():
                if v is not None:
                    temp_c = f"{v:.1f}"
                    break

            writer.writerow({
                "label": format_label(ws_bytes),
                "array_size": p["array_elements"],
                "working_set_bytes": ws_bytes,
                "ntimes": p["ntimes"],
                "copy_min": get("Copy", "min_mb_s"), "copy_med": get("Copy", "median_mb_s"), "copy_max": get("Copy", "max_mb_s"),
                "scale_min": get("Scale", "min_mb_s"), "scale_med": get("Scale", "median_mb_s"), "scale_max": get("Scale", "max_mb_s"),
                "add_min": get("Add", "min_mb_s"), "add_med": get("Add", "median_mb_s"), "add_max": get("Add", "max_mb_s"),
                "triad_min": get("Triad", "min_mb_s"), "triad_med": get("Triad", "median_mb_s"), "triad_max": get("Triad", "max_mb_s"),
                "cpu_freq_mhz": freq_mhz, "temp_c": temp_c,
            })

    print(f"\nСохранено:")
    print(f"  JSON: {json_path.relative_to(ROOT)}")
    print(f"  CSV:  {csv_path.relative_to(ROOT)}")
    print(f"  LOG:  {log_path.relative_to(ROOT)}")


def cmd_cache_sweep(args):
    cores = parse_cpuset(args.cpuset)
    cluster = cluster_label(cores)
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]

    # Генерируем точки измерений по той же логике, что и в bash-скрипте
    ws_list_kb = generate_ws_list(args.ws_min_kb, args.ws_max_kb, args.ppo)

    print(f"=== cache_sweep: {args.variant}/{args.dtype} cores={args.cpuset} ({cluster}) "
          f"— {len(ws_list_kb)} точек, {args.ws_min_kb}KB..{args.ws_max_kb}KB ===")

    build_info = get_build_info(args.variant, args.dtype)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    record = {
        "run_id": f"cache_sweep_{args.variant}_{args.dtype}_{args.threads}t_cores{args.cpuset}_{timestamp}",
        "machine": "aibox_k3", "benchmark": "stream_dynamic", "experiment_type": "cache_sweep",
        "variant": args.variant, "data_type": args.dtype, "cluster": cluster,
        "repeats": args.repeats, "warmup_runs": args.warmup, "timestamp": timestamp,
        "compiler": build_info["compiler"], "compiler_flags": build_info["compiler_flags"],
        "memory_frequency_hz": MEMORY_FREQUENCY_HZ, "theoretical_peak_mb_s": THEORETICAL_PEAK_MB_S,
        "x_axis": {"kind": "ws_kb", "label": "Working set size (KB)"},
        "points": [],
    }

    all_logs = []
    for idx, kb in enumerate(ws_list_kb, 1):
        ws_bytes = kb * 1024
        array_elements = max(1, ws_bytes // (3 * bpe))
        
        # Расчет NTIMES по той же логике, что и в bash-скрипте
        ntimes_point = calculate_ntimes(ws_bytes)

        result = measure_point(binary, args.cpuset, args.threads, cores,
                                array_elements, ntimes_point, args.repeats, args.warmup)
        all_logs.extend(result["logs"])

        triad = result["aggregate"]["Triad"]["median_mb_s"] if result["aggregate"]["Triad"] else None
        print(f"  [{idx}/{len(ws_list_kb)}] WS={kb:>8d}KB ntimes={ntimes_point:>5d} Triad={triad} MB/s")

        record["points"].append({
            "x": kb, "threads": args.threads, "cpuset": args.cpuset,
            "array_elements": array_elements, "ntimes": ntimes_point,
            "aggregate": result["aggregate"], "validates_all": result["validates_all"],
            "cpu_freq_before_hz": result["cpu_freq_before_hz"], "temp_before_c": result["temp_before_c"],
        })

    save_all(record, all_logs, cluster)


def main():
    ap = argparse.ArgumentParser(description="Сборщик STREAM cache_sweep (без графиков)")
    ap.add_argument("--variant", required=True, choices=["novec", "opt_serial", "omp"])
    ap.add_argument("--dtype", required=True, choices=["fp64", "fp32", "fp16"])
    ap.add_argument("--cpuset", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ws-min-kb", type=int, default=64)
    ap.add_argument("--ws-max-kb", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--ppo", type=int, default=6)
    args = ap.parse_args()
    cmd_cache_sweep(args)


if __name__ == "__main__":
    main()