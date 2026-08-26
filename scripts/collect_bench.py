#!/usr/bin/env python3
"""
scripts/collect_bench.py

Единый сборщик данных STREAM — заменяет старый run_bench.py целиком.
Только измерение и сохранение, НИ ОДНОЙ строчки про графики (это теперь
отдельно, scripts/plot_run.py).

Все три режима (standard, cache_sweep, scaling) пишут ОДНУ и ту же
JSON-схему — с явным "x_axis" (что именно меняется по точкам: размер
рабочего набора или число потоков) — чтобы графики строились одним и
тем же кодом, без разделения на "разные виды графиков под разные режимы".

  ./collect_bench.py standard --variant opt_serial --dtype fp64 \
      --cpuset 0 --repeats 5 --ws-mb 200

  ./collect_bench.py cache_sweep --variant novec --dtype fp64 \
      --cpuset 0 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536 --ppo 12

  ./collect_bench.py scaling --variant omp --dtype fp64 \
      --target-cluster x100 --repeats 5

ВАЖНО про CSV: пишется ТОЛЬКО в режиме cache_sweep, в НЕИЗМЕНЁННОЙ
схеме (это точка синхронизации с другой командой — колонки трогать
нельзя). У standard/scaling нет осмысленного соответствия этой схеме
(нет колонки под число потоков), поэтому CSV там не пишется вообще —
только наш JSON.
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

CLUSTER_POOLS = {
    "x100": list(range(0, 8)),
    "a100": list(range(8, 16)),
    "mixed": list(range(0, 16)),
}
DEFAULT_THREAD_LIST = {
    "x100": [1, 2, 4, 8],
    "a100": [1, 2, 4, 8],
    "mixed": [1, 2, 4, 8, 16],
}

THEORETICAL_PEAK_MB_S = 51200.0
MEMORY_FREQUENCY_HZ = None


def cpuset_str_from_cores(cores):
    if len(cores) == 1:
        return str(cores[0])
    return f"{cores[0]}-{cores[-1]}"


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


def save_common(record, all_logs, experiment_type, cluster):
    """Сохраняет JSON (наша единая схема) в results/raw/<experiment_type>/<cluster>/."""
    raw_dir = ROOT / "results" / "raw" / experiment_type / cluster
    log_dir = ROOT / "results" / "logs" / experiment_type / cluster
    for d in (raw_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"{record['run_id']}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (log_dir / f"{record['run_id']}.log").write_text("\n".join(all_logs))
    print(f"\nСохранено (JSON): {json_path.relative_to(ROOT)}")
    return json_path


CSV_COLUMNS = [
    "label", "array_size", "working_set_bytes", "ntimes",
    "copy_min", "copy_med", "copy_max",
    "scale_min", "scale_med", "scale_max",
    "add_min", "add_med", "add_max",
    "triad_min", "triad_med", "triad_max",
    "cpu_freq_mhz", "temp_c",
]


def save_shared_csv(record, csv_path):
    """НЕИЗМЕНЁННАЯ схема — точка синхронизации с другой командой.
    Вызывается только из cache_sweep."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for p in record["points"]:
            ws_bytes = int(p["x"] * 1024)  # x в cache_sweep — это ws_kb
            agg = p["aggregate"]

            def get(fn, key):
                return "" if agg.get(fn) is None else f"{agg[fn][key]:.1f}"

            freq_mhz = ""
            for v in (p.get("cpu_freq_before_hz") or {}).values():
                if v is not None:
                    freq_mhz = f"{v / 1_000_000:.0f}"
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
    print(f"Сохранено (CSV, общая схема): {csv_path.relative_to(ROOT)}")


def base_record(experiment_type, args, cluster, x_axis_kind, x_axis_label):
    build_info = get_build_info(args.variant, args.dtype)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return {
        "run_id": None,  # заполняется вызывающей функцией
        "machine": "aibox_k3", "benchmark": "stream_dynamic",
        "experiment_type": experiment_type,
        "variant": args.variant, "data_type": args.dtype,
        "cluster": cluster, "repeats": args.repeats, "warmup_runs": args.warmup,
        "timestamp": timestamp,
        "compiler": build_info["compiler"], "compiler_flags": build_info["compiler_flags"],
        "memory_frequency_hz": MEMORY_FREQUENCY_HZ,
        "theoretical_peak_mb_s": THEORETICAL_PEAK_MB_S,
        "x_axis": {"kind": x_axis_kind, "label": x_axis_label},
        "points": [],
    }


# ---------- standard: одна точка (вырожденный случай общего графика) ----------

def cmd_standard(args):
    cores = parse_cpuset(args.cpuset)
    cluster = cluster_label(cores)
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]
    ws_bytes = args.ws_mb * 1024 * 1024
    array_elements = ws_bytes // bpe

    print(f"=== standard: {args.variant}/{args.dtype} cores={args.cpuset} ({cluster}) "
          f"threads={args.threads} WS={args.ws_mb}MB/array ===")

    result = measure_point(binary, args.cpuset, args.threads, cores,
                            array_elements, args.ntimes, args.repeats, args.warmup)
    for fn in FUNCTIONS:
        agg = result["aggregate"][fn]
        print(f"  {fn:8s} median={agg['median_mb_s']:.1f} MB/s min={agg['min_mb_s']:.1f} max={agg['max_mb_s']:.1f}")

    record = base_record("standard", args, cluster, "ws_kb", "Working set size (KB)")
    record["run_id"] = f"standard_{args.variant}_{args.dtype}_{args.threads}t_{args.ws_mb}MB_cores{args.cpuset}_{record['timestamp']}"
    record["points"] = [{
        "x": args.ws_mb * 1024, "threads": args.threads, "cpuset": args.cpuset,
        "array_elements": array_elements, "ntimes": args.ntimes,
        "aggregate": result["aggregate"], "validates_all": result["validates_all"],
        "cpu_freq_before_hz": result["cpu_freq_before_hz"], "temp_before_c": result["temp_before_c"],
    }]

    save_common(record, result["logs"], "standard", cluster)
    if not result["validates_all"]:
        print("ВНИМАНИЕ: валидация провалена в одном из повторов!", file=sys.stderr)
        sys.exit(2)


# ---------- cache_sweep: много точек по WS + CSV (неизменная схема) ----------

def cmd_cache_sweep(args):
    cores = parse_cpuset(args.cpuset)
    cluster = cluster_label(cores)
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]

    n = int(math.log2(args.ws_max_kb / args.ws_min_kb) * args.ppo) + 1
    ws_list_kb = sorted(set(int(round(args.ws_min_kb * (2 ** (i / args.ppo)))) for i in range(n + 1)))
    ws_list_kb = [kb for kb in ws_list_kb if args.ws_min_kb <= kb <= args.ws_max_kb]

    print(f"=== cache_sweep: {args.variant}/{args.dtype} cores={args.cpuset} ({cluster}) "
          f"— {len(ws_list_kb)} точек, {args.ws_min_kb}KB..{args.ws_max_kb}KB ===")

    record = base_record("cache_sweep", args, cluster, "ws_kb", "Working set size (KB)")
    record["run_id"] = f"cache_sweep_{args.variant}_{args.dtype}_{args.threads}t_cores{args.cpuset}_{record['timestamp']}"

    all_logs = []
    for idx, kb in enumerate(ws_list_kb, 1):
        ws_bytes = kb * 1024
        array_elements = max(1, ws_bytes // (3 * bpe))
        ntimes_point = int(args.ntimes * args.ntimes_ref_kb / kb)
        ntimes_point = max(args.ntimes_min, min(ntimes_point, args.ntimes_max))

        result = measure_point(binary, args.cpuset, args.threads, cores,
                                array_elements, ntimes_point, args.repeats, args.warmup)
        all_logs.extend(result["logs"])

        triad = result["aggregate"]["Triad"]["median_mb_s"] if result["aggregate"]["Triad"] else None
        print(f"  [{idx}/{len(ws_list_kb)}] WS={kb:>8d}KB ntimes={ntimes_point:>4d} Triad={triad} MB/s")

        record["points"].append({
            "x": kb, "threads": args.threads, "cpuset": args.cpuset,
            "array_elements": array_elements, "ntimes": ntimes_point,
            "aggregate": result["aggregate"], "validates_all": result["validates_all"],
            "cpu_freq_before_hz": result["cpu_freq_before_hz"], "temp_before_c": result["temp_before_c"],
        })

    json_path = save_common(record, all_logs, "cache_sweep", cluster)
    csv_path = json_path.with_suffix(".csv")
    save_shared_csv(record, csv_path)


# ---------- scaling: точки по числу потоков ----------

def cmd_scaling(args):
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]
    pool = CLUSTER_POOLS[args.target_cluster]
    thread_list = ([int(x) for x in args.threads_list.split(",")]
                    if args.threads_list else DEFAULT_THREAD_LIST[args.target_cluster])
    ws_bytes = args.ws_mb * 1024 * 1024
    array_elements = ws_bytes // bpe

    print(f"=== scaling: {args.variant}/{args.dtype} target_cluster={args.target_cluster} "
          f"threads={thread_list} WS={args.ws_mb}MB/array ===")

    record = base_record("scaling", args, args.target_cluster, "threads", "Число потоков")
    record["run_id"] = f"scaling_{args.variant}_{args.dtype}_{args.target_cluster}_{args.ws_mb}MB_{record['timestamp']}"

    all_logs = []
    for t in thread_list:
        if t > len(pool):
            print(f"  [threads={t}] ПРОПУЩЕНО — в пуле только {len(pool)} ядер")
            continue
        cores = pool[:t]
        cpuset = cpuset_str_from_cores(cores)

        result = measure_point(binary, cpuset, t, cores, array_elements,
                                args.ntimes, args.repeats, args.warmup)
        all_logs.extend(result["logs"])

        triad = result["aggregate"]["Triad"]["median_mb_s"] if result["aggregate"]["Triad"] else None
        print(f"  [threads={t:>2d}] cpuset={cpuset:<8s} Triad={triad} MB/s")

        record["points"].append({
            "x": t, "threads": t, "cpuset": cpuset,
            "array_elements": array_elements, "ntimes": args.ntimes,
            "aggregate": result["aggregate"], "validates_all": result["validates_all"],
            "cpu_freq_before_hz": result["cpu_freq_before_hz"], "temp_before_c": result["temp_before_c"],
        })

    save_common(record, all_logs, "scaling", args.target_cluster)


def main():
    ap = argparse.ArgumentParser(description="Единый сборщик STREAM (без графиков)")
    sub = ap.add_subparsers(dest="mode", required=True)

    common_flags = [
        ("--variant", dict(required=True, choices=["novec", "opt_serial", "omp"])),
        ("--dtype", dict(required=True, choices=["fp64", "fp32", "fp16"])),
        ("--repeats", dict(type=int, default=5)),
        ("--warmup", dict(type=int, default=1)),
        ("--ntimes", dict(type=int, default=20)),
    ]

    p_std = sub.add_parser("standard")
    for flag, kw in common_flags:
        p_std.add_argument(flag, **kw)
    p_std.add_argument("--cpuset", required=True)
    p_std.add_argument("--threads", type=int, default=1)
    p_std.add_argument("--ws-mb", type=int, default=200)
    p_std.set_defaults(func=cmd_standard)

    p_sweep = sub.add_parser("cache_sweep")
    for flag, kw in common_flags:
        p_sweep.add_argument(flag, **kw)
    p_sweep.add_argument("--cpuset", required=True)
    p_sweep.add_argument("--threads", type=int, default=1)
    p_sweep.add_argument("--ws-min-kb", type=int, default=8)
    p_sweep.add_argument("--ws-max-kb", type=int, default=2 * 1024 * 1024)
    p_sweep.add_argument("--ppo", type=int, default=6)
    p_sweep.add_argument("--ntimes-ref-kb", type=int, default=256)
    p_sweep.add_argument("--ntimes-min", type=int, default=5)
    p_sweep.add_argument("--ntimes-max", type=int, default=300)
    p_sweep.set_defaults(func=cmd_cache_sweep)

    p_scale = sub.add_parser("scaling")
    for flag, kw in common_flags:
        p_scale.add_argument(flag, **kw)
    p_scale.add_argument("--target-cluster", required=True, choices=["x100", "a100", "mixed"])
    p_scale.add_argument("--threads-list", default=None)
    p_scale.add_argument("--ws-mb", type=int, default=200)
    p_scale.set_defaults(func=cmd_scaling)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
