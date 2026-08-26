#!/usr/bin/env python3
"""
scripts/run_bench.py

Единый инструмент запуска STREAM (dynamic-версия, builds_dynamic/).
Один бинарник на variant x dtype обслуживает любой размер рабочего
набора — размер передаётся в рантайме через STREAM_ARRAY_SIZE, без
пересборки.

Режим "standard" — одна точка, WS заведомо больше всех кэшей,
сравнение с теоретическим пиком памяти. Один запуск = один bar-график.

  ./run_bench.py standard --variant opt_serial --dtype fp64 \
      --cpuset 0 --repeats 5 --ws-mb 200

Режим "cache_sweep" — плотная сетка WS от --ws-min-kb до --ws-max-kb,
один линейный график Bandwidth vs Working set size с границами
L1D/L2 текущего кластера.

  ./run_bench.py cache_sweep --variant novec --dtype fp64 \
      --cpuset 0 --repeats 3 --ws-min-kb 256 --ws-max-kb 3145728
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent

X100_CORES = set(range(0, 8))
A100_CORES = set(range(8, 16))
FUNCTIONS = ["Copy", "Scale", "Add", "Triad"]
COLORS = {"Copy": "#4C72B0", "Scale": "#DD8452", "Add": "#55A868", "Triad": "#C44E52"}
DTYPE_BYTES = {"fp64": 8, "fp32": 4, "fp16": 2}

CACHE_BOUNDARIES_KB = {
    "x100": {"L1D": 64, "L2": 4096},
    "a100": {"L1D": 64, "L2": 1024},
}

# Пулы ядер для scaling-эксперимента. Для thread_count=t берутся ПЕРВЫЕ t
# ядер пула (contiguous), а не равномерно распределяются между кластерами —
# то есть для "mixed" при t<=8 фактически задействуется только X100-часть.
# Это стандартная практика (растим число потоков от малого к полному), но
# явно фиксируем допущение, чтобы не быть неочевидным при интерпретации.
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

# BW = ширина шины (бит) x скорость (MT/s) / 8. На AIBOX-K3: 64 бит, 2 канала,
# 6400 MT/s -> 51.2 GB/s = 51200 MB/s. См. docs/note_fp16_anomaly.md
# (исправление паспорта: 2 канала, не 4 — само число BW не меняется).
THEORETICAL_PEAK_MB_S = 51200.0

# Частота памяти (DRAM/DMC) недоступна без root на этой сборке ОС: единственный
# /sys/class/devfreq узел — cac00000.imggpu (GPU), не контроллер памяти;
# dmesg заблокирован правами обычного пользователя. Честно фиксируем как
# unknown, как и требует задание, вместо того чтобы подставлять предположение.
MEMORY_FREQUENCY_HZ = None


def cpuset_str_from_cores(cores):
    """[0,1,2,3] -> '0-3'; [0] -> '0' (список предполагается contiguous)."""
    if len(cores) == 1:
        return str(cores[0])
    return f"{cores[0]}-{cores[-1]}"


# ---------- общие утилиты ----------

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
    result = {"rates_mb_s": {}, "validates": None, "array_size_elements": None}
    m = re.search(r"Array size = (\d+)", text)
    if m:
        result["array_size_elements"] = int(m.group(1))
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
    """Читает build_info.json рядом с бинарником (compiler, compiler_flags).
    Если файла нет (бинарник собран до появления этой фичи) — возвращает
    unknown вместо падения, как того требует задание (не выдумывать данные)."""
    info_path = ROOT / "builds_dynamic" / variant / dtype / "build_info.json"
    try:
        return json.loads(info_path.read_text())
    except Exception:
        return {"compiler": "unknown", "compiler_flags": "unknown"}


def format_mib(mib: float) -> str:
    if mib < 1:
        return f"{mib * 1024:.3g} KB"
    if mib < 1024:
        return f"{mib:.3g} MB"
    return f"{mib / 1024:.3g} GB"


def measure_point(binary, cpuset, threads, cores, array_elements, ntimes, repeats, warmup):
    """Прогоняет одну точку (repeats раз + warmup), возвращает aggregate + метаданные."""
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

    freq_after = read_cpu_freqs(cores)

    aggregate = {}
    for fn in FUNCTIONS:
        vals = runs_rates[fn]
        if vals:
            median_mb_s = statistics.median(vals)
            aggregate[fn] = {
                "median_mb_s": median_mb_s,
                "min_mb_s": min(vals),
                "max_mb_s": max(vals),
                "stdev_mb_s": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
                "theoretical_peak_mb_s": THEORETICAL_PEAK_MB_S,
                "efficiency_pct": median_mb_s / THEORETICAL_PEAK_MB_S * 100,
            }
        else:
            aggregate[fn] = None

    return {
        "aggregate": aggregate,
        "validates_all": validates_all,
        "cpu_freq_before_hz": freq_before,
        "cpu_freq_after_hz": freq_after,
        "temp_before_c": temp_before,
        "logs": logs,
    }


# ---------- режим: standard ----------

def cmd_standard(args):
    cores = parse_cpuset(args.cpuset)
    cluster = cluster_label(cores)
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]

    ws_bytes = args.ws_mb * 1024 * 1024
    array_elements = ws_bytes // bpe

    print(f"=== standard: {args.variant}/{args.dtype} cores={args.cpuset} "
          f"({cluster}) threads={args.threads} WS={args.ws_mb}MB/array ===")

    result = measure_point(binary, args.cpuset, args.threads, cores,
                            array_elements, args.ntimes, args.repeats, args.warmup)
    build_info = get_build_info(args.variant, args.dtype)

    for fn in FUNCTIONS:
        agg = result["aggregate"][fn]
        print(f"  {fn:8s} median={agg['median_mb_s']:.1f} MB/s "
              f"min={agg['min_mb_s']:.1f} max={agg['max_mb_s']:.1f}")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = (f"{args.variant}_{args.dtype}_{args.threads}t_{args.ws_mb}MB_"
              f"cores{args.cpuset.replace(',', '_')}_{timestamp}")

    raw_dir = ROOT / "results" / "raw" / "standard" / cluster
    log_dir = ROOT / "results" / "logs" / "standard" / cluster
    plot_dir = ROOT / "results" / "processed" / "standard" / cluster
    for d in (raw_dir, log_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id, "machine": "aibox_k3", "benchmark": "stream_dynamic",
        "experiment_type": "standard", "variant": args.variant, "data_type": args.dtype,
        "cpu_set": args.cpuset, "cluster": cluster, "threads": args.threads,
        "repeats": args.repeats, "warmup_runs": args.warmup,
        "array_size_elements": array_elements, "working_set_mb_per_array": args.ws_mb,
        "timestamp": timestamp, "any_failed_validation": not result["validates_all"],
        "compiler": build_info["compiler"], "compiler_flags": build_info["compiler_flags"],
        "memory_frequency_hz": MEMORY_FREQUENCY_HZ,
        "aggregate": result["aggregate"],
        "cpu_freq_before_hz": result["cpu_freq_before_hz"],
        "cpu_freq_after_hz": result["cpu_freq_after_hz"],
        "temp_before_c": result["temp_before_c"],
    }

    (raw_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (log_dir / f"{run_id}.log").write_text("\n".join(result["logs"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [fn for fn in FUNCTIONS if result["aggregate"][fn]]
    medians = [result["aggregate"][fn]["median_mb_s"] for fn in names]
    lo = [result["aggregate"][fn]["median_mb_s"] - result["aggregate"][fn]["min_mb_s"] for fn in names]
    hi = [result["aggregate"][fn]["max_mb_s"] - result["aggregate"][fn]["median_mb_s"] for fn in names]
    ax.bar(names, medians, color=[COLORS[f] for f in names], yerr=[lo, hi], capsize=5)
    ax.set_ylabel("Bandwidth, MB/s (median, усы = min..max)")

    # --- теоретический предел (добавлено) ---
    ax.axhline(THEORETICAL_PEAK_MB_S, color="black", linestyle="--", alpha=0.6,
               label=f"Теор. пик памяти ({THEORETICAL_PEAK_MB_S / 1000:.1f} GB/s)")

    ax.set_title(f"{args.variant} / {args.dtype} / {args.threads}t / cores {args.cpuset} "
                 f"({cluster})\nWS={args.ws_mb}MB per array, n={args.repeats}")
    ax.set_ylim(0, args.y_max_mb_s)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = plot_dir / f"{run_id}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nСохранено: {plot_path.relative_to(ROOT)}")
    if not result["validates_all"]:
        print("ВНИМАНИЕ: валидация провалена в одном из повторов!", file=sys.stderr)
        sys.exit(2)


# ---------- режим: cache_sweep ----------

def cmd_cache_sweep(args):
    cores = parse_cpuset(args.cpuset)
    cluster = cluster_label(cores)
    binary = get_binary(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]

    build_info = get_build_info(args.variant, args.dtype)

    n = int(math.log2(args.ws_max_kb / args.ws_min_kb) * args.ppo) + 1
    ws_list_kb = sorted(set(
        int(round(args.ws_min_kb * (2 ** (i / args.ppo)))) for i in range(n + 1)
    ))
    ws_list_kb = [kb for kb in ws_list_kb if args.ws_min_kb <= kb <= args.ws_max_kb]

    print(f"=== cache_sweep: {args.variant}/{args.dtype} cores={args.cpuset} "
          f"({cluster}) — {len(ws_list_kb)} точек, {args.ws_min_kb}KB..{args.ws_max_kb}KB ===")

    points = []
    all_logs = []
    for idx, kb in enumerate(ws_list_kb, 1):
        ws_bytes = kb * 1024
        array_elements = max(1, ws_bytes // (3 * bpe))

        # Адаптивный NTIMES: на маленьких WS один проход занимает микросекунды,
        # таймер даёт шумные показания — компенсируем большим числом повторов.
        # На больших WS, наоборот, снижаем NTIMES, иначе замер займёт часы.
        # База — args.ntimes при ws_ref_kb, масштабируем обратно пропорционально размеру.
        ntimes_point = int(args.ntimes * args.ntimes_ref_kb / kb)
        ntimes_point = max(args.ntimes_min, min(ntimes_point, args.ntimes_max))

        result = measure_point(binary, args.cpuset, args.threads, cores,
                                array_elements, ntimes_point, args.repeats, args.warmup)
        all_logs.extend(result["logs"])

        triad = result["aggregate"]["Triad"]["median_mb_s"] if result["aggregate"]["Triad"] else None
        print(f"  [{idx}/{len(ws_list_kb)}] WS={kb:>8d}KB  ntimes={ntimes_point:>4d}  "
              f"Triad={triad} MB/s  validates_all={result['validates_all']}")

        points.append({
            "ws_kb": kb, "ntimes": ntimes_point, "aggregate": result["aggregate"],
            "validates_all": result["validates_all"],
            "cpu_freq_before_hz": result["cpu_freq_before_hz"],
        })

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"cache_sweep_{args.variant}_{args.dtype}_{args.threads}t_cores{args.cpuset}_{timestamp}"

    raw_dir = ROOT / "results" / "raw" / "cache_sweep" / cluster
    log_dir = ROOT / "results" / "logs" / "cache_sweep" / cluster
    plot_dir = ROOT / "results" / "processed" / "cache_sweep" / cluster
    for d in (raw_dir, log_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id, "machine": "aibox_k3", "benchmark": "stream_dynamic",
        "experiment_type": "cache_sweep", "variant": args.variant, "data_type": args.dtype,
        "cpu_set": args.cpuset, "cluster": cluster, "threads": args.threads,
        "repeats": args.repeats, "warmup_runs": args.warmup, "timestamp": timestamp,
        "compiler": build_info["compiler"], "compiler_flags": build_info["compiler_flags"],
        "memory_frequency_hz": MEMORY_FREQUENCY_HZ,
        "points": points,
    }
    (raw_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (log_dir / f"{run_id}.log").write_text("\n".join(all_logs))

    x_mib = [p["ws_kb"] / 1024 for p in points]
    fig, ax = plt.subplots(figsize=(13, 6))
    for fn in FUNCTIONS:
        y = [p["aggregate"][fn]["median_mb_s"] if p["aggregate"][fn] else None for p in points]
        ax.plot(x_mib, y, marker="o", markersize=3, linewidth=1, label=fn, color=COLORS[fn])

    ax.set_xscale("log", base=2)
    ax.margins(x=0)
    ax.set_xlim(x_mib[0], x_mib[-1])

    tick_positions = [x_mib[0]]
    v = x_mib[0]
    while v < x_mib[-1]:
        v *= 2
        tick_positions.append(min(v, x_mib[-1]))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_mib(v) for v in tick_positions], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("Working set size")
    ax.set_ylabel("Bandwidth (MB/s)")
    ax.set_title(f"STREAM {args.variant}/{args.dtype} (dynamic) — Cache hierarchy "
                 f"(AIBOX-K3, {cluster}, cores {args.cpuset}, {len(points)} точек)")
    ax.set_ylim(0, args.y_max_mb_s)
    ax.grid(True, which="both", alpha=0.3)

    # --- теоретический предел (добавлено) ---
    ax.axhline(THEORETICAL_PEAK_MB_S, color="black", linestyle="--", alpha=0.6,
               label=f"Теор. пик памяти ({THEORETICAL_PEAK_MB_S / 1000:.1f} GB/s)")

    ax.legend()

    if cluster in CACHE_BOUNDARIES_KB:
        for name, kb in CACHE_BOUNDARIES_KB[cluster].items():
            mib = kb / 1024
            if x_mib[0] <= mib <= x_mib[-1]:
                ax.axvline(mib, color="red", linestyle=":", alpha=0.7)
                ax.text(mib, ax.get_ylim()[1] * 0.97, f"{name}\n{kb}K",
                        color="red", fontsize=8, ha="center", va="top")

    fig.tight_layout()
    plot_path = plot_dir / f"{run_id}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nСохранено: {plot_path.relative_to(ROOT)}")


# ---------- режим: scaling ----------

def cmd_scaling(args):
    binary = get_binary(args.variant, args.dtype)
    build_info = get_build_info(args.variant, args.dtype)
    bpe = DTYPE_BYTES[args.dtype]

    pool = CLUSTER_POOLS[args.target_cluster]
    thread_list = ([int(x) for x in args.threads_list.split(",")]
                    if args.threads_list else DEFAULT_THREAD_LIST[args.target_cluster])

    ws_bytes = args.ws_mb * 1024 * 1024
    array_elements = ws_bytes // bpe

    print(f"=== scaling: {args.variant}/{args.dtype} target_cluster={args.target_cluster} "
          f"threads={thread_list} WS={args.ws_mb}MB/array ===")
    print(f"    (для каждого t берутся первые t ядер пула {pool[0]}-{pool[-1]}, "
          f"contiguous от начала пула)")

    points = []
    all_logs = []
    for t in thread_list:
        if t > len(pool):
            print(f"  [threads={t}] ПРОПУЩЕНО — в пуле {args.target_cluster} "
                  f"только {len(pool)} ядер")
            continue
        cores = pool[:t]
        cpuset = cpuset_str_from_cores(cores)

        result = measure_point(binary, cpuset, t, cores, array_elements,
                                args.ntimes, args.repeats, args.warmup)
        all_logs.extend(result["logs"])

        triad = result["aggregate"]["Triad"]["median_mb_s"] if result["aggregate"]["Triad"] else None
        print(f"  [threads={t:>2d}] cpuset={cpuset:<8s} Triad={triad} MB/s "
              f"validates_all={result['validates_all']}")

        points.append({
            "threads": t, "cpuset": cpuset, "cores": cores,
            "aggregate": result["aggregate"],
            "validates_all": result["validates_all"],
            "cpu_freq_before_hz": result["cpu_freq_before_hz"],
            "temp_before_c": result["temp_before_c"],
        })

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"scaling_{args.variant}_{args.dtype}_{args.target_cluster}_{args.ws_mb}MB_{timestamp}"

    raw_dir = ROOT / "results" / "raw" / "scaling" / args.target_cluster
    log_dir = ROOT / "results" / "logs" / "scaling" / args.target_cluster
    plot_dir = ROOT / "results" / "processed" / "scaling" / args.target_cluster
    for d in (raw_dir, log_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id, "machine": "aibox_k3", "benchmark": "stream_dynamic",
        "experiment_type": "scaling", "variant": args.variant, "data_type": args.dtype,
        "target_cluster": args.target_cluster, "working_set_mb_per_array": args.ws_mb,
        "array_size_elements": array_elements, "repeats": args.repeats,
        "warmup_runs": args.warmup, "timestamp": timestamp,
        "compiler": build_info["compiler"], "compiler_flags": build_info["compiler_flags"],
        "memory_frequency_hz": MEMORY_FREQUENCY_HZ,
        "points": points,
    }
    (raw_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (log_dir / f"{run_id}.log").write_text("\n".join(all_logs))

    x_threads = [p["threads"] for p in points]
    fig, ax = plt.subplots(figsize=(9, 6))
    for fn in FUNCTIONS:
        y = [p["aggregate"][fn]["median_mb_s"] if p["aggregate"][fn] else None for p in points]
        lo = [p["aggregate"][fn]["median_mb_s"] - p["aggregate"][fn]["min_mb_s"]
              if p["aggregate"][fn] else 0 for p in points]
        hi = [p["aggregate"][fn]["max_mb_s"] - p["aggregate"][fn]["median_mb_s"]
              if p["aggregate"][fn] else 0 for p in points]
        ax.errorbar(x_threads, y, yerr=[lo, hi], marker="o", label=fn,
                     color=COLORS[fn], capsize=4)

    ax.axhline(THEORETICAL_PEAK_MB_S, color="black", linestyle="--", alpha=0.6,
               label=f"Теор. пик памяти ({THEORETICAL_PEAK_MB_S / 1000:.1f} GB/s)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_threads)
    ax.set_xticklabels([str(t) for t in x_threads])
    ax.set_xlabel("Число потоков")
    ax.set_ylabel("Bandwidth (MB/s)")
    ax.set_title(f"STREAM {args.variant}/{args.dtype} — Thread scaling "
                 f"(AIBOX-K3, target={args.target_cluster}, WS={args.ws_mb}MB)")
    ax.set_ylim(0, args.y_max_mb_s)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plot_path = plot_dir / f"{run_id}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nСохранено: {plot_path.relative_to(ROOT)}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Единый инструмент запуска STREAM (dynamic)")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = dict(
        variant=("--variant", dict(required=True, choices=["novec", "opt_serial", "omp"])),
        dtype=("--dtype", dict(required=True, choices=["fp64", "fp32", "fp16"])),
        cpuset=("--cpuset", dict(required=True)),
        threads=("--threads", dict(type=int, default=1)),
        repeats=("--repeats", dict(type=int, default=5)),
        warmup=("--warmup", dict(type=int, default=1)),
        ntimes=("--ntimes", dict(type=int, default=20)),
        y_max=("--y-max-mb-s", dict(type=float, default=100_000,
                                     help="Фиксированный потолок оси Y в MB/s "
                                          "(по умолчанию 100000 = 100 GB/s), "
                                          "единый для сравнимости графиков между собой")),
    )

    p_std = sub.add_parser("standard", help="Одна точка WS >> LLC, сравнение с пиком памяти")
    for flag, kwargs in common.values():
        p_std.add_argument(flag, **kwargs)
    p_std.add_argument("--ws-mb", type=int, default=200,
                        help="Размер ОДНОГО массива в MB (по умолчанию 200, "
                             "суммарно x3 — с запасом больше всех кэшей)")
    p_std.set_defaults(func=cmd_standard)

    p_sweep = sub.add_parser("cache_sweep", help="Плотная сетка WS + график границ кэшей")
    for flag, kwargs in common.values():
        p_sweep.add_argument(flag, **kwargs)
    p_sweep.add_argument("--ws-min-kb", type=int, default=8,
                          help="Минимальный WS в КБ (по умолчанию 8 — заведомо "
                               "меньше L1D=64KB, чтобы увидеть плато и границу)")
    p_sweep.add_argument("--ws-max-kb", type=int, default=2 * 1024 * 1024)  # 2GB
    p_sweep.add_argument("--ppo", type=int, default=6, help="Точек на одно удвоение размера")
    p_sweep.add_argument("--ntimes-ref-kb", type=int, default=256,
                          help="Опорный размер (КБ), для которого --ntimes задаёт "
                               "число повторов буквально; для меньших WS повторы "
                               "растут обратно пропорционально размеру, для больших — падают")
    p_sweep.add_argument("--ntimes-min", type=int, default=5,
                          help="Нижняя граница NTIMES (не опускаться ниже, иначе "
                               "мало данных для median/min/max)")
    p_sweep.add_argument("--ntimes-max", type=int, default=300,
                          help="Верхняя граница NTIMES (не разгонять до бесконечности "
                               "на маленьких WS, иначе замер займёт вечность)")
    p_sweep.set_defaults(func=cmd_cache_sweep)

    p_scale = sub.add_parser("scaling", help="Масштабирование по числу потоков")
    for flag, kwargs in common.values():
        if flag in ("--cpuset", "--threads"):
            continue  # scaling сам управляет cpuset/threads через target-cluster
        p_scale.add_argument(flag, **kwargs)
    p_scale.add_argument("--target-cluster", required=True,
                          choices=["x100", "a100", "mixed"],
                          help="x100: ядра 0-7; a100: ядра 8-15; mixed: 0-15")
    p_scale.add_argument("--threads-list", default=None,
                          help="Через запятую, напр. '1,2,4,8'. По умолчанию "
                               "берётся стандартный набор для target-cluster")
    p_scale.add_argument("--ws-mb", type=int, default=200,
                          help="Размер ОДНОГО массива в MB (по умолчанию 200)")
    p_scale.set_defaults(func=cmd_scaling)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()