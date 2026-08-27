#!/usr/bin/env python3
"""
plot_cache.py (multi-file)

Строит график Bandwidth vs Working set size.

ГЛАВНОЕ ПРАВИЛО (если передано РОВНО 2 файла — основной сценарий):
    file1 = AIBOX-K3, file2 = RK3588 (Orange Pi)
Платформа задаётся ПОЗИЦИЕЙ аргумента, а не угадывается по имени файла —
у наших собственных CSV (из collect_bench.py) в имени нет слов "aibox"/
"x100"/"a100" вообще, только номер ядра (cores0/cores8), поэтому
угадывание по ключевым словам молча путало AIBOX с RK3588.

Кластер ВНУТРИ платформы всё ещё определяется автоматически:
  - AIBOX: по номеру ядра в имени файла (coresN, N<=7 -> X100, N>=8 -> A100)
  - RK3588: по ключевым словам eff/perf/all (как в файлах коллег)

Если файлов не 2 (один файл, или 3+) — используется старое поведение
"best-effort" угадывание платформы по ключевым словам в имени.

Использование (основной сценарий):
    python3 plot_cache.py aibox_file.csv orangepi_file.csv

Прочие случаи:
    python3 plot_cache.py file1.csv                    # один файл
    python3 plot_cache.py f1.csv f2.csv f3.csv          # 3+ файлов, best-effort
"""
import csv
import sys
import os
import re
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === НАСТРОЙКИ ===
THEORETICAL_MAX_BW_GBPS = 33.8      # LPDDR4X-4266 (RK3588)
THEORETICAL_MAX_BW_GBPS_2 = 51.2    # LPDDR5-6400 (AIBOX-K3)
# =================

MARKERS = {"copy": "o", "scale": "s", "add": "^", "triad": "D"}
METRICS = ["copy", "scale", "add", "triad"]

ACID_ACCENT = "#FFEA00"   # кислотно-жёлтый — акцент AIBOX (границы кэша, пик 51.2)
DARK_ACCENT = "#0B2545"   # тёмно-синий — акцент RK3588 (границы кэша, пик 33.8)

PLATFORM_STYLE = {
    "aibox_x100": {"linewidth": 1.0, "markersize": 4,
                    "colors": ["#42A5F5", "#FFB74D", "#66BB6A", "#EF5350"], "accent": ACID_ACCENT},
    "aibox_a100": {"linewidth": 1.0, "markersize": 4,
                    "colors": ["#42A5F5", "#FFB74D", "#66BB6A", "#EF5350"], "accent": ACID_ACCENT},
    "rk3588_perf": {"linewidth": 3.2, "markersize": 6,
                     "colors": ["#0D47A1", "#E65100", "#1B5E20", "#7A0C0C"], "accent": DARK_ACCENT},
    "rk3588_eff": {"linewidth": 3.2, "markersize": 6,
                    "colors": ["#0D47A1", "#E65100", "#1B5E20", "#7A0C0C"], "accent": DARK_ACCENT},
    "rk3588_all": {"linewidth": 3.2, "markersize": 6,
                    "colors": ["#0D47A1", "#E65100", "#1B5E20", "#7A0C0C"], "accent": DARK_ACCENT},
}

# Границы кэшей — размеры ОДНОГО ядра (доступно одному потоку), не сумма
# по всем ядрам чипа. L1i не учитывается — STREAM его не нагружает.
CACHE_BOUNDARIES_BY_PLATFORM = {
    "aibox_a100": {"L1D": 64, "L2": 1024},
    "aibox_x100": {"L1D": 64, "L2": 4096},
    "rk3588_perf": {"L1D": 64, "L2": 512, "L3": 3072},
    "rk3588_eff": {"L1D": 32, "L2": 128, "L3": 3072},
    "rk3588_all": {"L1D": 64, "L2": 512, "L3": 3072},
}

SHOW_CACHE_BOUNDARY = {
    "aibox_x100": True, "aibox_a100": False,
    "rk3588_perf": True, "rk3588_eff": False, "rk3588_all": False,
}

PLATFORM_DISPLAY_NAMES = {
    "aibox_a100": "AIBOX A100", "aibox_x100": "AIBOX X100",
    "rk3588_perf": "RK3588 Cortex-A76", "rk3588_eff": "RK3588 Cortex-A55",
    "rk3588_all": "RK3588 (all cores)",
}


def detect_aibox_cluster(file_title):
    """X100 vs A100 — по номеру ядра в имени файла (coresN), это у наших
    CSV есть ВСЕГДА (в отличие от текста 'x100'/'a100', которого может не
    быть). N<=7 -> X100, N>=8 -> A100."""
    m = re.search(r'cores?(\d+)', file_title)
    if m:
        core_num = int(m.group(1))
        return "aibox_a100" if core_num >= 8 else "aibox_x100"
    if "a100" in file_title:
        return "aibox_a100"
    if "x100" in file_title:
        return "aibox_x100"
    print(f"  ПРЕДУПРЕЖДЕНИЕ: не удалось определить X100/A100 по имени "
          f"'{file_title}' (нет ни coresN, ни 'x100'/'a100') — беру X100 по умолчанию.",
          file=sys.stderr)
    return "aibox_x100"


def detect_rk3588_cluster(file_title):
    """eff/perf/all — по ключевым словам (как в файлах коллег)."""
    if "perf" in file_title:
        return "rk3588_perf"
    if "eff" in file_title:
        return "rk3588_eff"
    if "all" in file_title:
        return "rk3588_all"
    print(f"  ПРЕДУПРЕЖДЕНИЕ: не удалось определить eff/perf/all по имени "
          f"'{file_title}' — беру Cortex-A76 (perf) по умолчанию.", file=sys.stderr)
    return "rk3588_perf"


def detect_platform(file_title):
    """Best-effort угадывание платформы по имени файла целиком — только
    для случая, когда файлов НЕ РОВНО 2 (тогда позиционное правило
    'file1=AIBOX, file2=RK3588' неприменимо)."""
    ft = file_title.lower()
    if "a100" in ft or "x100" in ft or "aibox" in ft:
        return detect_aibox_cluster(ft)
    return detect_rk3588_cluster(ft)


def safe_float(s):
    if s is None:
        return math.nan
    s = s.strip()
    if s == "":
        return math.nan
    try:
        v = float(s)
    except ValueError:
        return math.nan
    if math.isnan(v) or math.isinf(v):
        return math.nan
    return v


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: safe_float(r.get("working_set_bytes", "0")) or 0)
    return rows


def make_tags(file_title, platform):
    """'Умные теги' для легенды — платформа передаётся ЯВНО (уже
    определена снаружи, по позиции или по best-effort), не угадывается
    здесь заново."""
    tags = [PLATFORM_DISPLAY_NAMES[platform]]

    core_match = re.search(r'(\d+)_?cores?', file_title)
    if core_match:
        tags.append(f"{core_match.group(1)}-Core")

    if "novec" in file_title:
        tags.append("Non-vectorized")
    elif "vec" in file_title:
        tags.append("Vectorized")

    if "fp64" in file_title or "double" in file_title:
        tags.append("FP64")
    elif "fp32" in file_title or "float" in file_title:
        tags.append("FP32")
    elif "int64" in file_title:
        tags.append("INT64")
    elif "int32" in file_title:
        tags.append("INT32")

    return ", ".join(tags) if tags else file_title


def main():
    if len(sys.argv) < 2:
        print(f"Использование: {sys.argv[0]} <csv1> [csv2] ...")
        print(f"  Если РОВНО 2 файла: file1 = AIBOX-K3, file2 = RK3588 (Orange Pi)")
        sys.exit(1)

    paths = sys.argv[1:]

    # --- определяем платформу КАЖДОГО файла ---
    platforms = []
    if len(paths) == 2:
        print("Режим: 2 файла — платформа задана ПОЗИЦИЕЙ (файл 1 = AIBOX-K3, файл 2 = RK3588/Orange Pi).")
        t0 = os.path.splitext(os.path.basename(paths[0]))[0].lower()
        t1 = os.path.splitext(os.path.basename(paths[1]))[0].lower()
        platforms = [detect_aibox_cluster(t0), detect_rk3588_cluster(t1)]
    else:
        for p in paths:
            t = os.path.splitext(os.path.basename(p))[0].lower()
            platforms.append(detect_platform(t))

    files_data = []
    for path, platform in zip(paths, platforms):
        rows = read_csv(path)
        file_title = os.path.splitext(os.path.basename(path))[0].lower()
        label = make_tags(file_title, platform)
        sizes_mb = [safe_float(r["working_set_bytes"]) / 1048576 for r in rows]
        series = {m: [safe_float(r[f"{m}_med"]) for r in rows] for m in METRICS}
        files_data.append({
            "label": label, "sizes_mb": sizes_mb, "series": series,
            "title": file_title, "platform": platform,
        })
        n_bad = sum(1 for m in METRICS for v in series[m] if math.isnan(v))
        print(f"  [{path}] -> платформа={PLATFORM_DISPLAY_NAMES[platform]}"
              + (f", пропущено значений (inf/nan/пусто): {n_bad}" if n_bad else ""))

    fig, ax = plt.subplots(figsize=(19, 8))

    for fd in files_data:
        style = PLATFORM_STYLE[fd["platform"]]
        for mi, m in enumerate(METRICS):
            label = f"{fd['label']} — {m.capitalize()}"
            ax.plot(fd["sizes_mb"], fd["series"][m], marker=MARKERS[m], linestyle="-",
                     markersize=style["markersize"], linewidth=style["linewidth"],
                     color=style["colors"][mi], label=label, zorder=3)

    ax.set_xscale("log", base=2)

    all_vals = [v for fd in files_data for m in METRICS for v in fd["series"][m] if not math.isnan(v)]
    peak_mbps = max(THEORETICAL_MAX_BW_GBPS, THEORETICAL_MAX_BW_GBPS_2) * 1000
    y_max = max(max(all_vals) * 1.15 if all_vals else 0, peak_mbps * 1.10)
    ax.set_ylim(0, y_max)

    def format_size(x, pos):
        if x >= 1024:
            return f"{x/1024:g} GB"
        elif x >= 1:
            return f"{x:g} MB"
        else:
            return f"{x*1024:g} KB"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_size))
    ax.set_xticks([0.015625, 0.0625, 0.25, 1, 4, 16, 64, 256, 1024, 2048])

    if len(files_data) == 1:
        title = f"STREAM Benchmark: {files_data[0]['label']}"
    else:
        title = "STREAM Benchmark — сравнение (" + " vs ".join(fd["label"] for fd in files_data) + ")"
    ax.set_title(title, pad=15, fontsize=13)
    ax.set_xlabel("Working set size")
    ax.set_ylabel("Bandwidth (MB/s)")

    ax.grid(True, which="major", ls="-", alpha=0.5, zorder=0)
    ax.grid(True, which="minor", ls=":", alpha=0.25, zorder=0)
    ax.minorticks_on()

    boundary_count = 0
    for fd in files_data:
        if not SHOW_CACHE_BOUNDARY.get(fd["platform"], False):
            continue
        style = PLATFORM_STYLE[fd["platform"]]
        accent = style["accent"]
        boundaries = CACHE_BOUNDARIES_BY_PLATFORM[fd["platform"]]
        label_y = ax.get_ylim()[1] * (0.55 + 0.12 * boundary_count)
        for name, kb in boundaries.items():
            mb = kb / 1024
            ax.axvline(x=mb, color=accent, ls="--", alpha=0.8, linewidth=1.6, zorder=2)
            ax.text(mb, label_y, f"{fd['label']}\n{name} ({kb}K)", rotation=90,
                     va="bottom", ha="center", color=accent, fontsize=8, fontweight="bold")
        boundary_count += 1

    max_bw_mbps = THEORETICAL_MAX_BW_GBPS * 1000
    max_bw_mbps_2 = THEORETICAL_MAX_BW_GBPS_2 * 1000
    ax.axhline(y=max_bw_mbps, color=DARK_ACCENT, linestyle="-.", linewidth=2.2, alpha=0.85, zorder=2)
    ax.text(x=ax.get_xlim()[1], y=max_bw_mbps + (ax.get_ylim()[1] * 0.012),
             s=f"Theoretical RAM Max: {THEORETICAL_MAX_BW_GBPS} GB/s (LPDDR4X, RK3588)",
             color=DARK_ACCENT, va="bottom", ha="right", fontsize=10, fontweight="bold")
    ax.axhline(y=max_bw_mbps_2, color=ACID_ACCENT, linestyle="-.", linewidth=2.2, alpha=0.95, zorder=2)
    ax.text(x=ax.get_xlim()[1], y=max_bw_mbps_2 + (ax.get_ylim()[1] * 0.012),
             s=f"Theoretical RAM Max: {THEORETICAL_MAX_BW_GBPS_2} GB/s (LPDDR5, AIBOX-K3)",
             color="#B8A000", va="bottom", ha="right", fontsize=10, fontweight="bold")

    ncol = 4 if len(files_data) <= 2 else 3
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5,
               ncol=ncol, borderaxespad=0, frameon=True)

    plt.tight_layout()

    if len(files_data) == 1:
        output_file = f"{files_data[0]['title']}_graph.png"
    else:
        output_file = "compare_" + "_vs_".join(fd["title"] for fd in files_data) + ".png"
        if len(output_file) > 150:
            output_file = "compare_" + "_vs_".join(f"file{i+1}" for i in range(len(files_data))) + ".png"

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён в файл: {output_file}")


if __name__ == "__main__":
    main()
