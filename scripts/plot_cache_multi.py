#!/usr/bin/env python3
"""
plot_cache.py (multi-file)

Строит график Bandwidth vs Working set size.

ГЛАВНОЕ ПРАВИЛО (если передано РОВНО 2 файла — основной сценарий):
    file1 = SpacemiT K3, file2 = RK3588 (Orange Pi)
Платформа задаётся ПОЗИЦИЕЙ аргумента, а не угадывается по имени файла —
у наших собственных CSV (из collect_bench.py) в имени нет слов "aibox"/
"x100"/"a100" вообще, только номер ядра (cores0/cores8), поэтому
угадывание по ключевым словам молча путало AIBOX с RK3588.

Кластер ВНУТРИ платформы всё ещё определяется автоматически:
  - SpacemiT K3: по номеру ядра в имени файла (coresN, N<=7 -> X100, N>=8 -> A100)
  - RK3588: по ключевым словам eff/perf/all (как в файлах коллег)

Если файлов не 2 (один файл, или 3+) — используется старое поведение
"best-effort" угадывание платформы по ключевым словам в имени.

Использование (основной сценарий):
    python3 plot_cache.py k3_file.csv orangepi_file.csv

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
THEORETICAL_MAX_BW_GBPS = 33.8      # LPDDR4X-4266 (RK3588 / Orange Pi)
THEORETICAL_MAX_BW_GBPS_2 = 51.2    # LPDDR5-6400 (SpacemiT K3 / AIBOX)
# =================

MARKERS = {"copy": "o", "scale": "s", "add": "^", "triad": "D"}
METRICS = ["copy", "scale", "add", "triad"]

# Более читаемые цвета (светлые, контрастные)
COLORS = {
    "k3_x100": ["#1E88E5", "#FB8C00", "#43A047", "#D32F2F"],  # синий, оранж, зеленый, красный
    "k3_a100": ["#1976D2", "#F57C00", "#388E3C", "#C62828"],
    "rk3588_perf": ["#0D47A1", "#E65100", "#1B5E20", "#7A0C0C"],
    "rk3588_eff": ["#1565C0", "#EF6C00", "#2E7D32", "#B71C1C"],
    "rk3588_all": ["#0D47A1", "#E65100", "#1B5E20", "#7A0C0C"],
}

PLATFORM_STYLE = {
    "k3_x100": {"linewidth": 2.0, "markersize": 6,
                    "colors": COLORS["k3_x100"], "accent": "#000000"},
    "k3_a100": {"linewidth": 2.0, "markersize": 6,
                    "colors": COLORS["k3_a100"], "accent": "#000000"},
    "rk3588_perf": {"linewidth": 2.5, "markersize": 7,
                     "colors": COLORS["rk3588_perf"], "accent": "#000000"},
    "rk3588_eff": {"linewidth": 2.5, "markersize": 7,
                    "colors": COLORS["rk3588_eff"], "accent": "#000000"},
    "rk3588_all": {"linewidth": 2.5, "markersize": 7,
                    "colors": COLORS["rk3588_all"], "accent": "#000000"},
}

# Границы кэшей — размеры ОДНОГО ядра (доступно одному потоку), не сумма
# по всем ядрам чипа. L1i не учитывается — STREAM его не нагружает.
CACHE_BOUNDARIES_BY_PLATFORM = {
    "k3_a100": {"L1D": 64, "L2": 1024},
    "k3_x100": {"L1D": 64, "L2": 4096},
    "rk3588_perf": {"L1D": 64, "L2": 512, "L3": 3072},
    "rk3588_eff": {"L1D": 32, "L2": 128, "L3": 3072},
    "rk3588_all": {"L1D": 64, "L2": 512, "L3": 3072},
}

SHOW_CACHE_BOUNDARY = {
    "k3_x100": True, "k3_a100": False,
    "rk3588_perf": True, "rk3588_eff": False, "rk3588_all": False,
}

PLATFORM_DISPLAY_NAMES = {
    "k3_a100": "SpacemiT K3 A100", "k3_x100": "SpacemiT K3 X100",
    "rk3588_perf": "RK3588 Cortex-A76", "rk3588_eff": "RK3588 Cortex-A55",
    "rk3588_all": "RK3588 (all cores)",
}

# Отображение для подписей кешей
CACHE_DISPLAY_NAMES = {
    "k3_x100": "SpacemiT K3 X100",
    "rk3588_perf": "RK3588 Cortex-A76",
}


def detect_k3_cluster(file_title):
    """X100 vs A100 — по номеру ядра в имени файла (coresN), это у наших
    CSV есть ВСЕГДА (в отличие от текста 'x100'/'a100', которого может не
    быть). N<=7 -> X100, N>=8 -> A100."""
    m = re.search(r'cores?(\d+)', file_title)
    if m:
        core_num = int(m.group(1))
        return "k3_a100" if core_num >= 8 else "k3_x100"
    if "a100" in file_title:
        return "k3_a100"
    if "x100" in file_title:
        return "k3_x100"
    print(f"  ПРЕДУПРЕЖДЕНИЕ: не удалось определить X100/A100 по имени "
          f"'{file_title}' (нет ни coresN, ни 'x100'/'a100') — беру X100 по умолчанию.",
          file=sys.stderr)
    return "k3_x100"


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
    'file1=SpacemiT K3, file2=RK3588' неприменимо)."""
    ft = file_title.lower()
    if "a100" in ft or "x100" in ft or "k3" in ft or "aibox" in ft:
        return detect_k3_cluster(ft)
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


def extract_test_conditions(file_title):
    """Извлекает условия теста из имени файла"""
    conditions = []
    
    # Количество ядер
    core_match = re.search(r'(\d+)_?cores?', file_title)
    if core_match:
        conditions.append(f"{core_match.group(1)}-Core")
    
    # Тип данных
    if "fp64" in file_title or "double" in file_title:
        conditions.append("FP64")
    elif "fp32" in file_title or "float" in file_title:
        conditions.append("FP32")
    elif "int64" in file_title:
        conditions.append("INT64")
    elif "int32" in file_title:
        conditions.append("INT32")
    
    # Векторизация
    if "novec" in file_title:
        conditions.append("Non-vectorized")
    elif "vec" in file_title:
        conditions.append("Vectorized")
    
    return conditions


def main():
    if len(sys.argv) < 2:
        print(f"Использование: {sys.argv[0]} <csv1> [csv2] ...")
        print(f"  Если РОВНО 2 файла: file1 = SpacemiT K3, file2 = RK3588 (Orange Pi)")
        sys.exit(1)

    paths = sys.argv[1:]

    # --- определяем платформу КАЖДОГО файла ---
    platforms = []
    if len(paths) == 2:
        print("Режим: 2 файла — платформа задана ПОЗИЦИЕЙ (файл 1 = SpacemiT K3, файл 2 = RK3588/Orange Pi).")
        t0 = os.path.splitext(os.path.basename(paths[0]))[0].lower()
        t1 = os.path.splitext(os.path.basename(paths[1]))[0].lower()
        platforms = [detect_k3_cluster(t0), detect_rk3588_cluster(t1)]
    else:
        for p in paths:
            t = os.path.splitext(os.path.basename(p))[0].lower()
            platforms.append(detect_platform(t))

    files_data = []
    all_conditions = []
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
        
        # Собираем условия для всех файлов
        all_conditions.extend(extract_test_conditions(file_title))

    fig, ax = plt.subplots(figsize=(16, 10))

    for fd in files_data:
        style = PLATFORM_STYLE[fd["platform"]]
        # Упрощаем label для легенды: только ядро и операция
        platform_name = PLATFORM_DISPLAY_NAMES[fd["platform"]]
        for mi, m in enumerate(METRICS):
            # Короткое имя для легенды
            label = f"{platform_name} {m.capitalize()}"
            ax.plot(fd["sizes_mb"], fd["series"][m], marker=MARKERS[m], linestyle="-",
                     markersize=style["markersize"], linewidth=style["linewidth"],
                     color=style["colors"][mi], label=label, zorder=3)

    ax.set_xscale("log", base=2)

    # Ограничиваем X от 64KB (0.0625 MB) до максимального значения
    ax.set_xlim(left=0.0625)  # 64 KB

    # Устанавливаем Y до 200000 MB/s
    ax.set_ylim(0, 200000)

    def format_size(x, pos):
        if x >= 1024:
            return f"{x/1024:g} GB"
        elif x >= 1:
            return f"{x:g} MB"
        else:
            return f"{x*1024:g} KB"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_size))
    # Устанавливаем метки от 64KB до 2048MB
    ax.set_xticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Формируем заголовок с условиями в одних скобках без повторений
    # Убираем дубликаты из условий
    unique_conditions = sorted(set(all_conditions))
    conditions_str = ", ".join(unique_conditions) if unique_conditions else ""
    
    # Формируем список платформ
    platform_names = [PLATFORM_DISPLAY_NAMES[p] for p in platforms]
    
    if len(files_data) == 1:
        if conditions_str:
            title = f"STREAM Benchmark: {platform_names[0]} ({conditions_str})"
        else:
            title = f"STREAM Benchmark: {platform_names[0]}"
    else:
        if conditions_str:
            title = f"STREAM Benchmark: {' vs '.join(platform_names)} ({conditions_str})"
        else:
            title = f"STREAM Benchmark: {' vs '.join(platform_names)}"
    
    ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel("Working set size", fontsize=12, fontweight='bold')
    ax.set_ylabel("Bandwidth (MB/s)", fontsize=12, fontweight='bold')

    ax.grid(True, which="major", ls="-", alpha=0.3, zorder=0, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.15, zorder=0, color='gray')
    ax.minorticks_on()

    # Отображение границ кэшей справа от линий (черным цветом)
    boundary_count = 0
    for fd in files_data:
        if not SHOW_CACHE_BOUNDARY.get(fd["platform"], False):
            continue
        style = PLATFORM_STYLE[fd["platform"]]
        boundaries = CACHE_BOUNDARIES_BY_PLATFORM[fd["platform"]]
        
        # Получаем максимальное значение Y для позиционирования текста
        y_max_plot = ax.get_ylim()[1]
        
        # Определяем подпись для платформы
        if fd["platform"] in CACHE_DISPLAY_NAMES:
            platform_label = CACHE_DISPLAY_NAMES[fd["platform"]]
        else:
            platform_label = PLATFORM_DISPLAY_NAMES[fd["platform"]]
        
        for name, kb in boundaries.items():
            mb = kb / 1024
            # Вертикальная линия (черная)
            ax.axvline(x=mb, color='black', ls="--", alpha=0.5, linewidth=1.5, zorder=2)
            # Текст справа от линии, чуть ниже верхней части графика
            y_pos = y_max_plot * 0.92 - (boundary_count * 0.035 * y_max_plot)
            # Размещаем текст справа от линии с небольшим отступом
            x_pos = mb * 1.02  # 2% отступ справа
            # Подпись с названием платформы и кеша
            label_text = f"{platform_label}: {name} ({kb}K)"
            ax.text(x_pos, y_pos, label_text, 
                     va='top', ha='left', color='black', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
            boundary_count += 1

    # Теоретические максимумы с обновленными подписями
    max_bw_mbps = THEORETICAL_MAX_BW_GBPS * 1000
    max_bw_mbps_2 = THEORETICAL_MAX_BW_GBPS_2 * 1000
    
    # Orange Pi (RK3588) - LPDDR4X (синий)
    ax.axhline(y=max_bw_mbps, color='#0B2545', linestyle="-.", linewidth=2.0, alpha=0.7, zorder=2)
    ax.text(x=ax.get_xlim()[1] * 0.98, y=max_bw_mbps + 500,
             s=f"Theoretical Max Orange Pi 33.8 GB/s (LPDDR4X)",
             color='#0B2545', va='bottom', ha='right', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
    
    # SpacemiT K3 (AIBOX) - LPDDR5 (красный)
    ax.axhline(y=max_bw_mbps_2, color='#D32F2F', linestyle="-.", linewidth=2.0, alpha=0.8, zorder=2)
    ax.text(x=ax.get_xlim()[1] * 0.98, y=max_bw_mbps_2 + 500,
             s=f"Theoretical Max AIBOX 51.2 GB/s (LPDDR5)",
             color='#D32F2F', va='bottom', ha='right', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    # Легенда справа сверху
    ax.legend(loc='upper right', fontsize=9, frameon=True, 
               framealpha=0.9, edgecolor='gray', fancybox=True)

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