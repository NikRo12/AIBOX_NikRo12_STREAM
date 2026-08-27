#!/usr/bin/env python3
import csv
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === НАСТРОЙКИ ===
# Теоретическая максимальная пропускная способность памяти (в GB/s)
THEORETICAL_MAX_BW_GBPS = 33.8      # LPDDR4X-4266 (или ваша текущая)
THEORETICAL_MAX_BW_GBPS_2 = 51.2    # LPDDR5-6400 (дополнительная линия)
# =================

# Проверяем аргумент
if len(sys.argv) < 2:
    print(f"Использование: {sys.argv[0]} <csv-файл>")
    sys.exit(1)

filename = sys.argv[1]

# Получаем название файла без расширения
file_title = os.path.splitext(os.path.basename(filename))[0].lower()

# Чтение данных
with open(filename) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Оставляем деление на 1048576, чтобы масштаб был правильным
sizes_mb = [int(r["working_set_bytes"]) / 1048576 for r in rows]

copy_med  = [float(r["copy_med"])  for r in rows]
scale_med = [float(r["scale_med"]) for r in rows]
add_med   = [float(r["add_med"])   for r in rows]
triad_med = [float(r["triad_med"]) for r in rows]

# Создание графика
fig, ax = plt.subplots(figsize=(12, 7)) # Слегка увеличил высоту для заголовка

ax.plot(sizes_mb, copy_med,  "o-", label="Copy",  markersize=4, linewidth=1.5)
ax.plot(sizes_mb, scale_med, "s-", label="Scale", markersize=4, linewidth=1.5)
ax.plot(sizes_mb, add_med,   "^-", label="Add",   markersize=4, linewidth=1.5)
ax.plot(sizes_mb, triad_med, "D-", label="Triad", markersize=4, linewidth=1.5)

# Настройка оси X (Логарифмическая)
ax.set_xscale("log", base=2)

# Динамическая настройка оси Y в зависимости от типа ядер
if "perf" in file_title:
    y_max = 200000  # Для быстрых ядер Cortex-A76
elif "eff" in file_title:
    y_max = 40000   # Для энергоэффективных Cortex-A55
else:
    y_max = 150000  # Для всех ядер вместе

ax.set_ylim(0, y_max)

# Форматирование оси X
def format_size(x, pos):
    if x >= 1024:
        return f"{x/1024:g} GB"
    elif x >= 1:
        return f"{x:g} MB"
    else:
        return f"{x*1024:g} KB"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_size))

# Засечки X (равномерные в log2 пространстве с шагом x4)
ticks = [ 0.015625, 0.0625, 0.25, 1, 4, 16, 64, 256, 1024, 2048 ]
ax.set_xticks(ticks)

# ==========================================
# ФОРМИРОВАНИЕ УМНОГО ЗАГОЛОВКА
# ==========================================
tags = []

# 1. Тип ядер
if "perf" in file_title:
    tags.append("Cortex-A76")
elif "eff" in file_title:
    tags.append("Cortex-A55")
elif "all" in file_title:
    tags.append("All Cores")

# 2. Количество ядер (ищет паттерны вроде 1core, 4cores, 8_cores)
core_match = re.search(r'(\d+)_?cores?', file_title)
if core_match:
    tags.append(f"{core_match.group(1)}-Core")

# 3. Векторизация
if "novec" in file_title:
    tags.append("Non-vectorized")
elif "vec" in file_title:
    tags.append("Vectorized")

# 4. Тип данных (Добавил распознавание распространенных типов STREAM)
if "fp64" in file_title or "double" in file_title:
    tags.append("FP64 (Double)")
elif "fp32" in file_title or "float" in file_title:
    tags.append("FP32 (Float)")
elif "int64" in file_title:
    tags.append("INT64")
elif "int32" in file_title:
    tags.append("INT32")

# Собираем заголовок
base_title = "STREAM Benchmark: Rockchip RK3588"
if tags:
    # Если есть теги, пишем их на второй строке в скобках
    full_title = f"{base_title}\n({', '.join(tags)})"
else:
    # Если скрипт ничего не распознал, просто выводим имя файла
    full_title = f"{base_title}\n(File: {os.path.basename(filename)})"

ax.set_title(full_title, pad=15, fontsize=14)
ax.set_xlabel("Working set size")
ax.set_ylabel("Bandwidth (MB/s)")

# Сетка
ax.grid(True, which="major", ls="-", alpha=0.6)
ax.grid(True, which="minor", ls=":", alpha=0.3)
ax.minorticks_on()

# Отметки границ кэшей
caches = [
    (0.0625, "L1D (64 KB)"),
    (0.5, "L2 (512 KB)"),
    (3.0, "L3 (~3 MB)")
]

for x, name in caches:
    ax.axvline(x=x, color="gray", ls="--", alpha=0.8, linewidth=1.2)
    ax.text(
        x * 1.05,
        ax.get_ylim()[1] * 0.95,
        name,
        rotation=90,
        va="top",
        ha="left",
        color="dimgray",
        fontsize=10
    )

# ==========================================
# ЛИНИИ МАКСИМАЛЬНОЙ ТЕОРЕТИЧЕСКОЙ СКОРОСТИ
# ==========================================
# Переводим GB/s в MB/s (для RAM обычно используют метрические множители 1000)
max_bw_mbps = THEORETICAL_MAX_BW_GBPS * 1000
max_bw_mbps_2 = THEORETICAL_MAX_BW_GBPS_2 * 1000

# Первая линия (LPDDR4X-4266)
ax.axhline(y=max_bw_mbps, color="red", linestyle="-.", linewidth=2, alpha=0.7)
ax.text(
    x=ax.get_xlim()[1], 
    y=max_bw_mbps + (ax.get_ylim()[1] * 0.02),
    s=f"Theoretical RAM Max: {THEORETICAL_MAX_BW_GBPS} GB/s (LPDDR4X)", 
    color="red", 
    va="bottom", 
    ha="right", 
    fontsize=11, 
    fontweight="bold"
)

# Вторая линия (LPDDR5-6400)
ax.axhline(y=max_bw_mbps_2, color="blue", linestyle="-.", linewidth=2, alpha=0.7)
ax.text(
    x=ax.get_xlim()[1], 
    y=max_bw_mbps_2 + (ax.get_ylim()[1] * 0.02),
    s=f"Theoretical RAM Max: {THEORETICAL_MAX_BW_GBPS_2} GB/s (LPDDR5)", 
    color="blue", 
    va="bottom", 
    ha="right", 
    fontsize=11, 
    fontweight="bold"
)

ax.legend(loc="upper right")

plt.tight_layout()

# Имя выходного файла (сохраняем оригинальное имя)
output_file = f"{file_title}_graph.png"
plt.savefig(output_file, dpi=150)

print(f"График сохранен в файл: {output_file}")