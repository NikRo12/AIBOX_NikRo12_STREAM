#!/usr/bin/env python3
"""
scripts/plot_run.py

Единый график по JSON из collect_bench.py — ОДИН И ТОТ ЖЕ тип графика
(лог-ось X, точки/линия, опциональные границы кэшей, линия теор. пика)
для ЛЮБОГО режима: standard, cache_sweep, scaling. Не читает CSV вообще
(CSV — точка синхронизации с другой командой, не наш формат для чтения).

Что означает ось X, берётся прямо из JSON (record["x_axis"]) — скрипт
сам не решает "это cache_sweep, значит рисуй так-то": для ws_kb рисует
границы кэшей своей платформы и подписывает ось в KB/MB/GB, для threads
границы кэшей не рисует (не применимо) и подписывает ось как число
потоков. Standard (всегда одна точка) — тот же код, просто вырожденный
случай "линии из одной точки".

Работает с ОДНИМ или НЕСКОЛЬКИМИ JSON разом (для собственных сравнений,
например X100 vs A100 — не для сравнения с другой командой, у них
теперь свой инструмент поверх CSV).

Использование:
    python3 plot_run.py results/raw/cache_sweep/x100/cache_sweep_novec_fp64_..._....json -o x100_novec_fp64
    python3 plot_run.py results/raw/scaling/x100/scaling_omp_fp64_x100_200MB_....json \
                        results/raw/scaling/a100/scaling_omp_fp64_a100_200MB_....json \
                        -o x100_vs_a100_scaling
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FUNCS_ALL = ["Copy", "Scale", "Add", "Triad"]
MARKERS = {"Copy": "o", "Scale": "s", "Add": "^", "Triad": "D"}
RUN_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]

CACHE_BOUNDARIES_KB = {
    "x100": {"L1D": 64, "L2": 4096},
    "a100": {"L1D": 64, "L2": 1024},
}


def format_x_ws(kb):
    if kb >= 1024 ** 2:
        return f"{kb / 1024**2:g} GB"
    if kb >= 1024:
        return f"{kb / 1024:g} MB"
    return f"{kb:g} KB"


def load_run(path):
    record = json.loads(open(path, encoding="utf-8").read())
    if record["x_axis"]["kind"] == "threads":
        # число потоков само является переменной X — включать "Nt" в подпись
        # некорректно (в отличие от cache_sweep/standard, где потоки фиксированы)
        label = f"{record['variant']}/{record['data_type']} {record['cluster']}"
    else:
        threads = record["points"][0]["threads"]
        label = f"{record['variant']}/{record['data_type']} {record['cluster']} {threads}t"
    return record, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+")
    ap.add_argument("--labels", default=None,
                     help="Через запятую, по одному на файл (по умолчанию — автогенерация)")
    ap.add_argument("--functions", default=None,
                     help="Через запятую: Copy,Scale,Add,Triad. "
                          "По умолчанию: все 4 при одном файле, только Triad при нескольких")
    ap.add_argument("-o", "--out-prefix", default="run")
    ap.add_argument("--y-max-mb-s", type=float, default=None)
    args = ap.parse_args()

    runs = []
    labels = args.labels.split(",") if args.labels else None
    for i, path in enumerate(args.json_paths):
        record, auto_label = load_run(path)
        runs.append({"record": record, "label": labels[i] if labels else auto_label})

    if args.functions is not None:
        functions = [f.strip().capitalize() for f in args.functions.split(",")]
    elif len(runs) == 1:
        functions = FUNCS_ALL
    else:
        functions = ["Triad"]

    # x_axis должна быть одинакового СМЫСЛА у всех переданных файлов —
    # иначе накладывать друг на друга бессмысленно (KB поверх числа потоков)
    x_kinds = {r["record"]["x_axis"]["kind"] for r in runs}
    if len(x_kinds) > 1:
        raise SystemExit(f"ОШИБКА: нельзя совместить разные оси X в одном графике: {x_kinds}")
    x_kind = x_kinds.pop()
    x_label = runs[0]["record"]["x_axis"]["label"]

    def build_figure(mode):
        """mode: 'raw' (MB/s) или 'efficiency' (% от собственного пика)"""
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.set_xscale("log", base=2)

        all_vals, all_peaks = [], []
        for run in runs:
            for p in run["record"]["points"]:
                for fn in functions:
                    if p["aggregate"].get(fn):
                        all_vals.append(p["aggregate"][fn]["median_mb_s"])
            all_peaks.append(run["record"]["theoretical_peak_mb_s"])

        if mode == "raw":
            y_max = args.y_max_mb_s or max(
                max(all_vals) * 1.15 if all_vals else 0,
                max(all_peaks) * 1.08 if all_peaks else 100000,
            )
        else:
            y_max = 100

        for ri, run in enumerate(runs):
            record, label = run["record"], run["label"]
            color = RUN_COLORS[ri % len(RUN_COLORS)]
            peak = record["theoretical_peak_mb_s"]
            xs = [p["x"] for p in record["points"]]

            for fn in functions:
                ys = []
                for p in record["points"]:
                    if not p["aggregate"].get(fn):
                        ys.append(None)
                        continue
                    v = p["aggregate"][fn]["median_mb_s"]
                    ys.append(v if mode == "raw" else v / peak * 100)
                series_label = f"{label} — {fn}" if len(functions) > 1 else label
                ax.plot(xs, ys, marker=MARKERS[fn], markersize=4, linewidth=1.3,
                         color=color, label=series_label,
                         linestyle="-" if fn == "Triad" else "--")

            if mode == "raw":
                ax.axhline(peak, color=color, linestyle="--", alpha=0.4, linewidth=1.5)
                ax.text(max(xs) if xs else 1, peak, f" {label} теор. пик ({peak/1000:.1f} GB/s)",
                        color=color, fontsize=8, va="center")

            # границы кэшей — только когда ось X реально про размер данных,
            # и только если знаем границы для этого кластера
            if x_kind == "ws_kb" and record["cluster"] in CACHE_BOUNDARIES_KB:
                label_y = y_max * (0.30 + 0.10 * ri)
                for name, kb in CACHE_BOUNDARIES_KB[record["cluster"]].items():
                    ax.axvline(kb, color=color, linestyle=":", alpha=0.5, linewidth=1)
                    ax.text(kb, label_y, f"{label} {name} {kb}K",
                            color=color, fontsize=7, ha="center", va="bottom", rotation=90)

        ax.set_xlabel(x_label)
        if x_kind == "ws_kb":
            xs_all = [p["x"] for run in runs for p in run["record"]["points"]]
            ticks = sorted(set(xs_all))
            ax.set_xticks(ticks)
            ax.set_xticklabels([format_x_ws(v) for v in ticks], rotation=45, ha="right")
        ax.set_ylabel("Bandwidth (MB/s)" if mode == "raw" else "Эффективность, % от теор. пика")
        ax.set_ylim(0, y_max)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        title_kind = "сырая пропускная способность" if mode == "raw" else "эффективность от теор. пика"
        ax.set_title(f"STREAM — {title_kind}")
        fig.savefig(f"{args.out_prefix}_{mode}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    build_figure("raw")
    build_figure("efficiency")
    print(f"Сохранено: {args.out_prefix}_raw.png, {args.out_prefix}_efficiency.png")


if __name__ == "__main__":
    main()
