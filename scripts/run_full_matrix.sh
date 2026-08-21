#!/usr/bin/env bash
# run_full_matrix.sh
#
# Полный прогон STREAM по всем комбинациям (variant x dtype x кластер)
# для standard / cache_sweep / scaling, с последующим сбором ВСЕХ
# свежесозданных графиков в одну плоскую папку.
#
# Запускать из ~/stream_bench, обычным пользователем (не через sudo su).

set -uo pipefail  # без -e: одна неудачная точка не должна обрывать весь прогон

ROOT="$HOME/stream_bench"
cd "$ROOT" || { echo "Не найден $ROOT"; exit 1; }

# --- Проверка: march содержит zvfh (иначе fp16 снова сломается) ---
if ! grep -q "zvfh" scripts/build_dynamic.sh 2>/dev/null; then
    echo "ВНИМАНИЕ: zvfh не найден в scripts/build_dynamic.sh — fp16 может"
    echo "показать старую аномалию. Пересоберите build_dynamic.sh перед запуском."
    read -p "Продолжить всё равно? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

# --- Проверка доступа к A100 (cores 8-15) ---
CPUS_ALLOWED=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')
A100_AVAILABLE=1
if [[ "$CPUS_ALLOWED" != *"8-15"* && "$CPUS_ALLOWED" != *"0-15"* ]]; then
    echo "ВНИМАНИЕ: A100 (cores 8-15) недоступны в этой сессии (Cpus_allowed_list=$CPUS_ALLOWED)."
    echo "Точки на A100 будут пропущены. Для доступа — см. обходной путь через /proc/set_ai_thread."
    A100_AVAILABLE=0
fi

TS="$(date +%Y%m%dT%H%M%S)"
OUTDIR="$ROOT/all_graphs_$TS"
mkdir -p "$OUTDIR"
MARKER="$(mktemp)"
touch "$MARKER"

DTYPES=(fp64 fp32 fp16)
SINGLE_VARIANTS=(novec opt_serial)

run() { echo ">>> $*"; "$@"; echo ""; }

echo "########################################"
echo "# 1/3: STANDARD"
echo "########################################"

for variant in "${SINGLE_VARIANTS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
        run scripts/run_bench.py standard --variant "$variant" --dtype "$dtype" \
            --cpuset 0 --threads 1 --repeats 5
        if [[ $A100_AVAILABLE -eq 1 ]]; then
            run scripts/run_bench.py standard --variant "$variant" --dtype "$dtype" \
                --cpuset 8 --threads 1 --repeats 5
        fi
    done
done
for dtype in "${DTYPES[@]}"; do
    run scripts/run_bench.py standard --variant omp --dtype "$dtype" \
        --cpuset 0-7 --threads 8 --repeats 5
    if [[ $A100_AVAILABLE -eq 1 ]]; then
        run scripts/run_bench.py standard --variant omp --dtype "$dtype" \
            --cpuset 8-15 --threads 8 --repeats 5
    fi
done

echo "########################################"
echo "# 2/3: CACHE_SWEEP"
echo "########################################"

for variant in "${SINGLE_VARIANTS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
        run scripts/run_bench.py cache_sweep --variant "$variant" --dtype "$dtype" \
            --cpuset 0 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536
        if [[ $A100_AVAILABLE -eq 1 ]]; then
            run scripts/run_bench.py cache_sweep --variant "$variant" --dtype "$dtype" \
                --cpuset 8 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536
        fi
    done
done
echo "ПРИМЕЧАНИЕ: cache_sweep для omp известно нестабилен внутри кэша"
echo "(накладные расходы синхронизации потоков сопоставимы с полезной работой"
echo "на маленьких WS) — зона за пределами L2 (плато в RAM) остаётся надёжной."
for dtype in "${DTYPES[@]}"; do
    run scripts/run_bench.py cache_sweep --variant omp --dtype "$dtype" \
        --cpuset 0-7 --threads 8 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536
    if [[ $A100_AVAILABLE -eq 1 ]]; then
        run scripts/run_bench.py cache_sweep --variant omp --dtype "$dtype" \
            --cpuset 8-15 --threads 8 --repeats 3 --ws-min-kb 256 --ws-max-kb 65536
    fi
done

echo "########################################"
echo "# 3/3: SCALING (только omp — единственный вариант, чувствительный к потокам)"
echo "########################################"

for dtype in "${DTYPES[@]}"; do
    run scripts/run_bench.py scaling --variant omp --dtype "$dtype" \
        --target-cluster x100 --repeats 5
    if [[ $A100_AVAILABLE -eq 1 ]]; then
        run scripts/run_bench.py scaling --variant omp --dtype "$dtype" \
            --target-cluster a100 --repeats 5
    fi
done

echo "########################################"
echo "# Сбор всех новых графиков в одну папку"
echo "########################################"

find "$ROOT/results/processed" -name '*.png' -newer "$MARKER" -exec cp {} "$OUTDIR/" \;
rm -f "$MARKER"

COUNT=$(find "$OUTDIR" -name '*.png' | wc -l)
echo ""
echo "Готово. Собрано графиков: $COUNT"
echo "Папка: $OUTDIR"
