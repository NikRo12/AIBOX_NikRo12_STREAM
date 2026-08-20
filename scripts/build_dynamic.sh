#!/usr/bin/env bash
# scripts/build_dynamic.sh
#
# Сборка stream_dynamic.c — по одному бинарнику на комбинацию
# variant x dtype (размер массива больше не влияет на сборку,
# задаётся в рантайме через переменную окружения STREAM_ARRAY_SIZE).
#
# Использование:
#   ./build_dynamic.sh                  # вся матрица (9 бинарников)
#   ./build_dynamic.sh omp fp32         # только одна комбинация

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/STREAM_dynamic/stream_dynamic.c"
MARCH="rv64gcv_zvfh_zba_zbb_zbc_zbs"
ABI="lp64d"

declare -A VARIANT_FLAGS=(
    [novec]="-fno-tree-vectorize"
    [opt_serial]=""
    [omp]="-fopenmp"
)
declare -A DTYPE_DEFINE=(
    [fp64]="double"
    [fp32]="float"
    [fp16]="_Float16"
)

VARIANTS=("${!VARIANT_FLAGS[@]}")
DTYPES=("${!DTYPE_DEFINE[@]}")
if [[ $# -eq 2 ]]; then
    VARIANTS=("$1")
    DTYPES=("$2")
fi

if [[ ! -f "$SRC" ]]; then
    echo "ОШИБКА: не найден $SRC" >&2
    echo "Положите stream_dynamic.c в src/STREAM_dynamic/" >&2
    exit 1
fi

echo "=== Dynamic STREAM build matrix ==="
echo "SRC=$SRC"
echo "MARCH=$MARCH  ABI=$ABI"
echo "Variants: ${VARIANTS[*]}"
echo "Dtypes:   ${DTYPES[*]}"
echo "===================================="

FAILED=0

for variant in "${VARIANTS[@]}"; do
    for dtype in "${DTYPES[@]}"; do
        OUTDIR="$ROOT/builds_dynamic/$variant/$dtype"
        mkdir -p "$OUTDIR"
        OUT="$OUTDIR/stream_${variant}_${dtype}_dynamic"
        LOG="$OUTDIR/build.log"
        FLAGS="${VARIANT_FLAGS[$variant]}"
        TYPE_DEF="${DTYPE_DEFINE[$dtype]}"

        # shellcheck disable=SC2086
        if gcc -O3 -march="$MARCH" -mabi="$ABI" $FLAGS \
            -DSTREAM_TYPE="$TYPE_DEF" \
            "$SRC" -o "$OUT" -lm > "$LOG" 2>&1; then
            if [[ -s "$LOG" ]]; then
                echo "OK (с warnings, см. $LOG)  $variant/$dtype"
            else
                echo "OK (чисто)                  $variant/$dtype"
            fi

            # --- метаданные сборки для build_info.json (пункт 4: compiler/compiler_flags) ---
            GCC_VERSION="$(gcc --version | head -1)"
            FULL_FLAGS="-O3 -march=$MARCH -mabi=$ABI $FLAGS -DSTREAM_TYPE=$TYPE_DEF"
            BUILD_TS="$(date -Iseconds)"
            cat > "$OUTDIR/build_info.json" << INFO_EOF
{
  "variant": "$variant",
  "dtype": "$dtype",
  "compiler": "$GCC_VERSION",
  "compiler_flags": "$FULL_FLAGS",
  "source": "$SRC",
  "built_at": "$BUILD_TS"
}
INFO_EOF
        else
            echo "FAIL — см. $LOG             $variant/$dtype"
            FAILED=1
        fi
    done
done

echo ""
echo "=== Итог ==="
find "$ROOT/builds_dynamic" -mindepth 2 -type f -executable | sort

if [[ $FAILED -ne 0 ]]; then
    exit 1
fi
