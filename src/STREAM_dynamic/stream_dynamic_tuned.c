/*
 * stream_dynamic_tuned.c
 *
 * Версия stream_dynamic.c с ЯВНО ВЫНЕСЕННЫМИ функциями-кернелами
 * (аналог "TUNED" режима оригинального STREAM). Каждый кернел —
 * отдельная функция с __attribute__((noinline)), чтобы:
 *   1) в objdump был отдельный именованный символ на каждый кернел
 *   2) компилятор не мог инлайнить/сливать кернелы между собой,
 *      что усложнило бы анализ ассемблера
 *
 * Используется ТОЛЬКО для анализа (objdump/perf), не для автоматизации
 * замеров — параметры (размер, ntimes) те же env vars, что и в основной
 * версии, но нет отдельного места в run_bench.py.
 *
 * Сборка (пример, три варианта для сравнения):
 *   gcc -O3 -march=rv64gcv_zba_zbb_zbc_zbs -mabi=lp64d \
 *       -g -fno-omit-frame-pointer -fno-tree-vectorize \
 *       -DSTREAM_TYPE=double stream_dynamic_tuned.c -o tuned_novec_fp64 -lm
 *
 *   gcc -O3 -march=rv64gcv_zba_zbb_zbc_zbs -mabi=lp64d \
 *       -g -fno-omit-frame-pointer \
 *       -DSTREAM_TYPE=double stream_dynamic_tuned.c -o tuned_opt_serial_fp64 -lm
 *
 *   gcc -O3 -march=rv64gcv_zba_zbb_zbc_zbs -mabi=lp64d \
 *       -g -fno-omit-frame-pointer -fopenmp \
 *       -DSTREAM_TYPE=double stream_dynamic_tuned.c -o tuned_omp_fp64 -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

static double mysecond(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.0e-6;
}

static long getenv_long(const char *name, long fallback) {
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    return atol(v);
}

/* --- Кернелы: каждый — отдельная noinline функция, символ виден в objdump --- */

__attribute__((noinline))
void kernel_copy(STREAM_TYPE * restrict c, const STREAM_TYPE * restrict a, long n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < n; j++) c[j] = a[j];
}

__attribute__((noinline))
void kernel_scale(STREAM_TYPE * restrict b, const STREAM_TYPE * restrict c,
                   STREAM_TYPE scalar, long n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < n; j++) b[j] = scalar * c[j];
}

__attribute__((noinline))
void kernel_add(STREAM_TYPE * restrict c, const STREAM_TYPE * restrict a,
                 const STREAM_TYPE * restrict b, long n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < n; j++) c[j] = a[j] + b[j];
}

__attribute__((noinline))
void kernel_triad(STREAM_TYPE * restrict a, const STREAM_TYPE * restrict b,
                   const STREAM_TYPE * restrict c, STREAM_TYPE scalar, long n) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < n; j++) a[j] = b[j] + scalar * c[j];
}

/* --- Диагностические кернелы: проверка гипотезы "запись дороже чтения
   из-за DRAM bus turnaround" (переключение шины между read/write) --- */

__attribute__((noinline))
void kernel_store_only(STREAM_TYPE * restrict dst, STREAM_TYPE val, long n) {
    /* Чистая запись, без единого чтения из памяти (val — из регистра). */
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < n; j++) dst[j] = val;
}

__attribute__((noinline))
STREAM_TYPE kernel_load_only(const STREAM_TYPE * restrict src, long n) {
    /* Чистое чтение — суммируем, чтобы компилятор не выкинул цикл целиком
       (без использования результата -O3 удалит "бесполезный" код). */
    STREAM_TYPE sum = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (long j = 0; j < n; j++) sum += src[j];
    return sum;
}

int main(void) {
    long array_size = getenv_long("STREAM_ARRAY_SIZE", -1);
    int ntimes = (int)getenv_long("STREAM_NTIMES", 20);
    long offset = getenv_long("STREAM_OFFSET", 0);

    if (array_size <= 0) {
        fprintf(stderr, "ERROR: STREAM_ARRAY_SIZE must be set (env var).\n");
        return 1;
    }

    const char *HLINE = "-------------------------------------------------------------\n";
    STREAM_TYPE *a, *b, *c;
    long total_elems = array_size + offset;
    size_t bytes = (size_t)total_elems * sizeof(STREAM_TYPE);

    if (posix_memalign((void **)&a, 64, bytes) != 0 ||
        posix_memalign((void **)&b, 64, bytes) != 0 ||
        posix_memalign((void **)&c, 64, bytes) != 0) {
        fprintf(stderr, "ERROR: malloc failed for array_size=%ld\n", array_size);
        return 1;
    }

    double times[4][ntimes];
    double avgtime[4] = {0}, maxtime[4] = {0};
    double mintime[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
    const char *label[4] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};
    double bytes_per_kernel[4] = {
        2.0 * sizeof(STREAM_TYPE) * array_size,
        2.0 * sizeof(STREAM_TYPE) * array_size,
        3.0 * sizeof(STREAM_TYPE) * array_size,
        3.0 * sizeof(STREAM_TYPE) * array_size,
    };

    printf("%s", HLINE);
    printf("STREAM (dynamic, tuned/extracted kernels)\n");
    printf("%s", HLINE);
    printf("This system uses %d bytes per array element.\n", (int)sizeof(STREAM_TYPE));
    printf("Array size = %ld (elements), Offset = %ld (elements)\n", array_size, offset);
    printf("Each kernel will be executed %d times.\n", ntimes);
    printf("%s", HLINE);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < array_size; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    STREAM_TYPE scalar = 3.0;

    /* Изоляция одного кернела для perf: STREAM_KERNEL_ONLY=copy|scale|add|triad.
       По умолчанию (не задано) — гоняем все четыре, как раньше. */
    const char *kernel_only = getenv("STREAM_KERNEL_ONLY");

    /* --- Диагностические режимы: чистое чтение / чистая запись,
       отдельная ветка (не вписываются в таблицу из 4 стандартных кернелов,
       у них другая формула трафика — 1x bytes, а не 2x/3x). --- */
    if (kernel_only && strcmp(kernel_only, "store_only") == 0) {
        double mint = DBL_MAX, maxt = 0, avgt = 0;
        double tt[ntimes];
        for (int k = 0; k < ntimes; k++) {
            double t = mysecond();
            kernel_store_only(b, 7.0, array_size);
            tt[k] = mysecond() - t;
        }
        for (int k = 1; k < ntimes; k++) {
            avgt += tt[k];
            mint = tt[k] < mint ? tt[k] : mint;
            maxt = tt[k] > maxt ? tt[k] : maxt;
        }
        avgt /= (double)(ntimes - 1);
        double bytes_moved = 1.0 * sizeof(STREAM_TYPE) * array_size;
        printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
        printf("StoreOnly:  %12.1f  %11.6f  %11.6f  %11.6f\n",
               bytes_moved / mint / 1.0e6, avgt, mint, maxt);
        printf("%s", HLINE);
        printf("Validation skipped (diagnostic store-only kernel)\n");
        printf("%s", HLINE);
        free(a); free(b); free(c);
        return 0;
    }

    if (kernel_only && strcmp(kernel_only, "load_only") == 0) {
        double mint = DBL_MAX, maxt = 0, avgt = 0;
        double tt[ntimes];
        volatile STREAM_TYPE sink = 0;  /* чтобы компилятор не выкинул сумму */
        for (int k = 0; k < ntimes; k++) {
            double t = mysecond();
            sink = kernel_load_only(a, array_size);
            tt[k] = mysecond() - t;
        }
        (void)sink;
        for (int k = 1; k < ntimes; k++) {
            avgt += tt[k];
            mint = tt[k] < mint ? tt[k] : mint;
            maxt = tt[k] > maxt ? tt[k] : maxt;
        }
        avgt /= (double)(ntimes - 1);
        double bytes_moved = 1.0 * sizeof(STREAM_TYPE) * array_size;
        printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
        printf("LoadOnly:   %12.1f  %11.6f  %11.6f  %11.6f\n",
               bytes_moved / mint / 1.0e6, avgt, mint, maxt);
        printf("%s", HLINE);
        printf("Validation skipped (diagnostic load-only kernel)\n");
        printf("%s", HLINE);
        free(a); free(b); free(c);
        return 0;
    }

    int run_copy  = !kernel_only || strcmp(kernel_only, "copy")  == 0;
    int run_scale = !kernel_only || strcmp(kernel_only, "scale") == 0;
    int run_add   = !kernel_only || strcmp(kernel_only, "add")   == 0;
    int run_triad = !kernel_only || strcmp(kernel_only, "triad") == 0;

    for (int k = 0; k < ntimes; k++) {
        double t;

        if (run_copy) {
            t = mysecond();
            kernel_copy(c, a, array_size);
            times[0][k] = mysecond() - t;
        }

        if (run_scale) {
            t = mysecond();
            kernel_scale(b, c, scalar, array_size);
            times[1][k] = mysecond() - t;
        }

        if (run_add) {
            t = mysecond();
            kernel_add(c, a, b, array_size);
            times[2][k] = mysecond() - t;
        }

        if (run_triad) {
            t = mysecond();
            kernel_triad(a, b, c, scalar, array_size);
            times[3][k] = mysecond() - t;
        }
    }

    for (int k = 1; k < ntimes; k++) {
        for (int i = 0; i < 4; i++) {
            avgtime[i] += times[i][k];
            mintime[i] = times[i][k] < mintime[i] ? times[i][k] : mintime[i];
            maxtime[i] = times[i][k] > maxtime[i] ? times[i][k] : maxtime[i];
        }
    }

    int ran[4] = {run_copy, run_scale, run_add, run_triad};

    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int i = 0; i < 4; i++) {
        if (!ran[i]) continue;
        avgtime[i] /= (double)(ntimes - 1);
        printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n",
               label[i], bytes_per_kernel[i] / mintime[i] / 1.0e6,
               avgtime[i], mintime[i], maxtime[i]);
    }
    printf("%s", HLINE);

    /* Валидация корректна только если отработали ВСЕ ЧЕТЫРЕ кернела подряд —
       аналитический расчёт ожидаемых значений предполагает полную цепочку.
       В режиме STREAM_KERNEL_ONLY (для изолированного perf) валидацию
       пропускаем — это ожидаемо, не ошибка. */
    if (kernel_only) {
        printf("Validation skipped (STREAM_KERNEL_ONLY=%s — isolated single-kernel run)\n",
               kernel_only);
        printf("%s", HLINE);
        free(a); free(b); free(c);
        return 0;
    }

    STREAM_TYPE aj = 1.0, bj = 2.0, cj = 0.0;
    for (int k = 0; k < ntimes; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    double epsilon = (sizeof(STREAM_TYPE) == 4) ? 1.e-6 :
                      (sizeof(STREAM_TYPE) == 8) ? 1.e-13 : 1.e-6;

    double aSumErr = 0, bSumErr = 0, cSumErr = 0;
    for (long j = 0; j < array_size; j++) {
        aSumErr += fabs((double)a[j] - (double)aj);
        bSumErr += fabs((double)b[j] - (double)bj);
        cSumErr += fabs((double)c[j] - (double)cj);
    }
    double aAvgErr = aSumErr / (double)array_size;
    double bAvgErr = bSumErr / (double)array_size;
    double cAvgErr = cSumErr / (double)array_size;

    int failed = (fabs(aAvgErr / aj) > epsilon) ||
                 (fabs(bAvgErr / bj) > epsilon) ||
                 (fabs(cAvgErr / cj) > epsilon);

    if (!failed) {
        printf("Solution Validates: avg error less than %e on all three arrays\n", epsilon);
    } else {
        printf("Failed Validation on array a, b, or c\n");
    }
    printf("%s", HLINE);

    free(a); free(b); free(c);
    return failed ? 1 : 0;
}
