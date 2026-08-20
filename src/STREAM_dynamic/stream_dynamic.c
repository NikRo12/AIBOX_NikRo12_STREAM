/*
 * stream_dynamic.c
 *
 * Собственная реализация алгоритма STREAM Benchmark (McCalpin) с
 * рантайм-конфигурируемым размером массива через malloc — вместо
 * статических массивов фиксированного на этапе компиляции размера.
 *
 * Причина: статические массивы на RISC-V ограничены ~2GB суммарного
 * смещения (R_RISCV_PCREL_HI20 relocation), malloc-память этому
 * ограничению не подвержена. Также один бинарник теперь обслуживает
 * ЛЮБОЙ размер рабочего набора — не нужно пересобирать под каждую точку.
 *
 * Формат вывода намеренно совпадает с оригинальным stream.c, чтобы
 * существующие скрипты парсинга (regex по "Copy:", "Solution Validates"
 * и т.д.) продолжали работать без изменений.
 *
 * Параметры через переменные окружения:
 *   STREAM_ARRAY_SIZE  — размер каждого массива в элементах (обязательно)
 *   STREAM_NTIMES      — число повторов измерения (по умолчанию 20)
 *   STREAM_OFFSET      — смещение для избежания aliasing (по умолчанию 0)
 *
 * Компиляция — как и раньше:
 *   gcc -O3 -march=... -mabi=... [-fno-tree-vectorize|-fopenmp] \
 *       -DSTREAM_TYPE=double stream_dynamic.c -o stream_dynamic
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

int main(void) {
    long array_size = getenv_long("STREAM_ARRAY_SIZE", -1);
    int ntimes = (int)getenv_long("STREAM_NTIMES", 20);
    long offset = getenv_long("STREAM_OFFSET", 0);

    if (array_size <= 0) {
        fprintf(stderr, "ERROR: STREAM_ARRAY_SIZE must be set to a positive "
                        "number of elements (env var).\n");
        return 1;
    }

    const char *HLINE = "-------------------------------------------------------------\n";
    STREAM_TYPE *a, *b, *c;
    long total_elems = array_size + offset;
    size_t bytes = (size_t)total_elems * sizeof(STREAM_TYPE);

    if (posix_memalign((void **)&a, 64, bytes) != 0 ||
        posix_memalign((void **)&b, 64, bytes) != 0 ||
        posix_memalign((void **)&c, 64, bytes) != 0) {
        fprintf(stderr, "ERROR: malloc failed for array_size=%ld (%.1f MiB "
                        "per array)\n", array_size, bytes / (1024.0 * 1024.0));
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
    printf("STREAM (dynamic) version — custom malloc-based implementation\n");
    printf("%s", HLINE);
    printf("This system uses %d bytes per array element.\n", (int)sizeof(STREAM_TYPE));
    printf("%s", HLINE);
    printf("Array size = %ld (elements), Offset = %ld (elements)\n", array_size, offset);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
           bytes / (1024.0 * 1024.0), bytes / (1024.0 * 1024.0 * 1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
           3.0 * bytes / (1024.0 * 1024.0), 3.0 * bytes / (1024.0 * 1024.0 * 1024.0));
    printf("Each kernel will be executed %d times.\n", ntimes);
    printf("%s", HLINE);

    /* Инициализация — как в оригинальном STREAM */
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (long j = 0; j < array_size; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    STREAM_TYPE scalar = 3.0;

    for (int k = 0; k < ntimes; k++) {
        double t;

        t = mysecond();
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (long j = 0; j < array_size; j++) c[j] = a[j];
        times[0][k] = mysecond() - t;

        t = mysecond();
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (long j = 0; j < array_size; j++) b[j] = scalar * c[j];
        times[1][k] = mysecond() - t;

        t = mysecond();
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (long j = 0; j < array_size; j++) c[j] = a[j] + b[j];
        times[2][k] = mysecond() - t;

        t = mysecond();
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (long j = 0; j < array_size; j++) a[j] = b[j] + scalar * c[j];
        times[3][k] = mysecond() - t;
    }

    /* Best/avg/min/max, первая итерация исключается — как в оригинале */
    for (int k = 1; k < ntimes; k++) {
        for (int i = 0; i < 4; i++) {
            avgtime[i] += times[i][k];
            mintime[i] = times[i][k] < mintime[i] ? times[i][k] : mintime[i];
            maxtime[i] = times[i][k] > maxtime[i] ? times[i][k] : maxtime[i];
        }
    }

    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (int i = 0; i < 4; i++) {
        avgtime[i] /= (double)(ntimes - 1);
        printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n",
               label[i], bytes_per_kernel[i] / mintime[i] / 1.0e6,
               avgtime[i], mintime[i], maxtime[i]);
    }
    printf("%s", HLINE);

    /* --- Валидация: аналитически вычисляем ожидаемые значения --- */
    STREAM_TYPE aj = 1.0, bj = 2.0, cj = 0.0;
    for (int k = 0; k < ntimes; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    double epsilon;
    if (sizeof(STREAM_TYPE) == 4) {
        epsilon = 1.e-6;
    } else if (sizeof(STREAM_TYPE) == 8) {
        epsilon = 1.e-13;
    } else {
        printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n", (unsigned long)sizeof(STREAM_TYPE));
        epsilon = 1.e-6;
    }

    double aSumErr = 0, bSumErr = 0, cSumErr = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:aSumErr,bSumErr,cSumErr)
    #endif
    for (long j = 0; j < array_size; j++) {
        aSumErr += fabs((double)a[j] - (double)aj);
        bSumErr += fabs((double)b[j] - (double)bj);
        cSumErr += fabs((double)c[j] - (double)cj);
    }
    double aAvgErr = aSumErr / (double)array_size;
    double bAvgErr = bSumErr / (double)array_size;
    double cAvgErr = cSumErr / (double)array_size;

    int failed = 0;
    if (fabs(aAvgErr / aj) > epsilon) failed = 1;
    if (fabs(bAvgErr / bj) > epsilon) failed = 1;
    if (fabs(cAvgErr / cj) > epsilon) failed = 1;

    if (!failed) {
        printf("Solution Validates: avg error less than %e on all three arrays\n", epsilon);
    } else {
        printf("Failed Validation on array a, b, or c\n");
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
               (double)aj, aAvgErr, fabs(aAvgErr / aj));
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
               (double)bj, bAvgErr, fabs(bAvgErr / bj));
        printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",
               (double)cj, cAvgErr, fabs(cAvgErr / cj));
    }
    printf("%s", HLINE);

    free(a);
    free(b);
    free(c);
    return failed ? 1 : 0;
}
