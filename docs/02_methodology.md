# Методология и автоматизация (ТЗ §4)

Покрывает только STREAM (пропускная способность памяти) на AIBOX-K3.
GEMM-автоматизация (compute-bound часть ТЗ §4/§5) — не в этом
репозитории, см. [`08_status_vs_assignment.md`](08_status_vs_assignment.md).

## 1. Структура репозитория

```
src/STREAM/stream.c                      — эталонный STREAM (McCalpin), без изменений
src/STREAM_dynamic/stream_dynamic.c      — размер массива и NTIMES задаются в рантайме
                                            через переменные окружения, не пересборкой
src/STREAM_dynamic/stream_dynamic_tuned.c — кернелы вынесены в noinline-функции с
                                            собственными именами (kernel_copy, kernel_scale,
                                            ...) — нужно для objdump/perf annotate по
                                            конкретному кернелу (см. docs/04_perf_profiling.md)

tools/vlen_probe/vlen_probe.c            — эмпирическое чтение VLEN/VLMAX из CSR

scripts/build_dynamic.sh                 — сборка всей матрицы variant×dtype
scripts/run_bench.py                     — единая точка входа: standard | cache_sweep | scaling

builds_dynamic/{variant}/{dtype}/        — бинарники + build_info.json (в .gitignore)

results/{raw,logs,processed}/{standard,cache_sweep,scaling}/{x100,a100,mixed}/
sysinfo/standard/{x100,a100}/            — снимки cpufreq/temp для ранних (legacy) прогонов

perf_evidence/{asm,perf_stat,perf_record,run_outputs,binaries}/ — доказательная база
                                            для docs/04_perf_profiling.md
```

## 2. Матрица сборки (ТЗ: минимум 3 варианта)

`scripts/build_dynamic.sh` собирает **9 бинарников** — декартово
произведение 3 вариантов × 3 типов данных, единая база флагов
`-O3 -march=rv64gcv_zvfh_zba_zbb_zbc_zbs -mabi=lp64d`:

| Вариант | Доп. флаги | Требование ТЗ |
|---|---|---|
| `novec` | `-fno-tree-vectorize` | без автовекторизации |
| `opt_serial` | (нет доп. флагов) | оптимизированная однопоточная |
| `omp` | `-fopenmp` | оптимизированная многопоточная |

Типы данных: `fp64` (`double`), `fp32` (`float`), `fp16` (`_Float16`).
Каждая сборка пишет `build_info.json` рядом с бинарником (`compiler`,
`compiler_flags`, `built_at`) — `run_bench.py` читает его и кладёт
значения в каждую raw-запись, а не в предположение по умолчанию.

## 3. Раннер `scripts/run_bench.py` — три режима

### `standard` — сравнение с теоретическим пиком (ТЗ §3, стандартный STREAM)

Один рабочий набор (по умолчанию 200 МБ на массив ×3 массива, заведомо
больше LLC). Пример:

```bash
run_bench.py standard --variant opt_serial --dtype fp64 --cpuset 0 --repeats 5 --ws-mb 200
```

### `cache_sweep` — граница кэшей (ТЗ §3, исследование кэш-иерархии)

Логарифмическая сетка размеров рабочего набора (`--ws-min-kb`/`--ws-max-kb`,
по умолчанию 8 КБ .. 2 ГБ, 6 точек на удвоение). Для каждой точки
`array_elements = ws_bytes // (3 × bytes_per_element)` — совпадает с
формулой ТЗ (`Рабочий набор = 3 × число элементов × размер элемента`).
`NTIMES` адаптивный: обратно пропорционален размеру WS (иначе на малых
WS таймер шумит, на больших — замер занимает часы).

### `scaling` — масштабирование по потокам (ТЗ §3, п. «Масштабирование»)

```bash
run_bench.py scaling --variant omp --dtype fp64 --target-cluster x100
```

`--target-cluster` = `x100` (ядра 0-7) | `a100` (ядра 8-15) | `mixed`
(0-15) — прямое соответствие требованию ТЗ «отдельно протестировать
только производительные / только энергоэффективные / все ядра вместе»
применительно к двум разнородным кластерам X100/A100. Список потоков по
умолчанию — `1,2,4,8` (`16` дополнительно для `mixed`). Для каждого `t`
берутся **первые `t` ядер** пула (contiguous от начала), не
распределяются равномерно — явно задокументированное допущение в коде.

**Известная проблема**: `scaling --target-cluster mixed` не работает —
`taskset: Invalid argument` на любом вызове в рамках этого прогона,
включая тривиальный `threads=1, cpuset=0`. См.
[`07_open_questions.md`](07_open_questions.md#mixed-scaling).

## 4. CPU affinity, повторы, статистика

- Привязка к ядрам — `taskset -c <cpuset>` (не OpenMP affinity/`sched_setaffinity`).
- Каждая точка: `--warmup` прогонов без записи (по умолчанию 1) +
  `--repeats` измеряемых прогонов (по умолчанию 5, соответствует
  требованию ТЗ «не менее пяти раз»).
- По `repeats` прогонам сохраняются `median_mb_s`, `min_mb_s`, `max_mb_s`,
  `stdev_mb_s`, `n` — не только среднее.
- До и после каждой точки снимается `cpu_freq_{before,after}_hz`
  (`/sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq`) и
  `temp_before_c` (`/sys/class/thermal/thermal_zone*`).

## 5. Формат хранения результатов

Один запуск CLI (одна точка `standard`, один sweep `cache_sweep`, одна
серия `scaling`) = один `run_id`:

```
run_id = {variant}_{dtype}_{threads}t_{ws}_{cpuset}_{timestamp}   (standard)
run_id = cache_sweep_{variant}_{dtype}_{threads}t_{cpuset}_{timestamp}
run_id = scaling_{variant}_{dtype}_{target_cluster}_{ws}_{timestamp}
```

`run_id` связывает три файла без сопоставления по времени модификации:

```
results/raw/{type}/{cluster}/{run_id}.json   — первичные данные, вручную не редактируются
results/logs/{type}/{cluster}/{run_id}.log   — stdout+stderr всех repeats-прогонов
results/processed/{type}/{cluster}/{run_id}.png — график, строится из raw в том же вызове
```

Поля raw JSON (см. пример `results/raw/standard/x100/*.json`):
`run_id, machine, benchmark, experiment_type, variant, data_type,
cpu_set, cluster, threads, repeats, array_size_elements/working_set_mb,
compiler, compiler_flags, memory_frequency_hz, aggregate{Copy/Scale/Add/Triad:
{median_mb_s, min_mb_s, max_mb_s, stdev_mb_s, n, theoretical_peak_mb_s,
efficiency_pct}}, cpu_freq_before_hz, cpu_freq_after_hz, temp_before_c`.

Неизвестные значения (`memory_frequency_hz`, отсутствующий `build_info.json`
у старых бинарников) сохраняются как `null`/`"unknown"`, а не подставляются
по догадке — прямое соответствие требованию ТЗ.

## 6. Perf-артефакты — воспроизводимый сбор

`docs/04_perf_profiling.md` построен на артефактах `perf_evidence/`,
которые не редактируются вручную:

```
perf_evidence/asm/         — objdump конкретных кернелов
perf_evidence/perf_stat/   — perf stat по каждому кернелу отдельно (STREAM_KERNEL_ONLY)
perf_evidence/perf_record/ — perf record/report/annotate
perf_evidence/run_outputs/ — вывод диагностических запусков (LoadOnly/StoreOnly/FP16)
perf_evidence/binaries/    — бинарники-источники (в .gitignore, регенерируются)
```

## 7. Актуальные инструменты

В этом репозитории `scripts/` содержит 2 актуальных файла:
`build_dynamic.sh` и `run_bench.py` — оба описаны выше, ими
воспроизводятся все текущие измерения. Часть самых ранних записей
(`results/raw/standard/*_20260818*` и первые точки `cache_sweep`)
собрана более ранними версиями инструментов, которые в текущем
состоянии репозитория уже не присутствуют — это не мешает
воспроизводимости актуальной матрицы экспериментов, но объясняет,
почему у части старых raw JSON нет `build_info.json`
(`compiler`/`compiler_flags` в таких записях — `"unknown"`, см. §5).
