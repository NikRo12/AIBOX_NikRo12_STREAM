# STREAM Benchmark — AIBOX-K3

Исследование пропускной способности памяти на AIBOX-K3 (SpacemiT K3,
RISC-V 64, кластеры X100/A100). Часть группового проекта
`sb26-performance-analysis`;

**Начать здесь:**
- [`docs/stage_02_progress_report.md`](docs/stage_02_progress_report.md) —
  что сделано, чем подтверждено, что осталось; шпаргалка для защиты.
- [`docs/findings/00_SUMMARY.md`](docs/findings/00_SUMMARY.md) — сжатое
  объяснение разрыва между теоретическим и измеренным пиком памяти,
  со ссылками на детальные находки `01`–`05`.

## Структура каталогов

```
docs/
  stage_01_structure.md      — исходная структура проекта (журнал, этап 1)
  stage_01b_vlen_probe.md    — эмпирическая проверка длины векторного регистра
  stage_02_progress_report.md — сводка прогресса + проверка данных + Q&A для защиты
  findings/                  — цепочка находок по разрыву теория/практика (00 = сводка, 01-05 = детали)

src/
  STREAM/stream.c                 — эталонный STREAM (McCalpin), без изменений
  STREAM_dynamic/stream_dynamic.c — тот же STREAM, но размер массива и NTIMES
                                     задаются в рантейме через переменные окружения
  STREAM_dynamic/stream_dynamic_tuned.c — кернелы (copy/scale/add/triad/load_only/
                                     store_only) вынесены в noinline-функции с
                                     собственными именами — нужно для objdump/perf
                                     annotate по конкретному кернелу (см. perf_evidence/)

scripts/
  build_dynamic.sh            — актуальная сборка (builds_dynamic/, march включает zvfh)
  run_bench.py                — актуальный раннер: standard | cache_sweep | scaling

tools/
  vlen_probe/vlen_probe.c     — читает VLEN/VLMAX напрямую из CSR, см. stage_01b

builds/, builds_dynamic/      — скомпилированные бинарники (в .gitignore, не в git,
                                 регенерируются скриптами из scripts/)

results/
  raw/{standard,cache_sweep,scaling}/{x100,a100,mixed}/*.json — первичные данные,
                                 руками не редактируются
  logs/...                    — stdout/stderr соответствующих прогонов
  processed/...                — графики, сгенерированные из raw/ (run_bench.py)
  Все три связаны общим run_id — см. stage_01_structure.md

sysinfo/standard/{x100,a100}/ — снимки cpufreq/temp для ранних (legacy-скрипты)
                                 прогонов standard. Текущий run_bench.py пишет то же
                                 самое (cpu_freq_before_hz/temp_before_c) прямо внутрь
                                 results/raw/*.json — отдельные sysinfo-файлы для
                                 cache_sweep/scaling больше не создаются, это не баг,
                                 а изменившийся формат хранения

perf_evidence/
  asm/         — objdump конкретных кернелов (что доказывает находки 01-04)
  perf_stat/   — perf stat по каждому кернелу отдельно
  perf_record/ — perf record/report/annotate (находка про 80% времени на store)
  run_outputs/ — вывод диагностических запусков (LoadOnly/StoreOnly/FP16-с-zvfh)
  binaries/    — бинарники, из которых получены asm/perf_stat/perf_record
                 (в .gitignore, регенерируются scripts/collect_perf_evidence.sh)
```

## Быстрый старт (на плате AIBOX-K3)

```bash
scripts/build_dynamic.sh                 # собрать всю матрицу variant x dtype
scripts/run_bench.py standard --variant opt_serial --dtype fp64 \
    --cpuset 0 --repeats 5 --ws-mb 200
scripts/run_bench.py cache_sweep --variant novec --dtype fp64 --cpuset 0
scripts/run_bench.py scaling --variant omp --dtype fp64 --target-cluster x100
```

## Известные ограничения (подробности в stage_02_progress_report.md)

- Только AIBOX-K3 — Orange Pi 5 Plus / Banana Pi не измерялись в этой части.
- `scaling --target-cluster mixed` (оба кластера, 0-15) не работает —
  `taskset: Invalid argument`, см. stage_02, раздел 3.6.
- perf/дизассемблер собраны для X100.
