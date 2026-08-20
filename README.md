# STREAM Benchmark — AIBOX-K3

Исследование пропускной способности памяти на AIBOX-K3 (SpacemiT K3,
RISC-V 64, кластеры X100/A100). Часть группового проекта
`sb26-performance-analysis`;

**Начать здесь:** [`docs/08_status_vs_assignment.md`](docs/08_status_vs_assignment.md) —
статус по каждому разделу ТЗ + ответы на контрольные вопросы защиты.

## Документация (по этапам исследования, ТЗ)

| Файл | Раздел ТЗ | Содержание |
|---|---|---|
| [`docs/01_hardware_passport.md`](docs/01_hardware_passport.md) | §2 | Паспорт платформы: SoC, ядра X100/A100, VLEN, кэши, память, теор. пик BW — с пометкой подтверждено/предположение |
| [`docs/02_methodology.md`](docs/02_methodology.md) | §4 | Структура репозитория, матрица сборки, `run_bench.py`, схема `run_id`, формат raw JSON |
| [`docs/03_bandwidth_results.md`](docs/03_bandwidth_results.md) | §3 | Результаты STREAM: standard/cache_sweep/scaling, таблицы, ссылки на графики и raw-данные |
| [`docs/04_perf_profiling.md`](docs/04_perf_profiling.md) | §6 | `perf stat`/`annotate`, IPC, cache miss rate, ключевые фрагменты RVV-ассемблера |
| [`docs/05_root_cause_analysis.md`](docs/05_root_cause_analysis.md) | — | Цепочка причин разрыва теория/практика, внешняя валидация |
| [`docs/06_anomalies_and_fixes.md`](docs/06_anomalies_and_fixes.md) | §9 | FP16 ×88 → исправлено (`zvfh`); Copy=`memcpy()` — реальный эффект vs артефакт |
| [`docs/07_open_questions.md`](docs/07_open_questions.md) | — | Открытые вопросы, известный баг `scaling mixed`, что осталось |
| [`docs/08_status_vs_assignment.md`](docs/08_status_vs_assignment.md) | §9 | Чеклист по разделам ТЗ, ответы на контрольные вопросы |

## Структура каталогов

```
docs/  — см. таблицу выше

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
  vlen_probe/vlen_probe.c     — читает VLEN/VLMAX напрямую из CSR, см. docs/01_hardware_passport.md

builds/, builds_dynamic/      — скомпилированные бинарники (в .gitignore, не в git,
                                 регенерируются скриптами из scripts/)

results/
  raw/{standard,cache_sweep,scaling}/{x100,a100,mixed}/*.json — первичные данные,
                                 руками не редактируются
  logs/...                    — stdout/stderr соответствующих прогонов
  processed/...                — графики, сгенерированные из raw/ (run_bench.py)
  Все три связаны общим run_id — см. docs/02_methodology.md

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
                 (в .gitignore)
```

## Быстрый старт (на плате AIBOX-K3)

```bash
scripts/build_dynamic.sh                 # собрать всю матрицу variant x dtype
scripts/run_bench.py standard --variant opt_serial --dtype fp64 \
    --cpuset 0 --repeats 5 --ws-mb 200
scripts/run_bench.py cache_sweep --variant novec --dtype fp64 --cpuset 0
scripts/run_bench.py scaling --variant omp --dtype fp64 --target-cluster x100
```

## Известные ограничения (подробности в docs/07_open_questions.md)

- Только AIBOX-K3 — Orange Pi 5 Plus / Banana Pi не измерялись в этой части.
- `scaling --target-cluster mixed` (оба кластера, 0-15) не работает —
  `taskset: Invalid argument`, см. [`docs/07_open_questions.md`](docs/07_open_questions.md#scaling---target-cluster-mixed-не-работает).
- perf/дизассемблер собраны для X100, A100 не профилировался.
