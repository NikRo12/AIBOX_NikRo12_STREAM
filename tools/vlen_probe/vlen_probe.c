#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <sched.h>

/* CSR vlenb (0xC22) — read-only, VLEN в байтах. Доступен из user-mode. */
static inline uint64_t read_vlenb(void) {
    uint64_t v;
    asm volatile ("csrr %0, vlenb" : "=r"(v));
    return v;
}

/* vsetvli с rs1=x0 -> vl = VLMAX для заданных SEW/LMUL (спецификация RVV). */
static inline uint64_t vlmax_e8m1(void)  { uint64_t vl; asm volatile ("vsetvli %0, zero, e8,  m1, ta, ma" : "=r"(vl)); return vl; }
static inline uint64_t vlmax_e16m1(void) { uint64_t vl; asm volatile ("vsetvli %0, zero, e16, m1, ta, ma" : "=r"(vl)); return vl; }
static inline uint64_t vlmax_e32m1(void) { uint64_t vl; asm volatile ("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl)); return vl; }
static inline uint64_t vlmax_e64m1(void) { uint64_t vl; asm volatile ("vsetvli %0, zero, e64, m1, ta, ma" : "=r"(vl)); return vl; }

/* Те же SEW, но LMUL=8 — максимальная группировка регистров,
   актуально для микроядер GEMM (этап 8). */
static inline uint64_t vlmax_e32m8(void) { uint64_t vl; asm volatile ("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vl)); return vl; }
static inline uint64_t vlmax_e64m8(void) { uint64_t vl; asm volatile ("vsetvli %0, zero, e64, m8, ta, ma" : "=r"(vl)); return vl; }

int main(void) {
    int cpu = sched_getcpu();
    uint64_t vlenb = read_vlenb();

    printf("=== VLEN probe ===\n");
    printf("running on logical CPU (hart): %d\n", cpu);
    printf("VLENB (CSR)     = %lu bytes\n", vlenb);
    printf("VLEN  (derived) = %lu bits\n", vlenb * 8);
    printf("\n-- VLMAX by SEW, LMUL=1 (empirical via vsetvli) --\n");
    printf("SEW=8  (int8)   : VLMAX = %lu elements\n", vlmax_e8m1());
    printf("SEW=16 (fp16)   : VLMAX = %lu elements\n", vlmax_e16m1());
    printf("SEW=32 (fp32)   : VLMAX = %lu elements\n", vlmax_e32m1());
    printf("SEW=64 (fp64)   : VLMAX = %lu elements\n", vlmax_e64m1());
    printf("\n-- VLMAX with LMUL=8 (grouped registers) --\n");
    printf("SEW=32,LMUL=8 (fp32): VLMAX = %lu elements\n", vlmax_e32m8());
    printf("SEW=64,LMUL=8 (fp64): VLMAX = %lu elements\n", vlmax_e64m8());

    return 0;
}
