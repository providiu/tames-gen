/*
 * tame_phase.c
 * Pollard's kangaroo tame walk: generates the scored DP database
 * for use by kangaroo_wild.c (wild phase).
 *
 * Author: providiu
 * License: MIT (see LICENSE file)
 *
 * Uses modular inversion code by Jean Luc Pons (GPL-3.0)
 * Scored DP technique based on: D.J. Bernstein and T. Lange,
 * "Computing small discrete logarithms faster", INDOCRYPT 2012
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <stddef.h>

#include "G.h"
#include "fast_inv.h"

// ==========================================
// HUGEPAGE ALLOCATION HELPERS
// ==========================================
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

static size_t hp_ht_actual_size = 0;

/* Allocate anonymous hugepage-backed memory. Tries 1GB, 2MB, then regular+THP. */
static void* hugepage_alloc(size_t size, size_t *actual_size) {
    void *p;
    /* Try 1GB hugepages first */
    size_t sz_1g = (size + (1UL << 30) - 1) & ~((1UL << 30) - 1);
    p = mmap(NULL, sz_1g, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB | MAP_POPULATE, -1, 0);
    if (p != MAP_FAILED) {
        printf("[HUGEPAGE] %.2f GB on 1GB hugepages (%zu pages)\n",
               sz_1g / (1024.0*1024*1024), sz_1g >> 30);
        *actual_size = sz_1g;
        return p;
    }
    /* Try 2MB hugepages */
    size_t sz_2m = (size + (2UL << 20) - 1) & ~((2UL << 20) - 1);
    p = mmap(NULL, sz_2m, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB | MAP_POPULATE, -1, 0);
    if (p != MAP_FAILED) {
        printf("[HUGEPAGE] %.2f GB on 2MB hugepages (%zu pages)\n",
               sz_2m / (1024.0*1024*1024), sz_2m >> 21);
        *actual_size = sz_2m;
        return p;
    }
    /* Fallback: regular pages + THP hint */
    p = mmap(NULL, size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
        madvise(p, size, MADV_HUGEPAGE);
#endif
        printf("[HUGEPAGE] Fallback to regular pages + THP (%.2f GB)\n",
               size / (1024.0*1024*1024));
        *actual_size = size;
        return p;
    }
    *actual_size = 0;
    return MAP_FAILED;
}

/* =========================================================
 * COMPILE-TIME CONSTANTS
 * ========================================================= */

#define MAX_BATCH_K       128
#define MAX_NUM_WORKERS   512
#define MAX_DIST_BYTES    16

/* Walk params (defaults, overridable via CLI) */
#define DEFAULT_GLOBAL_BITS      16
#define DEFAULT_LOCAL_BITS        8
#define DEFAULT_JUMP_TABLE_BITS   9
#define DEFAULT_HISTORY_SIZE      4
#define DEFAULT_ESCAPE_TABLE_SIZE 128
#define DEFAULT_ESCAPE_MULT      2000
#define DEFAULT_MIN_DP_STEPS      1
#define DEFAULT_VITA         (1ULL << 23)
#define DEFAULT_TARGET_DP    400000000ULL
#define DEFAULT_BATCH_K      20
#define DEFAULT_SEED         42
#define DEFAULT_TRUNC_BITS    0

#define TRAINING_PARAMS_MAGIC 0x5452414E5041524DULL  /* "TRANPARM" */

/* Hash table segment locking */
#define NUM_SEGMENTS       (1 << 16)   /* 64K segments */
#define SEGMENT_MASK       (NUM_SEGMENTS - 1)

/* Visited DP cap per kangaroo */
#define VISITED_DP_CAP 1024

/* =========================================================
 * TYPES
 * ========================================================= */

typedef struct { uint64_t x[4]; uint64_t y[4]; } Point;
typedef struct { int64_t dist; Point pt; } JumpEntry;

#pragma pack(push, 1)
typedef struct {
    uint32_t global_bits;
    uint32_t range_bits_low;
    uint32_t range_bits_high;
    uint32_t jump_table_seed;
    uint64_t scored_target_dp;
    double   q_hat;
    double   R_factor;
    uint64_t N_sel;
    uint64_t reserved1;
    uint32_t local_bits;
    uint32_t jump_table_bits;
    uint32_t history_size;
    uint32_t escape_table_size;
    uint32_t escape_mult;
    uint32_t min_dp_steps;
    uint32_t hash_index_bits;
    uint32_t trunc_bits;
} TrainingParamsV2;
#pragma pack(pop)

/* Hash table slot: open-addressing with linear probing */
typedef struct {
    uint64_t x_hi;          /* key: x[1] of canonical DP point */
    uint64_t dist_lo;       /* low 64 bits of absolute distance */
    int64_t  dist_hi;       /* high 64 bits (signed) */
    uint32_t count;          /* visit count for scoring */
    uint32_t _pad;
} HTSlot;  /* 32 bytes */

/* For sorting in scored mode */
typedef struct {
    uint64_t x_hi;
    __int128 dist;
    uint32_t count;
    uint32_t fingerprint;
    uint64_t bucket;
} DPEntry;

/* =========================================================
 * EC CONSTANTS
 * ========================================================= */

static const uint64_t Gx[4] = {0x59f2815b16f81798, 0x029bfcdb2dce28d9,
                                0x55a06295ce870b07, 0x79be667ef9dcbbac};
static const uint64_t Gy[4] = {0x9c47d08ffb10d4b8, 0xfd17b448a6855419,
                                0x5da4fbfc0e1108a8, 0x483ada7726a3c465};

static const uint64_t p_half[4] = {0xffffffff7ffffe17, 0xffffffffffffffff,
                                    0xffffffffffffffff, 0x7fffffffffffffff};

/* =========================================================
 * GLOBALS
 * ========================================================= */

/* Parameters (set from CLI) */
static uint32_t g_global_bits      = DEFAULT_GLOBAL_BITS;
static uint32_t g_local_bits       = DEFAULT_LOCAL_BITS;
static uint32_t g_range_bits_low   = 74;
static uint32_t g_range_bits_high  = 75;
static uint32_t g_jump_table_seed  = DEFAULT_SEED;
static uint32_t g_jump_table_bits  = DEFAULT_JUMP_TABLE_BITS;
static uint32_t g_jump_table_size  = (1U << DEFAULT_JUMP_TABLE_BITS);
static uint32_t g_history_size     = DEFAULT_HISTORY_SIZE;
static uint32_t g_escape_table_size = DEFAULT_ESCAPE_TABLE_SIZE;
static uint32_t g_escape_mult      = DEFAULT_ESCAPE_MULT;
static uint32_t g_min_dp_steps     = DEFAULT_MIN_DP_STEPS;
static uint32_t g_trunc_bits       = DEFAULT_TRUNC_BITS;
static uint32_t g_hash_index_bits  = 0;   /* auto */
static uint64_t g_target_dp        = DEFAULT_TARGET_DP;
static uint64_t g_vita             = DEFAULT_VITA;
static int      g_scored           = 0;
static int      g_num_workers      = 0;   /* 0 = auto */
static int      g_batch_k          = DEFAULT_BATCH_K;
static char     g_prefix[256]      = "";

/* Derived */
static uint32_t g_dist_bytes       = 0;
static uint32_t g_local_buf1_size  = 0;
static uint32_t g_local_buf1_mask  = 0;
static __int128 g_midpoint;

/* Jump / escape tables */
static JumpEntry* jump_table  = NULL;
static JumpEntry* escape_table = NULL;

/* Hash table */
static HTSlot*    g_ht        = NULL;
static uint64_t   g_ht_size   = 0;    /* power of 2 */
static uint64_t   g_ht_mask   = 0;

/* Segment spinlocks (cacheline-spaced to avoid false sharing) */
typedef struct { volatile int lock; char pad[60]; } SpinLock;
static SpinLock* g_segments = NULL;

/* Atomic counters */
static volatile uint64_t g_dp_count      = 0;
static volatile uint64_t g_collect_target = 0;  /* set before workers start */
static volatile uint64_t g_total_steps   = 0;
static volatile int       g_shutdown      = 0;

/* Checkpoint */
static int    g_checkpoint_minutes = 30;   /* default: every 30 min */
static int    g_resume             = 0;    /* --resume flag */
static char   g_checkpoint_file[512] = "";

/* Disk-backed hash table */
static int    g_use_disk           = 0;
static char   g_disk_path[1024]    = "";
static char   g_disk_ht_file[2048] = "";
static char   g_disk_meta_file[2048] = "";

#define CHECKPOINT_MAGIC 0x434B50544D455354ULL  /* "CKPTMEST" */

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;
    uint64_t ht_size;        /* must match */
    uint64_t dp_count;       /* DPs collected so far */
    uint64_t total_steps;    /* total steps done */
    uint32_t global_bits;    /* sanity check */
    uint32_t range_low;
    uint32_t range_high;
    uint32_t pad;
} CheckpointHeader;
#pragma pack(pop)

/* Sparse checkpoint: only saves non-empty hash table slots.
 * Format: header + N × { uint64_t slot_index; HTSlot data; }
 * At 0.2% fill: ~600MB instead of 512GB.
 * At 50% fill (final): ~340GB instead of 512GB. */

typedef struct {
    uint64_t idx;
    HTSlot   slot;
} CheckpointEntry;  /* 8 + 32 = 40 bytes */

static void checkpoint_save(const char* path, uint64_t dp_count_now, uint64_t steps_now) {
    char tmp_path[520];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE* f = fopen(tmp_path, "wb");
    if (!f) { printf("[CHECKPOINT] WARNING: cannot open %s for writing\n", tmp_path); return; }
    setvbuf(f, NULL, _IOFBF, 64 * 1024 * 1024);

    CheckpointHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = CHECKPOINT_MAGIC;
    hdr.ht_size     = g_ht_size;
    hdr.dp_count    = dp_count_now;
    hdr.total_steps = steps_now;
    hdr.global_bits = g_global_bits;
    hdr.range_low   = g_range_bits_low;
    hdr.range_high  = g_range_bits_high;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    /* Write only non-empty slots */
    uint64_t written = 0;
    CheckpointEntry ent;
    for (uint64_t i = 0; i < g_ht_size; i++) {
        if (g_ht[i].count > 0) {
            ent.idx = i;
            ent.slot = g_ht[i];
            if (fwrite(&ent, sizeof(ent), 1, f) != 1) goto fail;
            written++;
        }
    }
    fclose(f);

    /* Atomic rename */
    rename(tmp_path, path);
    double file_mb = (sizeof(hdr) + written * sizeof(CheckpointEntry)) / (1024.0*1024);
    printf("[CHECKPOINT] Saved %lu DPs (%lu slots, %.1f MB) to %s\n",
           (unsigned long)dp_count_now, (unsigned long)written, file_mb, path);
    return;

fail:
    printf("[CHECKPOINT] WARNING: write error, checkpoint incomplete\n");
    fclose(f);
    unlink(tmp_path);
}

static int checkpoint_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("[CHECKPOINT] No checkpoint file found: %s\n", path); return 0; }

    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        printf("[CHECKPOINT] ERROR: cannot read header\n"); fclose(f); return -1;
    }
    if (hdr.magic != CHECKPOINT_MAGIC) {
        printf("[CHECKPOINT] ERROR: bad magic\n"); fclose(f); return -1;
    }
    if (hdr.ht_size != g_ht_size) {
        printf("[CHECKPOINT] ERROR: ht_size mismatch (checkpoint=%lu, current=%lu)\n",
               (unsigned long)hdr.ht_size, (unsigned long)g_ht_size);
        fclose(f); return -1;
    }
    if (hdr.global_bits != g_global_bits || hdr.range_low != g_range_bits_low ||
        hdr.range_high != g_range_bits_high) {
        printf("[CHECKPOINT] ERROR: parameter mismatch\n"); fclose(f); return -1;
    }

    /* Read sparse entries and insert into hash table */
    setvbuf(f, NULL, _IOFBF, 64 * 1024 * 1024);
    CheckpointEntry ent;
    uint64_t loaded = 0;
    while (fread(&ent, sizeof(ent), 1, f) == 1) {
        if (ent.idx < g_ht_size) {
            g_ht[ent.idx] = ent.slot;
            loaded++;
        }
    }
    fclose(f);

    g_dp_count = hdr.dp_count;
    printf("[CHECKPOINT] Resumed: %lu DPs (%lu slots) from %s\n",
           (unsigned long)hdr.dp_count, (unsigned long)loaded, path);
    return 1;
}

/* Disk-backed mode: save/load small metadata file */
#define DISK_META_MAGIC 0x444D455441464C45ULL  /* "DMETAFLE" */

#pragma pack(push, 1)
typedef struct {
    uint64_t magic;
    uint64_t dp_count;
    uint64_t total_steps;
    uint64_t ht_size;
    uint32_t global_bits;
    uint32_t range_low;
    uint32_t range_high;
    uint32_t pad;
} DiskMetaHeader;
#pragma pack(pop)

static void disk_meta_save(uint64_t dp_count_now, uint64_t steps_now) {
    char tmp_path[2060];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", g_disk_meta_file);
    FILE* f = fopen(tmp_path, "wb");
    if (!f) { printf("[DISK] WARNING: cannot save metadata\n"); return; }
    DiskMetaHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = DISK_META_MAGIC;
    hdr.dp_count    = dp_count_now;
    hdr.total_steps = steps_now;
    hdr.ht_size     = g_ht_size;
    hdr.global_bits = g_global_bits;
    hdr.range_low   = g_range_bits_low;
    hdr.range_high  = g_range_bits_high;
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);
    rename(tmp_path, g_disk_meta_file);
    printf("[DISK] Metadata saved: %lu DPs\n", (unsigned long)dp_count_now);
}

static int disk_meta_load(void) {
    FILE* f = fopen(g_disk_meta_file, "rb");
    if (!f) { printf("[DISK] No metadata file found: %s\n", g_disk_meta_file); return 0; }
    DiskMetaHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        printf("[DISK] ERROR: cannot read metadata\n"); fclose(f); return -1;
    }
    fclose(f);
    if (hdr.magic != DISK_META_MAGIC) {
        printf("[DISK] ERROR: bad metadata magic\n"); return -1;
    }
    if (hdr.ht_size != g_ht_size) {
        printf("[DISK] ERROR: ht_size mismatch (meta=%lu, current=%lu)\n",
               (unsigned long)hdr.ht_size, (unsigned long)g_ht_size);
        return -1;
    }
    if (hdr.global_bits != g_global_bits || hdr.range_low != g_range_bits_low ||
        hdr.range_high != g_range_bits_high) {
        printf("[DISK] ERROR: parameter mismatch\n"); return -1;
    }
    g_dp_count = hdr.dp_count;
    printf("[DISK] Resumed: %lu DPs from %s\n", (unsigned long)hdr.dp_count, g_disk_meta_file);
    return 1;
}

/* Worker step counters (false-sharing avoidance) */
static volatile uint64_t worker_steps[MAX_NUM_WORKERS * 8];

/* =========================================================
 * INLINE HELPERS
 * ========================================================= */

static inline int cmp256(const uint64_t* a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

static inline int y_is_greater_than_half(const uint64_t* y) {
    if (y[3] > 0x7fffffffffffffffULL) return 1;
    if (y[3] < 0x7fffffffffffffffULL) return 0;
    return cmp256(y, p_half) > 0;
}

static inline void point_copy(Point* dst, const Point* src) {
    memcpy(dst, src, sizeof(Point));
}

static inline int point_is_zero(const Point* p) {
    return !p->x[0] && !p->x[1] && !p->x[2] && !p->x[3];
}

static void scalar_mult(Point* result, const uint64_t* k, const Point* P) {
    (void)P; /* always multiplies G via mul() */
    uint64_t jx[4], jy[4], jz[4], kp[4];
    memcpy(kp, k, 32);
    mul(jx, jy, jz, kp);
    uint64_t invz[4];
    fast_modinv(invz, jz);
    jac_to_aff(result->x, result->y, jx, jy, invz);
}

static void point_add(Point* result, const Point* P, const Point* Q) {
    if (point_is_zero(P)) { point_copy(result, Q); return; }
    if (point_is_zero(Q)) { point_copy(result, P); return; }
    uint64_t jax[4], jay[4], jaz[4] = {1,0,0,0};
    memcpy(jax, P->x, 32);
    memcpy(jay, P->y, 32);
    aecc_add_ja(jax, jay, jaz, Q->x, Q->y);
    uint64_t invz[4];
    fast_modinv(invz, jaz);
    jac_to_aff(result->x, result->y, jax, jay, invz);
}

static void point_neg(Point* result, const Point* P) {
    memcpy(result->x, P->x, 32);
    aecc_sub(result->y, pp, P->y);
}

static void batch_inv_with_scratch(uint64_t invs[][4], uint64_t vals[][4],
                                    int n, uint64_t products[][4]) {
    if (n == 0) return;
    if (n == 1) { fast_modinv(invs[0], vals[0]); return; }
    memcpy(products[0], vals[0], 32);
    for (int i = 1; i < n; i++)
        aecc_mul(products[i], products[i-1], vals[i]);
    uint64_t inv_all[4];
    fast_modinv(inv_all, products[n-1]);
    for (int i = n-1; i > 0; i--) {
        aecc_mul(invs[i], inv_all, products[i-1]);
        aecc_mul(inv_all, inv_all, vals[i]);
    }
    memcpy(invs[0], inv_all, 32);
}

static void batch_point_add(Point* points, __int128* dists,
                             const int* jump_idx, int n, JumpEntry* jt,
                             uint64_t dx[][4], uint64_t inv_dx[][4],
                             uint64_t products[][4]) {
    for (int i = 0; i < n; i++)
        aecc_sub(dx[i], jt[jump_idx[i]].pt.x, points[i].x);
    batch_inv_with_scratch(inv_dx, dx, n, products);
    for (int i = 0; i < n; i++) {
        uint64_t dy[4], lambda[4], lambda2[4], new_x[4], new_y[4], tmp[4];
        aecc_sub(dy, jt[jump_idx[i]].pt.y, points[i].y);
        aecc_mul(lambda, dy, inv_dx[i]);
        aecc_sqr(lambda2, lambda);
        aecc_sub(tmp, lambda2, points[i].x);
        aecc_sub(new_x, tmp, jt[jump_idx[i]].pt.x);
        aecc_sub(tmp, points[i].x, new_x);
        aecc_mul(new_y, lambda, tmp);
        aecc_sub(new_y, new_y, points[i].y);
        memcpy(points[i].x, new_x, 32);
        memcpy(points[i].y, new_y, 32);
        dists[i] += (__int128)jt[jump_idx[i]].dist;
    }
}

static uint64_t rand64(unsigned int* seed) {
    return ((uint64_t)rand_r(seed) << 32) | rand_r(seed);
}

static inline uint64_t xorshift64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static inline uint32_t hash_fingerprint32(uint64_t x_hi) {
    uint64_t h = x_hi * 0xc6a4a7935bd1e995ULL;
    h ^= h >> 32;
    h *= 0x9e3779b97f4a7c15ULL;
    h ^= h >> 29;
    return (uint32_t)(h & 0xFFFFFFFF);
}

static inline uint64_t hash_for_bucket(uint64_t x_hi, uint64_t mask) {
    uint64_t h = x_hi * 0x9e3779b97f4a7c15ULL;
    h ^= h >> 33;
    h *= 0xc6a4a7935bd1e995ULL;
    h ^= h >> 29;
    return h & mask;
}

/* =========================================================
 * SPINLOCK
 * ========================================================= */

static inline void spin_lock(SpinLock* s) {
    while (__sync_lock_test_and_set(&s->lock, 1))
        while (s->lock) __builtin_ia32_pause();
}

static inline void spin_unlock(SpinLock* s) {
    __sync_lock_release(&s->lock);
}

/* =========================================================
 * HASH TABLE OPERATIONS
 * ========================================================= */

/* Hash for HT probe position */
static inline uint64_t ht_hash(uint64_t x_hi) {
    uint64_t h = x_hi;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/*
 * Insert or update a DP in the hash table.
 * Returns 1 if this was a NEW insert, 0 if existing slot was updated (count++).
 */
static int ht_insert(uint64_t x_hi, __int128 dist) {
    uint64_t idx = ht_hash(x_hi) & g_ht_mask;
    uint64_t seg = idx & SEGMENT_MASK;

    spin_lock(&g_segments[seg]);

    for (;;) {
        HTSlot* s = &g_ht[idx];
        if (s->count == 0) {
            /* Empty slot — insert */
            s->x_hi    = x_hi;
            s->dist_lo = (uint64_t)dist;
            s->dist_hi = (int64_t)(dist >> 64);
            s->count   = 1;
            spin_unlock(&g_segments[seg]);
            return 1;
        }
        if (s->x_hi == x_hi) {
            /* Existing — increment count, keep first distance */
            s->count++;
            spin_unlock(&g_segments[seg]);
            return 0;
        }
        idx = (idx + 1) & g_ht_mask;
        /* If we crossed into another segment, we still hold the original lock.
         * For correctness under linear probing this is fine as long as we
         * hold *at least* the segment of the initial probe. */
    }
}

/* =========================================================
 * JUMP TABLE INIT (must match wild phase exactly)
 * ========================================================= */

static void init_jump_table(void) {
    unsigned int seed = g_jump_table_seed;
    long double high_val = powl(2.0L, g_range_bits_high);
    long double low_val  = powl(2.0L, g_range_bits_low);
    double W = (double)(high_val - low_val) / 2.0;

    double gap = W / (double)g_target_dp;
    uint64_t opt = (uint64_t)(gap) / sqrt((double)(1ULL << g_global_bits));
    if (opt < 1) opt = 1;

    printf("[JUMP_TABLE] target_dp=%lu gap=%.2e opt=%lu (2^%.1f)\n",
           (unsigned long)g_target_dp, gap, (unsigned long)opt,
           opt > 0 ? log2((double)opt) : 0);

    Point G;
    memcpy(G.x, Gx, 32);
    memcpy(G.y, Gy, 32);

    for (uint32_t i = 0; i < g_jump_table_size; i++) {
        uint64_t d = (opt / 2) + (rand64(&seed) % opt);
        if (d == 0) d = 1;
        jump_table[i].dist = d;
        uint64_t k[4] = {d, 0, 0, 0};
        scalar_mult(&jump_table[i].pt, k, &G);
    }
    for (uint32_t i = 0; i < g_escape_table_size; i++) {
        uint64_t d = (opt * g_escape_mult) + (rand64(&seed) % (opt * g_escape_mult));
        escape_table[i].dist = d;
        uint64_t k[4] = {d, 0, 0, 0};
        scalar_mult(&escape_table[i].pt, k, &G);
    }
}

/* =========================================================
 * SIGNAL HANDLER
 * ========================================================= */

static void sigint_handler(int sig) {
    (void)sig;
    printf("\n[SIGINT] Ctrl+C received, shutting down...\n");
    g_shutdown = 1;
}

/* =========================================================
 * WORKER THREAD (TAME WALK)
 * ========================================================= */

typedef struct {
    int wid;
} WorkerArg;

static void* tame_worker(void* arg) {
    WorkerArg* wa = (WorkerArg*)arg;
    int wid = wa->wid;
    int BATCH = g_batch_k;

    uint64_t rng = (uint64_t)wid * 0x9e3779b97f4a7c15ULL + time(NULL) + (uint64_t)wid * 12345;

    /* Local copy of jump table */
    JumpEntry* local_jt = malloc(g_jump_table_size * sizeof(JumpEntry));
    memcpy(local_jt, jump_table, sizeof(JumpEntry) * g_jump_table_size);

    uint64_t g_mask     = (1ULL << g_global_bits) - 1;
    uint64_t l_mask     = (1ULL << g_local_bits) - 1;
    uint64_t table_mask = g_jump_table_size - 1;
    uint64_t esc_mask   = g_escape_table_size - 1;

    Point    *cx  = calloc(BATCH, sizeof(Point));
    __int128 *cd  = calloc(BATCH, sizeof(__int128));
    int      *life = calloc(BATCH, sizeof(int));
    int      *jump_idx = calloc(BATCH, sizeof(int));
    int      *needs_respawn = calloc(BATCH, sizeof(int));
    uint64_t *steps_since_respawn = calloc(BATCH, sizeof(uint64_t));

    /* Scratch buffers for batch point addition */
    uint64_t (*batch_dx)[4]      = calloc(BATCH, sizeof(uint64_t[4]));
    uint64_t (*batch_inv_dx)[4]  = calloc(BATCH, sizeof(uint64_t[4]));
    uint64_t (*batch_products)[4]= calloc(BATCH, sizeof(uint64_t[4]));

    /* History buffer */
    Point    *hist_pt_flat   = calloc((size_t)BATCH * g_history_size, sizeof(Point));
    __int128 *hist_dist_flat = calloc((size_t)BATCH * g_history_size, sizeof(__int128));
    int      *hist_idx       = calloc(BATCH, sizeof(int));
    #define HIST_PT(i, h)   hist_pt_flat[(size_t)(i) * g_history_size + (h)]
    #define HIST_DIST(i, h) hist_dist_flat[(size_t)(i) * g_history_size + (h)]

    uint64_t *last_canonical_x0 = calloc(BATCH, sizeof(uint64_t));
    uint64_t *last_canonical_x1 = calloc(BATCH, sizeof(uint64_t));

    /* local_buf1 */
    uint64_t *local_buf1_flat = calloc((size_t)BATCH * g_local_buf1_size, sizeof(uint64_t));
    int      *local_buf1_idx  = calloc(BATCH, sizeof(int));
    #define LB1(i, j) local_buf1_flat[(size_t)(i) * g_local_buf1_size + (j)]

    int *escape_idx_arr       = calloc(BATCH, sizeof(int));
    int *consecutive_escapes  = calloc(BATCH, sizeof(int));

    /* Visited DP tracking */
    uint64_t (*visited_dp)[VISITED_DP_CAP] = calloc(BATCH, sizeof(*visited_dp));
    int *visited_dp_cnt = calloc(BATCH, sizeof(int));

    Point G_pt;
    memcpy(G_pt.x, Gx, 32);
    memcpy(G_pt.y, Gy, 32);

    /* Midpoint as __int128 */
    __int128 mid = g_midpoint;

    /* Half range for uniform random offset */
    unsigned __int128 hi_val = (unsigned __int128)1 << g_range_bits_high;
    unsigned __int128 lo_val = (unsigned __int128)1 << g_range_bits_low;
    unsigned __int128 half_range = (hi_val - lo_val) / 2;

    /* Mark all kangaroos for initial spawn */
    for (int i = 0; i < BATCH; i++) {
        needs_respawn[i] = 1;
        life[i] = 0;
        hist_idx[i] = 0;
    }

    uint64_t steps = 0;

    while (!g_shutdown) {
        /* Check if we've reached collect target */
        if (__sync_fetch_and_add(&g_dp_count, 0) >= g_collect_target) {
            break;
        }

        /* Respawn kangaroos that need it */
        int respawn_count = 0;
        for (int i = 0; i < BATCH; i++) {
            if (needs_respawn[i]) {
                respawn_count++;
                steps_since_respawn[i] = 1;
                visited_dp_cnt[i] = 0;

                /* Tame walk: start at midpoint + uniform random offset.
                 * cd[i] = midpoint + offset (ABSOLUTE scalar) */
                unsigned __int128 r = ((unsigned __int128)xorshift64(&rng) << 64)
                                      | xorshift64(&rng);
                __int128 offset = (__int128)(r % ((unsigned __int128)half_range * 2 + 1))
                                  - (__int128)half_range;
                __int128 abs_d = mid + offset;
                cd[i] = abs_d;

                /* Compute abs_d * G */
                __int128 ad = (abs_d >= 0) ? abs_d : -abs_d;
                uint64_t k[4] = {0,0,0,0};
                k[0] = (uint64_t)ad;
                k[1] = (uint64_t)((unsigned __int128)ad >> 64);
                scalar_mult(&cx[i], k, &G_pt);
                if (abs_d < 0)
                    point_neg(&cx[i], &cx[i]);

                /* Y-canonicalization */
                if (y_is_greater_than_half(cx[i].y)) {
                    aecc_sub(cx[i].y, pp, cx[i].y);
                    cd[i] = -cd[i];
                }

                life[i] = 1;
                hist_idx[i] = 0;
                memset(&HIST_PT(i, 0), 0, sizeof(Point) * g_history_size);
                memset(&HIST_DIST(i, 0), 0, sizeof(__int128) * g_history_size);
                jump_idx[i] = ((cx[i].x[0] * 0x9e3779b97f4a7c15ULL) >> g_global_bits) & table_mask;
                local_buf1_idx[i] = 0;
                memset(&LB1(i, 0), 0, g_local_buf1_size * sizeof(uint64_t));
                last_canonical_x0[i] = 0;
                last_canonical_x1[i] = 0;
                escape_idx_arr[i] = 0;
                consecutive_escapes[i] = 0;
                needs_respawn[i] = 0;
            }
        }

        /* Batch point addition */
        batch_point_add(cx, cd, jump_idx, BATCH, local_jt,
                        batch_dx, batch_inv_dx, batch_products);

        /* Post-step processing */
        for (int i = 0; i < BATCH; i++) {
            /* Y-canonicalization */
            if (y_is_greater_than_half(cx[i].y)) {
                aecc_sub(cx[i].y, pp, cx[i].y);
                cd[i] = -cd[i];
            }
            life[i]++;
            steps_since_respawn[i]++;

            /* Life limit */
            if (life[i] >= (int)g_vita) {
                needs_respawn[i] = 1;
                continue;
            }

            uint64_t xv = cx[i].x[0];

            /* History cycle detection */
            int in_hist = 0;
            int found_hist_idx = -1;
            uint64_t curr_x0 = cx[i].x[0], curr_x1 = cx[i].x[1];
            for (uint32_t h = 0; h < g_history_size; h++) {
                if (HIST_PT(i, h).x[0] == curr_x0 &&
                    HIST_PT(i, h).x[1] == curr_x1 &&
                    (curr_x0 || curr_x1)) {
                    in_hist = 1;
                    found_hist_idx = h;
                    break;
                }
            }

            if (in_hist) {
                if (consecutive_escapes[i] >= 16) {
                    needs_respawn[i] = 1;
                    continue;
                }
                /* Find canonical point in cycle */
                uint64_t min_x0 = curr_x0, min_x1 = curr_x1;
                __int128 canonical_dist = cd[i];
                Point canonical_pt;
                point_copy(&canonical_pt, &cx[i]);
                int idx = found_hist_idx;
                while (idx != hist_idx[i]) {
                    if (HIST_PT(i, idx).x[1] < min_x1 ||
                        (HIST_PT(i, idx).x[1] == min_x1 &&
                         HIST_PT(i, idx).x[0] < min_x0)) {
                        min_x0 = HIST_PT(i, idx).x[0];
                        min_x1 = HIST_PT(i, idx).x[1];
                        canonical_dist = HIST_DIST(i, idx);
                        point_copy(&canonical_pt, &HIST_PT(i, idx));
                    }
                    idx = (idx + 1) % g_history_size;
                }
                if (min_x0 == last_canonical_x0[i] &&
                    min_x1 == last_canonical_x1[i]) {
                    escape_idx_arr[i]++;
                }
                last_canonical_x0[i] = min_x0;
                last_canonical_x1[i] = min_x1;
                int ei = (escape_idx_arr[i] + (min_x0 >> 40)) & esc_mask;
                point_add(&cx[i], &canonical_pt, &escape_table[ei].pt);
                cd[i] = canonical_dist + escape_table[ei].dist;
                if (y_is_greater_than_half(cx[i].y)) {
                    aecc_sub(cx[i].y, pp, cx[i].y);
                    cd[i] = -cd[i];
                }
                consecutive_escapes[i]++;
                hist_idx[i] = 0;
                memset(&HIST_PT(i, 0), 0, sizeof(Point) * g_history_size);
                memset(&HIST_DIST(i, 0), 0, sizeof(__int128) * g_history_size);
                xv = cx[i].x[0];
            }

            if (!in_hist) consecutive_escapes[i] = 0;

            /* Update history */
            point_copy(&HIST_PT(i, hist_idx[i]), &cx[i]);
            HIST_DIST(i, hist_idx[i]) = cd[i];
            hist_idx[i] = (hist_idx[i] + 1) % g_history_size;

            /* Local buf1 cycle detection */
            if ((xv & l_mask) == 0 && (xv & g_mask) != 0) {
                uint64_t fp = cx[i].x[0] * 0x9e3779b97f4a7c15ULL
                            ^ cx[i].x[1] * 0xc6a4a7935bd1e995ULL;
                fp ^= fp >> 33;
                int in_buf1 = 0;
                for (uint32_t j = 0; j < g_local_buf1_size; j++) {
                    if (LB1(i, j) == fp && fp != 0) { in_buf1 = 1; break; }
                }
                if (in_buf1) {
                    needs_respawn[i] = 1;
                    continue;
                }
                LB1(i, local_buf1_idx[i]) = fp;
                local_buf1_idx[i] = (local_buf1_idx[i] + 1) % g_local_buf1_size;
            }

            /* DP check */
            if ((xv & g_mask) == 0) {
                /* Check visited DP */
                uint64_t xhi = cx[i].x[1];
                int already = 0;
                for (int v = 0; v < visited_dp_cnt[i]; v++) {
                    if (visited_dp[i][v] == xhi) { already = 1; break; }
                }
                if (already) {
                    needs_respawn[i] = 1;
                    continue;
                }
                if (visited_dp_cnt[i] < VISITED_DP_CAP)
                    visited_dp[i][visited_dp_cnt[i]++] = xhi;

                /* Insert into hash table. Store RELATIVE offset from midpoint,
                   because kangaroo_wild collision formula is:
                   base_key = td[m] + midpoint - pend_dist[k] */
                int is_new = ht_insert(xhi, cd[i] - g_midpoint);
                if (is_new) {
                    __sync_fetch_and_add(&g_dp_count, 1);
                }
            }

            jump_idx[i] = ((xv * 0x9e3779b97f4a7c15ULL) >> g_global_bits) & table_mask;
        }

        steps += BATCH + respawn_count;
        worker_steps[wid * 8] = steps;
    }

    #undef LB1
    #undef HIST_PT
    #undef HIST_DIST

    free(local_jt);
    free(cx); free(cd); free(life); free(jump_idx);
    free(needs_respawn); free(steps_since_respawn);
    free(batch_dx); free(batch_inv_dx); free(batch_products);
    free(hist_pt_flat); free(hist_dist_flat); free(hist_idx);
    free(last_canonical_x0); free(last_canonical_x1);
    free(local_buf1_flat); free(local_buf1_idx);
    free(escape_idx_arr); free(consecutive_escapes);
    free(visited_dp); free(visited_dp_cnt);

    return NULL;
}

/* =========================================================
 * SORTING COMPARATORS
 * ========================================================= */

static int cmp_by_count_desc(const void* a, const void* b) {
    const DPEntry* ea = (const DPEntry*)a;
    const DPEntry* eb = (const DPEntry*)b;
    if (ea->count > eb->count) return -1;
    if (ea->count < eb->count) return 1;
    return 0;
}

static int cmp_by_bucket(const void* a, const void* b) {
    const DPEntry* ea = (const DPEntry*)a;
    const DPEntry* eb = (const DPEntry*)b;
    if (ea->bucket < eb->bucket) return -1;
    if (ea->bucket > eb->bucket) return 1;
    /* Tie-break by fingerprint for deterministic order */
    if (ea->fingerprint < eb->fingerprint) return -1;
    if (ea->fingerprint > eb->fingerprint) return 1;
    return 0;
}

/* =========================================================
 * WRITE OUTPUT FILES
 * ========================================================= */

static void write_output(DPEntry* entries, uint64_t count, uint64_t hash_index_size,
                          uint64_t hash_index_mask, uint32_t hash_index_bits,
                          double q_hat, double R_factor) {
    char fname[512];

    /* Compute dist_bytes */
    uint32_t dist_bytes = g_dist_bytes;

    printf("[OUTPUT] Writing %lu entries (dist_bytes=%u, trunc=%u)\n",
           (unsigned long)count, dist_bytes, g_trunc_bits);

    /* ---- training_params.bin ----
     * Uses "{HIGH}_training_{GLOBAL}_params.bin" naming
     * (different pattern from the other files which use g_prefix). */
    snprintf(fname, sizeof(fname), "%u_training_%u_params.bin",
             g_range_bits_high, g_global_bits);
    printf("[OUTPUT] %s\n", fname);
    FILE* f = fopen(fname, "wb");
    if (!f) { perror("fopen training_params"); exit(1); }
    uint64_t magic = TRAINING_PARAMS_MAGIC;
    fwrite(&magic, sizeof(uint64_t), 1, f);
    TrainingParamsV2 tp;
    memset(&tp, 0, sizeof(tp));
    tp.global_bits      = g_global_bits;
    tp.range_bits_low   = g_range_bits_low;
    tp.range_bits_high  = g_range_bits_high;
    tp.jump_table_seed  = g_jump_table_seed;
    tp.scored_target_dp = count;
    tp.q_hat            = q_hat;
    tp.R_factor         = R_factor;
    tp.N_sel            = count;
    tp.reserved1        = 0;
    tp.local_bits       = g_local_bits;
    tp.jump_table_bits  = g_jump_table_bits;
    tp.history_size     = g_history_size;
    tp.escape_table_size = g_escape_table_size;
    tp.escape_mult      = g_escape_mult;
    tp.min_dp_steps     = g_min_dp_steps;
    tp.hash_index_bits  = hash_index_bits;
    tp.trunc_bits       = g_trunc_bits;
    fwrite(&tp, sizeof(TrainingParamsV2), 1, f);
    fclose(f);

    /* ---- Sort entries by bucket for output ---- */
    for (uint64_t i = 0; i < count; i++) {
        entries[i].fingerprint = hash_fingerprint32(entries[i].x_hi);
        entries[i].bucket = hash_for_bucket(entries[i].x_hi, hash_index_mask);
    }
    printf("[OUTPUT] Sorting %lu entries by bucket...\n", (unsigned long)count);
    qsort(entries, count, sizeof(DPEntry), cmp_by_bucket);

    /* ---- tame_db.bin ---- */
    snprintf(fname, sizeof(fname), "%s_tame_db.bin", g_prefix);
    printf("[OUTPUT] %s\n", fname);
    f = fopen(fname, "wb");
    if (!f) { perror("fopen tame_db"); exit(1); }
    /* Use large stdio buffer for write performance */
    setvbuf(f, NULL, _IOFBF, 4 * 1024 * 1024);
    uint8_t buf[MAX_DIST_BYTES];
    for (uint64_t i = 0; i < count; i++) {
        __int128 d = entries[i].dist;
        /* Truncate low bits (arithmetic right shift) */
        if (g_trunc_bits > 0) {
            d >>= g_trunc_bits;
        }
        /* Write as little-endian signed (two's complement).
         * On little-endian systems, __int128 is already two's complement,
         * so we just extract the low dist_bytes directly. */
        memset(buf, 0, sizeof(buf));
        unsigned __int128 raw = (unsigned __int128)d;
        for (uint32_t b = 0; b < dist_bytes; b++)
            buf[b] = (uint8_t)(raw >> (b * 8));
        fwrite(buf, dist_bytes, 1, f);
    }
    fclose(f);

    /* ---- fingerprints.bin ---- */
    snprintf(fname, sizeof(fname), "%s_fingerprints.bin", g_prefix);
    printf("[OUTPUT] %s\n", fname);
    f = fopen(fname, "wb");
    if (!f) { perror("fopen fingerprints"); exit(1); }
    setvbuf(f, NULL, _IOFBF, 4 * 1024 * 1024);
    for (uint64_t i = 0; i < count; i++) {
        uint32_t fp = entries[i].fingerprint;
        fwrite(&fp, sizeof(uint32_t), 1, f);
    }
    fclose(f);

    /* ---- bucket_offsets.bin ---- */
    snprintf(fname, sizeof(fname), "%s_bucket_offsets.bin", g_prefix);
    printf("[OUTPUT] %s\n", fname);
    f = fopen(fname, "wb");
    if (!f) { perror("fopen bucket_offsets"); exit(1); }
    int use_u32 = (count <= 0xFFFFFFFFULL);
    /* Build bucket offsets by scanning sorted entries */
    if (use_u32) {
        uint32_t* boff = calloc(hash_index_size + 1, sizeof(uint32_t));
        if (!boff) { perror("calloc bucket_offsets"); exit(1); }
        /* Count entries per bucket */
        for (uint64_t i = 0; i < count; i++) {
            uint64_t b = entries[i].bucket;
            boff[b + 1]++;
        }
        /* Prefix sum */
        for (uint64_t b = 1; b <= hash_index_size; b++)
            boff[b] += boff[b - 1];
        fwrite(boff, sizeof(uint32_t), hash_index_size + 1, f);
        free(boff);
    } else {
        uint64_t* boff = calloc(hash_index_size + 1, sizeof(uint64_t));
        if (!boff) { perror("calloc bucket_offsets"); exit(1); }
        for (uint64_t i = 0; i < count; i++) {
            uint64_t b = entries[i].bucket;
            boff[b + 1]++;
        }
        for (uint64_t b = 1; b <= hash_index_size; b++)
            boff[b] += boff[b - 1];
        fwrite(boff, sizeof(uint64_t), hash_index_size + 1, f);
        free(boff);
    }
    fclose(f);

    printf("[OUTPUT] Done. Bucket offsets: %s (%lu buckets)\n",
           use_u32 ? "uint32" : "uint64", (unsigned long)hash_index_size);
}

/* =========================================================
 * HELP
 * ========================================================= */

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("  -n N          Target DP count (default: %lu)\n", (unsigned long)DEFAULT_TARGET_DP);
    printf("  -g BITS       Global bits / DP mask (default: %u)\n", DEFAULT_GLOBAL_BITS);
    printf("  -r LOW HIGH   Range bits (default: 74 75)\n");
    printf("  -w N          Worker threads (default: auto-detect)\n");
    printf("  -b N          Batch size per worker (default: %d)\n", DEFAULT_BATCH_K);
    printf("  -t BITS       Truncation bits for distances (default: 0)\n");
    printf("  -H BITS       Hash index bits for bucket offsets (default: auto)\n");
    printf("  --scored      Enable scoring (select top N by visit count)\n");
    printf("  --seed S      Jump table seed (default: %u)\n", DEFAULT_SEED);
    printf("  --prefix P    Output file prefix (default: auto)\n");
    printf("  --vita N      Life limit per walk (default: %lu)\n", (unsigned long)DEFAULT_VITA);
    printf("  --checkpoint M  Checkpoint interval in minutes (default: 30)\n");
    printf("  --resume      Resume from checkpoint file\n");
    printf("  help          Show this help\n");
}

/* =========================================================
 * MAIN
 * ========================================================= */

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGINT, sigint_handler);

    /* Parse CLI */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "help") || !strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_help(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            g_target_dp = strtoull(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "-g") && i + 1 < argc) {
            g_global_bits = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 2 < argc) {
            g_range_bits_low  = atoi(argv[++i]);
            g_range_bits_high = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            g_num_workers = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-b") && i + 1 < argc) {
            g_batch_k = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            g_trunc_bits = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-H") && i + 1 < argc) {
            g_hash_index_bits = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--scored")) {
            g_scored = 1;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            g_jump_table_seed = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) {
            strncpy(g_prefix, argv[++i], sizeof(g_prefix) - 1);
        } else if (!strcmp(argv[i], "--vita") && i + 1 < argc) {
            g_vita = strtoull(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--resume")) {
            g_resume = 1;
        } else if (!strcmp(argv[i], "--checkpoint") && i + 1 < argc) {
            g_checkpoint_minutes = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--disk") && i + 1 < argc) {
            strncpy(g_disk_path, argv[++i], sizeof(g_disk_path) - 1);
            g_use_disk = 1;
        } else {
            fprintf(stderr, "[WARN] Unknown argument: %s\n", argv[i]);
        }
    }

    /* Validate */
    if (g_range_bits_high <= g_range_bits_low) {
        fprintf(stderr, "[ERROR] range_bits_high (%u) must be > range_bits_low (%u)\n",
                g_range_bits_high, g_range_bits_low);
        return 1;
    }
    if (g_global_bits < 1 || g_global_bits > 30) {
        fprintf(stderr, "[ERROR] global_bits (%u) out of range [1,30]\n", g_global_bits);
        return 1;
    }
    if (g_batch_k < 1) g_batch_k = 1;
    if (g_batch_k > MAX_BATCH_K) g_batch_k = MAX_BATCH_K;

    /* Auto-detect workers */
    if (g_num_workers <= 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpu < 1) ncpu = 1;
        g_num_workers = (int)(ncpu > MAX_NUM_WORKERS ? MAX_NUM_WORKERS : ncpu);
        printf("[AUTO] Detected %ld CPUs, using %d workers\n", ncpu, g_num_workers);
    }
    if (g_num_workers > MAX_NUM_WORKERS) g_num_workers = MAX_NUM_WORKERS;

    /* Auto prefix */
    if (g_prefix[0] == '\0') {
        snprintf(g_prefix, sizeof(g_prefix), "%u_scored_%u",
                 g_range_bits_high, g_global_bits);
    }

    /* Derived params */
    g_jump_table_size = 1U << g_jump_table_bits;
    if (g_trunc_bits > 0)
        g_dist_bytes = (g_range_bits_high - g_trunc_bits + 7) / 8;
    else
        g_dist_bytes = (g_range_bits_high + 7) / 8;
    if (g_dist_bytes > MAX_DIST_BYTES) {
        fprintf(stderr, "[ERROR] dist_bytes=%u exceeds MAX_DIST_BYTES=%d\n",
                g_dist_bytes, MAX_DIST_BYTES);
        return 1;
    }

    g_local_buf1_size = ((1U << (g_global_bits - g_local_bits)) * 4);
    g_local_buf1_mask = g_local_buf1_size - 1;

    /* Midpoint = (2^high + 2^low) / 2 */
    unsigned __int128 hi_val = (unsigned __int128)1 << g_range_bits_high;
    unsigned __int128 lo_val = (unsigned __int128)1 << g_range_bits_low;
    g_midpoint = (__int128)((hi_val + lo_val) / 2);

    /* Auto hash_index_bits */
    if (g_hash_index_bits == 0) {
        /* At least 20, sized so avg bucket < 16 entries */
        uint64_t needed = g_target_dp / 8;
        g_hash_index_bits = 20;
        while ((1ULL << g_hash_index_bits) < needed && g_hash_index_bits < 32)
            g_hash_index_bits++;
    }
    uint64_t hash_index_size = 1ULL << g_hash_index_bits;
    uint64_t hash_index_mask = hash_index_size - 1;

    printf("============================================================\n");
    printf("  TAME PHASE - Scored DP Database Generator\n");
    printf("============================================================\n");
    printf("  Range:          2^%u - 2^%u\n", g_range_bits_low, g_range_bits_high);
    printf("  Global bits:    %u (DP mask: 0x%lx)\n", g_global_bits,
           (unsigned long)((1ULL << g_global_bits) - 1));
    printf("  Target DPs:     %lu (%.1fM)\n", (unsigned long)g_target_dp,
           g_target_dp / 1e6);
    printf("  Scored mode:    %s\n", g_scored ? "YES" : "NO");
    printf("  Workers:        %d x %d = %d kangaroos\n",
           g_num_workers, g_batch_k, g_num_workers * g_batch_k);
    printf("  Jump table:     %u entries (bits=%u, seed=%u)\n",
           g_jump_table_size, g_jump_table_bits, g_jump_table_seed);
    printf("  Escape table:   %u entries (mult=%u)\n",
           g_escape_table_size, g_escape_mult);
    printf("  History:        %u\n", g_history_size);
    printf("  Local buf1:     %u\n", g_local_buf1_size);
    printf("  Vita (life):    %lu\n", (unsigned long)g_vita);
    printf("  Dist bytes:     %u (trunc=%u)\n", g_dist_bytes, g_trunc_bits);
    printf("  Hash index:     %u bits (%lu buckets)\n",
           g_hash_index_bits, (unsigned long)hash_index_size);
    printf("  Prefix:         %s\n", g_prefix);
    if (g_use_disk)
        printf("  Disk mode:      %s\n", g_disk_path);
    printf("  Midpoint:       0x%lx%016lx\n",
           (unsigned long)(uint64_t)((unsigned __int128)g_midpoint >> 64),
           (unsigned long)(uint64_t)g_midpoint);
    printf("============================================================\n");

    /* Allocate jump/escape tables */
    jump_table   = calloc(g_jump_table_size, sizeof(JumpEntry));
    escape_table = calloc(g_escape_table_size, sizeof(JumpEntry));
    if (!jump_table || !escape_table) {
        fprintf(stderr, "[ERROR] Cannot allocate jump/escape tables\n");
        return 1;
    }

    printf("[INIT] Building jump table...\n");
    init_jump_table();

    /* Allocate hash table.
     * For scored mode, we want more DPs than target (collect ~3x target).
     * For unscored mode, collect exactly target. */
    uint64_t collect_target = g_scored ? g_target_dp * 3 : g_target_dp;
    g_collect_target = collect_target;
    /* HT size: next power of 2 above collect_target * 2 for ~50% load */
    g_ht_size = 1;
    while (g_ht_size < collect_target * 2)
        g_ht_size <<= 1;
    g_ht_mask = g_ht_size - 1;

    size_t ht_bytes = (size_t)g_ht_size * sizeof(HTSlot);
    printf("[INIT] Allocating hash table: %lu slots (%.2f GB)\n",
           (unsigned long)g_ht_size, ht_bytes / (1024.0 * 1024.0 * 1024.0));

    if (g_use_disk) {
        /* ---- DISK-BACKED HASH TABLE ---- */
        snprintf(g_disk_ht_file, sizeof(g_disk_ht_file),
                 "%s/%s_hashtable.bin", g_disk_path, g_prefix);
        snprintf(g_disk_meta_file, sizeof(g_disk_meta_file),
                 "%s/%s_meta.bin", g_disk_path, g_prefix);

        int is_resume_disk = 0;
        int fd = -1;

        if (g_resume) {
            fd = open(g_disk_ht_file, O_RDWR);
            if (fd >= 0) {
                struct stat st;
                fstat(fd, &st);
                if ((size_t)st.st_size == ht_bytes) {
                    is_resume_disk = 1;
                    printf("[DISK] Resuming from existing hash table: %s (%.2f TB)\n",
                           g_disk_ht_file, ht_bytes / (1024.0*1024*1024*1024));
                } else {
                    printf("[DISK] File size mismatch (file=%ld, expected=%zu), creating new\n",
                           (long)st.st_size, ht_bytes);
                    close(fd);
                    fd = -1;
                }
            }
        }

        if (fd < 0) {
            printf("[DISK] Creating hash table file: %s (%.2f TB)\n",
                   g_disk_ht_file, ht_bytes / (1024.0*1024*1024*1024));
            fd = open(g_disk_ht_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                perror("[DISK] Cannot create hash table file");
                return 1;
            }
            printf("[DISK] Allocating file with fallocate...\n");
            if (fallocate(fd, 0, 0, ht_bytes) < 0) {
                printf("[DISK] fallocate failed, trying ftruncate...\n");
                if (ftruncate(fd, ht_bytes) < 0) {
                    perror("[DISK] Cannot set file size");
                    close(fd);
                    return 1;
                }
            }
            printf("[DISK] File allocated successfully\n");
        }

        g_ht = mmap(NULL, ht_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (g_ht == MAP_FAILED) {
            perror("[DISK] mmap failed");
            return 1;
        }
        hp_ht_actual_size = ht_bytes;

        /* No memset needed: fallocate/ftruncate gives zeros, resume has data */
        if (!is_resume_disk) {
            printf("[DISK] Hash table zero-filled by filesystem\n");
        }

        /* Resume: load metadata */
        if (is_resume_disk) {
            int rc = disk_meta_load();
            if (rc < 0) {
                fprintf(stderr, "[ERROR] Failed to load disk metadata, aborting\n");
                return 1;
            }
            if (rc > 0) {
                printf("[DISK] Resuming from %lu / %lu DPs (%.1f%%)\n",
                       (unsigned long)g_dp_count, (unsigned long)collect_target,
                       g_dp_count * 100.0 / collect_target);
            }
        }
    } else {
        /* ---- RAM-BACKED HASH TABLE (original) ---- */
        g_ht = hugepage_alloc(ht_bytes, &hp_ht_actual_size);
        if (g_ht == MAP_FAILED) {
            g_ht = mmap(NULL, ht_bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_ht == MAP_FAILED) {
                perror("mmap hash table");
                return 1;
            }
            hp_ht_actual_size = ht_bytes;
        }
        memset(g_ht, 0, ht_bytes);

        /* Build checkpoint filename */
        snprintf(g_checkpoint_file, sizeof(g_checkpoint_file),
                 "%s_checkpoint.bin", g_prefix);

        /* Resume from checkpoint if requested */
        if (g_resume) {
            int rc = checkpoint_load(g_checkpoint_file);
            if (rc < 0) {
                fprintf(stderr, "[ERROR] Failed to load checkpoint, aborting\n");
                return 1;
            }
            if (rc > 0) {
                printf("[CHECKPOINT] Resuming from %lu / %lu DPs (%.1f%%)\n",
                       (unsigned long)g_dp_count, (unsigned long)collect_target,
                       g_dp_count * 100.0 / collect_target);
            }
        }
    }

    /* Allocate segment spinlocks */
    g_segments = calloc(NUM_SEGMENTS, sizeof(SpinLock));
    if (!g_segments) {
        fprintf(stderr, "[ERROR] Cannot allocate segment locks\n");
        return 1;
    }

    /* Launch workers */
    printf("[RUN] Starting %d workers...\n", g_num_workers);
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    memset((void*)worker_steps, 0, sizeof(worker_steps));

    pthread_t threads[MAX_NUM_WORKERS];
    WorkerArg wargs[MAX_NUM_WORKERS];
    for (int i = 0; i < g_num_workers; i++) {
        wargs[i].wid = i;
        pthread_create(&threads[i], NULL, tame_worker, &wargs[i]);
    }

    /* Progress reporting loop */
    uint64_t last_dp = 0;
    struct timespec last_time = ts;
    double last_checkpoint_time = 0;
    double checkpoint_interval = g_checkpoint_minutes * 60.0;
    if (g_use_disk)
        printf("[DISK] Auto-sync every %d minutes (no worker pause needed)\n",
               g_checkpoint_minutes);
    else
        printf("[CHECKPOINT] Auto-save every %d minutes to %s\n",
               g_checkpoint_minutes, g_checkpoint_file);

    while (!g_shutdown) {
        usleep(5000000);  /* 5 seconds */
        if (g_shutdown) break;

        uint64_t dp_now = __sync_fetch_and_add(&g_dp_count, 0);
        if (dp_now >= collect_target) break;

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - ts.tv_sec) + (now.tv_nsec - ts.tv_nsec) / 1e9;
        double dt = (now.tv_sec - last_time.tv_sec) +
                    (now.tv_nsec - last_time.tv_nsec) / 1e9;

        uint64_t total_st = 0;
        for (int i = 0; i < g_num_workers; i++)
            total_st += worker_steps[i * 8];

        double fill = (double)dp_now / (double)g_ht_size * 100.0;
        double dp_rate = (dp_now - last_dp) / dt;
        double eta = (dp_rate > 0) ? (collect_target - dp_now) / dp_rate : 0;

        printf("[PROGRESS] DPs: %lu / %lu (%.1f%%) | %.2fM steps/s | "
               "%.0f DP/s | fill: %.1f%% | ETA: %.0fs | elapsed: %.0fs\n",
               (unsigned long)dp_now, (unsigned long)collect_target,
               dp_now * 100.0 / collect_target,
               total_st / elapsed / 1e6,
               dp_rate, fill, eta, elapsed);

        /* Periodic checkpoint */
        if (elapsed - last_checkpoint_time >= checkpoint_interval && dp_now > 0) {
            if (g_use_disk) {
                /* Disk mode: no need to pause workers, data is already on disk */
                msync(g_ht, ht_bytes, MS_ASYNC);
                disk_meta_save(dp_now, total_st);
                last_checkpoint_time = elapsed;
            } else {
                g_shutdown = 1;  /* pause workers briefly */
                for (int i = 0; i < g_num_workers; i++)
                    pthread_join(threads[i], NULL);

                checkpoint_save(g_checkpoint_file, dp_now, total_st);
                last_checkpoint_time = elapsed;

                /* Restart workers */
                g_shutdown = 0;
                for (int i = 0; i < g_num_workers; i++)
                    pthread_create(&threads[i], NULL, tame_worker, &wargs[i]);
            }
        }

        last_dp = dp_now;
        last_time = now;
    }

    /* Signal shutdown and join */
    g_shutdown = 1;
    for (int i = 0; i < g_num_workers; i++)
        pthread_join(threads[i], NULL);

    struct timespec te;
    clock_gettime(CLOCK_MONOTONIC, &te);
    double total_elapsed = (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec) / 1e9;
    uint64_t total_steps_final = 0;
    for (int i = 0; i < g_num_workers; i++)
        total_steps_final += worker_steps[i * 8];

    uint64_t final_dp = __sync_fetch_and_add(&g_dp_count, 0);

    /* Save checkpoint on Ctrl+C (incomplete run) */
    if (final_dp < collect_target) {
        printf("\n[CHECKPOINT] Interrupted — saving progress...\n");
        if (g_use_disk) {
            msync(g_ht, ht_bytes, MS_SYNC);
            disk_meta_save(final_dp, total_steps_final);
            printf("[DISK] Hash table synced to disk. Resume with: --disk %s --resume\n", g_disk_path);
        } else {
            checkpoint_save(g_checkpoint_file, final_dp, total_steps_final);
            printf("[CHECKPOINT] Resume later with: --resume\n");
        }
        munmap(g_ht, hp_ht_actual_size);
        free(g_segments);
        free(jump_table);
        free(escape_table);
        return 0;
    }

    printf("\n============================================================\n");
    printf("  COLLECTION COMPLETE\n");
    printf("============================================================\n");
    printf("  Unique DPs:     %lu\n", (unsigned long)final_dp);
    printf("  Total steps:    %.2fM\n", total_steps_final / 1e6);
    printf("  Time:           %.1fs\n", total_elapsed);
    printf("  Rate:           %.2fM steps/s\n", total_steps_final / total_elapsed / 1e6);
    printf("============================================================\n");

    /* Extract DPs from hash table */
    printf("[EXTRACT] Scanning hash table for DPs...\n");
    DPEntry* all_dps = malloc((size_t)(final_dp + 1024) * sizeof(DPEntry));
    if (!all_dps) {
        fprintf(stderr, "[ERROR] Cannot allocate DPEntry array\n");
        return 1;
    }
    uint64_t n_extracted = 0;
    uint64_t total_count_sum = 0;
    for (uint64_t i = 0; i < g_ht_size; i++) {
        if (g_ht[i].count > 0) {
            all_dps[n_extracted].x_hi  = g_ht[i].x_hi;
            __int128 d = ((__int128)g_ht[i].dist_hi << 64) | g_ht[i].dist_lo;
            all_dps[n_extracted].dist   = d;
            all_dps[n_extracted].count  = g_ht[i].count;
            total_count_sum += g_ht[i].count;
            n_extracted++;
        }
    }
    printf("[EXTRACT] Found %lu unique DPs (total visits: %lu)\n",
           (unsigned long)n_extracted, (unsigned long)total_count_sum);

    /* Free hash table */
    munmap(g_ht, ht_bytes);
    g_ht = NULL;
    free(g_segments);
    g_segments = NULL;

    /* Scoring / selection */
    double q_hat = 1.0;
    double R_factor = 1.0;
    uint64_t N_sel = g_target_dp;
    if (N_sel > n_extracted) N_sel = n_extracted;

    if (g_scored && n_extracted > N_sel) {
        printf("[SCORE] Sorting %lu DPs by visit count...\n",
               (unsigned long)n_extracted);
        qsort(all_dps, n_extracted, sizeof(DPEntry), cmp_by_count_desc);

        /* Compute average count for ALL DPs */
        double avg_count_all = (double)total_count_sum / (double)n_extracted;

        /* Compute average count for selected top N */
        uint64_t sel_count_sum = 0;
        for (uint64_t i = 0; i < N_sel; i++)
            sel_count_sum += all_dps[i].count;
        double avg_count_sel = (double)sel_count_sum / (double)N_sel;

        q_hat = avg_count_sel / avg_count_all;
        R_factor = q_hat;

        printf("[SCORE] avg_count_all=%.4f avg_count_sel=%.4f\n",
               avg_count_all, avg_count_sel);
        printf("[SCORE] q_hat=%.6f R_factor=%.1f\n", q_hat, R_factor);
        printf("[SCORE] Selected top %lu of %lu DPs\n",
               (unsigned long)N_sel, (unsigned long)n_extracted);
        printf("[SCORE] Min selected count=%u, max=%u\n",
               all_dps[N_sel - 1].count, all_dps[0].count);
    } else {
        printf("[SCORE] Unscored mode: taking %lu of %lu DPs\n",
               (unsigned long)N_sel, (unsigned long)n_extracted);
        q_hat = 1.0;
        R_factor = 1.0;
    }

    /* Write output files */
    write_output(all_dps, N_sel, hash_index_size, hash_index_mask,
                 g_hash_index_bits, q_hat, R_factor);

    /* Remove checkpoint/disk files after successful completion */
    if (g_use_disk) {
        printf("[DISK] Removing hash table file: %s\n", g_disk_ht_file);
        unlink(g_disk_ht_file);
        unlink(g_disk_meta_file);
    } else {
        unlink(g_checkpoint_file);
    }

    free(all_dps);
    free(jump_table);
    free(escape_table);

    printf("\n============================================================\n");
    printf("  TAME PHASE COMPLETE\n");
    printf("============================================================\n");
    printf("  Output prefix:  %s\n", g_prefix);
    printf("  DPs written:    %lu\n", (unsigned long)N_sel);
    printf("  q_hat:          %.6f\n", q_hat);
    printf("  R_factor:       %.1f\n", R_factor);
    printf("  Total time:     %.1fs\n", total_elapsed);
    printf("============================================================\n");

    return 0;
}
