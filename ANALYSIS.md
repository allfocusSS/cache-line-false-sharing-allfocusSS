# Cache Line & False Sharing Mechanics — Q1–Q5

> **Full specification run**. All numerical values below are copied from this run's
> `cache_lab_manifest.json` (or the smoke manifest). **Do not submit numbers measured
> on somebody else's computer as your own.** Re-run the script on the target host;
> this report is regenerated automatically.

## Run context and method

- **GitHub identity:** `allfocusSS`
- **Host:** `MacBook-Air-Zaksylyk.local`; `Apple M2`; `arm64`; `macOS-27.2-arm64-arm-64bit-Mach-O`
- **Logical / physical CPU cores:** 8 / 8
- **Reported coherence line size:** 128 bytes; **instance seed:** `0x39b27437d80b45bb`
- **Thread count:** 4; **iterations per worker per layout:** 60,000,000
- **Total logical updates per layout:** 240,000,000; **timing API:** `time.perf_counter_ns()`
- **Native implementation:** NumPy 64-byte-or-better-aligned uint64 buffer; `@njit(nogil=True)`;
  OS threads and best-effort Linux affinity (`affinity_pinned` in the JSON manifest).
- **Compilation:** Numba kernels were warmed up before measured intervals. Each layout was timed once;
  timing variation due to OS scheduling, CPU frequency, thermal state, and hardware topology is possible.

**Important: this host reported 128-byte coherence lines.** The 64-byte stride=8 layout has only two physical lines for four workers, so it still suffers false sharing. The stride=16 layout gives each worker a separate 128-byte line. The PDF's 64-byte-line assumption does not apply.

## Q1. MESI protocol state transitions and RFO

Assume both caches initially have **I (Invalid)** for the shared line containing
`buffer[0]` and `buffer[1]` in the stride=1, 64-byte-line case:

| Event | Core 0 line state | Core 1 line state | Explanation |
|---|---|---|---|
| Before access | I | I | Neither has a valid copy |
| Core 0 requests ownership and stores to `buffer[0]` | M | I | Core 0 obtains the line via Read-For-Ownership (RFO); a modified line is exclusive to its owner |
| Core 1 requests ownership and stores to `buffer[1]` | I | M | Core 0 must relinquish the modified line; coherence transfers/invalidates ownership before Core 1's store can commit |
| Core 0 stores again | M | I | Ownership moves back, creating line ping-pong |

If a clean line were first fetched by a reader, it might enter **E (Exclusive)**
and then become **S (Shared)** after another read; a subsequent write upgrades S to M
and invalidates the other shared copies. A private E line can become M without a competing
ownership transfer. The key point is that coherence tracks the whole *line*, not the individual
8-byte addresses. The shared-line RFO may make Core 1's pending store wait for ownership;
queued writes can temporarily sit in its store buffer or be limited by buffer capacity.
**Not every store creates a new RFO:** stores can continue locally while a core retains M.

## Q2. Quantitative slowdown and coherence overhead

| Layout | Spacing | Elapsed time (ns) | ns/update, aggregate | Cache lines occupied | Voluntary / involuntary switches |
|---|---:|---:|---:|---:|---:|
| A - contiguous | 8 B | 131,300,750 | 0.547086 | 1 | 0 / 9 |
| B - half-line | 32 B | 129,930,500 | 0.541377 | 1 | 0 / 49 |
| C - padded | 64 B | 127,643,750 | 0.531849 | 2 | 0 / 39 |
| D - prefetch-safe | 128 B | 126,492,375 | 0.527052 | 4 | 0 / 49 |
| Private-register baseline | 64 B | 9,806,208 | 0.040859 | 2 | 0 / 12 |

The elapsed measurements are **wall-clock nanoseconds for all 4 workers**.
The reported `ns/update` is `elapsed_ns / (4 × 60,000,000)`; it is an
aggregate-throughput metric, **not** the physical latency of one isolated store.

- **Stride=1 relative to stride=8:** 131,300,750 / 127,643,750 = **1.028650× slower**.
- **Stride=1 relative to private baseline:** 131,300,750 / 9,806,208 = **13.389554×**.
- **Stride=8 relative to private baseline:** 127,643,750 / 9,806,208 = **13.016627×**.
- `ns/update` includes native-loop overhead, actual stores, cross-core contention,
  scheduling, and memory hierarchy behavior. It cannot uniquely isolate an RFO cost.

The contiguous case makes 1 distinct physical
line(s) for four writers, whereas stride=8 occupies
2. In a 64-byte-line system, this typically
explains a large fraction of the slowdown from ownership ping-pong. The private register
baseline makes one final shared-memory store per worker, so it eliminates repeated writes;
Numba/LLVM can also vectorize or otherwise optimize that local arithmetic, making it a
**lower-bound control**, not a pure measure of cache hit latency. Context switches are measured with resource.getrusage(RUSAGE_SELF), i.e. process-level deltas including benchmark orchestration, not hardware RFO counts.

## Q3. Adjacent-line prefetching and 128-byte pairing

The measured durations are **127,643,750 ns** at stride=8 (64 B between counters) and
**126,492,375 ns** at stride=16 (128 B between counters). stride=16 was 0.91% faster than stride=8 by wall time, a ratio of 1.0091× (stride8/stride16).

Some CPUs have adjacent-line spatial prefetchers that fetch a neighboring 64-byte
line in a larger 128-byte neighborhood. This can affect bandwidth, cache residency,
and measurement noise. **Prefetching a clean neighboring line alone does not generate
write-ownership invalidations** and is not proof of false sharing. True secondary
invalidations occur when counters actually share a hardware coherence granule (for
example, on a host with 128-byte coherence lines), or other concurrent agents
write the relevant line. The reported `128`-byte line size and observed
`distinct_cache_lines` are more reliable than assuming 64-byte padding suffices on
all ARM64/x86 hosts. A small stride=8/16 timing difference in one run should not be
interpreted as conclusive evidence of a particular prefetcher implementation.

## Q4. LLVM IR: register promotion versus real memory stores

Compilers normally optimize `counter += increment` by keeping the counter in a register
and storing once at the end of a loop. A loop-carried arithmetic dependency **by itself
does not prevent this optimization**. A plain `buffer[slot] += increment` in an
optimized native loop can therefore accidentally benchmark register arithmetic rather
than cache coherence.

This implementation uses an Numba `@intrinsic` that writes the LLVM instructions
`load volatile i64` and `store volatile i64` for each loop iteration. Exact lines
observed in `worker_strided_accumulate.inspect_llvm()` on this run:

```llvm
%counter_previous = load volatile i64, ptr %sunkaddr2, align 8
store volatile i64 %.204, ptr %sunkaddr2, align 8
```

There were **1 distinct volatile load sites**
and **1 distinct volatile store sites**
(the optimizer may unroll the loop, producing multiple static sites). This is stronger
verification than relying on the source-level data dependency. The compiled function
has **1 Numba nopython signature(s)**
and uses `nogil=True`, so the measured inner loop executes as native code without
per-iteration Python operations. The `inspect_llvm()` module also contains normal
Python-entry wrapper code; that does **not** mean the native loop falls back to Python.

The private baseline has exactly **1 native i64 store site**
(`store i64 %local_counter.2.0.lcssa, ptr %.217, align 8`), after the register-local loop.

The final counter values matched bit-for-bit across all five modes:
`[150000000, 510000000, 450000000, 210000000]`. Expected per worker is the integer
sum `(base_increment × iterations + floor(iterations / 2)) mod 2^64`, with
`base_increment = 1 + ((seed XOR (seed >> 32)) AND 7)`.

## Q5. CPython GIL versus genuine multi-core false sharing

In a standard GIL-enabled CPython build, multiple Python threads performing a
pure Python increment loop generally do **not** execute their Python bytecodes in
parallel on multiple cores. The GIL serializes the Python interpreter work; extra
threads add interpreter lock handoffs, scheduling overhead, and context switching.
Thus they do not produce the same sustained, simultaneous cross-core store contention.

Here Numba's `@njit(nogil=True)` compiles the update kernel to machine code and
releases the GIL while each `threading.Thread` runs it. Different cores can genuinely
write separate counters at the same time. When their counters share one cache line,
the coherency protocol repeatedly transfers ownership; when they use separate lines,
that mechanism mostly disappears. This distinction is why a pure Python threading
benchmark is not a valid substitute for the required GIL-free experiment.

## Verification and limitations

- **Expected vs. observed state parity:** True.
- **LLVM volatile memory accesses found:** True.
- **Submission-level `verification_passed`:** True.
- **Environment:** these numbers belong to **MacBook-Air-Zaksylyk.local / Apple M2**.
  Execute again on your own MacBook and submit that generated JSON and ANALYSIS.md.
- Affinity on macOS cannot be set with the Linux `os.sched_setaffinity()` API;
  when unavailable, the manifest records the unpinned status instead of inventing it.
- CPU caching/coherency mechanisms are inferred from topology and elapsed time.
  This laboratory does not directly count RFO bus transactions or hardware PMU events.
