#!/usr/bin/env python3
"""GIL-free, cache-line false-sharing laboratory.

Task mapping:
  1. init_system_context
  2. worker_strided_accumulate and a guaranteed volatile native load/store
  3. run_stride_benchmark for strides 1, 4, 8, 16
  4. worker_private_accumulate register baseline
  5. audit_llvm_and_parity plus machine-specific JSON manifest

The default runs the required 60,000,000 updates per thread.  --smoke uses
small input and MUST NOT be submitted as the full benchmark manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numba
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir
from llvmlite.ir import instructions as llvm_instructions

try:
    import resource
except ImportError:
    resource = None

# GitHub handle previously supplied with the user's projects; change if needed.
GIT_USERNAME = "allfocusSS"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
UINT64_MASK = (1 << 64) - 1


def _sysctl_value(key: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def detect_cache_line_size() -> int:
    """Read the host L1D coherency line size, falling back to 64 B."""
    linux_path = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
    try:
        if linux_path.is_file():
            value = int(linux_path.read_text(encoding="utf-8").strip())
            if value > 0 and value % 8 == 0:
                return value
    except (ValueError, OSError):
        pass
    if sys.platform == "darwin":
        try:
            value = int(_sysctl_value("hw.cachelinesize") or "0")
            if value > 0 and value % 8 == 0:
                return value
        except ValueError:
            pass
    return 64


def detect_cpu_model() -> str:
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            value = _sysctl_value(key)
            if value:
                return value
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith(("model name", "Hardware", "Processor")):
                    return line.split(":", 1)[-1].strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().machine


def _physical_cpu_count(logical: int) -> int | None:
    if sys.platform == "darwin":
        result = _sysctl_value("hw.physicalcpu")
        if result and result.isdigit():
            return int(result)
    if sys.platform.startswith("linux"):
        groups = set()
        try:
            for cpu in range(logical):
                root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                pkg = (root / "physical_package_id").read_text().strip()
                core = (root / "core_id").read_text().strip()
                groups.add((pkg, core))
            if groups:
                return len(groups)
        except OSError:
            pass
    return None


def _candidate_cores() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        return []
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        return []
    # Prefer distinct physical cores when Linux's topology is readable.
    groups = []
    seen = set()
    remaining = []
    for cpu in allowed:
        try:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            pkg = (root / "physical_package_id").read_text().strip()
            core = (root / "core_id").read_text().strip()
            physical = (pkg, core)
        except OSError:
            physical = ("unknown", cpu)
        if physical not in seen:
            groups.append(cpu)
            seen.add(physical)
        else:
            remaining.append(cpu)
    return groups + remaining


def init_system_context(git_user: str) -> dict:
    """Task 1: system topology, identity-bound seed, best-effort pinning."""
    if not git_user or git_user.strip() in ("", "REPLACE_WITH_YOUR_GIT_USERNAME"):
        raise ValueError("Provide your real GitHub handle in GIT_USERNAME or --git-username")
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires >= {NUM_THREADS} logical cores; found {logical_cores}")
    cache_line = detect_cache_line_size()
    cpu_model = detect_cpu_model()
    cpu_signature = f"{platform.machine()}::{cpu_model}::{logical_cores}"
    seed_material = f"{git_user.strip()}::{cpu_signature}"
    instance_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    candidate_cores = _candidate_cores()
    if candidate_cores and len(candidate_cores) < NUM_THREADS:
        raise RuntimeError("Not enough schedulable CPUs in the current affinity mask")
    return {
        "git_username": git_user.strip(),
        "cpu_model": cpu_model,
        "cpu_signature": cpu_signature,
        "architecture": platform.machine(),
        "operating_system": platform.platform(),
        "hostname": platform.node(),
        "logical_cores": logical_cores,
        "physical_cores": _physical_cpu_count(logical_cores),
        "cache_line_size_bytes": cache_line,
        "instance_seed": instance_seed,
        "instance_seed_hex": f"0x{instance_seed:016x}",
        "affinity_api_supported": hasattr(os, "sched_setaffinity"),
        "affinity_core_candidates": candidate_cores[:NUM_THREADS],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
    }


# llvmlite's high-level IRBuilder.load/store have no public `volatile=` option.
# Emit explicit LLVM `load volatile` / `store volatile` instructions via a
# small extension of their printer. LLVM verifies the generated module.
# These instructions cannot be legally elided or promoted into a register.
class _VolatileLoad(llvm_instructions.LoadInstr):
    def descr(self, buf):
        pos = len(buf)
        super().descr(buf)
        buf[pos] = buf[pos].replace("load ", "load volatile ", 1)


class _VolatileStore(llvm_instructions.StoreInstr):
    def descr(self, buf):
        pos = len(buf)
        super().descr(buf)
        buf[pos] = buf[pos].replace("store ", "store volatile ", 1)


@intrinsic
def _volatile_add_u64(typing_context, address, increment):
    """One native volatile read-modify-write at the uint64 memory address."""
    if address != types.uintp or increment != types.uint64:
        return None
    signature = types.void(address, increment)

    def codegen(context, builder, sig, args):
        address_value, increment_value = args
        ptr = builder.inttoptr(address_value, ir.IntType(64).as_pointer())
        previous = _VolatileLoad(builder.block, ptr, name="counter_previous")
        previous.align = 8
        builder._insert(previous)
        updated = builder.add(previous, increment_value)
        stored = _VolatileStore(builder.block, updated, ptr)
        stored.align = 8
        builder._insert(stored)
        return context.get_dummy_value()

    return signature, codegen


@njit(nogil=True)
def _step_for_seed(thread_seed):
    return np.uint64((thread_seed ^ (thread_seed >> np.uint64(32))) & np.uint64(7)) + np.uint64(1)


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    """Task 2: update buffer[slot] on EVERY loop iteration, not a register copy.

    The intrinsic uses buffer.ctypes.data + slot*8 to address the actual
    buffer[slot] cell, and enforces one LLVM volatile load/store per step.
    """
    increment = _step_for_seed(thread_seed)
    address_of_buffer_slot = np.uintp(buffer.ctypes.data) + np.uintp(slot * 8)
    for step_index in range(iters):
        # Alternating increments keep the private baseline from collapsing
        # into a single multiplication, while both kernels do identical work.
        _volatile_add_u64(address_of_buffer_slot, increment + np.uint64(step_index & 1))


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    """Task 4: pure register accumulation followed by ONE memory store."""
    increment = _step_for_seed(thread_seed)
    local_counter = np.uint64(0)
    for step_index in range(iters):
        local_counter += increment + np.uint64(step_index & 1)
    buffer[slot] = local_counter


def aligned_uint64_array(length: int, alignment: int = 64) -> np.ndarray:
    if length <= 0 or alignment < 64 or alignment % 8 != 0:
        raise ValueError("Invalid length or alignment")
    extra = alignment // 8 + 1
    owner = np.zeros(length + extra, dtype=np.uint64)
    offset_bytes = (-owner.ctypes.data) % alignment
    offset_words = offset_bytes // 8
    array = owner[offset_words:offset_words + length]
    if array.ctypes.data % alignment != 0 or not array.flags.c_contiguous:
        raise AssertionError("Could not allocate aligned contiguous uint64 buffer")
    return array


def _thread_seed(instance_seed: int, thread_index: int) -> int:
    return (instance_seed + 0x9E3779B97F4A7C15 * (thread_index + 1)) & UINT64_MASK


def _expected_counter(thread_seed: int, iters: int) -> int:
    step = ((thread_seed ^ (thread_seed >> 32)) & 7) + 1
    return (step * iters + (iters // 2)) & UINT64_MASK


def _context_switches() -> dict | None:
    if resource is None:
        return None
    info = resource.getrusage(resource.RUSAGE_SELF)
    return {"voluntary": int(info.ru_nvcsw), "involuntary": int(info.ru_nivcsw)}


def _delta_switches(before, after):
    if before is None or after is None:
        return None
    return {key: after[key] - before[key] for key in ("voluntary", "involuntary")}


def _compile_warmup() -> None:
    buf = aligned_uint64_array(4)
    worker_strided_accumulate(buf, 0, 12, np.uint64(123456))
    buf[:] = 0
    worker_private_accumulate(buf, 0, 12, np.uint64(123456))


def _run_one(context: dict, stride: int, iters: int, private: bool = False) -> dict:
    thread_count = NUM_THREADS
    line_size = context["cache_line_size_bytes"]
    alignment = math.lcm(64, line_size)
    slots = [i * stride for i in range(thread_count)]
    buffer = aligned_uint64_array(slots[-1] + 1, alignment)
    start_ready = threading.Barrier(thread_count + 1)
    start_event = threading.Event()
    errors: list[BaseException] = []
    actual_affinity = [None for _ in range(thread_count)]
    try_affinity = bool(context["affinity_api_supported"] and context["affinity_core_candidates"])
    kernel = worker_private_accumulate if private else worker_strided_accumulate
    seeds = [_thread_seed(context["instance_seed"], index) for index in range(thread_count)]

    def thread_target(index):
        try:
            # os.sched_setaffinity(0, ...) binds only the CALLING Linux thread.
            if try_affinity:
                requested = context["affinity_core_candidates"][index]
                try:
                    os.sched_setaffinity(0, {requested})
                    actual_affinity[index] = sorted(os.sched_getaffinity(0))
                except OSError:
                    actual_affinity[index] = None
            start_ready.wait(timeout=30)
            start_event.wait(timeout=30)
            kernel(buffer, slots[index], iters, np.uint64(seeds[index]))
        except BaseException as exc:
            errors.append(exc)
            try:
                start_ready.abort()
            except Exception:
                pass

    workers = [threading.Thread(target=thread_target, args=(index,), name=f"lab-worker-{index}")
               for index in range(thread_count)]
    for worker in workers:
        worker.start()
    try:
        start_ready.wait(timeout=30)
    except threading.BrokenBarrierError as exc:
        for worker in workers:
            worker.join(timeout=1)
        raise RuntimeError("Thread did not reach launch barrier") from exc
    switches_before = _context_switches()
    start_ns = time.perf_counter_ns()
    start_event.set()
    for worker in workers:
        worker.join()
    elapsed_ns = time.perf_counter_ns() - start_ns
    switches_after = _context_switches()
    if errors:
        raise RuntimeError("One or more Numba workers failed") from errors[0]

    observed = [int(buffer[index]) for index in slots]
    expected = [_expected_counter(seed, iters) for seed in seeds]
    exact = observed == expected
    if not exact:
        raise AssertionError(f"State parity failure, stride={stride}, private={private}: {observed} != {expected}")
    # Whole-line mapping accounts for the host line width, not an assumed 64 B.
    absolute_lines = [(buffer.ctypes.data + 8 * slot) // line_size for slot in slots]
    return {
        "layout": "private" if private else f"stride_{stride}",
        "stride_elements": stride,
        "stride_bytes": stride * 8,
        "iterations_per_thread": iters,
        "total_updates": iters * thread_count,
        "elapsed_ns": elapsed_ns,
        "ns_per_update": elapsed_ns / (iters * thread_count),
        "context_switch_delta": _delta_switches(switches_before, switches_after),
        "buffer_address_alignment_bytes": alignment,
        "slots": slots,
        "relative_cache_line_ids": [a - absolute_lines[0] for a in absolute_lines],
        "distinct_cache_lines": len(set(absolute_lines)),
        "observed_counter_values": observed,
        "expected_counter_values": expected,
        "parity_passed": exact,
        "thread_affinity_masks": actual_affinity,
        "affinity_pinned": all(a is not None for a in actual_affinity),
    }


def run_stride_benchmark(context: dict, iters: int) -> list[dict]:
    """Task 3 and Task 4: four shared-stride layouts and local baseline."""
    records = []
    for stride in STRIDES:
        record = _run_one(context, stride=stride, iters=iters)
        records.append(record)
        print(f"stride={stride:2}  {record['elapsed_ns']:>14,} ns  "
              f"{record['ns_per_update']:.3f} ns/update  "
              f"lines={record['distinct_cache_lines']}  parity=PASS", flush=True)
    record = _run_one(context, stride=8, iters=iters, private=True)
    records.append(record)
    print(f"private     {record['elapsed_ns']:>14,} ns  "
          f"{record['ns_per_update']:.3f} ns/update  parity=PASS", flush=True)
    return records


def audit_llvm_and_parity(records: list[dict]) -> dict:
    """Task 5: require native Numba mode, real LLVM volatile i64 stores, exact parity."""
    if not worker_strided_accumulate.nopython_signatures:
        raise AssertionError("Kernel has no Numba nopython signatures")
    if not worker_private_accumulate.nopython_signatures:
        raise AssertionError("Private baseline has no Numba nopython signatures")
    signature = worker_strided_accumulate.signatures[0]
    module_ir = worker_strided_accumulate.inspect_llvm(signature)
    store_lines = [line.strip() for line in module_ir.splitlines()
                   if re.search(r"\bstore volatile i64\b", line)]
    load_lines = [line.strip() for line in module_ir.splitlines()
                  if re.search(r"\bload volatile i64\b", line)]
    if not store_lines or not load_lines:
        raise AssertionError("LLVM failed to emit volatile i64 loads AND stores: cache test is invalid")
    private_ir = worker_private_accumulate.inspect_llvm(worker_private_accumulate.signatures[0])
    private_store_lines = [line.strip() for line in private_ir.splitlines()
                           if re.search(r"\bstore i64\b", line)]
    if len(private_store_lines) != 1 or "store volatile i64" in private_ir:
        raise AssertionError("Private kernel must have one ordinary final i64 store")
    native_functions = [line for line in module_ir.splitlines()
                        if line.startswith("define ") and "worker_strided_accumulate" in line]
    if not native_functions:
        raise AssertionError("Native function definition not present in LLVM module")
    if len(records) != 5 or {r["layout"] for r in records} != {
        "stride_1", "stride_4", "stride_8", "stride_16", "private"
    }:
        raise AssertionError("All four layouts and baseline are required")
    expected = records[0]["expected_counter_values"]
    parity = all(r["parity_passed"] and r["observed_counter_values"] == expected
                 for r in records)
    if not parity:
        raise AssertionError("At least one layout differs in exact final uint64 values")
    return {
        "parity_across_all_five_layouts": parity,
        "native_numba_nopython_signatures": len(worker_strided_accumulate.nopython_signatures),
        "llvm_volatile_i64_store_sites": len(store_lines),
        "llvm_volatile_i64_load_sites": len(load_lines),
        "llvm_volatile_store_example": store_lines[0],
        "llvm_volatile_load_example": load_lines[0],
        "private_single_final_i64_store": private_store_lines[0],
        "private_final_i64_store_sites": len(private_store_lines),
        "native_function_definition": native_functions[0],
        "cpython_wrapper_note": "inspect_llvm includes a CPython entry wrapper; the benchmark loop is a nopython nogil native function.",
        "llvm_verified": True,
    }


def build_manifest(context: dict, records: list[dict], llvm: dict, iters: int) -> dict:
    measurements = {record["layout"]: record for record in records}
    s1 = measurements["stride_1"]["elapsed_ns"]
    s8 = measurements["stride_8"]["elapsed_ns"]
    s16 = measurements["stride_16"]["elapsed_ns"]
    ratios = {
        "stride_1_over_stride_8": s1 / s8,
        "stride_8_over_stride_16": s8 / s16,
        "stride_1_over_private": s1 / measurements["private"]["elapsed_ns"],
        "stride_8_over_private": s8 / measurements["private"]["elapsed_ns"],
    }
    full = iters == ITERATIONS_PER_THREAD and NUM_THREADS == 4
    return {
        "specification": "Cache Line & False Sharing Mechanics in Multi-Core Python, Sufyan bin Uzayr",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "full" if full else "smoke_or_custom",
        "required_iterations_per_thread": ITERATIONS_PER_THREAD,
        "required_threads": 4,
        "system_context": context,
        "benchmark": {
            "threads": NUM_THREADS,
            "iterations_per_thread": iters,
            "layouts": records,
            "ratios": ratios,
        },
        "llvm_audit": llvm,
        "verification_passed": bool(full and llvm["parity_across_all_five_layouts"] and llvm["llvm_verified"]),
        "provenance_warning": "Measurements are valid ONLY for the host recorded under system_context. Re-run on your own Mac before submission.",
        "interpretation_note": (
            "Cache-line count is computed using measured host line size. On 128-byte lines, "
            "stride=8 may still share physical lines. Volatile loads/stores are proven in LLVM; "
            "the register-private baseline can be optimized by LLVM."
        ),
    }


def write_analysis(manifest: dict, output: Path) -> None:
    """Create host-specific forensic answers using this run's actual measurements.

    Intentionally overwrite ANALYSIS.md on a new FULL run so its numbers can
    never silently remain from a different machine's manifest.
    """
    ctx = manifest["system_context"]
    bench = manifest["benchmark"]
    audit = manifest["llvm_audit"]
    rows = {r["layout"]: r for r in bench["layouts"]}
    ratios = bench["ratios"]
    table_lines = [
        "| Layout | Spacing | Elapsed time (ns) | ns/update, aggregate | Cache lines occupied | Voluntary / involuntary switches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("stride_1", "A - contiguous"),
                       ("stride_4", "B - half-line"),
                       ("stride_8", "C - padded"),
                       ("stride_16", "D - prefetch-safe"),
                       ("private", "Private-register baseline")):
        row = rows[key]
        sw = row["context_switch_delta"]
        switch_str = f"{sw['voluntary']} / {sw['involuntary']}" if sw else "unavailable"
        table_lines.append(
            f"| {label} | {row['stride_bytes']} B | {row['elapsed_ns']:,} | "
            f"{row['ns_per_update']:.6f} | {row['distinct_cache_lines']} | {switch_str} |"
        )
    table = "\n".join(table_lines)
    s1 = rows["stride_1"]["elapsed_ns"]
    s8 = rows["stride_8"]["elapsed_ns"]
    s16 = rows["stride_16"]["elapsed_ns"]
    private = rows["private"]["elapsed_ns"]
    width = ctx["cache_line_size_bytes"]
    threads = bench["threads"]
    iters = bench["iterations_per_thread"]
    full = manifest["verification_passed"]
    provenance = ("**Full specification run**" if full
                  else "**SMOKE/CUSTOM run (NOT valid as full submission)**")
    if width == 64:
        width_note = (
            "On this run, the host reported **64-byte coherence lines**. "
            "The four stride=8 slots occupy separate lines; stride=16 also occupies four lines."
        )
    elif width == 128:
        width_note = (
            "**Important: this host reported 128-byte coherence lines.** "
            "The 64-byte stride=8 layout has only two physical lines for four workers, "
            "so it still suffers false sharing. The stride=16 layout gives each worker "
            "a separate 128-byte line. The PDF's 64-byte-line assumption does not apply."
        )
    else:
        width_note = (
            f"This host reported **{width}-byte coherence lines**, different from the PDF's "
            "nominal 64 B. Interpret the distinct-line counts in the table instead of "
            "assuming stride=8 is always sharing-free."
        )
    if s16 < s8:
        compare = (f"stride=16 was {(s8/s16 - 1)*100:.2f}% faster than stride=8 "
                   f"by wall time, a ratio of {s8/s16:.4f}× (stride8/stride16).")
    else:
        compare = (f"stride=16 was {(s16/s8 - 1)*100:.2f}% slower than stride=8 "
                   f"by wall time, a ratio of {s8/s16:.4f}× (stride8/stride16).")
    context_switch_note = (
        "Context switches are measured with resource.getrusage(RUSAGE_SELF), i.e. process-level "
        "deltas including benchmark orchestration, not hardware RFO counts."
    )
    content = f"""# Cache Line & False Sharing Mechanics — Q1–Q5

> {provenance}. All numerical values below are copied from this run's
> `cache_lab_manifest.json` (or the smoke manifest). **Do not submit numbers measured
> on somebody else's computer as your own.** Re-run the script on the target host;
> this report is regenerated automatically.

## Run context and method

- **GitHub identity:** `{ctx['git_username']}`
- **Host:** `{ctx['hostname']}`; `{ctx['cpu_model']}`; `{ctx['architecture']}`; `{ctx['operating_system']}`
- **Logical / physical CPU cores:** {ctx['logical_cores']} / {ctx['physical_cores'] if ctx['physical_cores'] is not None else 'unknown'}
- **Reported coherence line size:** {width} bytes; **instance seed:** `{ctx['instance_seed_hex']}`
- **Thread count:** {threads}; **iterations per worker per layout:** {iters:,}
- **Total logical updates per layout:** {threads*iters:,}; **timing API:** `time.perf_counter_ns()`
- **Native implementation:** NumPy 64-byte-or-better-aligned uint64 buffer; `@njit(nogil=True)`;
  OS threads and best-effort Linux affinity (`affinity_pinned` in the JSON manifest).
- **Compilation:** Numba kernels were warmed up before measured intervals. Each layout was timed once;
  timing variation due to OS scheduling, CPU frequency, thermal state, and hardware topology is possible.

{width_note}

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

{table}

The elapsed measurements are **wall-clock nanoseconds for all {threads} workers**.
The reported `ns/update` is `elapsed_ns / ({threads} × {iters:,})`; it is an
aggregate-throughput metric, **not** the physical latency of one isolated store.

- **Stride=1 relative to stride=8:** {s1:,} / {s8:,} = **{ratios['stride_1_over_stride_8']:.6f}× slower**.
- **Stride=1 relative to private baseline:** {s1:,} / {private:,} = **{ratios['stride_1_over_private']:.6f}×**.
- **Stride=8 relative to private baseline:** {s8:,} / {private:,} = **{ratios['stride_8_over_private']:.6f}×**.
- `ns/update` includes native-loop overhead, actual stores, cross-core contention,
  scheduling, and memory hierarchy behavior. It cannot uniquely isolate an RFO cost.

The contiguous case makes {rows['stride_1']['distinct_cache_lines']} distinct physical
line(s) for four writers, whereas stride=8 occupies
{rows['stride_8']['distinct_cache_lines']}. In a 64-byte-line system, this typically
explains a large fraction of the slowdown from ownership ping-pong. The private register
baseline makes one final shared-memory store per worker, so it eliminates repeated writes;
Numba/LLVM can also vectorize or otherwise optimize that local arithmetic, making it a
**lower-bound control**, not a pure measure of cache hit latency. {context_switch_note}

## Q3. Adjacent-line prefetching and 128-byte pairing

The measured durations are **{s8:,} ns** at stride=8 (64 B between counters) and
**{s16:,} ns** at stride=16 (128 B between counters). {compare}

Some CPUs have adjacent-line spatial prefetchers that fetch a neighboring 64-byte
line in a larger 128-byte neighborhood. This can affect bandwidth, cache residency,
and measurement noise. **Prefetching a clean neighboring line alone does not generate
write-ownership invalidations** and is not proof of false sharing. True secondary
invalidations occur when counters actually share a hardware coherence granule (for
example, on a host with 128-byte coherence lines), or other concurrent agents
write the relevant line. The reported `{width}`-byte line size and observed
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
{audit['llvm_volatile_load_example']}
{audit['llvm_volatile_store_example']}
```

There were **{audit['llvm_volatile_i64_load_sites']} distinct volatile load sites**
and **{audit['llvm_volatile_i64_store_sites']} distinct volatile store sites**
(the optimizer may unroll the loop, producing multiple static sites). This is stronger
verification than relying on the source-level data dependency. The compiled function
has **{audit['native_numba_nopython_signatures']} Numba nopython signature(s)**
and uses `nogil=True`, so the measured inner loop executes as native code without
per-iteration Python operations. The `inspect_llvm()` module also contains normal
Python-entry wrapper code; that does **not** mean the native loop falls back to Python.

The private baseline has exactly **{audit["private_final_i64_store_sites"]} native i64 store site**\n(`{audit["private_single_final_i64_store"]}`), after the register-local loop.\n\nThe final counter values matched bit-for-bit across all five modes:
`{rows['stride_1']['observed_counter_values']}`. Expected per worker is the integer
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

- **Expected vs. observed state parity:** {audit['parity_across_all_five_layouts']}.
- **LLVM volatile memory accesses found:** {audit['llvm_verified']}.
- **Submission-level `verification_passed`:** {full}.
- **Environment:** these numbers belong to **{ctx['hostname']} / {ctx['cpu_model']}**.
  Execute again on your own MacBook and submit that generated JSON and ANALYSIS.md.
- Affinity on macOS cannot be set with the Linux `os.sched_setaffinity()` API;
  when unavailable, the manifest records the unpinned status instead of inventing it.
- CPU caching/coherency mechanisms are inferred from topology and elapsed time.
  This laboratory does not directly count RFO bus transactions or hardware PMU events.
"""
    output.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-username", default=GIT_USERNAME,
                        help="Your GitHub username used for deterministic host-bound seed")
    parser.add_argument("--iterations", type=int, default=ITERATIONS_PER_THREAD,
                        help="Per-thread updates, default mandatory 60,000,000")
    parser.add_argument("--smoke", action="store_true", help="Quick self-test, not valid for submission")
    parser.add_argument("--output", type=Path, default=None, help="Override JSON manifest filename")
    args = parser.parse_args()
    iters = 100_000 if args.smoke else args.iterations
    if iters <= 0:
        parser.error("--iterations must be positive")
    output = args.output or Path("cache_lab_smoke_manifest.json" if args.smoke else "cache_lab_manifest.json")
    context = init_system_context(args.git_username)
    print("GitHub:", context["git_username"])
    print("Host:", context["hostname"], "|", context["cpu_model"])
    print("Cache line:", context["cache_line_size_bytes"], "bytes")
    print("Seed:", context["instance_seed_hex"])
    print("Affinity candidates:", context["affinity_core_candidates"] or "not supported")
    print("JIT warm-up (excluded from measured durations)...", flush=True)
    _compile_warmup()
    records = run_stride_benchmark(context, iters)
    llvm = audit_llvm_and_parity(records)
    manifest = build_manifest(context, records, llvm, iters)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    analysis_output = output.parent / ("ANALYSIS_SMOKE.md" if args.smoke else "ANALYSIS.md")
    write_analysis(manifest, analysis_output)
    print("LLVM:", llvm["llvm_volatile_i64_store_sites"], "volatile stores;",
          llvm["llvm_volatile_i64_load_sites"], "volatile loads")
    print("Verification:", manifest["verification_passed"], "| output:", output, "| report:", analysis_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
