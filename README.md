# Cache Line & False Sharing Mechanics in Multi-Core Python

**Instructor:** Sufyan bin Uzayr  
**Git repository name:** `cache-line-false-sharing-allfocusSS`  
**Author identity used in runtime seed:** `allfocusSS` (edit or override if different)

This project implements all five programming tasks and all five forensic questions from the supplied four-page PDF. The annex in that PDF ends during Task 1, so `false_sharing_lab.py` is a fully working reconstructed deliverable rather than a modification of a complete starter file.

## Files

- `false_sharing_lab.py` — Task 1–5 implementation; emits a JSON manifest and matching Markdown report.
- `cache_lab_manifest.json` — full 4-thread, 60,000,000-updates-per-thread benchmark **run on the provided execution environment (Linux Xeon)**. Its telemetry **is not from your MacBook**.
- `ANALYSIS.md` — automatically generated answers Q1–Q5 from that very manifest, including measured nanoseconds and native LLVM IR examples.
- `test_lab.py` — self-tests with correct parity, memory alignment, and host line mapping.
- `requirements.txt` — tested NumPy/Numba/llvmlite versions.

## Run on your own MacBook (important for submission)

From Terminal in this repository directory:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest -v test_lab
python false_sharing_lab.py --git-username allfocusSS
```

**Crucial:** the default command runs all **60 million iterations for each of 4 threads** under strides 1, 4, 8, 16, plus the register-private baseline. It will overwrite both `cache_lab_manifest.json` **and** `ANALYSIS.md` with the actual **MacBook** CPU, system seed, timings, context-switch deltas, state parity, and LLVM verification evidence. Use the generated MacBook files for submission, not the supplied Linux telemetry.

To test faster without overwriting the final output:

```bash
python false_sharing_lab.py --smoke
```

This writes `cache_lab_smoke_manifest.json` and `ANALYSIS_SMOKE.md`, where `verification_passed` is **false** because the benchmark was not the full run. Smoke output must not be submitted as the complete deliverable.

**Why the CPU line size matters:** the assignment assumes a 64-byte line; some Apple Silicon CPUs may report another size, such as 128 bytes. The script queries `sysctl -n hw.cachelinesize` and interprets sharing using your *actual* line size. A 64-byte stride cannot be assumed to prevent sharing when the physical line is larger.

**Affinity:** Linux supports thread-level `os.sched_setaffinity`; this is used when permitted. macOS has no equivalent `os.sched_setaffinity` API, so the manifest explicitly records that no Linux-style thread pinning occurred rather than claiming it did. Standard `resource.getrusage()` process context switches are collected where supported. The code still runs true Numba `nogil=True` workers in separate threads on macOS.

## Check the results

```bash
python - <<'PY'
import json
m = json.load(open('cache_lab_manifest.json'))
print('verification_passed:', m['verification_passed'])
print('host:', m['system_context']['cpu_model'])
print('reported cache line:', m['system_context']['cache_line_size_bytes'], 'bytes')
for entry in m['benchmark']['layouts']:
    print(entry['layout'], entry['elapsed_ns'], 'ns', entry['ns_per_update'], 'ns/update')
print('LLVM proof:', m['llvm_audit']['llvm_volatile_store_example'])
PY
```

Ensure `verification_passed: True`. The `ANALYSIS.md` table and ratios must match the current machine's generated JSON.

## Task-to-file correspondence

| PDF requirement | Implementation |
|---|---|
| Task 1: host CPU/line size, deterministic identity seed, affinity | `init_system_context` |
| Task 2: native GIL-free per-step RMW in shared buffer | `worker_strided_accumulate` + `_volatile_add_u64` |
| Task 3: four layouts, parallel 60M/thread, timing, context switches | `run_stride_benchmark` |
| Task 4: private register accumulation, single final store | `worker_private_accumulate` |
| Task 5: exact state parity, native volatile load/store LLVM audit, JSON manifest | `audit_llvm_and_parity`, `build_manifest` |
| Q1–Q5: MESI, metrics, prefetcher, LLVM, GIL | generated `ANALYSIS.md` |

### Technical note on LLVM and fairness

A naïve Numba loop of `buffer[slot] += increment` can be optimized by LLVM into one register accumulation and one final store, spoiling a false-sharing benchmark. The Numba intrinsic here explicitly emits **LLVM `load volatile i64` and `store volatile i64`** on *every* loop iteration. The unit test and full-run audit check for those native instructions. The register-private baseline can be vectorized and is only a lower-bound control for repeated-store overhead; it is **not** a direct measure of one memory write's latency. Neither `resource.getrusage` nor this benchmark measures RFO bus transactions directly.

This code uses an internal llvmlite IR instruction extension because the high-level IRBuilder does not provide a public volatile flag. The requirements file pins the tested LLVM binding version.

## Git submission

After you have run the benchmark **on your own computer** and verified the output:

```bash
cd cache-line-false-sharing-allfocusSS
git init
git add false_sharing_lab.py cache_lab_manifest.json ANALYSIS.md README.md requirements.txt test_lab.py
git commit -m "Complete cache line false sharing laboratory"

# Requires GitHub CLI (gh) to be installed and authenticated:
gh auth status
gh repo create cache-line-false-sharing-allfocusSS --private --source=. --remote=origin --push
```

If the instructor requires a **public** repository to view it, use `--public` instead of `--private`, or grant the instructor access according to their instructions. Share the resulting GitHub repository URL with the instructor.

**Project status:** source files and reference full-run output are prepared. Creating/pushing a repository on *your* GitHub account and generating the required user-MacBook telemetry need to be done from an authenticated MacBook Terminal. No commit or push has been performed on your behalf.
