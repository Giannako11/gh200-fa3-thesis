# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-10-15T11:51:50+00:00 — Qwen baseline
- Run: `thesis/outputs/qwen_baseline_20261015_091635_497401787`
- Exit code: 130 (0 = success).

## 2026-10-20T23:58:36+00:00 — Qwen baseline
- Run: `thesis/outputs/qwen_baseline_20261020_234840_957642857`
- Exit code: 0 (0 = success).

## 2026-10-29 — TMA / CPU memory microbenchmark

- Added standalone CUDA 12.8/Hopper TMA probes under `thesis/microbenchmarks/tma_cpu/`
  and `thesis/scripts/run_tma_cpu.sh`: HBM control, mapped pinned host memory,
  and CPU-first-touch registered memory, each compared with ordinary GPU loads.
- Records correctness, CUDA failure stage, NUMA residency queries, source/code
  hashes, PTX instruction evidence and optional cache-sensitive timing samples.
- Added CPU-only controller tests. CUDA compilation and GH200 execution remain
  to be validated on grace2; these changes are AI-assisted code for review.

## 2026-10-30 — TMA code annotations

- Added Greek stage comments to `thesis/microbenchmarks/tma_cpu/tma_cpu.cu`:
  (1) host allocation and known values, (2) ordinary GPU-load control,
  (3) TMA descriptor/transfer/wait/output, (4) element-by-element verification,
  and (5) NUMA page queries before and after the kernels.
- Removed the introductory AI-related source comment. Executable code, trial
  parameters and result formats are unchanged; GH200 validation is still pending.

## 2026-10-30 — Απλούστερα σχόλια και πηγή του TMA παραδείγματος

- Απλοποιήθηκαν τα σχόλια στα τέσσερα αρχεία κώδικα του TMA πειράματος,
  με σαφή διάκριση CPU thread, GPU thread και GPU block.
- Στην αρχή του `tma_cpu.cu` προστέθηκε ο σύνδεσμος του παραδείγματος NVIDIA
  CUDA 12.8.1, ενότητα 7.29.2, στο οποίο βασίζονται ο TMA kernel και το barrier.
  Προστέθηκε επίσης η τεκμηρίωση για mapped host memory.
- Η λογική, τα μεγέθη δοκιμών και τα αποτελέσματα δεν άλλαξαν.
  Το προηγούμενο GH200 run `tma_cpu_20261030T123257_533342Z`, στο commit
  `7a27765`, πέρασε και τις έξι δοκιμές των 4 KiB. Σε αυτό το run επιβεβαιώθηκε
  η θέση των host σελίδων πριν και μετά, ενώ το αίτημα `mbind` απέτυχε.
- Για την αλλαγή σχολίων πέρασαν οι έξι υπάρχοντες έλεγχοι του Python runner,
  ο έλεγχος σύνταξης Bash και η σύγκριση της λογικής πριν/μετά.
