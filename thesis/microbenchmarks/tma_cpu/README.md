# TMA από μνήμη CPU στο GH200 — αρχικό microbenchmark

Σκοπός: να ελέγξουμε αν το Hopper TMA μπορεί να διαβάσει συγκεκριμένες host
allocations στο grace2. Πρόκειται για ανεξάρτητο πείραμα μεταφοράς δεδομένων,
χωρίς Qwen, αλλαγές στο FA3 ή εγκατάσταση dependencies.

Ο κώδικας είναι AI-assisted προσχέδιο προς ανθρώπινο review. Οι CPU-only έλεγχοι
του runner δεν αποτελούν επιβεβαίωση compilation CUDA ή λειτουργίας σε GH200.
Δεν συνοδεύεται από επινοημένες μετρήσεις. Η πραγματική επαλήθευση γίνεται στο
διαθέσιμο slot του grace2.

## Εκτέλεση

Μέσα σε tmux, από το ενημερωμένο repository:

```bash
bash thesis/scripts/run_tma_cpu.sh
```

Το Bash ενεργοποιεί μόνο το υπάρχον Conda `fa3`, αναζητά το CUDA Toolkit 12.8.1
στο `$CONDA_PREFIX/cuda-12.8.1` ή `/usr/local/cuda-12.8` και καλεί τον controller.
Δεν χρησιμοποιεί το διαφορετικό `/usr/local/cuda` χωρίς έλεγχο έκδοσης.

Ο controller ελέγχει αρχικά για άλλες GPU compute διεργασίες, δημιουργεί μοναδικό
`thesis/outputs/tma_cpu_<UTC>/`, καταγράφει πηγαίο κώδικα/commit/compiler και
κάνει compilation με `-arch=sm_90a`. Ελέγχει ότι το παραγόμενο PTX περιέχει
`cp.async.bulk.tensor.2d`. Τα binaries/PTX μένουν στον ignored φάκελο `build/`
δίπλα στον κώδικα. Δεν γίνεται commit/push ή αλλαγή driver από τον runner.

Η προεπιλογή είναι έξι μικρές δοκιμές των 4 KiB με 3 timed επαναλήψεις:

| Πηγή | Ordinary loads | TMA |
|---|---|---|
| HBM / `cudaMalloc` | control | control TMA kernel |
| `cudaHostAllocMapped` | host accessibility | host TMA |
| `mmap` + `cudaHostRegisterMapped` | host accessibility | host TMA |

Κάθε συνδυασμός έχει δική του διεργασία και CUDA context. Επιπλέον, μέσα σε κάθε
worker, ordinary loads ελέγχουν την **ίδια allocation** πριν από τον επιλεγμένο
kernel. Η είσοδος host ΔΕΝ αντιγράφεται στην HBM. Το output στην HBM επιστρέφει
με `cudaMemcpy` στην CPU αποκλειστικά για την επαλήθευση όλων των στοιχείων.

Μετά από επιτυχημένο smoke και έλεγχο των logs μπορεί να ζητηθεί:

```bash
bash thesis/scripts/run_tma_cpu.sh --bench
```

Αυτό εκτελεί ξανά το smoke και, μόνο αν περάσουν όλες οι περιπτώσεις, προσθέτει
1 MiB και 64 MiB με 20 timed επαναλήψεις. Δεν αλλάζει το περιβάλλον. Η πρόοδος
εμφανίζεται ως `[1/6]`, `[2/6]` κ.λπ. (ή έως `[18/18]` στο bench).

## Πώς λειτουργεί ο CUDA kernel

1. Η CPU γεμίζει την πηγή με επαναλήψιμο uint32 pattern. Τα δεδομένα ελέγχουν
   καθαρά τη μεταφορά bits, χωρίς αριθμητική BF16 ή sparsity σε αυτή τη φάση.
2. Για HBM αντιγράφει τις τιμές στην allocation του `cudaMalloc`. Για host,
   ανακτά τη GPU-visible διεύθυνση με `cudaHostGetDevicePointer`.
3. Δημιουργεί `CUtensorMap` για πίνακα πλάτους 32 στοιχείων. Κάθε CTA φορτώνει
   ένα διαφορετικό tile 32x32 (4 KiB), χωρίς swizzle.
4. Ένα thread εκκινεί `cp_async_bulk_tensor_2d_global_to_shared`. Το block
   περιμένει το transaction με `cuda::barrier`, όπως στο παράδειγμα CUDA 12.8.
5. Τα threads αποθηκεύουν το tile σε διαφορετική περιοχή του output.
6. Η CPU συγκρίνει κάθε στοιχείο. Πριν από το μετρημένο correctness launch,
   το output μηδενίζεται ώστε το ordinary-load control να μη συγκαλύπτει αποτυχία.
7. CUDA events μετρούν τον επιλεγμένο kernel. Οι επαναλήψεις διαβάζουν την ίδια
   πηγή και η ordinary-load προδοκιμή έχει ήδη ζεστάνει τις caches.

## CPU allocation και φυσική τοποθέτηση

Το πρόγραμμα διακρίνει CUDA memory type από φυσική NUMA τοποθέτηση. Ελέγχει ότι
η CUDA χαρακτηρίζει τις host allocations ως host memory. Καταγράφει τον NUMA
node της CPU που εκτελεί το πρόγραμμα και ερωτά τη θέση **όλων** των source
pages με `move_pages`, πριν και μετά τους kernels, χωρίς αίτημα migration.

Στην registered περίπτωση επιχειρεί `mbind(MPOL_BIND)` πριν από το CPU first
touch και καταγράφει αν η πολιτική εφαρμόστηκε. Στην pinned περίπτωση η επιλογή
τοποθέτησης αρχίζει από το CUDA API. Δεν ζητούνται sudo ή αλλαγές NUMA του συστήματος.

Αν το page query απαγορεύεται ή αποτυγχάνει, η τοποθέτηση αναφέρεται ως άγνωστη.
Ένα `correctness_pass` αποδεικνύει σωστά δεδομένα, **όχι από μόνο του CPU DRAM
residency**. Το `cpu_residency_verified_before_and_after` απαιτεί όλες τις
σελίδες στον συγκεκριμένο CPU node και στις δύο μετρήσεις. Ακόμη και αυτό δεν
είναι συνεχής παρακολούθηση residency ή απόδειξη ότι κάθε timed read πέρασε από
το NVLink-C2C: μπορεί να εξυπηρετηθεί από cache.

## Αποτελέσματα και σφάλματα

- `manifest.json`: commit, hashes, compiler, build commands και επιλογές.
- `run.log`, `build.log`, `gpu-before.txt`, `numa-topology.txt`: συνολικό log.
- `tma-instruction.txt`: εντολές TMA που βρέθηκαν στο παραγόμενο PTX.
- `results.csv`, `results.json`, `summary.json`, `status.txt`: αποτελέσματα.
- `<bytes>_<source>_<method>/worker.log` και `result.json`: κάθε περίπτωση.
- Στο `thesis/CHANGELOG.MD` προστίθεται η διαδρομή και το exit code του run.

Exit 0: όλες οι ζητούμενες correctness περιπτώσεις ολοκληρώθηκαν. Το summary
αναφέρει χωριστά την τεκμηρίωση CPU residency. Exit 2: αποτυχία, timeout ή
ελλιπές πείραμα, που χρειάζεται ανάγνωση του worker log. Αποτυχία descriptor
encoding και αποτυχία kernel διακρίνονται από την τελευταία ένδειξη `STAGE`.
Αποτυχία HBM control δεν επιτρέπει συμπέρασμα για CPU TMA.

Κάθε worker έχει εξωτερικό timeout 45 s (`--timeout 90` το αλλάζει). Σε timeout,
signal termination ή αποτυχία HBM control σταματούν οι υπόλοιπες δοκιμές. Μια
διεργασία με σφάλμα CUDA δεν επαναχρησιμοποιείται. Timeout/kill δεν αποδεικνύει
ότι η CPU μνήμη δεν υποστηρίζεται. Ο εξωτερικός timeout δεν εγγυάται ανάκτηση
μιας GPU αν έχει συμβεί driver/hardware hang. Η GPU πρέπει να είναι διαθέσιμη
σε συνεννόηση με το εργαστήριο· ο αρχικός process check δεν αποτελεί κράτηση.

## Ερμηνεία χρόνων

`median_kernel_ms` είναι η upper median των CUDA event samples. Το
`effective_source_gb_s` είναι `source_bytes / kernel_seconds / 1e9` και
**δεν** είναι άμεσα μετρημένο bandwidth CPU DRAM, NVLink ή μόνο του TMA. Ο kernel
περιλαμβάνει και stores στο output. Τα μικρά working sets και οι επαναλήψεις
μπορεί να μετρούν κυρίως GPU cache. Για δημοσιεύσιμη ανάλυση streaming DRAM
χρειάζονται ξεχωριστό cache-controlled πρωτόκολλο και κατάλληλο profiling.

## Πηγές σχεδιασμού

- CUDA 12.8.1 Programming Guide, 7.29.2 (descriptor, alignment και barrier):
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
- Tensor map API:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
- Mapped host memory, Programming Guide 3.2.6.3:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mapped-memory

Η υποστήριξη TMA για αυτές τις host allocations δεν θεωρείται δεδομένη από τα
παραπάνω: είναι το ερώτημα του πραγματικού πειράματος.
