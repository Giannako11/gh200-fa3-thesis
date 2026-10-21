# Qwen32B — πλήρες μοντέλο, μνήμη ανά T και όριο χωρητικότητας

Το `run_qwen_stress.sh` εκτελεί το επόμενο πείραμα μετά το επιτυχημένο baseline `QWEN_INITIAL_FORWARD_OK`. Χρησιμοποιεί **ολόκληρο το Qwen/Qwen2.5-32B-Instruct με pretrained βάρη**, BF16, batch1, FA3, συνθετικά token IDs και ένα πλήρες causal prefill ανά T. Δεν είναι training ή επαναλαμβανόμενο autoregressive decode, ούτε το παλιό attention-kernel benchmark. Η μέτρηση χρόνου είναι βοηθητική, μία συγχρονισμένη εκτέλεση μετά από μικρό warmup, όχι στατιστικά τεκμηριωμένο throughput benchmark.

## Αμετάβλητο πρωτόκολλο

- Παίρνει **την πραγματική revision και το αποθηκευμένο model-config.json** από το πιο πρόσφατο επιτυχημένο `thesis/outputs/qwen_baseline_*`. Προαιρετικά δίνεται `--baseline-dir /absolute/path`. Δεν ερωτά το Hugging Face για νέο latest revision.
- Απαιτεί το υπάρχον περιβάλλον `fa3`: PyTorch2.11.0+cu128, Transformers4.57.1, flash-attn-3 3.0.1, GPU capability9.0. Δεν εγκαθιστά dependencies, δεν αλλάζει CUDA/FA3 και δεν κατεβάζει βάρη. Cache: το ίδιο `$HOME/models/huggingface`, με `local_files_only=True` και offline flags.
- Όλα τα floating parameters παραμένουν BF16 στη `cuda:0`, χωρίς quantization ή CPU/disk weight offload. Καταγράφει πραγματικό πλήθος/bytes παραμέτρων και συσκευή.
- Διατηρεί YaRN factor4, original positions32768 και configured context131072 για **όλες** τις δοκιμές. Δεν αλλάζει model/RoPE για να περάσει ένα μεγαλύτερο T.
- `use_cache=True`, άρα περιλαμβάνεται το KV cache ολόκληρου του prefix. `logits_to_keep=1`: το transformer επεξεργάζεται όλα τα T tokens, ενώ το vocabulary projection κρατά μόνο την τελευταία θέση, όπως στο baseline.
- Κάθε trial έχει καινούρια Python/CUDA διεργασία. Προηγούμενα KV caches, outputs ή αποτυχημένες allocations δεν παραμένουν στην επόμενη δοκιμή. Μόνο τα ήδη αποθηκευμένα βάρη διαβάζονται ξανά· δεν αντιγράφονται/σώζονται νέα checkpoints.
- Μετρά actual calls στις εγκατεστημένες FA3 entry functions και ελέγχει μία ανά layer σε warmup και επιτυχημένο measured prefill. Είναι απόδειξη του Python dispatch path, όχι GPU profiler trace ή σύγκριση αριθμητικής ορθότητας με reference attention.

## Σάρωση και τερματισμός

Αρχίζει από **1024,2048,4096,8192,16384,32768,65536,131072**. Αν όλα πετύχουν, συνεχίζει262144,524288,1048576. Το1M είναι ρητό όριο τερματισμού, όχι υπόσχεση ότι χωράει. Αν φτάσει εκεί χωρίς OOM, αναφέρει `max_t_reached_without_oom`· δεν επινοεί όριο OOM.

Πάνω από131072 κάθε row γράφει **`beyond_128k_capacity_only`**: πρόκειται για extrapolative stress test χωρητικότητας με το ίδιο model configuration. Δεν δηλώνεται υποστηριζόμενο/ποιοτικά ορθό inference μεγαλύτερου context. Αν αποτύχει από άλλο όριο του μοντέλου/kernel, καταγράφεται error και σταματά, χωρίς να βαφτιστεί CUDA OOM. Το μοντέλο παρέχει131072 context με τις επίσημες οδηγίες YaRN: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

Στο πρώτο CUDA OOM κατά `input_allocation` ή `prefill`, δεν δοκιμάζει αυθαίρετα μεγαλύτερα T. Από προεπιλογή κάνει binary refinement στο παρατηρημένο διάστημα success/OOM, με βήμα1024 tokens. Το summary αναφέρει **μεγαλύτερο παρατηρημένο success** και **μικρότερο παρατηρημένο capacity OOM**, με ρητή υπόθεση μονοτονικής χωρητικότητας. Δεν αποδεικνύει αποτέλεσμα για κάθε ενδιάμεσο μήκος. `--no-refine` αφήνει μόνο το αρχικό coarse διάστημα.

Timeout ανά trial:30 λεπτά. Συνολικός χρόνος ανά invocation:6 ώρες. Οι τιμές μπορούν να αλλάξουν στην πρώτη εκτέλεση με `--timeout-seconds` και `--max-wall-seconds`. Timeout, killed process, GPU busy, import/configuration/model-load error **δεν** αποτελούν context-capacity OOM. Η διεργασία που δημιουργήθηκε από τον runner τερματίζεται σε timeout/διακοπή· δεν γίνεται kill ξένων jobs.

## Εκτέλεση στο grace2

Αποθήκευσε τον runner μέσα στο repository, κάτω από `thesis/scripts/`, και μείνε στο **main**. Δεν χρειάζεται `git pull`, checkout, reset ή επανεκτέλεση του baseline. Σε tmux:

```bash
bash "$HOME/repos/gh200-fa3-thesis/thesis/scripts/run_qwen_stress.sh"
```

Για να περιορίσεις τη δοκιμή στο υποστηριζόμενο context:

```bash
bash "$HOME/repos/gh200-fa3-thesis/thesis/scripts/run_qwen_stress.sh" --max-t 131072
```

Οι παραπάνω εντολές χρησιμοποιούν την υπάρχουσα Conda εγκατάσταση στο `$HOME/miniconda3`. Ο runner ενεργοποιεί μόνο το ήδη εγκατεστημένο `fa3`. Αποτυγχάνει καθαρά αν λείπει baseline/cache, οι εκδόσεις διαφέρουν ή χρησιμοποιείται άλλη GPU. Δεν επιχειρεί αυτόματη επιδιόρθωση του περιβάλλοντος. Προηγούμενο επιτυχημένο baseline παραμένει αναγκαίο, αλλά δεν εγγυάται ότι όλα τα μεγαλύτερα T θα εκτελεστούν.

## Αποτελέσματα

Νέος μοναδικός φάκελος: `thesis/outputs/qwen_stress_<UTC>_<unique>/`. Ο runner τυπώνει την ακριβή διαδρομή στην αρχή και στο τέλος.

| Αρχείο / μέτρηση | Τι σημαίνει |
|---|---|
|`results.csv` / `results.jsonl`|Μία ολοκληρωμένη εγγραφή ανά πραγματική απόπειρα. Γράφεται και γίνεται fsync μετά από κάθε trial. Περιλαμβάνει success/OOM/error/timeout, T,B, χρόνο, bytes/GiB, stage και μήνυμα σφάλματος.|
|`T_<...>/result.json` / `worker.log`|Πλήρη στοιχεία και log της συγκεκριμένης δοκιμής. Στο OOM διατηρείται το μήνυμα της αποτυχημένης allocation.|
|`summary.json`|Stop reason, μεγαλύτερο παρατηρημένο success, μικρότερο capacity OOM και ανάλυση του διαστήματος.|
|`manifest.json` / `baseline-model-*.json`|Pinned revision, αμετάβλητο configuration, protocol/code hashes, cache, flags, όρια και ορισμοί μετρήσεων.|
|`baseline_allocated_bytes`|GPU tensor allocations μετά τη φόρτωση και το16-token warmup, με warmup cache/output απελευθερωμένα. Περιλαμβάνει τα βάρη.|
|`peak_allocated_bytes`|Ακριβής counter reading του μέγιστου PyTorch tensor allocation, περιλαμβανομένων βαρών, input, KV cache και προσωρινών tensors, από πριν το input έως το τέλος του συγχρονισμένου prefill.|
|`peak_reserved_bytes`|Μέγιστη GPU μνήμη που διαχειρίστηκε ο PyTorch caching allocator στην ίδια περίοδο. Δεν προστίθεται στην allocated: είναι διαφορετική μέτρηση της ίδιας διαχείρισης μνήμης.|
|`peak_minus_baseline_bytes`|Πρόσθετο peak tensor allocation σε σχέση με το after-load/warmup baseline. Δεν ταυτίζεται μόνο με KV cache.|
|`device_total_bytes`, `device_used_before/after_bytes`|Πραγματική συσκευή/free/used memory από CUDA στα αναφερόμενα σημεία. Δεν υποθέτει HBM capacity από το εμπορικό όνομα GH200480GB.|
|`gpu_used_sampled_peak_mib` / `gpu-samples.csv`|Προσεγγιστικό device-wide peak μέσω nvidia-smi ανά500ms, περιλαμβάνει και loading. Μπορεί να χάσει βραχύβια peaks και περιλαμβάνει allocations εκτός PyTorch. Δεν παρουσιάζεται ως ακριβές μέγιστο. Αν άγνωστο, παραμένει κενό.|
|`cpu_process_peak_rss_bytes`|Linux ru_maxrss high-water mark ολόκληρης της worker διεργασίας, περιλαμβανομένου loading.|
|`forward_sampled_peak_rss_bytes`|Δειγματοληπτούμενο CPU RSS ανά50ms μόνο γύρω από το prefill. Δεν είναι ακριβές instantaneous peak.|
|`expected_kv_bytes`|Θεωρητικό μέγεθος BF16 KV από την αρχιτεκτονική και το T, όχι ανεξάρτητα μετρημένη KV allocation.|

Bytes είναι οι πρωτογενείς μετρήσεις. GiB = bytes /2^30· τα MiB του nvidia-smi είναι άλλη μονάδα. Δεν αθροίζουμε CPU RSS με GPU usage ως μοναδική φυσική μνήμη. Στο OOM το allocation που απορρίφθηκε **δεν** περιλαμβάνεται στο καταγεγραμμένο peak, επομένως δεν ισχυριζόμαστε ότι γνωρίζουμε την ακριβή μνήμη που θα απαιτούσε η επιτυχής ολοκλήρωση.

Ορισμοί PyTorch2.11: https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html και https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_reserved.html

## Αν διακοπεί

```bash
bash "$HOME/repos/gh200-fa3-thesis/thesis/scripts/run_qwen_stress.sh" \
  --resume /absolute/path/to/thesis/outputs/qwen_stress_<...>
```

Χρησιμοποίησε το ίδιο αρχικό runner και μόνο `--resume <directory>`. Φορτώνεται το αποθηκευμένο πρωτόκολλο, παραλείπονται τα ολοκληρωμένα success/capacity-OOM trials και επανεκτελείται η μη ολοκληρωμένη δοκιμή σε νέα διεργασία. Παλιό incomplete trial directory διατηρείται με suffix. Ολοκληρωμένο worker artifact που δεν πρόλαβε να περάσει στο index ανακτάται. Ένα torn τελευταίο JSONL append διατηρείται χωριστά και το πλήρες worker artifact επανεισάγεται. Το κλείσιμο του SSH χωρίς tmux μπορεί να διακόψει το job· το resume αποκαθιστά την πρόοδο, δεν κρατά ζωντανό ένα κλειστό session.

## Επαλήθευση αυτού του παραδοτέου

`bash -n`, Python compilation και CPU-only unit/integration checks καλύπτουν scheduling/refinement, όρια, baseline/Yarn/revision επιλογή, διατήρηση bytes/results, torn-tail recovery και πραγματικές child-process περιπτώσεις timeout/SIGKILL/import failure. Δεν έχει εκτελεστεί από τον συντάκτη στη GH200 ή με τα πραγματικά βάρη· ο πρώτος πραγματικός έλεγχος γίνεται στο διαθέσιμο slot του grace2. Δεν παραδίδονται συνθετικά performance/memory αποτελέσματα.

Τα νέα files προστίθενται στο main χωρίς αλλαγή baseline runner, kernels, dependency locks ή ιστορικών outputs. Το local stress run δεν κάνει αυτόματα commit/push τα πειραματικά δεδομένα· τα διατηρεί πλήρως κάτω από `thesis/outputs` για έλεγχο και μετέπειτα δημοσίευση.
