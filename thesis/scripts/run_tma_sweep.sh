#!/usr/bin/env bash
# Οι παράμετροι αλλάζουν εδώ ή μπροστά από την εντολή εκτέλεσης.
set -Eeuo pipefail
repo="$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
script="$(realpath "${BASH_SOURCE[0]}")"
src="$repo/thesis/microbenchmarks/tma_sweep/tma_sweep.cu"
readme="$repo/thesis/microbenchmarks/tma_sweep/README.md"
SIZES="${SIZES:-4KiB,8KiB,16KiB,32KiB,64KiB,128KiB,256KiB,512KiB,1MiB,2MiB,4MiB,8MiB,16MiB,32MiB,64MiB,128MiB,256MiB,512MiB,1GiB,2GiB,4GiB,8GiB,16GiB}"
TILES="${TILES:-32x32,32x128,64x64,128x32,64x128,128x64,128x128}"
SOURCES="${SOURCES:-hbm,pinned,registered}"
GPU_THREADS="${GPU_THREADS:-128}"
TIMEOUT_S="${TIMEOUT_S:-900}"
mode="${1:-run}"
[[ $# -eq 0 || ($# -eq 1 && "$mode" == --plan) || ($# -eq 2 && "$mode" == --resume) ]] || {
    echo 'Usage: bash run_tma_sweep.sh [--plan | --resume RUN_FOLDER]' >&2; exit 2;
}
[[ "$GPU_THREADS" =~ ^(64|128|256)$ && "$TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || exit 2

bytes() {
    [[ "$1" =~ ^([1-9][0-9]*)(KiB|MiB|GiB)$ ]] || return 2
    local n="${BASH_REMATCH[1]}" unit="${BASH_REMATCH[2]}" factor
    case "$unit" in KiB) factor=1024;; MiB) factor=1048576;; GiB) factor=1073741824;; esac
    (( n <= 16777216 )) || return 2
    local result=$((n*factor))
    (( result >= 4096 && result <= 17179869184 && (result & (result-1)) == 0 )) || return 2
    echo "$result"
}

IFS=, read -r -a size_labels <<< "$SIZES"
IFS=, read -r -a tiles <<< "$TILES"
IFS=, read -r -a sources <<< "$SOURCES"
sizes=()
declare -A seen=()
for label in "${size_labels[@]}"; do
    n="$(bytes "$label")" || { echo "Invalid size: $label" >&2; exit 2; }
    [[ ! -v "seen[size:$n]" ]] || { echo 'Duplicate size' >&2; exit 2; }
    seen["size:$n"]=1; sizes+=("$n")
done
for tile in "${tiles[@]}"; do
    [[ ",$tile," =~ ,(32x32|32x128|64x64|128x32|64x128|128x64|128x128), ]] || exit 2
    [[ ! -v "seen[tile:$tile]" ]] || exit 2
    seen["tile:$tile"]=1
done
for source in "${sources[@]}"; do
    [[ "$source" =~ ^(hbm|pinned|registered)$ && ! -v "seen[source:$source]" ]] || exit 2
    seen["source:$source"]=1
done

# Μετράμε μόνο συνδυασμούς όπου η είσοδος αποτελείται από ολόκληρα tiles.
groups=0
for n in "${sizes[@]}"; do
    for tile in "${tiles[@]}"; do
        h="${tile%x*}"; w="${tile#*x}"; tile_bytes=$((h*w*4))
        if (( n >= tile_bytes && n % tile_bytes == 0 )); then
            groups=$((groups+${#sources[@]}))
        fi
    done
done
(( groups > 0 )) || { echo 'No valid configurations' >&2; exit 2; }
expected=$((groups*3))
echo "Plan: $groups allocation groups, $expected kernel measurements; one launch each, no warmup."
[[ "$mode" != --plan ]] || exit 0

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fa3
set -u
toolkit=''
for candidate in "$CONDA_PREFIX/cuda-12.8.1" /usr/local/cuda-12.8; do
    if [[ -x "$candidate/bin/nvcc" ]] && [[ "$("$candidate/bin/nvcc" --version)" == *'release 12.8,'* ]]; then
        toolkit="$candidate"; break
    fi
done
[[ -n "$toolkit" ]] || { echo 'CUDA Toolkit 12.8 not found' >&2; exit 2; }
for command in nvidia-smi timeout flock sha256sum awk cmp; do command -v "$command" >/dev/null; done
mkdir -p "$repo/thesis/outputs"
exec 9>"$repo/thesis/outputs/.tma_sweep.lock"
flock -n 9 || { echo 'Another sweep is running from this repo' >&2; exit 2; }
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ ! "$apps" =~ [0-9] ]] || { echo 'GPU compute process detected' >&2; exit 2; }
params="$(printf '%s\n' "SIZES=$SIZES" "TILES=$TILES" "SOURCES=$SOURCES" "GPU_THREADS=$GPU_THREADS" "TIMEOUT_S=$TIMEOUT_S" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}")"
hashes="$(sha256sum "$src" "$script" "$readme" | awk '{print $1}')"
compiler="$("$toolkit/bin/nvcc" --version)"
identity="$(nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader)"
if [[ "$mode" == --resume ]]; then
    out="$(realpath "$2")"
    [[ "$out" == "$repo/thesis/outputs/"* ]] || exit 2
    for item in config source_hashes compiler gpu_identity; do
        case "$item" in config) current="$params";; source_hashes) current="$hashes";; compiler) current="$compiler";; gpu_identity) current="$identity";; esac
        cmp -s "$out/$item.txt" <(printf '%s\n' "$current") || {
            echo "Resume mismatch: $item" >&2; exit 2;
        }
    done
else
    out="$(mktemp -d "$repo/thesis/outputs/tma_sweep_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    printf '%s\n' "$params" > "$out/config.txt"
    printf '%s\n' "$hashes" > "$out/source_hashes.txt"
    printf '%s\n' "$compiler" > "$out/compiler.txt"
    printf '%s\n' "$identity" > "$out/gpu_identity.txt"
    cp "$src" "$out/tma_sweep.cu"
    cp "$script" "$out/runner-used.sh"
    cp "$readme" "$out/README.md"
    git -C "$repo" rev-parse HEAD > "$out/base_commit.txt"
fi
archived="$(sha256sum "$out/tma_sweep.cu" "$out/runner-used.sh" "$out/README.md" | awk '{print $1}')"
[[ "$archived" == "$hashes" ]] || { echo 'Saved source files do not match' >&2; exit 2; }
exec > >(tee -a "$out/run.log") 2>&1
echo "Outputs: $out"
trap 'result=$?; printf "exit_code=%s\n" "$result" > "$out/status.txt"; echo "Outputs: $out"' EXIT
nvidia-smi > "$out/gpu-before.txt"
lscpu > "$out/numa-topology.txt"
mkdir -p "$out/build" "$out/cases"
binary="$out/build/tma_sweep"
echo 'Compiling CUDA benchmark...'
"$toolkit/bin/nvcc" -std=c++17 -O3 -lineinfo -arch=sm_90a "$out/tma_sweep.cu" \
    -ptx -o "$out/build/tma_sweep.ptx" > "$out/build.log" 2>&1
"$toolkit/bin/nvcc" -std=c++17 -O3 -lineinfo -arch=sm_90a "$out/tma_sweep.cu" \
    -L"$toolkit/lib64/stubs" -lcuda -o "$binary" >> "$out/build.log" 2>&1
grep -F 'cp.async.bulk.tensor.2d' "$out/build/tma_sweep.ptx" > "$out/tma-instructions.txt"
header="$("$binary" --header)"

valid_rows() {
    awk -F, -v head="$header" -v n="$n" -v h="$h" -v w="$w" -v s="$source" -v t="$GPU_THREADS" '
        NR==1 {if ($0!=head) bad=1; next}
        {if (NF<23 || $1!=n || $2!=h || $3!=w || $4!=h*w*4 || $5!=s || $7!=t) bad=1;
         if ($6!="loads_direct" && $6!="loads_shared" && $6!="tma_shared") bad=1;
         if (++seen[$6]!=1) bad=1;
         if ($11!="ok" && $11!="error" && $11!="not_run" && $11!="device_fault" && $11!="worker_error") bad=1;
         if ($11=="ok" && ($9!~/^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || $9+0<=0)) bad=1}
        END {if (bad || NR!=4) exit 1}' "$folder/rows.csv"
}

# Ξαναφτιάχνουμε το συνολικό CSV από τις ολοκληρωμένες ομάδες.
summary() {
    printf '%s\n' "$header" > "$out/results.csv.tmp"
    for folder_done in "$out"/cases/*; do
        [[ -f "$folder_done/done" ]] || continue
        tail -n +2 "$folder_done/rows.csv" >> "$out/results.csv.tmp"
    done
    mv "$out/results.csv.tmp" "$out/results.csv"
    awk -F, -v expected="$expected" '
        NR>1 {n++; key=$5 "/" $6; total[key]++;
              if ($11=="ok") {ok++; passed[key]++} else {bad++; if (bad<=20)
                  errors=errors sprintf("- %s bytes, tile %sx%s, %s, %s: %s\n", $1,$2,$3,$5,$6,$11)}
              if ($11=="ok" && $5!="hbm") {host++; if ($14==1) placed++}}
        END {print "# TMA sweep summary\n";
             print "- Expected measurements: " expected;
             print "- Recorded measurements: " n+0;
             print "- Correct outputs: " ok+0;
             print "- Failed or not run: " bad+0;
             print "- Missing measurements: " expected-n;
             print "- Correct host outputs with CPU page placement verified: " placed+0 " / " host+0;
             print "\n## Results by source and kernel\n";
             for (key in total) print "- " key ": " passed[key]+0 " / " total[key] " correct";
             if (bad) print "\n## First failures\n\n" errors;
             print "\nOne timed launch per row. No warmup or cache control.";
             print "GPU events include the complete copy kernel and its output writes.";
             print "Correct output and CPU page placement are separate checks."}' \
        "$out/results.csv" > "$out/summary.md.tmp"
    mv "$out/summary.md.tmp" "$out/summary.md"
}

active=''
stop() {
    trap - INT TERM
    if [[ -n "$active" ]]; then kill -TERM "$active" 2>/dev/null || true; wait "$active" || true; fi
    exit 130
}
trap stop INT TERM
trap 'result=$?; summary; printf "exit_code=%s\n" "$result" > "$out/status.txt"; echo "Outputs: $out"' EXIT
finished=0
started=$SECONDS
# Η μπάρα μετρά ολοκληρωμένες ομάδες, μαζί με όσες κρατάμε από προηγούμενο run.
progress() {
    local completed="$1" width=30 filled bar empty
    filled=$((completed*width/groups))
    printf -v bar '%*s' "$filled" ''
    printf -v empty '%*s' "$((width-filled))" ''
    printf 'Progress: [%s%s] %3d%% (%d/%d groups) elapsed=%ss\n' \
        "${bar// /#}" "${empty// /-}" "$((completed*100/groups))" \
        "$completed" "$groups" "$((SECONDS-started))"
}
progress 0
for si in "${!sizes[@]}"; do
    n="${sizes[si]}"
    for ti in "${!tiles[@]}"; do
        tile="${tiles[ti]}"; h="${tile%x*}"; w="${tile#*x}"
        (( n >= h*w*4 && n % (h*w*4) == 0 )) || continue
        for pi in "${!sources[@]}"; do
            source="${sources[pi]}"
            folder="$out/cases/${n}_${tile}_${source}"
            rotation=$(((si+ti+pi)%3))
            mkdir -p "$folder"
            finished=$((finished+1))
            if [[ -f "$folder/done" ]]; then
                valid_rows || { echo "Invalid saved case: $folder" >&2; exit 2; }
                progress "$finished"
                continue
            fi
            echo "[$finished/$groups] ${size_labels[si]} tile=$tile source=$source elapsed=$((SECONDS-started))s"
            rc=0
            timeout --kill-after=5s "${TIMEOUT_S}s" "$binary" "$n" "$h" "$w" \
                "$source" "$GPU_THREADS" "$rotation" "$folder/rows.csv" > "$folder/worker.log" 2>&1 &
            active=$!
            wait "$active" || rc=$?
            active=''
            trustworthy=1
            if (( rc != 0 && rc != 2 && rc != 3 )); then trustworthy=0; fi
            if (( rc != 0 )) && [[ -f "$folder/rows.csv" ]] &&
                awk -F, 'NR>1 && $11!="ok" {bad=1} END {exit bad}' "$folder/rows.csv"; then
                trustworthy=0
            fi
            if (( !trustworthy )) || [[ ! -f "$folder/rows.csv" ]] || ! valid_rows; then
                [[ ! -f "$folder/rows.csv" ]] || mv "$folder/rows.csv" "$folder/rows.incomplete.csv"
                timeout --kill-after=5s 10s "$binary" --failure "$n" "$h" "$w" "$source" "$GPU_THREADS" "$rotation" \
                    "$folder/rows.csv" "worker_exit_${rc}_missing_or_invalid_result"
                [[ "$rc" != 0 ]] || rc=2
            fi
            touch "$folder/done"
            summary
            progress "$finished"
            # Σε GPU fault, timeout ή signal σταματάμε αντί να συνεχίζουμε σε άγνωστη κατάσταση.
            if (( rc == 3 || rc == 124 || rc >= 128 )); then exit "$rc"; fi
        done
    done
done
awk -F, -v expected="$expected" 'NR>1 {n++; if ($11!="ok") bad++} END {exit (n!=expected || bad>0)}' "$out/results.csv"
