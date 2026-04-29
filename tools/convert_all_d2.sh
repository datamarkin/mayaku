#!/usr/bin/env bash
# Drive the full D2 parity pipeline for the 12 model-zoo configs listed in
# tools/d2_model_zoo.tsv. For each row:
#   1. Download the .pkl to d2_pkls/ if missing.
#   2. Convert it into models/coco/<task>/<config_name>.pth (skipped when
#      the .pth is newer than the .pkl).
#   3. Run `mayaku eval` against COCO val2017 with --output to capture
#      metrics.json.
#   4. Compare the actual APs to the expected APs from the manifest;
#      flag any deviation outside ±AP_TOLERANCE as a fail.
# After all rows are processed, write docs/d2_parity_report.md and exit
# non-zero if any model failed.
#
# Flags:
#   --only NAME      Process only the row whose config_name == NAME.
#   --skip-eval      Download + convert but skip the eval step.
#   --gpu N          Set CUDA_VISIBLE_DEVICES=N for the eval subprocess.
#   --tolerance F    Override AP_TOLERANCE (default 0.3).
#
# Usage:
#   bash tools/convert_all_d2.sh
#   bash tools/convert_all_d2.sh --only mask_rcnn_R_50_FPN_3x
#   bash tools/convert_all_d2.sh --gpu 1 --tolerance 0.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="$REPO_ROOT/tools/d2_model_zoo.tsv"
PKL_DIR="$REPO_ROOT/d2_pkls"
EVAL_DIR="$PKL_DIR/.eval"
REPORT="$REPO_ROOT/docs/d2_parity_report.md"
COCO_IMAGES="/mnt/markin/Intranet/coco/val2017"
COCO_ANNOT_DIR="/mnt/markin/Intranet/coco/annotations"

ONLY=""
SKIP_EVAL=0
GPU=""
AP_TOLERANCE="0.3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --tolerance) AP_TOLERANCE="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "$MANIFEST" ]]; then
    echo "manifest not found: $MANIFEST" >&2
    exit 2
fi

mkdir -p "$PKL_DIR" "$EVAL_DIR" "$(dirname "$REPORT")"

# Per-row results accumulated as TSV lines for the final report.
RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT
FAILED=0
PROCESSED=0

# Compare a measured AP to expected; emit "PASS"/"FAIL"/"SKIP" and the delta.
compare_ap() {
    local actual="$1" expected="$2"
    if [[ "$expected" == "-" || "$actual" == "-" ]]; then
        echo "SKIP	-"
        return
    fi
    python3 - "$actual" "$expected" "$AP_TOLERANCE" <<'PY'
import sys
actual, expected, tol = (float(x) for x in sys.argv[1:4])
delta = actual - expected
status = "PASS" if abs(delta) <= tol else "FAIL"
print(f"{status}\t{delta:+.2f}")
PY
}

# Read metrics.json and extract the AP for a given task; "-" if absent.
# pycocotools returns AP as a fraction in [0, 1]; D2 MODEL_ZOO publishes
# it as a percentage (e.g. 40.2). Multiply by 100 to compare on the same
# scale.
extract_ap() {
    local metrics_json="$1" task="$2"
    if [[ ! -f "$metrics_json" ]]; then
        echo "-"
        return
    fi
    python3 - "$metrics_json" "$task" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
task = sys.argv[2]
if task in data and "AP" in data[task]:
    print(f"{data[task]['AP'] * 100:.2f}")
else:
    print("-")
PY
}

process_row() {
    local config_name="$1" family="$2" mayaku_cfg="$3" pkl_url="$4"
    local gt_json="$5" exp_box="$6" exp_mask="$7" exp_kpt="$8"

    local task_subdir
    case "$family" in
        faster_rcnn)   task_subdir="detection" ;;
        mask_rcnn)     task_subdir="segmentation" ;;
        keypoint_rcnn) task_subdir="keypoints" ;;
        *) echo "unknown family: $family" >&2; return 1 ;;
    esac

    local pkl_path="$PKL_DIR/$(basename "$pkl_url")"
    local pth_path="$REPO_ROOT/models/coco/$task_subdir/$config_name.pth"
    local out_dir="$EVAL_DIR/$config_name"
    local gt_path="$COCO_ANNOT_DIR/$gt_json"

    PROCESSED=$((PROCESSED + 1))
    echo
    echo "=== [$PROCESSED] $config_name ($family) ==="

    # 1. Download.
    if [[ ! -f "$pkl_path" ]]; then
        echo "[download] $pkl_url"
        wget --quiet --show-progress -O "$pkl_path.partial" "$pkl_url"
        mv "$pkl_path.partial" "$pkl_path"
    else
        echo "[download] cached: $pkl_path"
    fi

    # 2. Convert (idempotent: skip if .pth is newer than .pkl).
    mkdir -p "$(dirname "$pth_path")"
    if [[ -f "$pth_path" && "$pth_path" -nt "$pkl_path" ]]; then
        echo "[convert]  cached: $pth_path"
    else
        echo "[convert]  $pkl_path -> $pth_path"
        python3 "$REPO_ROOT/tools/convert_d2_checkpoint.py" "$pkl_path" -o "$pth_path"
    fi

    if [[ "$SKIP_EVAL" == "1" ]]; then
        echo "[eval]     skipped (--skip-eval)"
        printf "%s\t%s\t%s\t-\t%s\t-\t-\t%s\t-\t-\t%s\t-\t-\tSKIPPED\n" \
            "$config_name" "$family" "$exp_box" "$exp_mask" "$exp_kpt" "convert-only" >> "$RESULTS_FILE"
        return 0
    fi

    # 3. Eval.
    if [[ ! -f "$gt_path" ]]; then
        echo "[eval]     SKIP — annotations not found: $gt_path"
        printf "%s\t%s\t%s\tFAIL\t%s\t-\t-\t%s\t-\t-\t%s\t-\t-\tnoannot\n" \
            "$config_name" "$family" "$exp_box" "$exp_mask" "$exp_kpt" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        return 0
    fi

    mkdir -p "$out_dir"
    local log="$out_dir/eval.log"
    local metrics="$out_dir/metrics.json"

    # Eval cache: skip if metrics.json is newer than the .pth (means it
    # was produced by *this* checkpoint and not stale).
    if [[ -f "$metrics" && "$metrics" -nt "$pth_path" ]]; then
        echo "[eval]     cached: $metrics"
    else
        echo "[eval]     $mayaku_cfg --weights $pth_path --json $(basename "$gt_path")"
        local mayaku_cmd=(mayaku eval "$mayaku_cfg"
            --weights "$pth_path"
            --json "$gt_path"
            --images "$COCO_IMAGES"
            --output "$out_dir"
            --device cuda)

        local rc=0
        if [[ -n "$GPU" ]]; then
            # CUDA_DEVICE_ORDER=PCI_BUS_ID makes index N match `nvidia-smi`,
            # not torch's default fastest-first ordering — so on a mixed
            # 3060+3090 host, --gpu 1 reliably picks the 3090.
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
                "${mayaku_cmd[@]}" >"$log" 2>&1 || rc=$?
        else
            "${mayaku_cmd[@]}" >"$log" 2>&1 || rc=$?
        fi
        if (( rc != 0 )); then
            echo "[eval]     FAILED (exit $rc); see $log"
            printf "%s\t%s\t%s\tFAIL\t%s\t-\t-\t%s\t-\t-\t%s\t-\t-\teval-rc=%d\n" \
                "$config_name" "$family" "$exp_box" "$exp_mask" "$exp_kpt" "$rc" >> "$RESULTS_FILE"
            FAILED=$((FAILED + 1))
            return 0
        fi
    fi

    # 4. Parse + compare.
    local act_box act_mask act_kpt
    act_box=$(extract_ap "$metrics" bbox)
    act_mask=$(extract_ap "$metrics" segm)
    act_kpt=$(extract_ap "$metrics" keypoints)

    local box_status box_delta mask_status mask_delta kpt_status kpt_delta
    IFS=$'\t' read -r box_status box_delta < <(compare_ap "$act_box" "$exp_box")
    IFS=$'\t' read -r mask_status mask_delta < <(compare_ap "$act_mask" "$exp_mask")
    IFS=$'\t' read -r kpt_status kpt_delta < <(compare_ap "$act_kpt" "$exp_kpt")

    local notes=""
    if [[ "$box_status" == "FAIL" || "$mask_status" == "FAIL" || "$kpt_status" == "FAIL" ]]; then
        FAILED=$((FAILED + 1))
        notes=$(diagnose "$family" "$box_delta" "$mask_delta" "$kpt_delta")
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$config_name" "$family" \
        "$exp_box" "$box_status" "$act_box" "$box_delta" \
        "$exp_mask" "$mask_status" "$act_mask" "$mask_delta" \
        "$exp_kpt" "$kpt_status" "$act_kpt" "$kpt_delta" >> "$RESULTS_FILE"
    if [[ -n "$notes" ]]; then
        printf "    note: %s\n" "$notes"
    fi
    echo "[eval]     box=$act_box (exp $exp_box, $box_status $box_delta) mask=$act_mask kpt=$act_kpt"
}

# Heuristic root-cause notes per docs/possible_bugs.md categories.
diagnose() {
    local family="$1" box_delta="$2" mask_delta="$3" kpt_delta="$4"
    python3 - "$family" "$box_delta" "$mask_delta" "$kpt_delta" <<'PY'
import sys
fam, b, m, k = sys.argv[1:5]
def fnum(x):
    try:
        return float(x)
    except ValueError:
        return None
b, m, k = fnum(b), fnum(m), fnum(k)
notes = []
if b is not None and b < -1.5:
    notes.append("large negative box AP delta — check stem channel-order flip and stride_in_1x1 (see docs/possible_bugs.md)")
if fam == "mask_rcnn" and m is not None and (b is None or abs(b) < 0.5) and m < -1.0:
    notes.append("mask AP off while box AP is fine — check mask head rename / num_conv")
if fam == "keypoint_rcnn" and k is not None and (b is None or abs(b) < 0.5) and k < -1.0:
    notes.append("keypoint AP off while box AP is fine — check keypoint head rename / score_lowres -> deconv")
if b is not None and m is not None and b < -1.0 and m < -1.0 and abs(b - m) < 0.5:
    notes.append("box+mask off by similar amount — check FrozenBN replacement on the backbone")
print("; ".join(notes) or "no heuristic match — check the eval log")
PY
}

write_report() {
    local total
    total=$(wc -l <"$RESULTS_FILE" | awk '{print $1}')
    local passed=$((total - FAILED))
    {
        echo "# Detectron2 Parity Report"
        echo
        echo "Generated by \`tools/convert_all_d2.sh\`. Tolerance: ±${AP_TOLERANCE} AP."
        echo
        if (( FAILED == 0 )); then
            echo "**${total}/${total} passed.**"
        else
            echo "**${passed}/${total} passed — see fails below.**"
        fi
        echo
        echo "| config | family | exp box | box | Δ | exp mask | mask | Δ | exp kpt | kpt | Δ |"
        echo "|---|---|---|---|---|---|---|---|---|---|---|"
        while IFS=$'\t' read -r name family eb bs ab bd em ms am md ek ks ak kd notes; do
            local box_cell mask_cell kpt_cell
            box_cell="$ab ($bs)"
            mask_cell="$am ($ms)"
            kpt_cell="$ak ($ks)"
            echo "| $name | $family | $eb | $box_cell | $bd | $em | $mask_cell | $md | $ek | $kpt_cell | $kd |"
        done <"$RESULTS_FILE"
        if (( FAILED > 0 )); then
            echo
            echo "## Failure notes"
            echo
            while IFS=$'\t' read -r name family eb bs ab bd em ms am md ek ks ak kd notes; do
                if [[ "$bs" == "FAIL" || "$ms" == "FAIL" || "$ks" == "FAIL" ]]; then
                    echo "- **$name**: $notes"
                fi
            done <"$RESULTS_FILE"
        fi
    } >"$REPORT"
    echo
    echo "Wrote $REPORT (${passed}/${total} passed, ${FAILED} failed)."
}

# Stream the manifest, skipping comments and blanks.
while IFS=$'\t' read -r config_name family mayaku_cfg pkl_url gt_json exp_box exp_mask exp_kpt; do
    case "$config_name" in
        ""|"#"*) continue ;;
    esac
    if [[ -n "$ONLY" && "$config_name" != "$ONLY" ]]; then
        continue
    fi
    process_row "$config_name" "$family" "$mayaku_cfg" "$pkl_url" \
        "$gt_json" "$exp_box" "$exp_mask" "$exp_kpt"
done <"$MANIFEST"

if (( PROCESSED == 0 )); then
    echo "no rows processed (manifest empty? --only filtered everything?)" >&2
    exit 2
fi

write_report

if (( FAILED > 0 )); then
    echo
    echo "$FAILED of $PROCESSED model(s) failed parity check (see $REPORT)." >&2
    exit 1
fi
