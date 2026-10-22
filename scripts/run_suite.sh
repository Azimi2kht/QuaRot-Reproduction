#!/usr/bin/env bash
set -euo pipefail
cd /home/sazimido25/quarot-reproduction
run() {
    local experiment="$1"
    shift
    if [[ -s "results/$experiment/metrics.json" ]]; then
        echo "Already completed: $experiment"
        return
    fi
    echo "Starting $experiment at $(date -Is)"
    EXPERIMENT_NAME="$experiment" .venv/bin/python -u scripts/run_experiment.py \
      --model HuggingFaceTB/SmolLM2-1.7B --bsz 1 --seed 0 --save_name "$experiment" "$@" \
      > "logs/$experiment.log" 2>&1
    echo "Completed $experiment at $(date -Is)"
}
run baseline
run rotation_only --rotate --fp32_had
quant=(--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_clip_ratio 0.9 --k_groupsize 64 --v_groupsize 64 --k_asym --v_asym --k_clip_ratio 0.95 --v_clip_ratio 0.95)
run quarot_4bit --rotate --fp32_had "${quant[@]}"
run unrotated_4bit "${quant[@]}"
