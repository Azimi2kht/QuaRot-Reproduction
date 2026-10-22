#!/usr/bin/env bash
set -euo pipefail
cd /home/sazimido25/quarot-reproduction
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HADAMARD_BACKEND=native
unset TASK_LIMIT
common=(--model HuggingFaceTB/SmolLM2-1.7B --bsz 1 --seed 0 --lm_eval --lm_eval_batch_size auto:4 --tasks piqa winogrande hellaswag arc_easy arc_challenge lambada_openai)
if [[ ! -s results/baseline/zero_shot.json ]]; then
 TASKS_ONLY=1 EXPERIMENT_NAME=baseline .venv/bin/python -u scripts/run_experiment.py "${common[@]}" --save_name baseline_tasks > logs/baseline_tasks.log 2>&1
fi
if [[ ! -s results/quarot_4bit/zero_shot.json ]]; then
 TASKS_ONLY=1 EXPERIMENT_NAME=quarot_4bit .venv/bin/python -u scripts/run_experiment.py "${common[@]}" --save_name quarot_tasks \
 --rotate --fp32_had --load_qmodel_path results/quarot_4bit/model.pt \
 --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --w_clip --a_clip_ratio 0.9 \
 --k_groupsize 64 --v_groupsize 64 --k_asym --v_asym --k_clip_ratio 0.95 --v_clip_ratio 0.95 > logs/quarot_tasks.log 2>&1
fi
