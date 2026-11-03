#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_dir_1="$HOME/pp_data/pick_place_toys/pick_toys_simrobot_100/processed_data_pkl/expert_demos"
bc_weight_1="$HOME/Point-Policy/point_policy/exp_local/2026.05.22/pick_toys_simrobot_100/deterministic/223855_hidden_dim_256/snapshot/100000.pt"

TASK_NAME_1="pick_place_toys"
PORT_1=8765

echo "Starting server 1 on port $PORT_1..."
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/point_policy_server.py" \
    --bc_weight "$bc_weight_1" \
    --port "$PORT_1" \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" \
    "suite.use_robot_points=true" "suite.use_object_points=true" \
    "experiment=eval_point_policy" "suite/task/franka_env=${TASK_NAME_1}" \
    "data_dir=${data_dir_1}" &
PID1=$!

echo "Server running: PID1=$PID1 (port $PORT_1)"
echo "Press Ctrl+C to stop"

trap "echo 'Shutting down...'; kill $PID1 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "Server shut down."
