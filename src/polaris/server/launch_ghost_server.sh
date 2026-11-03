#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POLICY_PATH1="${POLICY_PATH:-$HOME/lerobot/outputs/train/pick_toys/checkpoints/last/pretrained_model}"
PORT1="${PORT:-8767}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"

echo "Starting GHOST server..."
echo "  Policy:            $POLICY_PATH1"
echo "  Port:              $PORT1"
echo "  Open-loop horizon: $OPEN_LOOP_HORIZON"

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/ghost_server.py" \
    --policy_path "$POLICY_PATH1" \
    --open_loop_horizon "$OPEN_LOOP_HORIZON" \
    --port "$PORT1" &
PID=$!

echo "GHOST server running: PID=$PID (port $PORT1)"
echo "Press Ctrl+C to stop"

trap "echo 'Shutting down...'; kill $PID 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "GHOST server shut down."
