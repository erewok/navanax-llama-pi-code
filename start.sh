#!/bin/bash
# Legacy wrapper — use llama-launch.sh for model switching
# This defaults to Qwen3.6 35B-A3B Q6_K (best balance for M5 Pro / 64GB)
exec /Users/erewok/llama/llama-launch.sh qwen3-35b-q8
