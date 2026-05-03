#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# llama-launch.sh — Launch llama-server with model switching
# Optimized for MacBook Pro M5 Pro / 64GB unified memory
# =============================================================================
#
# MODELS:
#   qwen3-35b-q6    Qwen3.6 35B-A3B MoE, Q6_K (~25GB) — best balance
#   qwen3-35b-q8    Qwen3.6 35B-A3B MoE, Q8_0 (~37GB) — max quality
#   gemma4-31b-q8   Gemma 4 31B, Q8_0 (~35GB) — frontier quality, dense
#   gemma4-27b-q8   Gemma 4 27B A4B, Q8_0 (~28GB) — fast MoE, great for coding
#
# USAGE:
#   ./llama-launch.sh                 # interactive model selector
#   ./llama-launch.sh qwen3-35b-q6    # launch Qwen3 Q6
#   ./llama-launch.sh qwen3-35b-q8    # launch Qwen3 Q8
#   ./llama-launch.sh gemma4-31b-q8   # launch Gemma4 31B Q8
#   ./llama-launch.sh gemma4-27b-q8   # launch Gemma4 27B Q8
#   ./llama-launch.sh list            # show available models
#   ./llama-launch.sh stop            # stop running llama-server
#   ./llama-launch.sh status          # show current status
# =============================================================================

LLAMA_DIR="/Users/erewok/llama"
CACHE_DIR="$LLAMA_DIR/cache"
LOG_DIR="$LLAMA_DIR/logs"
PID_FILE="$LLAMA_DIR/llama-server.pid"
PORT=3333
HOST="127.0.0.1"

mkdir -p "$CACHE_DIR" "$LOG_DIR"

# --- Model definitions ---
# Each model has: hf_repo, quant, name, threads, batch, ubatch, context,
#                 kv_cache_k, kv_cache_v, reasoning, max_tokens, context_window

declare -A MODEL_HF
declare -A MODEL_NAME
declare -A MODEL_THREADS
declare -A MODEL_BATCH
declare -A MODEL_UBATCH
declare -A MODEL_CTX
declare -A MODEL_CTK
declare -A MODEL_CTV
declare -A MODEL_REASONING
declare -A MODEL_MAX_TOKENS
declare -A MODEL_CTX_WINDOW

# Qwen3.6 35B-A3B MoE — Q6_K (25GB)
# Best balance of quality + speed + headroom on 64GB
MODEL_HF[qwen3-35b-q6]="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K"
MODEL_NAME[qwen3-35b-q6]="Qwen3.6 35B-A3B Q6_K (Local)"
MODEL_THREADS[qwen3-35b-q6]=10
MODEL_BATCH[qwen3-35b-q6]=1024
MODEL_UBATCH[qwen3-35b-q6]=256
MODEL_CTX[qwen3-35b-q6]=131072
MODEL_CTK[qwen3-35b-q6]="q8_0"
MODEL_CTV[qwen3-35b-q6]="q8_0"
MODEL_REASONING[qwen3-35b-q6]=true
MODEL_MAX_TOKENS[qwen3-35b-q6]=8192
MODEL_CTX_WINDOW[qwen3-35b-q6]=131072

# Qwen3.6 35B-A3B MoE — Q8_0 (37GB)
# Maximum quality, tight on memory — use only if you need the best
MODEL_HF[qwen3-35b-q8]="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"
MODEL_NAME[qwen3-35b-q8]="Qwen3.6 35B-A3B Q8_0 (Local)"
MODEL_THREADS[qwen3-35b-q8]=10
MODEL_BATCH[qwen3-35b-q8]=768
MODEL_UBATCH[qwen3-35b-q8]=192
MODEL_CTX[qwen3-35b-q8]=131072
MODEL_CTK[qwen3-35b-q8]="q8_0"
MODEL_CTV[qwen3-35b-q8]="q8_0"
MODEL_REASONING[qwen3-35b-q8]=true
MODEL_MAX_TOKENS[qwen3-35b-q8]=8192
MODEL_CTX_WINDOW[qwen3-35b-q8]=131072

# Gemma 4 31B — Q8_0 (~35GB)
# Dense model, all 31B active. Frontier-level quality.
# Tight on 64GB: ~35GB model + ~12GB KV (131k) + ~10GB OS = ~57GB
# NOTE: Download with: llama-launch.sh gemma4-31b-q8 (auto-downloads)
MODEL_HF[gemma4-31b-q8]="unsloth/gemma-4-31B-it-GGUF:Q8_0"
MODEL_NAME[gemma4-31b-q8]="Gemma 4 31B Q8_0 (Local)"
MODEL_THREADS[gemma4-31b-q8]=10
MODEL_BATCH[gemma4-31b-q8]=768
MODEL_UBATCH[gemma4-31b-q8]=192
MODEL_CTX[gemma4-31b-q8]=131072
MODEL_CTK[gemma4-31b-q8]="q8_0"
MODEL_CTV[gemma4-31b-q8]="q8_0"
MODEL_REASONING[gemma4-31b-q8]=true
MODEL_MAX_TOKENS[gemma4-31b-q8]=8192
MODEL_CTX_WINDOW[gemma4-31b-q8]=131072

# Gemma 4 27B A4B — Q8_0 (~28GB)
# MoE architecture (~4B active per token). Fastest option.
# NOTE: Download with: llama-launch.sh gemma4-27b-q8 (auto-downloads)
MODEL_HF[gemma4-27b-q8]="unsloth/gemma-4-26B-A4B-it-GGUF:Q8_0"
MODEL_NAME[gemma4-27b-q8]="Gemma 4 27B A4B Q8_0 (Local)"
MODEL_THREADS[gemma4-27b-q8]=10
MODEL_BATCH[gemma4-27b-q8]=1024
MODEL_UBATCH[gemma4-27b-q8]=256
MODEL_CTX[gemma4-27b-q8]=131072
MODEL_CTK[gemma4-27b-q8]="q8_0"
MODEL_CTV[gemma4-27b-q8]="q8_0"
MODEL_REASONING[gemma4-27b-q8]=true
MODEL_MAX_TOKENS[gemma4-27b-q8]=8192
MODEL_CTX_WINDOW[gemma4-27b-q8]=131072

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; }

# --- Check if llama-server is running ---
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # Stale PID file
        rm -f "$PID_FILE"
    fi
    return 1
}

# --- Stop llama-server ---
do_stop() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        info "Stopping llama-server (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        # Wait up to 10 seconds
        for i in $(seq 1 20); do
            if ! kill -0 "$pid" 2>/dev/null; then
                info "Stopped."
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 0.5
        done
        warn "Graceful stop timed out. Force killing..."
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
    else
        info "llama-server is not running."
    fi
}

# --- Show status ---
do_status() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo -e "${BOLD}llama-server is running${NC}"
        echo -e "  PID:    $pid"
        echo -e "  Port:   $PORT"
        echo -e "  Log:    $LOG_DIR/llama-server.log"
        echo -e "  Cache:  $CACHE_DIR"
        # Show memory usage
        if command -v ps &>/dev/null; then
            local mem
            mem=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f MB", $1/1024}')
            echo -e "  Memory: $mem"
        fi
    else
        echo -e "${YELLOW}llama-server is not running.${NC}"
        echo ""
        echo "Available models:"
        for key in "${!MODEL_NAME[@]}"; do
            echo -e "  ${CYAN}${key}${NC}  →  ${MODEL_NAME[$key]}"
        done
    fi
}

# --- List models ---
do_list() {
    echo -e "${BOLD}Available models:${NC}"
    echo ""
    printf "  %-20s %s\n" "KEY" "DESCRIPTION"
    printf "  %-20s %s\n" "────────────────────" "────────────────────────────────────────"
    for key in $(echo "${!MODEL_NAME[@]}" | tr ' ' '\n' | sort); do
        printf "  %-20s %s\n" "$key" "${MODEL_NAME[$key]}"
    done
    echo ""
    echo -e "${DIM}Usage: ./llama-launch.sh <model-key>${NC}"
}

# --- Download model if needed ---
ensure_model() {
    local model_key="$1"
    local hf_id="${MODEL_HF[$model_key]}"
    local manifest_file="$CACHE_DIR/manifest=${hf_id//[:\/]/=}.json"

    # Check if model files exist
    if [[ -f "$manifest_file" ]]; then
        info "Model manifest found, checking files..."
        # The manifest file existence is a good proxy for downloaded model
        return 0
    fi

    warn "Model not found. Downloading $hf_id ..."
    info "This may take a while. Subsequent launches will be instant."
    llama-server -hf "$hf_id" --host "$HOST" --port "$((PORT + 1))" --no-hang &
    local dl_pid=$!
    # Wait for download to complete (llama-server exits after download)
    wait "$dl_pid" 2>/dev/null || true
    info "Download complete."
}

# --- Launch llama-server ---
do_launch() {
    local model_key="$1"

    if [[ -z "${MODEL_HF[$model_key]+x}" ]]; then
        error "Unknown model: $model_key"
        echo "Available: $(echo "${!MODEL_NAME[@]}" | tr ' ' ', ')"
        exit 1
    fi

    if is_running; then
        error "llama-server is already running. Stop it first:"
        echo "  ./llama-launch.sh stop"
        exit 1
    fi

    local hf_id="${MODEL_HF[$model_key]}"
    local name="${MODEL_NAME[$model_key]}"
    local threads="${MODEL_THREADS[$model_key]}"
    local batch="${MODEL_BATCH[$model_key]}"
    local ubatch="${MODEL_UBATCH[$model_key]}"
    local ctx="${MODEL_CTX[$model_key]}"
    local ctk="${MODEL_CTK[$model_key]}"
    local ctv="${MODEL_CTV[$model_key]}"
    local reasoning="${MODEL_REASONING[$model_key]}"
    local max_tokens="${MODEL_MAX_TOKENS[$model_key]}"
    local ctx_window="${MODEL_CTX_WINDOW[$model_key]}"

    info "Launching: $name"
    info "  HF repo:    $hf_id"
    info "  Threads:    $threads"
    info "  Batch:      $batch / $ubatch"
    info "  Context:    $ctx"
    info "  KV cache:   K=$ctk, V=$ctv"
    info "  Max tokens: $max_tokens"
    info "  Reasoning:  $reasoning"
    echo ""

    # Ensure model is downloaded
    ensure_model "$model_key"

    # Build reasoning-format flag
    local reasoning_flag=""
    if [[ "$reasoning" == "true" ]]; then
        reasoning_flag="--reasoning-format deepseek"
    fi

    # Export environment
    export LLAMA_CACHE="$CACHE_DIR"
    export LLAMA_SERVER_LOG_FILE="$LOG_DIR/llama-server.log"

    # Launch
    exec llama-server \
        -hf "$hf_id" \
        --host "$HOST" \
        --port "$PORT" \
        --jinja \
        $reasoning_flag \
        -ngl 99 \
        -fa auto \
        -c "$ctx" \
        -t "$threads" \
        -b "$batch" \
        -ub "$ubatch" \
        -n "$max_tokens" \
        --temp 0.0 \
        --cache-type-k "$ctk" \
        --cache-type-v "$ctv" \
        --slot-save-path "$LLAMA_CACHE" \
        --log-file "$LLAMA_SERVER_LOG_FILE" \
        --mlock \
        --prio 2 \
        --poll 100 \
        --poll-batch 1 \
        --log-disable
}

# --- Interactive model selector ---
do_select() {
    if is_running; then
        error "llama-server is already running. Stop it first:"
        echo "  $0 stop"
        exit 1
    fi

    echo -e "${BOLD}llama-launch.sh${NC} — llama-server launcher for M5 Pro / 64GB"
    echo ""
    echo -e "${BOLD}Select a model:${NC}"
    echo ""

    # Build numbered list
    local -a keys=()
    for key in $(echo "${!MODEL_NAME[@]}" | tr ' ' '\n' | sort); do
        keys+=("$key")
    done

    for i in "${!keys[@]}"; do
        local idx=$((i + 1))
        echo -e "  ${CYAN}${idx}${NC}) ${keys[$i]}  →  ${MODEL_NAME[${keys[$i]}]}"
    done
    echo ""
    echo -e "  ${DIM}q) Quit${NC}"
    echo ""
    printf "  ${BOLD}Choose [1-%d]:${NC} " "${#keys[@]}"
    read -r choice

    case "$choice" in
        [1-9]|[1-9][0-9])
            local idx=$((choice - 1))
            if (( idx >= 0 && idx < ${#keys[@]} )); then
                do_launch "${keys[$idx]}"
            else
                error "Invalid selection."
                do_select
            fi
            ;;
        q|Q)
            info "Goodbye."
            exit 0
            ;;
        *)
            error "Invalid selection. Try again."
            echo ""
            do_select
            ;;
    esac
}

# --- Main ---
case "${1:-}" in
    qwen3-35b-q6|qwen3-35b-q8|gemma4-31b-q8|gemma4-27b-q8)
        do_launch "$1"
        ;;
    stop)
        do_stop
        ;;
    status)
        do_status
        ;;
    list)
        do_list
        ;;
    *)
        do_select
        ;;
esac
