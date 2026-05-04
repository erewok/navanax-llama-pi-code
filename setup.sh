#!/bin/bash
set -euo pipefail

# Bootstrap script for local LLM coding agent on Apple Silicon Macs (M1-M5)
# Installs: llama.cpp (llama-server), pi-coding-agent
# Model and params auto-tuned based on available unified memory

LLAMA_PORT=3333
LLAMA_HOST=127.0.0.1

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; exit 1; }

# --- Preflight checks ---
if [[ "$(uname -s)" != "Darwin" ]]; then
    error "This script is for macOS only."
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    error "Apple Silicon (arm64) required. Detected: $ARCH"
fi

RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
info "Detected ${RAM_GB}GB unified memory"

# --- Memory selection ---
# Each tier picks a model that fits comfortably, leaving headroom for OS + KV cache.
#
# 16 GB — Qwen3 8B dense, Q8_0 (8.7 GB model, ~3 GB free for KV + OS)
# 24 GB — Qwen3 8B dense, BF16 (16.4 GB model, ~3.5 GB free) — full precision small model
# 32 GB — Qwen 3.6 35B MoE, Q4_K_M (22.1 GB model, ~5.5 GB free)
# 64 GB — Qwen 3.6 35B MoE, Q8_0 (36.9 GB model, ~23 GB free)
# 128 GB — Qwen 3.6 35B MoE, BF16 (69.4 GB model, ~54 GB free)
#
# KV cache quantization is tuned per tier:
#   tight memory → q4_0 (75% savings vs f16)
#   moderate     → q8_0 (50% savings vs f16)
#   plenty       → f16  (no quantization)
#
# Context sizes are aggressive — set to maximize the KV cache for the
# pi coding agent harness, which uses these values to determine compaction
# thresholds. Assumes ~3 GB for macOS + 1 GB headroom.

echo ""
echo -e "${BOLD}How much unified memory does this Mac have?${NC}"
echo ""
echo -e "  ${CYAN}1)${NC}  16 GB   →  Qwen3 8B Q8_0         ${DIM}(8.7 GB, dense, fast)${NC}"
echo -e "  ${CYAN}2)${NC}  24 GB   →  Qwen3 8B BF16          ${DIM}(16.4 GB, dense, full precision)${NC}"
echo -e "  ${CYAN}3)${NC}  32 GB   →  Qwen 3.6 35B-A3B Q4_K_M ${DIM}(22.1 GB, MoE, good balance)${NC}"
echo -e "  ${CYAN}4)${NC}  64 GB   →  Qwen 3.6 35B-A3B Q8_0  ${DIM}(36.9 GB, MoE, near-lossless)${NC}"
echo -e "  ${CYAN}5)${NC} 128 GB   →  Qwen 3.6 35B-A3B BF16  ${DIM}(69.4 GB, MoE, full precision)${NC}"
echo ""
read -rp "Select [1-5]: " mem_choice

case "$mem_choice" in
    1)
        MODEL_HF="unsloth/Qwen3-8B-GGUF:Q8_0"
        MODEL_DISPLAY="Qwen3 8B Q8_0 (Local)"
        QUANT_LABEL="Q8_0 (8-bit, 8.7 GB dense model)"
        MODEL_SIZE_GB=9
        CTX_SIZE=16384
        CACHE_TYPE_K="q4_0"
        CACHE_TYPE_V="q4_0"
        THREADS=8
        BATCH_SIZE=1024
        UBATCH_SIZE=256
        REASONING=false
        ;;
    2)
        MODEL_HF="unsloth/Qwen3-8B-GGUF:BF16"
        MODEL_DISPLAY="Qwen3 8B BF16 (Local)"
        QUANT_LABEL="BF16 (16-bit, 16.4 GB dense model)"
        MODEL_SIZE_GB=17
        CTX_SIZE=16384
        CACHE_TYPE_K="q8_0"
        CACHE_TYPE_V="q8_0"
        THREADS=8
        BATCH_SIZE=1024
        UBATCH_SIZE=256
        REASONING=false
        ;;
    3)
        MODEL_HF="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
        MODEL_DISPLAY="Qwen 3.6 35B Q4_K_M (Local)"
        QUANT_LABEL="Q4_K_M (4-bit, 22.1 GB MoE model)"
        MODEL_SIZE_GB=22
        CTX_SIZE=65536
        CACHE_TYPE_K="q4_0"
        CACHE_TYPE_V="q4_0"
        THREADS=10
        BATCH_SIZE=2048
        UBATCH_SIZE=512
        REASONING=true
        ;;
    4)
        MODEL_HF="unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"
        MODEL_DISPLAY="Qwen 3.6 35B Q8_0 (Local)"
        QUANT_LABEL="Q8_0 (8-bit, 36.9 GB MoE model)"
        MODEL_SIZE_GB=37
        CTX_SIZE=262144
        CACHE_TYPE_K="q4_0"
        CACHE_TYPE_V="q4_0"
        THREADS=12
        BATCH_SIZE=2048
        UBATCH_SIZE=512
        REASONING=true
        ;;
    5)
        MODEL_HF="unsloth/Qwen3.6-35B-A3B-GGUF:BF16"
        MODEL_DISPLAY="Qwen 3.6 35B BF16 (Local)"
        QUANT_LABEL="BF16 (16-bit, 69.4 GB MoE model)"
        MODEL_SIZE_GB=70
        CTX_SIZE=262144
        CACHE_TYPE_K="q8_0"
        CACHE_TYPE_V="q8_0"
        THREADS=12
        BATCH_SIZE=2048
        UBATCH_SIZE=512
        REASONING=true
        ;;
    *)
        error "Invalid selection. Please choose 1-5."
        ;;
esac

echo ""
info "Selected: ${QUANT_LABEL}"
info "Model:    ${MODEL_HF}"
info "Context:  ${CTX_SIZE} tokens"
info "KV cache: K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"

if (( MODEL_SIZE_GB >= RAM_GB )); then
    warn "Model size (${MODEL_SIZE_GB}GB) meets or exceeds your RAM (${RAM_GB}GB)."
    warn "The system needs memory for the OS, KV cache, and other apps."
    warn "This will likely be very slow due to swap. Consider a smaller option."
    echo ""
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

# --- Install Homebrew if missing ---
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# --- Install llama.cpp ---
if ! command -v llama-server &>/dev/null; then
    info "Installing llama.cpp via Homebrew..."
    brew install llama.cpp
else
    info "llama-server already installed: $(which llama-server)"
fi

# --- Install Node.js if missing ---
if ! command -v node &>/dev/null; then
    info "Installing Node.js via Homebrew..."
    brew install node
else
    info "Node.js already installed: $(node --version)"
fi

# --- Install pi-coding-agent ---
if ! npm list -g @mariozechner/pi-coding-agent &>/dev/null; then
    info "Installing pi-coding-agent..."
    npm install -g @mariozechner/pi-coding-agent
else
    info "pi-coding-agent already installed"
fi

# --- Install pi-subagents ---
pi install npm:@tintinweb/pi-subagents

# --- Configure pi to use local llama-server ---
PI_CONFIG_DIR="$HOME/.pi/agent"
PI_MODELS="$PI_CONFIG_DIR/models.json"

mkdir -p "$PI_CONFIG_DIR"

if [[ -f "$PI_MODELS" ]]; then
    warn "Pi models config already exists at $PI_MODELS — skipping."
    warn "To reconfigure, delete it and re-run this script."
else
    info "Writing pi models config to $PI_MODELS"
    cat > "$PI_MODELS" <<PIEOF
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://localhost:${LLAMA_PORT}/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "${MODEL_HF}",
          "name": "${MODEL_DISPLAY}",
          "reasoning": ${REASONING},
          "input": ["text"],
          "contextWindow": ${CTX_SIZE},
          "maxTokens": 4096,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
PIEOF
fi

# --- Create llama-server launch script ---
LLAMA_DIR="$HOME/llama"
LLAMA_SCRIPT="$LLAMA_DIR/start.sh"
mkdir -p "$LLAMA_DIR/cache" "$LLAMA_DIR/logs"

if [[ ! -f "$LLAMA_SCRIPT" ]]; then
    info "Writing llama-server launch script to $LLAMA_SCRIPT"
    cat > "$LLAMA_SCRIPT" <<LLAMAEOF
#!/bin/bash
export LLAMA_CACHE="${LLAMA_DIR}/cache"
export LLAMA_SERVER_LOG_FILE="${LLAMA_DIR}/logs/llama-server.log"

exec llama-server \\
    -hf ${MODEL_HF} \\
    --host ${LLAMA_HOST} \\
    --port ${LLAMA_PORT} \\
    --jinja \\
    --reasoning-format deepseek \\
    -ngl 99 \\
    -fa auto \\
    -c ${CTX_SIZE} \\
    -t ${THREADS} \\
    -b ${BATCH_SIZE} \\
    -ub ${UBATCH_SIZE} \\
    -n 4096 \\
    --temp 0.0 \\
    --cache-type-k ${CACHE_TYPE_K} \\
    --cache-type-v ${CACHE_TYPE_V} \\
    --slot-save-path "\$LLAMA_CACHE" \\
    --log-file "\$LLAMA_SERVER_LOG_FILE"
LLAMAEOF
    chmod +x "$LLAMA_SCRIPT"
else
    warn "llama-server launch script already exists at $LLAMA_SCRIPT"
fi

echo ""
info "Setup complete!"
echo ""
echo -e "  ${BOLD}Model:${NC}     ${MODEL_HF}"
echo -e "  ${BOLD}Quant:${NC}     ${QUANT_LABEL}"
echo -e "  ${BOLD}Context:${NC}   ${CTX_SIZE} tokens"
echo -e "  ${BOLD}KV cache:${NC}  K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
echo -e "  ${BOLD}Threads:${NC}   ${THREADS}"
echo -e "  ${BOLD}Batch:${NC}     ${BATCH_SIZE} / ${UBATCH_SIZE}"
echo ""
echo "  1. Start llama-server:"
echo "     ${LLAMA_SCRIPT}"
echo ""
echo "     (First run downloads the model. Subsequent runs use cache.)"
echo ""
echo "  2. In another terminal, start pi in your project directory:"
echo "     cd /path/to/your/project"
echo "     pi"
echo ""
echo "     Select '${MODEL_DISPLAY}' when prompted."
echo ""

