#!/bin/bash
set -euo pipefail

# Minimal bootstrap: install system tools, then hand off to `just bootstrap`.
# All model selection, server config, and pi-agent config are handled by
# the Python launcher (llama-launch.py) and TOML configs in launch_params/.

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}==>${NC} ${BOLD}$*${NC}"; }
info()  { echo -e "    $*"; }
warn()  { echo -e "    ${YELLOW}$*${NC}"; }
error() { echo -e "    ${RED}$*${NC}"; exit 1; }

# --- Preflight ---
step "Checking system requirements"

if [[ "$(uname -s)" != "Darwin" ]]; then
    error "This script is for macOS only."
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    error "Apple Silicon (arm64) required. Detected: $(uname -m)"
fi

RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
info "macOS on Apple Silicon with ${RAM_GB}GB unified memory"

# --- Homebrew ---
if command -v brew &>/dev/null; then
    info "Homebrew present"
else
    error "Homebrew is required but not found. Please install Homebrew first: https://brew.sh/"
fi

# --- Node.js ---
if command -v node &>/dev/null; then
    info "Node.js: already installed ($(node --version))"
else
    error "Node.js is required but not found. Please install Node.js first: https://nodejs.org/"
fi

# --- just ---
if command -v just &>/dev/null; then
    info "just: already installed ($(just --version))"
else
    error "just not found: proceeding with basic setup. You run the Python package directly."
fi

# --- llama.cpp ---
if command -v llama-server &>/dev/null; then
    info "llama-server: already installed ($(which llama-server))"
else
    step "Installing llama.cpp (local LLM inference server)"
    brew install llama.cpp
fi

# --- pi-coding-agent ---
if npm list -g @mariozechner/pi-coding-agent &>/dev/null 2>&1; then
    info "pi-coding-agent: already installed"
else
    step "Installing pi-coding-agent"
    npm install -g @mariozechner/pi-coding-agent
fi

# --- pi-subagents ---
step "Installing pi-subagents"
pi install npm:@tintinweb/pi-subagents

# --- Python environment ---
step "Setting up Python environment (just bootstrap)"
just bootstrap

# --- Done ---
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo ""
echo "  1. See available models and disable any that don't fit your Mac:"
echo "       just list"
echo "       just disable <model-key>"
echo ""
echo "  2. Launch a model:"
echo "       just run"
echo ""
echo "  3. In another terminal, start pi in your project:"
echo "       cd /path/to/your/project && pi"
echo ""
echo "  4. Copy the provided user-config.sample.toml to user-config.toml and customize as needed:"
echo "       cp user-config.sample.toml user-config.toml"
echo ""
echo "  See README.md for full usage details."
echo ""
