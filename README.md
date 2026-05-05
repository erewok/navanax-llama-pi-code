# navanax-llama-pi-code

I made this so that I could more easily run local LLMs on my MacBook Pros using [llama.cpp](https://github.com/ggerganov/llama.cpp) and talk to them with [pi-coding-agent](https://github.com/mariozechner/pi-coding-agent).

All models run entirely on-device via Apple Silicon's unified memory and Metal GPU: no cloud and no API keys required.

## Prerequisites for The Python Program

- macOS on Apple Silicon (M1–M5)
- [Homebrew](https://brew.sh)
- [just](https://github.com/casey/just) (`brew install just`)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)

## Llama.cpp and pi-code Prerequisites

I made a [`setup.sh`](./setup.sh) script to start running local models, and it will install the following:

- llama.cpp (via homebrew)
- pi-coding-agent (via npm)
- tintinweb/pi-subagents (via npm)

## Setup

```bash
# Install llama.cpp, Node.js, and pi-coding-agent and then load deps for this python program
just setup
```

## Quick Start

After setup, pick the models that fit your machine and disable the rest:

```bash
just list                         # See all models with sizes
just disable qwen35-122b-q6       # Too big? Disable it
just run                          # Launch — only enabled models appear
```

Enabling or disabling for the first time creates a `user-config.toml` (git-ignored) that remembers your choices. You only need to do this once — your selections persist across sessions. To re-enable a model later, run `just enable <key>`.

If you skip this step, all models are enabled by default. See [model-tuning.md](model-tuning.md) for which models fit your hardware.

## Usage

```bash
just run                        # Interactive model selector
just launch qwen3-35b-q8       # Launch a specific model
just stop                       # Stop the running server
just status                     # Check if server is running
just list                       # List all models (enabled and disabled)
just enable <model>             # Enable a model
just disable <model>            # Disable a model
just logs                       # Tail the server log
just generate-pi-models         # Rebuild pi-models.json from TOML configs
```

Once the server is running, open another terminal in your project directory and run `pi`. Select the local model when prompted.

## How It Works

Models are defined as TOML files in [`launch_params/`](launch_params/). Each file specifies the HuggingFace repo, quantization, and launch parameters tuned for a specific hardware profile. Add a new TOML file to that directory and it will available in the list and for running.

```
launch_params/           # Model configs (one TOML per model)
user-config.sample.toml  # Sample per-machine model selection
user-config.toml         # Your enabled/disabled models (git-ignored, auto-created)
llama-launch.py          # Python launcher with rich terminal UI
pi-models.json           # Generated config for pi-coding-agent
setup.sh                 # One-time system setup script
justfile                 # Task runner recipes
cache/                   # Downloaded GGUF models (git-ignored)
logs/                    # Server logs (git-ignored)
```

The launcher downloads models on first run via `llama-server -hf`, caches them locally, and starts the server with the parameters from the TOML config.

## Adding a New Model

Create a TOML file in `launch_params/`:

```toml
[hf]
repo = "unsloth/Some-Model-GGUF"
tag  = "Q6_K"

[meta]
name    = "Some Model Q6_K (Local)"
quant   = "Q6_K"
size_gb = 40
moe     = true
active_params_b = 10

[launch]
threads        = 12
batch          = 768
ubatch         = 192
context        = 131072
kv_cache_k     = "q8_0"
kv_cache_v     = "q8_0"
reasoning      = true
max_tokens     = 8192
context_window = 131072
```

Then run `just generate-models` to update `pi-models.json`.

See [model-tuning.md](model-tuning.md) for what each parameter means and recommended defaults for different Mac configurations.
