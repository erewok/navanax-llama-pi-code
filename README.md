# navanax-llama-pi-code

Run local LLMs on your MacBook Pro using [llama.cpp](https://github.com/ggerganov/llama.cpp) and talk to them with [pi-coding-agent](https://github.com/mariozechner/pi-coding-agent). All models run entirely on-device via Apple Silicon's unified memory and Metal GPU — no cloud, no API keys.

## Prerequisites

- macOS on Apple Silicon (M1–M5)
- [Homebrew](https://brew.sh)
- [just](https://github.com/casey/just) (`brew install just`)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)

## Setup

```bash
# Install llama.cpp, Node.js, and pi-coding-agent
just setup

# Install Python dependencies
just bootstrap
```

## Usage

```bash
just run                        # Interactive model selector
just launch qwen35-122b-q6     # Launch a specific model
just stop                       # Stop the running server
just status                     # Check if server is running
just list                       # List available models
just logs                       # Tail the server log
just generate-models            # Rebuild pi-models.json from TOML configs
```

Once the server is running, open another terminal in your project directory and run `pi`. Select the local model when prompted.

## How It Works

Models are defined as TOML files in [`launch_params/`](launch_params/). Each file specifies the HuggingFace repo, quantization, and launch parameters tuned for a specific hardware profile. Drop a new TOML file in that directory and it's automatically available.

```
launch_params/           # Model configs (one TOML per model)
llama-launch.py          # Python launcher with rich terminal UI
pi-models.json           # Generated config for pi-coding-agent
bootstrap.sh             # One-time system setup script
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
