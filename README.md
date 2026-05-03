# llama — Local LLM Server for M5 Pro / 64GB

Launch and manage [llama-server](https://github.com/ggerganov/llama.cpp) instances. Optimized for MacBook Pro M5 Pro with 64GB unified memory.

## Quick Start

```bash
./llama-launch              # Interactive model selector
./llama-launch qwen3-35b-q6 # Launch Qwen3.6 35B Q6_K directly
./llama-launch stop         # Stop the running server
./llama-launch status       # Check server status
./llama-launch list         # List available models
```

Or with `uv run`:

```bash
uv run python llama-launch.py              # Interactive model selector
uv run python llama-launch.py qwen3-35b-q6 # Launch directly
uv run python llama-launch.py stop         # Stop the server
```

## Available Models

| Model | Quant | Size | Active Params | Speed | Best For |
|-------|-------|------|---------------|-------|----------|
| **Qwen3.6 35B-A3B** | Q6_K | ~25GB | 3B MoE | ~15-25 tok/s | Default — best balance |
| **Qwen3.6 35B-A3B** | Q8_0 | ~37GB | 3B MoE | ~8-15 tok/s | Maximum quality |
| **Gemma 4 31B** | Q8_0 | ~35GB | 31B dense | ~10-20 tok/s | Frontier reasoning |
| **Gemma 4 27B A4B** | Q8_0 | ~28GB | ~4B MoE | ~20-35 tok/s | Fastest option |

## Architecture

```
llama-launch.py          # Python launcher (rich terminal UI)
llama-launch             # Wrapper script (runs via uv)
llama-launch.sh          # Legacy bash launcher (deprecated)
models.json              # OpenAI-compatible model config for pi-coding-agent
MODEL_TUNING.md          # Detailed parameter tuning rationale
cache/                   # Downloaded GGUF models
logs/                    # llama-server logs
```

### Launcher Commands

| Command | Description |
|---------|-------------|
| `./llama-launch` | Interactive model selector with table |
| `./llama-launch <model-key>` | Launch a specific model |
| `./llama-launch list` | Show all available models |
| `./llama-launch status` | Check if server is running (PID, memory, port) |
| `./llama-launch stop` | Graceful shutdown (force-kill after 10s timeout) |

**Model keys:** `qwen3-35b-q6`, `qwen3-35b-q8`, `gemma4-31b-q8`, `gemma4-27b-q8`

The launcher uses [rich](https://github.com/Textualize/rich) for beautiful terminal output — panels, tables, and styled prompts. Model parameters are loaded from TOML files in [launch_params/](launch_params/) (see [pyproject.toml](pyproject.toml) for dependencies).

## Integration with pi-coding-agent

After launching a model, run `pi` in your project and select the local provider. `models.json` is pre-configured with all four models at `http://localhost:3333/v1`.

## Hardware Notes

- **Threads:** 10 (performance cores only, leaves efficiency cores for macOS)
- **GPU:** Full offload (`-ngl 99`) via Metal
- **Context:** 128k tokens with q8_0 KV cache precision
- **Flash Attention:** Enabled for long-context performance
- **Memory lock:** Model pinned in RAM to prevent swapping

## Gemma 4 Setup

Gemma 4 models download automatically on first launch. To download manually:

```bash
huggingface-cli download unsloth/gemma-4-31b-it-GGUF \
  --include "*Q8_0*" \
  --local-dir /Users/erewok/llama/cache/
```

## Development

```bash
uv sync              # Install dependencies (rich)
uv run python llama-launch.py  # Run with uv
```

## Troubleshooting

See [MODEL_TUNING.md](MODEL_TUNING.md) for detailed troubleshooting, parameter tuning, and compaction settings.
