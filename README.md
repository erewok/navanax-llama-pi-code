# llama — Local LLM Server for M5 Pro / 64GB

Launch and manage [llama-server](https://github.com/ggerganov/llama.cpp) instances with zero cloud dependency. Optimized for MacBook Pro M5 Pro with 64GB unified memory.

## Quick Start

```bash
./llama-launch.sh              # Interactive model selector
./llama-launch.sh qwen3-35b-q6 # Launch Qwen3.6 35B Q6_K directly
./llama-launch.sh stop         # Stop the running server
./llama-launch.sh status       # Check server status
./llama-launch.sh list         # List available models
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
llama-launch.sh          # Launcher with model selector, stop, status
models.json              # OpenAI-compatible model config for pi-coding-agent
MODEL_TUNING.md          # Detailed parameter tuning rationale
cache/                   # Downloaded GGUF models
logs/                    # llama-server logs
```

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

## Troubleshooting

See [MODEL_TUNING.md](MODEL_TUNING.md) for detailed troubleshooting, parameter tuning, and compaction settings.
