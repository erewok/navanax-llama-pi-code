# Local Model Tuning Guide - M5 Pro / 64GB

## Quick Reference

```bash
# Launch models
./llama-launch.sh qwen3-35b-q6    # Best balance (recommended default)
./llama-launch.sh qwen3-35b-q8    # Max quality, tighter memory
./llama-launch.sh gemma4-31b-q8   # Frontier quality, dense 31B
./llama-launch.sh gemma4-27b-q8   # Fastest MoE, great for coding

# Management
./llama-launch.sh stop
./llama-launch.sh status
./llama-launch.sh list
```

## Model Comparison

| Model | Quant | Size | Active Params | Speed | Quality | Context |
|-------|-------|------|---------------|-------|---------|---------|
| Qwen3.6 35B-A3B | Q6_K | ~25GB | 3B active | ~15-25 tok/s | Excellent | 128k |
| Qwen3.6 35B-A3B | Q8_0 | ~37GB | 3B active | ~8-15 tok/s | Near-lossless | 128k |
| Gemma 4 31B | Q8_0 | ~35GB | 31B dense | ~10-20 tok/s | Frontier | 128k |
| Gemma 4 27B A4B | Q8_0 | ~28GB | ~4B active | ~20-35 tok/s | Excellent | 128k |

### When to use which:

- **Qwen3.6 Q6_K** — Default for coding work. MoE architecture means only 3B params active per token, so it's fast despite the 35B total. Q6_K is perceptually near-identical to Q8 for coding tasks.
- **Qwen3.6 Q8_0** — Use when you need maximum reasoning quality and can accept slower speeds. Full 128k context.
- **Gemma 4 31B** — Frontier-level quality. Dense model (all 31B active). Tightest on memory (~57GB total with KV) but best reasoning. Use for complex architectural decisions, code reviews, and deep reasoning tasks.
- **Gemma 4 27B A4B** — Fastest option. MoE architecture (~4B active per token). Excellent for code generation, refactoring, and general coding tasks. Download needed (see below).

## Parameter Tuning Rationale

### Threads: `-t 10`
M5 Pro has ~10 performance cores. Using all cores causes contention. 10 threads targets the performance cores only, leaving efficiency cores for macOS.

### GPU Layers: `-ngl 99`
Always offload everything to GPU on Apple Silicon. This is the single biggest performance lever.

### Batch Sizes
| Model | `-b` (prompt) | `-ub` (decode) | Why |
|-------|---------------|----------------|-----|
| Q6_K | 1024 | 256 | Good headroom with 25GB model |
| Q8_0 | 768 | 192 | Tighter memory, smaller batches prevent stalls |
| Gemma4 27B | 1024 | 256 | Smaller model, can handle larger batches |

### Context Window
| Model | Context | KV Cache Type | Why |
|-------|---------|---------------|-----|
| Q6_K | 128k | q8_0 | 25GB model + 8GB KV = ~33GB, fits in 64GB |
| Q8_0 | 128k | q8_0 | 37GB model + 8GB KV = ~45GB, tight but stable |
| Gemma4 31B | 128k | q8_0 | 35GB model + 12GB KV = ~47GB, tight but stable |
| Gemma4 27B | 128k | q8_0 | 28GB model + 8GB KV = ~36GB, fits comfortably |

### KV Cache Precision: `--cache-type-k q8_0 --cache-type-v q8_0`
Q8_0 KV cache gives 50% size reduction vs f16 with minimal quality loss. For Q8_0 model mode, consider `q4_0` if you hit memory pressure.

### Additional Performance Flags
- `--mlock` - Pins model in RAM, prevents swapping
- `--prio 2` - High process priority
- `--poll 100` - Maximum polling for Metal (reduces latency)
- `--poll-batch 1` - Minimal batch polling overhead
- `--log-disable` - Reduces I/O overhead from logging
- `-fa auto` - Flash Attention enabled (critical for long context)
- `--slot-save-path` - Persists KV cache between requests

## Compaction Settings (pi-coding-agent)

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 32768,
    "keepRecentTokens": 32000
  }
}
```

### Why these values?

- **`reserveTokens: 32768`** (was 16384) - Doubled reserve. With local models that are slower, you want more room for the LLM to complete long responses without triggering compaction mid-generation.
- **`keepRecentTokens: 32000`** (was 20000) - Kept more recent context. This means compaction happens less frequently, and when it does, it preserves more of your recent conversation.

### How compaction works here:

With Q6_K model (contextWindow: 131072):
- Compaction triggers when: `contextTokens > 131072 - 32768 = 98304`
- So you get ~96k tokens of conversation before compaction
- With keepRecentTokens at 32k, the last ~29k tokens are preserved

With Q8_0 model (contextWindow: 65536):
- Compaction triggers when: `contextTokens > 65536 - 32768 = 32768`
- So you get ~32k tokens before compaction
- This is why Q8_0 has a smaller context window - the model itself takes more memory

## Gemma 4 Setup

Gemma 4 models are not yet downloaded. To get them:

```bash
# Option 1: Let llama-launch.sh download automatically
./llama-launch.sh gemma4-31b-q8   # 31B frontier model
./llama-launch.sh gemma4-27b-q8   # 27B MoE model

# Option 2: Manual download
huggingface-cli download unsloth/gemma-4-31b-it-GGUF \
  --include "*Q8_0*" \
  --local-dir /Users/erewok/llama/cache/

huggingface-cli download unsloth/gemma-4-27b-it-GGUF \
  --include "*Q8_0*" \
  --local-dir /Users/erewok/llama/cache/
```

## Troubleshooting

### "Memory pressure" or slow performance
1. Check with `./llama-launch.sh status`
2. Switch to Q6_K if on Q8_0: `./llama-launch.sh qwen3-35b-q6`
3. Reduce context: edit the model entry in `llama-launch.sh`
4. Try Gemma 4 27B - it's the fastest option

### Compaction still happening too often
1. Check your models.json `contextWindow` matches llama-server `-c`
2. Increase `reserveTokens` in settings.json
3. Use Q6_K instead of Q8_0 for more headroom

### Model not found
```bash
# Check cache
ls -la /Users/erewok/llama/cache/

# Re-download
./llama-launch.sh stop
./llama-launch.sh qwen3-35b-q6
```

### Stale PID file
```bash
rm /Users/erewok/llama/llama-server.pid
```
