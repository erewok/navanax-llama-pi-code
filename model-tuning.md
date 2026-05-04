# Model Tuning Guide

This document explains the parameters in our TOML configs and provides recommended defaults for different MacBook Pro configurations.

## TOML Config Reference

### `[hf]` — Model Source

| Field | Description |
|-------|-------------|
| `repo` | HuggingFace repository (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF`) |
| `tag` | Quantization tag to download (e.g. `Q8_0`, `Q6_K`, `UD-Q4_K_M`) |

### `[meta]` — Model Metadata

| Field | Description |
|-------|-------------|
| `name` | Display name shown in the launcher and pi-coding-agent |
| `quant` | Quantization label for reference |
| `size_gb` | Approximate model file size in GB |
| `moe` | `true` if Mixture of Experts, `false` if dense |
| `active_params_b` | Billions of parameters active per token. For dense models this equals total params. For MoE models this is much smaller — e.g. a 35B MoE model might only activate 3B per token, which is why MoE models are fast despite their total size. |

### `[launch]` — Server Parameters

| Field | Flag | Description |
|-------|------|-------------|
| `threads` | `-t` | Number of CPU threads for inference. Target your Mac's performance cores (not efficiency cores). |
| `batch` | `-b` | Prompt processing batch size. Larger values process prompts faster but use more memory. |
| `ubatch` | `-ub` | Decode (token generation) micro-batch size. Smaller than `batch`. |
| `context` | `-c` | Maximum context window in tokens passed to llama-server. This is how much text the model can "see" at once. |
| `kv_cache_k` | `--cache-type-k` | Quantization for the KV cache key tensors. Options: `f16`, `q8_0`, `q4_0`. |
| `kv_cache_v` | `--cache-type-v` | Quantization for the KV cache value tensors. Same options as above. |
| `reasoning` | `--reasoning-format` | Enable chain-of-thought reasoning (uses `deepseek` format). |
| `max_tokens` | `-n` | Maximum tokens per response. |
| `context_window` | _(pi-models.json)_ | Context window reported to pi-coding-agent. This controls when the agent triggers compaction. Can differ from `context` if you want the agent to compact earlier than the server's hard limit. |

## Understanding KV Cache

The KV (key-value) cache stores the attention state for every token in your context. It's the second-largest memory consumer after the model weights, and it scales linearly with context length.

**KV cache size formula:**
```
KV bytes per token = 2 (K+V) x layers x kv_heads x head_dim x bytes_per_element
```

**Quantization options and their trade-offs:**

| Type | Bytes/element | Relative size | Quality | When to use |
|------|---------------|---------------|---------|-------------|
| `f16` | 2.0 | 100% | Best | You have plenty of RAM headroom |
| `q8_0` | 1.0 | 50% | Very good | Default choice — minimal quality loss |
| `q4_0` | 0.5 | 25% | Good | Memory-constrained, large context needed |

## Memory Budget

Total memory usage is roughly:

```
model weights + KV cache + batch buffers + OS overhead (~8GB)
```

To estimate whether a configuration fits, add:
1. Model size (from `size_gb` in the TOML)
2. KV cache size (depends on model architecture, context length, and KV quantization)
3. ~2-4 GB for batch buffers and scratch space
4. ~8 GB for macOS and other apps

If the total exceeds your unified memory, you'll hit swap and performance drops severely.

## Recommended Defaults by Hardware

### Performance Cores by Chip

| Chip | P-cores | E-cores | Recommended `threads` |
|------|---------|---------|----------------------|
| M1 | 4 | 4 | 4 |
| M1 Pro | 6–8 | 2 | 6–8 |
| M1 Max/Ultra | 8 | 2 | 8 |
| M2 | 4 | 4 | 4 |
| M2 Pro | 6–8 | 4 | 6–8 |
| M2 Max/Ultra | 8 | 4 | 8 |
| M3 | 4 | 4 | 4 |
| M3 Pro | 5–6 | 6 | 5–6 |
| M3 Max | 12 | 4 | 12 |
| M4 | 4 | 6 | 4 |
| M4 Pro | 10 | 4 | 10 |
| M4 Max | 14 | 4 | 14 |
| M5 Pro | 10 | 4 | 10 |

### 16 GB (M1/M2/M3/M4 base)

You have ~8 GB after macOS overhead. Only small models fit.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3 8B Q8_0 (~9 GB) | Largest that fits comfortably |
| `threads` | 4 | Base chips have 4 P-cores |
| `batch` | 512 | Conservative for limited RAM |
| `ubatch` | 128 | |
| `context` | 16384 | ~16k is realistic |
| `kv_cache_k` | `q4_0` | Aggressive quantization to save memory |
| `kv_cache_v` | `q4_0` | |

### 24 GB (M2/M3/M4 base configs)

~16 GB usable. Can fit a small model at full precision or a medium model quantized.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3 8B BF16 (~16 GB) | Full precision small model |
| `threads` | 4 | |
| `batch` | 1024 | |
| `ubatch` | 256 | |
| `context` | 16384 | |
| `kv_cache_k` | `q8_0` | |
| `kv_cache_v` | `q8_0` | |

### 32 GB (M_ Pro configs)

~24 GB usable. MoE models become viable.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3.6 35B-A3B Q4_K_M (~22 GB) | MoE, only 3B active — fast |
| `threads` | 6–10 | Depends on your Pro chip variant |
| `batch` | 1024 | |
| `ubatch` | 256 | |
| `context` | 65536 | ~64k context fits with q4_0 KV |
| `kv_cache_k` | `q4_0` | Needed to fit 64k context |
| `kv_cache_v` | `q4_0` | |

### 64 GB (M_ Pro/Max configs)

~56 GB usable. Comfortable for 35B-class models with large context.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3.6 35B-A3B Q8_0 (~37 GB) | Near-lossless quality |
| `threads` | 10–12 | Depends on your chip |
| `batch` | 768–1024 | |
| `ubatch` | 192–256 | |
| `context` | 131072 | Full 128k context |
| `kv_cache_k` | `q8_0` | Good balance of quality and size |
| `kv_cache_v` | `q8_0` | |

### 128 GB (M_ Max/Ultra configs)

~120 GB usable. Can run large MoE models with massive context windows.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3.5 122B-A10B Q6_K (~80 GB) | 10B active params, high quality |
| `threads` | 12–14 | Max has 12–14 P-cores |
| `batch` | 768 | Conservative due to large model |
| `ubatch` | 192 | |
| `context` | 262144 | Full 256k context fits |
| `kv_cache_k` | `q8_0` | Room for quality KV at this RAM level |
| `kv_cache_v` | `q8_0` | |

**Alternative for 128 GB:** Run the 35B-A3B at Q8_0 with `f16` KV cache and 256k context (~67 GB total). Maximum KV quality with a smaller, faster model.

## Other Launch Flags

These are set by the launcher and generally don't need per-model tuning:

| Flag | Value | Purpose |
|------|-------|---------|
| `-ngl 99` | Always | Offload all layers to Metal GPU. This is the single biggest performance lever on Apple Silicon. |
| `-fa auto` | Always | Flash attention — critical for long context performance. |
| `--mlock` | Always | Pin model in RAM to prevent macOS from swapping it out. |
| `--prio 2` | Always | High process priority. |
| `--temp 0.0` | Always | Deterministic output for coding tasks. |
| `--jinja` | Always | Enable Jinja2 chat templates. |
| `--poll 100` | Always | Maximum Metal polling frequency (reduces latency). |
| `--poll-batch 1` | Always | Minimal batch polling overhead. |

## Troubleshooting

**Slow performance / memory pressure:** Your model + KV cache exceeds available RAM. Switch to a smaller model, reduce context, or use more aggressive KV cache quantization.

**Compaction happening too often in pi:** Your `context_window` in the TOML may be too small. This value controls when pi-coding-agent starts compacting conversation history. Increase it if you have memory headroom, or decrease it if you're hitting swap.

**Model not found:** Models download automatically on first launch. If it fails, check your network connection and ensure you have enough disk space in the `cache/` directory.
