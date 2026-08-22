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

## The Metal Wired Limit (Apple Silicon)

The memory budget above tells you whether a model fits in *physical* RAM. On
Apple Silicon there is a second, lower ceiling: macOS caps how much unified
memory the GPU may wire down. The default is roughly 75% of RAM — about
**48 GB on a 64 GB machine**. Exceed it and llama-server dies during compute
with:

```
ggml_metal_synchronize: error: command buffer 1 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

This is *not* a shortage of physical RAM — it is the Metal allocation cap, and
it bites well before you run out of memory. The launcher always passes
`--mlock`, so the whole model wants to be wired.

Check and raise it (resets on reboot):

```sh
sysctl iogpu.wired_limit_mb          # 0 means "default"
sudo sysctl iogpu.wired_limit_mb=57344   # 56 GB
```

Raising it to 56 GB on a 64 GB machine leaves only ~8 GB for macOS and
everything else. It works, but with no margin — a browser with many tabs will
push you into swap. Prefer a model that fits under the default ceiling.

**Practical 64 GB ceiling:** keep `model + KV + ~2 GB compute buffers` under
~46 GB and you never have to touch the sysctl.

## Frontier Open-Weight Models: What Does Not Fit

Periodically worth rechecking as new quants land, but as of August 2026:

| Model | Total / active | Smallest GGUF | Fits 64 GB? |
|-------|----------------|---------------|-------------|
| Kimi K2.7-Code | 1T / 32B | 304 GB (UD-IQ1_M) | No — 5-6x over, even ternary builds are ~200 GB |
| DeepSeek V4-Flash | 284B / 13B | 82.5 GB (UD-IQ1_S) | No — ~35 GB over, and 1-bit wrecks coding reliability |
| DeepSeek V4-Pro | 1.6T / 49B | — | No |

DeepSeek ships no official GGUFs; all are community quants, and the FP8/MXFP4
builds need a llama.cpp fork with the `deepseek-v4-flash` arch — stock upstream
will not load them.

The largest thing that actually runs on 64 GB is **Qwen3.5 122B-A10B at 2-3 bit**
(see `qwen35-122b-q2kxl.toml` / `qwen35-122b-iq3s.toml`). Qwen3.5 is
hybrid-attention — only 12 of its 48 layers are full attention, with 2 KV heads
at head_dim 256 — so KV costs only ~12 KB/token at q8_0. That means 128k context
is ~1.6 GB and even 256k is ~3.2 GB; context is cheap here, weights are not.

Open question worth settling empirically: 122B-A10B at 2-bit versus
`qwen3-35b-q8` (37 GB, near-lossless). Bigger model at worse quantization is not
automatically the better coder. The 2-bit failure mode to watch for in pi-code is
not gibberish but drift — malformed tool calls, losing earlier constraints on
long multi-step edits.

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

## Monitoring GPU and RAM Usage

Keeping an eye on memory pressure and GPU utilization helps you know whether your model fits comfortably or is thrashing swap.

### System Monitoring (CPU + RAM)

- **btop** — Rich terminal UI with CPU, RAM, disk, and network. Install with `brew install btop`.
- **htop** — Lightweight alternative. Install with `brew install htop`.

Either will show you unified memory pressure and whether swap is active, which is the first sign your model config is too large.

### GPU Monitoring

On Apple Silicon, the GPU shares unified memory with the CPU, so standard NVIDIA tools like `nvidia-smi` don't apply.

You can try the following instead:

- **Activity Monitor** (built-in; no install required) — Open Activity Monitor → GPU History (Window → GPU History) to see real-time GPU usage per process. The Memory tab also shows memory pressure.
- **macmon** — Terminal-based Apple Silicon monitor that shows GPU utilization, power draw, and memory. Install with `brew install macmon`.
- **nvtop** - I found that it works okay and the graphs look good but they report it may be buggy. Install with `brew install nvtop`.

> **Note:** On Linux with NVIDIA GPUs, use **nvtop** (`apt install nvtop`) for real-time GPU/VRAM monitoring.

## Troubleshooting

**Slow performance / memory pressure:** Your model + KV cache exceeds available RAM. Switch to a smaller model, reduce context, or use more aggressive KV cache quantization.

**Compaction happening too often in pi:** Your `context_window` in the TOML may be too small. This value controls when pi-coding-agent starts compacting conversation history. Increase it if you have memory headroom, or decrease it if you're hitting swap.

**Model not found:** Models download automatically on first launch. If it fails, check your network connection and ensure you have enough disk space in the `cache/` directory.
