# Llama Run Notes

> ⚠️ **New system in place.** Use `./llama-launch.sh` for model switching.
> See [MODEL_TUNING.md](MODEL_TUNING.md) for full tuning guide.

## Model Selection

> ⚠️ **Use `./llama-launch.sh` for model switching.** See [MODEL_TUNING.md](MODEL_TUNING.md) for full details.

### Available Models

| Model | Quant | Size | Active Params | Use Case |
|-------|-------|------|---------------|----------|
| Qwen3.6 35B-A3B | Q6_K | ~25GB | 3B active | Default — best balance |
| Qwen3.6 35B-A3B | Q8_0 | ~37GB | 3B active | Max quality |
| Gemma 4 31B | Q8_0 | ~35GB | 31B dense | Frontier reasoning |
| Gemma 4 27B A4B | Q8_0 | ~28GB | ~4B active | Fastest coding |

### Qwen/Qwen3.6

Quantized models (min 6-bit to 8-bit):

- llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K
- llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K_XL
- llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0
- llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q8_K_XL

```sh
  llama-server \
    -m /path/to/qwen3-32b-q8_0.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl 99 \
    -fa \
    --jinja \
    --reasoning-format deepseek \
    -c 32768 \
    -t 8 \
    -b 2048 \
    -ub 512 \
    -n 4096 \
    --temp 0.0 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --slot-save-path ./cache \
    --log-file ./logs/llama-server.log
```


(not sure on difference)

Update your Claude Code or API parameters to exactly this:

- Temperature: 1.0
- Top_P: 0.95
- Top_K: 40

---
You’re trying to run a **very large MoE model (Qwen 3.6 35B A3B, Q8)** on a **64 GB Apple Silicon machine**, which is right on the edge of what’s practical—so the “best” parameters are really about **balancing memory pressure, GPU offload, and batching**, not just maxing everything out.

Here’s the grounded reality first:

* **Q8_0 for a 35B MoE is huge** (tens of GB). Even on unified memory, you’ll be tight once you add KV cache + large context. ([AI 織夢][1])
* Apple Silicon + llama.cpp uses **Metal + unified memory**, so `-ngl` effectively means “push as much as possible to GPU.” ([Qwen][2])
* For models in the ~30–35B class, people typically run:

  * **Q5–Q6 quant** for headroom
  * **full GPU offload (`-ngl -1` or large)**
  * **context ≤128k unless you accept slowdown** ([BSWEN][3])

---

# 🔧 Recommended baseline (for your exact setup)

Start here:

```bash
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0 \
  --jinja \
  --reasoning-format deepseek \
  -ngl 99 \
  -fa \
  -sm row \
  -t 12 \
  -b 1024 \
  -ub 256 \
  -c 65536 \
  -n 2048 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --repeat-penalty 1.05 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-mmap
```

---

# 🧠 Why these matter (and how to tune them)

## 1) GPU offload (critical on Apple Silicon)

* `-ngl 99` → offload all layers
* This is **the single biggest performance lever** on M-series
* Equivalent advice elsewhere: *“gpu_layers: -1 … critical for performance”* ([BSWEN][3])

👉 If you see memory pressure → reduce context, not GPU layers.

---

## 2) Threads (`-t`)

* Use **~performance cores only**
* On M5 Pro/Max guess: **10–12 threads**
* More ≠ better (causes contention)

---

## 3) Batch sizes (huge impact on throughput)

* `-b 1024` (prompt processing)
* `-ub 256` (decode micro-batch)

From real configs, these are **sweet spots for 30B+ models** ([Reddit][4])

👉 If you get stalls:

* drop to `-b 512`, `-ub 128`

---

## 4) Context length (`-c`)

This is where most people go wrong.

* 35B Q8 + KV cache = massive RAM usage
* Safe ranges:

| Context | Use case      |
| ------- | ------------- |
| 32k     | fastest       |
| 64k     | balanced      |
| 128k    | max practical |

Even though 128k is possible, it **hurts speed heavily** ([BSWEN][3])

👉 Start with `-c 65536`

---

## 5) Flash attention (`-fa`)

* Always enable on Apple Silicon
* Gives **noticeable speedup on long context**

---

## 6) KV cache precision

* `--cache-type-k q8_0`
* `--cache-type-v q8_0`

This:

* improves quality
* increases RAM usage

👉 If memory tight → switch to `q4_0`

---

## 7) mmap

* `--no-mmap` often improves stability for large models
* Especially on macOS unified memory

---

## 8) Sampling (Qwen defaults)

From official examples:

* temp 0.6
* top-k 20
* top-p 0.95 ([Qwen][2])

These are already near-optimal.

---

# ⚠️ The big constraint (important)

Running **Q8 35B MoE on 64GB** means:

* You will likely get:

  * **~5–12 tok/s**
  * possible memory pressure at high context
* If you push:

  * 128k context
  * large batch
    → expect slowdowns or swapping

---

# 🚀 If you actually want “best performance”

Be blunt: **Q8 is not the optimal choice for your machine.**

Better configs:

### Option A (recommended)

```bash
Q6_K or Q5_K_M version
```

* ~same quality perceptually
* **much faster + more stable**

### Option B (best balance)

```bash
IQ4_XS or Q5_K_M
```

* newer quantization = better efficiency ([AI 織夢][1])

---

# 🧪 Advanced tweaks (if you want to push it)

Try adding:

```bash
--parallel 2
--ctx-size 65536
--flash-attn on
--spec-type ngram-mod
--spec-ngram-size-n 16
```

These come from real-world tuning of Qwen3.x MoE configs (mixed results, but worth testing).

---

# ✅ Bottom line

For your hardware:

* **Max quality:** Q8 + reduced context + careful batching
* **Best real-world UX:** Q5/Q6 + larger context + higher throughput

👉 If you want, tell me:

* your exact M5 variant (Pro / Max / Ultra)
* target use (coding, chat, agents)

…and I’ll give you a *fully optimized command tailored to your chip + workload*.

[1]: https://weavai.app/blog/en/2026/04/24/llama-cpp-2026-guide-local-ai-inference-setup/?utm_source=chatgpt.com "llama.cpp 2026 Guide: Local AI Inference & Setup - WeavAI Blog"
[2]: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html?utm_source=chatgpt.com "llama.cpp - Qwen"
[3]: https://docs.bswen.com/blog/2026-03-21-best-local-llm-mac-64gb/?utm_source=chatgpt.com "What's the Best Local LLM for Coding on a Mac with 64GB RAM? | BSWEN"
[4]: https://www.reddit.com/r/LocalLLaMA/comments/1svlnyk/rtx_3090_27b_model_performance_issues_llamacpp/?utm_source=chatgpt.com "RTX 3090 + 27B model performance issues (llama.cpp) what am I doing wrong"

---
Good—that narrows things down a lot. An **M5 Pro + 64 GB unified memory** is powerful, but for a **Qwen 3.6 35B MoE in Q8**, you’re still operating in a constrained zone where *smart parameter tuning matters more than raw hardware*.

I’ll give you a configuration that’s actually tuned for **M5 Pro specifically**, not just generic advice.

---

# 🧠 First: what your M5 Pro is good at

* Strong Metal GPU → **full offload is essential**
* Limited vs Max/Ultra GPU cores → batching must be **moderate**
* 64 GB unified memory → enough, but **KV cache is your real limit**

---

# ✅ Best practical config (M5 Pro, Q8 model)

```bash
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0 \
  --jinja \
  --reasoning-format deepseek \
  -ngl 99 \
  -fa \
  -sm row \
  -t 10 \
  -b 768 \
  -ub 192 \
  -c 49152 \
  -n 2048 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --repeat-penalty 1.05 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-mmap
```

---

# 🔍 Why this is *specifically right* for M5 Pro

## 1) Threads (`-t 10`)

* M5 Pro likely has **10 performance cores**
* Going above that hurts performance (thread contention)
* Don’t use all cores blindly

---

## 2) GPU offload (`-ngl 99`)

* On Apple Silicon: always push everything to GPU
* This is where most of your speed comes from

---

## 3) Batch tuning (this is where people mess up)

### Your sweet spot:

* `-b 768`
* `-ub 192`

Why not higher?

* M5 Pro GPU < M5 Max
* Large batches → memory pressure + stalls

If you want to experiment:

* Faster prompt ingestion → try `-b 1024`
* More stable → drop to `-b 512`

---

## 4) Context (`-c 49152`) ← key decision

This is the *real optimization*.

With Q8:

* 64k works but is tight
* 48k is the **stability/performance sweet spot**

Going higher:

* 64k → OK but slower
* 128k → not worth it on M5 Pro

---

## 5) KV cache precision

```bash
--cache-type-k q8_0
--cache-type-v q8_0
```

* Best quality
* But heavy

👉 If you hit memory issues:

```bash
--cache-type-k q4_0
--cache-type-v q4_0
```

This is the **first knob to turn down**, not context.

---

## 6) Flash attention (`-fa`)

Mandatory. Big speed win at long context.

---

## ⚠️ Reality check (important)

With this setup, expect roughly:

* **5–10 tokens/sec**
* Occasional slowdown at long context
* High memory pressure but stable

If you try to push:

* higher batch
* 64k+ context
  → you’ll hit diminishing returns fast

---

# 🚀 If you want it to feel *fast*

Here’s the honest advice:

**Q8 is not the best choice for M5 Pro.**

Switching to:

* **Q6_K** → best balance
* **Q5_K_M** → best overall UX

lets you run:

```bash
-c 65536 or even 131072
-b 1024+
```

…and you’ll *feel* the difference immediately.

---

# 🧪 Optional advanced tweaks

If you want to squeeze more out:

```bash
--parallel 2
```

* Helps throughput for multi-requests

```bash
--mlock
```

* Prevents memory swapping (use if stable)

```bash
--numa distribute
```

* Sometimes helps on unified memory (minor effect)

---

# 🧭 Bottom line

* Your config above = **best stable Q8 setup for M5 Pro**
* Your bottleneck is **memory bandwidth + KV cache**, not compute
* The biggest upgrade isn’t parameters—it’s **dropping to Q6**

---

If you want, I can:

* benchmark-optimized configs (max tok/s vs max quality)
* or tune specifically for **coding vs chat vs agent workflows**

Those actually need *different parameter sets*, and that’s where you can squeeze out another ~20–30% performance.
