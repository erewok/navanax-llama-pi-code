#!/usr/bin/env python3
"""
llama-launch.py — Launch llama-server with model switching
Optimized for MacBook Pro M5 Pro / 64GB

MODELS:
  qwen3-35b-q6    Qwen3.6 35B-A3B MoE, Q6_K (~25GB) — best balance
  qwen3-35b-q8    Qwen3.6 35B-A3B MoE, Q8_0 (~37GB) — max quality
  gemma4-31b-q8   Gemma 4 31B, Q8_0 (~35GB) — frontier quality, dense
  gemma4-27b-q8   Gemma 4 27B A4B, Q8_0 (~28GB) — fast MoE, great for coding

USAGE:
  python llama-launch.py                 # interactive model selector
  python llama-launch.py qwen3-35b-q6    # launch Qwen3 Q6
  python llama-launch.py qwen3-35b-q8    # launch Qwen3 Q8
  python llama-launch.py gemma4-31b-q8   # launch Gemma4 31B Q8
  python llama-launch.py gemma4-27b-q8   # launch Gemma4 27B Q8
  python llama-launch.py list            # show available models
  python llama-launch.py stop            # stop running llama-server
  python llama-launch.py status          # show current status
"""

import os
import sys
import signal
import subprocess
import time
import tomllib
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# ─── Configuration ───────────────────────────────────────────────────────────

LLAMA_DIR = Path(os.path.dirname(__file__))
CACHE_DIR = LLAMA_DIR / "cache"
LOG_DIR = LLAMA_DIR / "logs"
PID_FILE = LLAMA_DIR / "llama-server.pid"
PORT = 3333
HOST = "127.0.0.1"
PARAMS_DIR = LLAMA_DIR / "launch_params"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


# ─── Model Definitions ───────────────────────────────────────────────────────

@dataclass
class Model:
    hf_repo: str
    hf_tag: str
    name: str
    quant: str
    size_gb: int
    moe: bool
    active_params_b: int
    threads: int = 10
    batch: int = 1024
    ubatch: int = 256
    context: int = 131072
    kv_cache_k: str = "q8_0"
    kv_cache_v: str = "q8_0"
    reasoning: bool = True
    max_tokens: int = 8192
    context_window: int = 131072

    @property
    def hf_id(self) -> str:
        """Full HuggingFace identifier: repo:tag"""
        return f"{self.hf_repo}:{self.hf_tag}"


def load_models() -> dict[str, Model]:
    """Load all models from TOML files in launch_params/."""
    models: dict[str, Model] = {}
    if not PARAMS_DIR.is_dir():
        return models

    for toml_file in sorted(PARAMS_DIR.glob("*.toml")):
        key = toml_file.stem  # e.g. "qwen3-35b-q6"
        with open(toml_file, "rb") as f:
            data = tomllib.load(f)

        hf = data["hf"]
        meta = data["meta"]
        launch = data["launch"]

        models[key] = Model(
            hf_repo=hf["repo"],
            hf_tag=hf["tag"],
            name=meta["name"],
            quant=meta["quant"],
            size_gb=meta["size_gb"],
            moe=meta["moe"],
            active_params_b=meta["active_params_b"],
            threads=launch["threads"],
            batch=launch["batch"],
            ubatch=launch["ubatch"],
            context=launch["context"],
            kv_cache_k=launch["kv_cache_k"],
            kv_cache_v=launch["kv_cache_v"],
            reasoning=launch["reasoning"],
            max_tokens=launch["max_tokens"],
            context_window=launch["context_window"],
        )

    return models


MODELS = load_models()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _manifest_file(model_key: str) -> Path:
    """Return the manifest file path for a model."""
    hf_id = MODELS[model_key].hf_id
    safe_name = hf_id.replace("/", "=").replace(":", "=")
    return CACHE_DIR / f"manifest={safe_name}.json"


def is_running() -> bool:
    """Check if llama-server is currently running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0: check existence
        return True
    except (ProcessLookupError, ValueError, OSError):
        # Stale PID file
        PID_FILE.unlink(missing_ok=True)
        return False


def get_pid() -> int | None:
    """Get the current PID if running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def get_memory_usage(pid: int) -> str:
    """Get RSS memory usage for a PID."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            rss_mb = int(result.stdout.strip()) / 1024
            return f"{rss_mb:.0f} MB"
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return "N/A"


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_stop() -> None:
    """Stop the running llama-server."""
    if not is_running():
        console.print(
            Panel(
                "[green]✓[/] llama-server is not running.",
                title="[bold]Status[/]",
                border_style="green",
            )
        )
        return

    pid = get_pid()
    console.print(
        Panel(
            f"Stopping llama-server (PID [bold]{pid}[/])...",
            title="[bold]Stopping[/]",
            border_style="yellow",
        )
    )

    # Graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        console.print("[green]✓[/] Process already exited.")
        return

    # Wait up to 10 seconds
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)
            console.print(Panel("[green]✓[/] Stopped gracefully.", border_style="green"))
            return
        time.sleep(0.5)

    # Force kill
    console.print("[yellow]⚠[/] Graceful stop timed out. Force killing...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    console.print(Panel("[green]✓[/] Force killed.", border_style="red"))


def cmd_status() -> None:
    """Show current status of llama-server."""
    if is_running():
        pid = get_pid()
        mem = get_memory_usage(pid)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="bold")
        table.add_row("Status", "[green]● Running[/]")
        table.add_row("PID", str(pid))
        table.add_row("Port", str(PORT))
        table.add_row("Log", str(LOG_DIR / "llama-server.log"))
        table.add_row("Cache", str(CACHE_DIR))
        table.add_row("Memory", mem)

        console.print(
            Panel(
                table,
                title="[bold green]llama-server[/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]llama-server is not running.[/]\n\n"
                "Run [bold]python llama-launch.py[/] to start.",
                title="[bold]Status[/]",
                border_style="yellow",
            )
        )


def cmd_list() -> None:
    """List all available models."""
    table = Table(title="[bold]Available Models[/]", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", style="cyan", width=18)
    table.add_column("Description", style="white", min_width=30)
    table.add_column("Quant", style="dim", width=8)
    table.add_column("Size", style="dim", width=8)

    for i, (key, model) in enumerate(sorted(MODELS.items()), 1):
        table.add_row(
            str(i),
            key,
            model.name,
            model.quant,
            f"~{model.size_gb}GB",
        )

    console.print(table)
    console.print(
        "\n[dim]Usage: python llama-launch.py <model-key>[/]\n"
        "[dim]       python llama-launch.py          # interactive selector[/]"
    )


def cmd_download(model_key: str) -> None:
    """Download a model if not already present."""
    manifest = _manifest_file(model_key)
    if manifest.exists():
        console.print("[green]✓[/] Model manifest found, skipping download.")
        return

    model = MODELS[model_key]
    console.print(
        Panel(
            f"Downloading [cyan]{model.hf_id}[/]\n"
            f"This may take a while. Subsequent launches will be instant.",
            title="[bold yellow]Downloading[/]",
            border_style="yellow",
        )
    )

    # llama-server with --hf downloads the model then exits
    dl_env = os.environ.copy()
    dl_env["LLAMA_CACHE"] = str(CACHE_DIR)

    try:
        proc = subprocess.run(
            [
                "llama-server",
                "-hf", model.hf_id,
                "--host", HOST,
                "--port", str(PORT + 1),
                "--no-hang",
            ],
            env=dl_env,
            capture_output=False,
        )
        console.print(Panel("[green]✓[/] Download complete.", border_style="green"))
    except FileNotFoundError:
        console.print("[red]✗[/] 'llama-server' command not found. Is it installed?")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Download interrupted.[/]")
        sys.exit(130)


def cmd_launch(model_key: str) -> None:
    """Launch llama-server with the specified model."""
    if model_key not in MODELS:
        console.print(
            Panel(
                f"Unknown model: [red]{model_key}[/]\n\n"
                f"Available: [cyan]{', '.join(sorted(MODELS.keys()))}[/]",
                title="[bold red]Error[/]",
                border_style="red",
            )
        )
        sys.exit(1)

    if is_running():
        console.print(
            Panel(
                "llama-server is already running.\n\n"
                "Stop it first:\n"
                "  [dim]python llama-launch.py stop[/]",
                title="[bold red]Error[/]",
                border_style="red",
            )
        )
        sys.exit(1)

    model = MODELS[model_key]

    # Print launch info
    info_lines = [
        f"[cyan]HF repo:[/]     {model.hf_id}",
        f"[cyan]Threads:[/]    {model.threads}",
        f"[cyan]Batch:[/]      {model.batch} / {model.ubatch}",
        f"[cyan]Context:[/]    {model.context}",
        f"[cyan]KV cache:[/]   K={model.kv_cache_k}, V={model.kv_cache_v}",
        f"[cyan]Max tokens:[/] {model.max_tokens}",
        f"[cyan]Reasoning:[/]  {'deepseek' if model.reasoning else 'disabled'}",
    ]

    console.print(
        Panel(
            "\n".join(info_lines),
            title=f"[bold]Launching: {model.name}[/]",
            border_style="green",
        )
    )

    # Download if needed
    cmd_download(model_key)

    # Build reasoning flag
    reasoning_flag = ["--reasoning-format", "deepseek"] if model.reasoning else []

    # Environment
    env = os.environ.copy()
    env["LLAMA_CACHE"] = str(CACHE_DIR)
    env["LLAMA_SERVER_LOG_FILE"] = str(LOG_DIR / "llama-server.log")

    # Build command
    cmd = [
        "llama-server",
        "-hf", model.hf_id,
        "--host", HOST,
        "--port", str(PORT),
        "--jinja",
        *reasoning_flag,
        "-ngl", "99",
        "-fa", "auto",
        "-c", str(model.context),
        "-t", str(model.threads),
        "-b", str(model.batch),
        "-ub", str(model.ubatch),
        "-n", str(model.max_tokens),
        "--temp", "0.0",
        "--cache-type-k", model.kv_cache_k,
        "--cache-type-v", model.kv_cache_v,
        "--slot-save-path", str(CACHE_DIR),
        "--log-file", str(LOG_DIR / "llama-server.log"),
        "--mlock",
        "--prio", "2",
        "--poll", "100",
        "--poll-batch", "1",
        "--log-disable",
    ]

    # Write PID file before launching (llama-server writes its own PID)
    # Use subprocess.run to keep the server in the foreground.
    # Inheriting stdout/stderr/stdin lets llama-server talk to the terminal
    # (like the bash script's `exec llama-server ...`), and Ctrl+C works
    # naturally because the signal goes straight to the server process.
    try:
        proc = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        console.print("[red]✗[/] 'llama-server' command not found. Is it installed?")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        sys.exit(130)
    finally:
        PID_FILE.unlink(missing_ok=True)


def cmd_select() -> None:
    """Interactive model selector."""
    if is_running():
        console.print(
            Panel(
                "llama-server is already running.\n\n"
                "Stop it first:\n"
                f"  [dim]python llama-launch.py stop[/]",
                title="[bold red]Error[/]",
                border_style="red",
            )
        )
        return

    # Header
    console.print()
    console.print(
        Panel(
            "[bold]llama-launch.py[/] — llama-server launcher for M5 Pro / 64GB",
            subtitle="[dim]Press Ctrl+C to exit[/]",
            border_style="cyan",
        )
    )

    # Model table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", style="cyan")
    table.add_column("Description", style="white")

    sorted_keys = sorted(MODELS.keys())
    for i, key in enumerate(sorted_keys, 1):
        table.add_row(str(i), key, MODELS[key].name)

    console.print(table)
    console.print()

    # Prompt
    while True:
        try:
            choice = Prompt.ask(
                "[bold]Choose[/]",
                choices=[str(i) for i in range(1, len(sorted_keys) + 1)] + ["q"],
                default="q",
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/]")
            return

        if choice == "q":
            console.print("[green]✓[/] Goodbye.")
            return

        idx = int(choice) - 1
        if 0 <= idx < len(sorted_keys):
            cmd_launch(sorted_keys[idx])
            return
        else:
            console.print(f"[red]✗[/] Invalid selection. Choose 1-{len(sorted_keys)} or q.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        cmd_select()
        return

    command = sys.argv[1].lower()

    match command:
        case "qwen3-35b-q6" | "qwen3-35b-q8" | "gemma4-31b-q8" | "gemma4-27b-q8":
            cmd_launch(command)
        case "stop":
            cmd_stop()
        case "status":
            cmd_status()
        case "list":
            cmd_list()
        case _:
            console.print(
                Panel(
                    f"Unknown command: [red]{command}[/]\n\n"
                    f"Available: [cyan]stop, status, list[/]\n"
                    f"Models: [cyan]{', '.join(sorted(MODELS.keys()))}[/]\n"
                    f"[dim]Run without arguments for interactive selector[/]",
                    title="[bold red]Error[/]",
                    border_style="red",
                )
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
