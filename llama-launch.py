#!/usr/bin/env python3
"""
llama-launch.py — Launch llama-server with model switching
Models are loaded dynamically from TOML files in launch_params/

USAGE:
  python llama-launch.py                 # interactive model selector
  python llama-launch.py <model-key>     # launch model by TOML filename
  python llama-launch.py list            # show available models
  python llama-launch.py stop            # stop running llama-server
  python llama-launch.py status          # show current status
  python llama-launch.py generate-models # regenerate models.json from TOMLs
  python llama-launch.py enable <key>    # enable a model
  python llama-launch.py disable <key>   # disable a model
"""

import json
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
USER_CONFIG = LLAMA_DIR / "user-config.toml"
USER_CONFIG_SAMPLE = LLAMA_DIR / "user-config.sample.toml"

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
    enabled: bool = True
    threads: int = 10
    batch: int = 1024
    ubatch: int = 256
    context: int = 131072
    kv_cache_k: str = "q8_0"
    kv_cache_v: str = "q8_0"
    reasoning: bool = True
    max_tokens: int = 8192
    context_window: int = 131072
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64

    @property
    def hf_id(self) -> str:
        """Full HuggingFace identifier: repo:tag"""
        return f"{self.hf_repo}:{self.hf_tag}"


def _load_user_config() -> dict[str, bool] | None:
    """Load enabled/disabled state from user-config.toml. Returns None if no file."""
    if not USER_CONFIG.exists():
        return None
    with open(USER_CONFIG, "rb") as f:
        data = tomllib.load(f)
    return {key: section.get("enabled", True) for key, section in data.items() if key != "launcher"}


def _load_default_model() -> str | None:
    """Load the default model key from the [launcher] section of user-config.toml."""
    if not USER_CONFIG.exists():
        return None
    with open(USER_CONFIG, "rb") as f:
        data = tomllib.load(f)
    return data.get("launcher", {}).get("default", None)


def load_all_models() -> dict[str, Model]:
    """Load all models from TOML files in launch_params/, with user-config applied."""
    models: dict[str, Model] = {}
    if not PARAMS_DIR.is_dir():
        return models

    user_config = _load_user_config()

    for toml_file in sorted(PARAMS_DIR.glob("*.toml")):
        key = toml_file.stem  # e.g. "qwen3-35b-q6"
        with open(toml_file, "rb") as f:
            data = tomllib.load(f)

        hf = data["hf"]
        meta = data["meta"]
        launch = data["launch"]

        enabled = user_config[key] if user_config and key in user_config else True

        models[key] = Model(
            hf_repo=hf["repo"],
            hf_tag=hf["tag"],
            name=meta["name"],
            quant=meta["quant"],
            size_gb=meta["size_gb"],
            moe=meta["moe"],
            active_params_b=meta["active_params_b"],
            enabled=enabled,
            threads=launch["threads"],
            batch=launch["batch"],
            ubatch=launch["ubatch"],
            context=launch["context"],
            kv_cache_k=launch["kv_cache_k"],
            kv_cache_v=launch["kv_cache_v"],
            reasoning=launch["reasoning"],
            max_tokens=launch["max_tokens"],
            context_window=launch["context_window"],
            repeat_penalty=launch.get("repeat_penalty", 1.0),
            repeat_last_n=launch.get("repeat_last_n", 64),
        )

    return models


ALL_MODELS = load_all_models()
MODELS = {k: m for k, m in ALL_MODELS.items() if m.enabled}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _manifest_file(model_key: str) -> Path:
    """Return the manifest file path for a model."""
    hf_id = ALL_MODELS[model_key].hf_id
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
        console.print("[red]✗[/] Invalid PID file content. Recommend deleting the PID file.")
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
    if pid is None:
        console.print("[red]✗[/] Could not read PID file. Deleting PID file and exiting.")
        PID_FILE.unlink(missing_ok=True)
        return

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
        if pid is None:
            console.print("[red]✗[/] Could not read PID file. Status unknown.")
            return

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
    """List all models, showing enabled/disabled status."""
    table = Table(title="[bold]All Models[/]", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", width=18)
    table.add_column("Description", min_width=30)
    table.add_column("Quant", style="dim", width=8)
    table.add_column("Size", style="dim", width=8)
    table.add_column("Enabled", width=7)
    table.add_column("Local", width=5)

    for i, (key, model) in enumerate(sorted(ALL_MODELS.items()), 1):
        downloaded = _manifest_file(key).exists()
        key_style = "cyan" if model.enabled else "dim"
        desc_style = "white" if model.enabled else "dim"
        table.add_row(
            str(i),
            f"[{key_style}]{key}[/]",
            f"[{desc_style}]{model.name}[/]",
            model.quant,
            f"~{model.size_gb}GB",
            "[green]yes[/]" if model.enabled else "[red]no[/]",
            "[green]✓[/]" if downloaded else "[dim]—[/]",
        )

    console.print(table)

    has_config = USER_CONFIG.exists()
    if has_config:
        console.print(f"\n[dim]Config: {USER_CONFIG}[/]")
    else:
        console.print(
            "\n[dim]No user-config.toml found — all models enabled.[/]\n"
            "[dim]Copy the sample to get started: cp user-config.sample.toml user-config.toml[/]"
        )

    console.print(
        "[dim]Usage: python llama-launch.py <model-key>[/]\n"
        "[dim]       python llama-launch.py enable/disable <key>[/]"
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

    # Use llama-cli to download the model — it fetches the GGUF and exits
    dl_env = os.environ.copy()
    dl_env["LLAMA_CACHE"] = str(CACHE_DIR)

    try:
        _proc = subprocess.run(
            [
                "llama-cli",
                "-hf", model.hf_id,
                "-c", "1",
                "-n", "0",
                "-p", "",
            ],
            env=dl_env,
            capture_output=True,
        )
        console.print(Panel("[green]✓[/] Download complete.", border_style="green"))
    except FileNotFoundError:
        console.print("[red]✗[/] 'llama-cli' command not found. Is llama.cpp installed?")
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
        f"[cyan]Repeat pen:[/] {model.repeat_penalty} (last {model.repeat_last_n} tokens)",
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
        *(["--repeat-penalty", str(model.repeat_penalty),
           "--repeat-last-n", str(model.repeat_last_n)] if model.repeat_penalty > 1.0 else []),
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
        _proc = subprocess.run(cmd, env=env)
    except FileNotFoundError:
        console.print("[red]✗[/] 'llama-server' command not found. Is it installed?")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        sys.exit(130)
    finally:
        PID_FILE.unlink(missing_ok=True)


def cmd_generate_models() -> None:
    """Generate pi-models.json for pi-coding-agent from TOML configs."""
    models_list = []
    for key in sorted(MODELS):
        m = MODELS[key]
        models_list.append({
            "id": m.hf_id,
            "name": m.name,
            "reasoning": m.reasoning,
            "input": ["text"],
            "contextWindow": m.context_window,
            "maxTokens": m.max_tokens,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        })

    models_json = {
        "providers": {
            "llama-cpp": {
                "baseUrl": f"http://{HOST}:{PORT}/v1",
                "api": "openai-completions",
                "apiKey": "none",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": models_list,
            }
        }
    }

    out_path = LLAMA_DIR / "pi-models.json"
    out_path.write_text(json.dumps(models_json, indent=2) + "\n")

    pi_dest = Path.home() / ".pi" / "agent" / "models.json"
    console.print(
        Panel(
            f"[green]✓[/] Wrote [bold]{len(models_list)}[/] models to [cyan]{out_path}[/]\n\n"
            f"To use with pi-coding-agent, copy it to your pi config directory:\n\n"
            f"  [bold]mkdir -p ~/.pi/agent[/]\n"
            f"  [bold]cp {out_path} {pi_dest}[/]\n\n"
            f"[dim]pi-coding-agent reads models from {pi_dest} at startup.[/]",
            title="[bold]pi-models.json[/]",
            border_style="green",
        )
    )


def _write_user_config(enabled_map: dict[str, bool]) -> None:
    """Write user-config.toml with current enabled/disabled state, preserving [launcher] settings."""
    default_model = _load_default_model()

    lines = [
        "# user-config.toml — Per-machine model selection",
        "#",
        "# Set enabled = true for models you want to use, false to hide them.",
        "# Disabled models won't appear in the launcher, won't download, and",
        "# won't be included in pi-models.json.",
        "#",
        "# See model-tuning.md for guidance on which models fit your hardware.",
        "",
    ]

    if default_model:
        lines += [
            "[launcher]",
            f'default = "{default_model}"',
            "",
        ]
    for key in sorted(ALL_MODELS):
        m = ALL_MODELS[key]
        enabled = enabled_map.get(key, True)
        moe_label = f"MoE ({m.active_params_b}B active)" if m.moe else f"dense ({m.active_params_b}B active)"
        lines.append(f"[{key}]  # {m.name} — ~{m.size_gb}GB, {moe_label}")
        lines.append(f"enabled = {'true' if enabled else 'false'}")
        lines.append("")

    USER_CONFIG.write_text("\n".join(lines))


def _ensure_user_config() -> dict[str, bool]:
    """Load user-config.toml, creating it from current state if it doesn't exist."""
    if USER_CONFIG.exists():
        config = _load_user_config()
        if config is not None:
            return config
    # Generate from ALL_MODELS — everything enabled by default
    enabled_map = {key: m.enabled for key, m in ALL_MODELS.items()}
    _write_user_config(enabled_map)
    return enabled_map


def cmd_enable(model_key: str) -> None:
    """Enable a model in user-config.toml."""
    if model_key not in ALL_MODELS:
        console.print(f"[red]Unknown model:[/] {model_key}")
        console.print(f"[dim]Available: {', '.join(sorted(ALL_MODELS.keys()))}[/]")
        sys.exit(1)

    enabled_map = _ensure_user_config()
    if enabled_map.get(model_key, True):
        console.print(f"[green]✓[/] [cyan]{model_key}[/] is already enabled.")
        return

    enabled_map[model_key] = True
    _write_user_config(enabled_map)
    console.print(f"[green]✓[/] Enabled [cyan]{model_key}[/]")


def cmd_disable(model_key: str) -> None:
    """Disable a model in user-config.toml."""
    if model_key not in ALL_MODELS:
        console.print(f"[red]Unknown model:[/] {model_key}")
        console.print(f"[dim]Available: {', '.join(sorted(ALL_MODELS.keys()))}[/]")
        sys.exit(1)

    enabled_map = _ensure_user_config()
    if not enabled_map.get(model_key, True):
        console.print(f"[yellow]✓[/] [cyan]{model_key}[/] is already disabled.")
        return

    enabled_map[model_key] = False
    _write_user_config(enabled_map)
    console.print(f"[yellow]✓[/] Disabled [cyan]{model_key}[/]")


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
            "[bold]llama-launch.py[/] — llama-server launcher",
            subtitle="[dim]Press Ctrl+C to exit[/]",
            border_style="cyan",
        )
    )

    default_key = _load_default_model()
    if default_key and default_key not in MODELS:
        default_key = None

    # Model table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", style="cyan")
    table.add_column("Description", style="white")

    sorted_keys = sorted(MODELS.keys())
    default_num = None
    for i, key in enumerate(sorted_keys, 1):
        is_default = key == default_key
        if is_default:
            default_num = i
        key_str = f"[bold cyan]{key} *[/]" if is_default else key
        desc_str = f"[bold white]{MODELS[key].name}[/]" if is_default else MODELS[key].name
        table.add_row(str(i), key_str, desc_str)

    console.print(table)
    console.print()

    prompt_default = str(default_num) if default_num else "q"

    # Prompt
    while True:
        try:
            choice = Prompt.ask(
                "[bold]Choose[/]",
                choices=[str(i) for i in range(1, len(sorted_keys) + 1)] + ["q"],
                default=prompt_default,
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
        case "stop":
            cmd_stop()
        case "status":
            cmd_status()
        case "list":
            cmd_list()
        case "generate-models":
            cmd_generate_models()
        case "enable":
            if len(sys.argv) < 3:
                console.print("[red]Usage:[/] python llama-launch.py enable <model-key>")
                sys.exit(1)
            cmd_enable(sys.argv[2])
        case "disable":
            if len(sys.argv) < 3:
                console.print("[red]Usage:[/] python llama-launch.py disable <model-key>")
                sys.exit(1)
            cmd_disable(sys.argv[2])
        case key if key in MODELS:
            cmd_launch(key)
        case key if key in ALL_MODELS:
            console.print(
                Panel(
                    f"[yellow]{key}[/] is disabled.\n\n"
                    f"Enable it first:\n"
                    f"  [bold]python llama-launch.py enable {key}[/]",
                    title="[bold yellow]Disabled Model[/]",
                    border_style="yellow",
                )
            )
            sys.exit(1)
        case _:
            console.print(
                Panel(
                    f"Unknown command: [red]{command}[/]\n\n"
                    f"Available: [cyan]stop, status, list, generate-models, enable, disable[/]\n"
                    f"Models: [cyan]{', '.join(sorted(MODELS.keys()))}[/]\n"
                    f"[dim]Run without arguments for interactive selector[/]",
                    title="[bold red]Error[/]",
                    border_style="red",
                )
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
