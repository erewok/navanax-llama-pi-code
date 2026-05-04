# just manual: https://github.com/casey/just#readme
_default:
    just --list

# Install dependencies used by this project
bootstrap python="3.14":
    uv venv --python {{python}}
    just sync

# Sync dependencies with environment
sync:
    uv sync

# Interactive model selector
run:
    uv run python llama-launch.py

# Launch a specific model by key (e.g. just launch qwen35-122b-q6)
launch model:
    uv run python llama-launch.py {{model}}

# List available models
list:
    uv run python llama-launch.py list

# Stop the running llama-server
stop:
    uv run python llama-launch.py stop

# Show llama-server status
status:
    uv run python llama-launch.py status

# Regenerate pi-models.json from TOML configs
generate-pi-models:
    uv run python llama-launch.py generate-models

# Enable a model (e.g. just enable qwen3-35b-q8)
enable model:
    uv run python llama-launch.py enable {{model}}

# Disable a model (e.g. just disable qwen35-122b-q6)
disable model:
    uv run python llama-launch.py disable {{model}}

# Tail the llama-server log
logs:
    tail -f logs/llama-server.log

# Run the full setup script (installs tools, Python env, etc.)
setup:
    bash setup.sh
