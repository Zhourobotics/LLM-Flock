# LLMs for Flocking

Multi-agent flocking simulation where each robot is controlled by an LLM.

Each agent can:
- generate an initial formation plan,
- execute movement decisions round by round,
- optionally adopt/propagate plans through influence-based consensus.

This repository contains the simulation runtime, plotting pipeline, and analysis notebook used for experiments in shape formation tasks.

Demo video:
- https://youtu.be/8zcPYqjWzYo?si=NVISFMOwd_StI_vB

## What Is In This Repo

- `main.py`: entrypoint; loads config + runs simulation + plots results.
- `simulation_runner.py`: orchestrates initialization, rounds, and saving.
- `simulation_engine.py`: per-round parallel agent decision step + influence calculation.
- `plan_manager.py`: initial planning + influence/plan adoption logic.
- `agents.py`: agent model clients, memory handling, and prompt calls.
- `graph.py`: animation/frame rendering and export.
- `data.py`: result serialization and dialog history export.
- `metrics.ipynb`: post-run evaluation and plotting utilities.

## Requirements

- Python 3.10+ (recommended)
- Linux/macOS shell environment (examples use `bash`)
- Provider API key(s) for the model provider you choose

Install dependencies:

```bash
pip install -r requirements.txt
```

Current runtime dependencies are maintained in `requirements.txt` (OpenAI/Anthropic clients, plotting, parsing, optional Pydantic AI wrapper, SciPy for notebook metrics).

## Quick Start

### 1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Configure keys

A template is provided at `secrets.example.yml`. Copy it once per provider you plan to use:

```bash
cp secrets.example.yml secrets.yml             # OpenAI (default)
cp secrets.example.yml secrets_claude.yml      # Anthropic
cp secrets.example.yml secrets_deepseek.yml    # DeepSeek API
cp secrets.example.yml secrets_qwen.yml        # Qwen
cp secrets.example.yml secrets_llama.yml       # Llama API
```

Then edit each copy and replace the placeholder strings under `api_keys` with your real keys. All real `secret*.yml` files are gitignored; the `.example.yml` template is the only one that ships.

The code loads these files lazily — missing files just warn at startup; a hard error only fires when you actually try to use a provider with no key configured. Multiple keys per provider are round-robined on each call.

### 3) Set experiment config

Edit `config.yaml` (default run settings) and make sure:
- `name` is unique (existing names are not overwritten),
- `mode` is `run` or `plot`,
- `model` + `model_company` match your desired provider.

### 4) Run

Run from config defaults:

```bash
python3 main.py
```

Or override with CLI:

```bash
python3 main.py \
  -m run \
  -n my_run_001 \
  -mc openai \
  -gpt gpt-5-mini \
  -ra low \
  -a 6 \
  -r 20 \
  -am influence
```

Plot an existing run:

```bash
python3 main.py -m plot -n my_run_001
```

## Configuration Model

Arguments are merged from:
1. CLI args
2. `config.yaml`

CLI values take priority over config values.

Get all available flags:

```bash
python3 main.py -h
```

Important flags:
- `-m, --mode`: `run` or `plot`
- `-n, --name`: experiment identifier
- `-a, --agents`: number of agents
- `-r, --rounds`: number of rounds
- `-form, --formation`: target shape description
- `-am, --agent_mode`: `influence`, `naive`, `basic`, `plan`
- `-mc, --model_company`: `openai`, `claude`, `deepseek`, `deepseek_api`, `qwen`, `llama_api`
- `-gpt, --model`: model name for provider
- `-mlim, --memory_limit`: per-agent memory horizon
- `--use_pydantic_ai`: optional Claude structured-output wrapper

## Agent Modes

- `basic`:
  - no initial planning phase,
  - movement decisions use formation instructions only.

- `naive`:
  - each agent generates its own plan initially,
  - no plan adoption/consensus during rounds.

- `influence`:
  - each agent generates an initial plan,
  - per round: influence is computed from neighbors in communication range,
  - plans can propagate through local leader selection and pairwise adoption.

- `plan`:
  - designated leader behavior (Agent 0 plan is propagated),
  - followers map to plan indices by ID with bounds checks.

## Outputs

Results are written under:

```text
results/<name>/
```

Key artifacts:
- `results`: pickled run object (agent histories + settings)
- `dialog_history.txt`: full prompt/response history per agent
- `animation.gif`: trajectory animation
- `frame_XX.pdf` / `frame_XX.png`: per-round snapshots
- `last.svg`: final frame export

Runtime logs:
- `performance.log`
- `api_performance.csv`

## Plotting Notes

- Rendering backend is non-interactive (`Agg`), so plots are saved to files.
- Matplotlib style names vary by version; the code auto-falls back if a style is unavailable.
- LaTeX text rendering is opt-in:
  - default behavior uses standard Matplotlib text (`text.usetex=False`),
  - set `FLOCKING_USE_TEX=1` to enable TeX rendering if your LaTeX installation is complete.

If you want LaTeX-quality figure text on Ubuntu:

```bash
sudo apt update
sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng
```

Then run with:

```bash
FLOCKING_USE_TEX=1 python3 main.py
```

## Common Issues

- `Test <name> already exists! Aborting.`  
  Use a new `--name` or remove the previous result directory.

- `No API keys configured for model '<provider>'...`  
  Add the matching secrets file and `api_keys` section for that provider.

- Provider secrets warning at startup  
  Safe to ignore unless you are using that provider in the current run.

- Slow runs with many agents  
  Reduce `agents`, `rounds`, or switch to a faster model.

## Batch Scripts

Helper scripts for sweeping seeds live under `scripts/`:
- `scripts/circle-10-gpt.sh` — sequential 10-seed sweep
- `scripts/ds_10_runs_circle.sh` — parallel sweep using tmux panes (requires `tmux` and a conda env named `llm-flocking`)

Both use placeholder test names and model settings; review and adjust before running.

## Development Notes

- `metrics.ipynb` contains post-processing and evaluation plots. It assumes at least one finished simulation under `results/<name>/` — run the simulation first, then open the notebook.
- `prompts.py` controls task/system instructions given to agents.
- Structured parsing models are in `structured_models.py` and `structured_parser.py`.
- OpenAI model branching in `agents.py` only supports the gpt-5 family on the Responses API; everything else (Claude, DeepSeek, Qwen, Llama API) goes through the chat-completions / Anthropic Messages paths.

## Contact

For project questions: Peihan Li (`pl525(at)drexel.edu`).
