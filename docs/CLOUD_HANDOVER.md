# Midas Cloud Migration — Handover Document

Purpose: get this project running on a fresh cloud VM (RunPod CPU pod or
similar) with no prior context.

Last updated: 2026-05-23 (after discovery v3 + strategic re-framing toward
slower-horizon "post-microstructure" research).

## 1. What this project is

**Midas** is a dual-loop framework for discovering alpha features in
intraday index-futures order-flow data:

- **Offline loop**: LLM (Claude via Anthropic API) proposes DSL expressions
  → multi-agent evaluator scores them → learnings written to a filesystem KB.
- **Online loop** (not yet exercised): per-bar monitor watches deployed
  features for IC decay, slippage, drawdown; auto-diagnoses when they break.

### Strategic direction (new — 2026-05-23)

Earlier work targeted continuous-edge microstructure signals at 1-second
horizons. That direction is **abandoned**. The competitive reality:

- HFT firms with co-located servers (microsecond round-trip to CME) have
  already arbitraged every signal that decays in <1s.
- You cannot out-react them physically; trying is a losing game.
- **But** the market state HFTs leave behind — post-flush book dynamics,
  post-VPIN-spike regimes, depth-replenishment patterns — carries information
  that decays over **minutes**, not milliseconds. Slower capital wins there.

So the framework's focus shifts to:

| What | Then (v1-3) | Now |
|---|---|---|
| Features | 1-second microstructure (OFI, VPIN, book imbalance, etc.) | **Same** — still summarise the order book |
| Forward return horizons | 1/4/8/24/48 minutes | **5/15/30/60/120 minutes** |
| Strategy style | Continuous always-on position scaled by signal | **Event-conditional**: wait for trigger, hold N minutes, exit |
| Evaluation | rankIC across ALL bars uniformly | **Gated rankIC**: score only on bars where the trigger fires, plus trigger frequency and per-trigger expected return |
| Instrument | NQ pull was planned | **Stay on MNQ data** — at minute-scale, NQ and MNQ track within fractions of a tick. Research on MNQ, trade NQ later if needed |

This re-framing **reuses everything we've already built** — the 14-column
1s-bar MNQ panel is the *input* to the new model; only the *target return*
and *evaluation mode* change.

## 2. Repo layout

```
Midas/
├── midas/                    # Source — Python package
│   ├── adapters/             # Databento loader, MBO extractor, panel loader,
│   │                         #   DSL evaluator, MLflow hook, Nautilus bars
│   ├── proposer.py           # LLM-backed expression generator
│   ├── evaluator.py          # AlphaEvaluator + MultiAgentEvaluator (6 agents)
│   ├── loops.py              # OfflineCompoundLoop.run() orchestrator
│   ├── monitor.py            # Online loop (not used yet)
│   ├── kb.py                 # Filesystem KB I/O
│   ├── factory.py            # create_midas() bootstrap + CLI
│   └── llm.py                # Anthropic/OpenAI provider boundary
├── scripts/
│   ├── pull_week.py          # Download MBO + ohlcv from Databento (deferred)
│   ├── extract_mbo_features.py  # Batch MBO -> 14-col parquet
│   └── run_discovery.py      # Discovery session driver
├── midas-kb-mnq/             # MNQ knowledge base (prompts, skills, thresholds, learnings)
├── market_data/              # NOT in repo — see §5
└── docs/CLOUD_HANDOVER.md    # this file
```

## 3. Cloud VM — recommended spec

GPU is irrelevant for this workload (LLM is API-hosted; all local work is
pandas/numpy/sortedcontainers). If CPU pod availability is poor, **a low-end
GPU pod with good CPU/RAM is fine — you just ignore the GPU.**

- **Provider**: RunPod (GPU pod with idle GPU is OK; AWS/GCP/Hetzner equivalent)
- **Image (RunPod GPU pods)**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  or any `-py3.11+ -devel -ubuntu22.04` PyTorch image. PyTorch itself is unused;
  pick by Python version. We need Python ≥ 3.11.
- **CPU**: 16+ vCPUs (x86_64; RunPod GPU pods are always x86_64)
- **RAM**: 32 GB (53 GB is fine, more is wasted)
- **Disk**: NVMe network volume, 250 GB. Always store under `/workspace`
  (network volume mount point — survives pod stop/restart). Container root
  is wiped on termination.
- **Network**: 1 Gbit+ (standard)
- **Estimated cost** (May 2026): RTX 4000 Ada / 16 vCPU / 53 GB / 250 GB
  net-storage at ~$0.29/hr is the current sweet spot. A discovery cycle
  ~12 min, an extraction ~10 min on 16 cores, so a full research loop
  costs well under $1.

### CPU performance note

RunPod GPU pods over-allocate vCPUs relative to physical cores, so per-vCPU
throughput is modest. For our bursty workloads (12-min discovery, 80-min
single-threaded extraction or ~10-min parallel) that's fine. If you ever
need sustained CPU performance for a longer run, redeploy onto a different
host or a CPU-optimised pod.

## 4. Initial install on a fresh RunPod GPU pod (or vanilla Ubuntu 22.04)

```bash
# --- 4.0 If using a RunPod PyTorch image, you may land inside a conda env.
#         Deactivate it so we don't accumulate PyTorch's dep tree alongside ours.
conda deactivate 2>/dev/null || true

# --- 4.1 System packages (most are pre-installed on PyTorch images;
#         apt install is a no-op for things already present) ---
apt update
apt install -y \
  python3.11 python3.11-venv python3.11-dev \
  git build-essential \
  curl ca-certificates \
  rsync openssh-client
# On a RunPod PyTorch image, python3.11 + git + build-essential are usually
# already there — the apt install just ensures it.

# --- 4.2 Workspace (use /workspace — that's the network volume mount
#         and survives pod stop/restart; container root is wiped) ---
mkdir -p /workspace && cd /workspace

# --- 4.3 Get the code ---
# Option A: clone from git (push from local first if you haven't already)
git clone <your-repo-url> Midas

# Option B: rsync the whole repo from local (skip if cloning)
#   On local Windows machine:
#     scp -r "D:/File Transfer/FX Trading/Midas" \
#       user@runpod-host:/workspace/Midas

cd /workspace/Midas

# --- 4.4 Python env (own venv — isolated from the pod's PyTorch env) ---
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# --- 4.5 Project deps ---
pip install -e ".[nq]"
# Pulls: anthropic, openai, pandas, numpy, scikit-learn, pyyaml,
#        sortedcontainers, vectorbt, mlflow, databento.
#
# nautilus_trader (the backtest workstream) is a separate extra because
# the latest release line requires Python >= 3.12 while the standard
# RunPod PyTorch image is on Python 3.11. Add nautilus only when you're
# ready to backtest:
#
#   pip install -e ".[backtest]"   # adds nautilus_trader >= 1.220 on top of [nq]

# --- 4.6 Sanity check imports ---
python -c "
import pandas, numpy, vectorbt, databento, anthropic, sortedcontainers, mlflow
from midas.adapters import make_mbo_data_fn, MboFeatureExtractor
from midas.factory import create_midas
print('OK: all imports clean')
"
```

If `nautilus_trader[databento]` fails to install (it's a large package with
several native deps), it's only needed for the future backtest workstream
— you can defer with `pip install -e .` (no extras) and still run discovery.

## 5. Transferring the existing data + KB from local

The feature parquets (~56 MB) and the MNQ knowledge base are small enough
to `scp` directly. The raw MBO `.dbn.zst` files (10 GB) are **not** needed
for discovery — only re-extraction. Leave them on the local machine; re-pull
fresh to the cloud only if you need to re-extract.

### Quick path: the bundled PowerShell script

```powershell
# From the repo root on your local Windows machine:
.\scripts\transfer_to_cloud.ps1
# default args: -RemoteHost runpod-mnq -RemoteRoot /workspace/Midas
# add  -SkipLearnings  if you want a fresh KB on the cloud side
```

The script verifies SSH, checks local files, creates the remote directory
tree, scp's the `.env` (chmod 600) + parquets + prior learnings, and
prints a verification listing from the remote.

### Manual path (if you prefer line-by-line)

On your local Windows machine (PowerShell):

```powershell
# --- 5.1 Feature parquets (14-col 1s bars) ---
$features = "D:\File Transfer\FX Trading\market_data\features\GLBX.MDP3\MNQ.c.0\bar_1s"
scp -r $features user@runpod-host:/workspace/Midas/market_data/features/GLBX.MDP3/MNQ.c.0/

# --- 5.2 Knowledge base (prompts, skills, thresholds, prior learnings) ---
scp -r "D:\File Transfer\FX Trading\Midas\midas-kb-mnq" user@runpod-host:/workspace/Midas/

# --- 5.3 The .env file (API keys — NEVER commit to git) ---
scp "D:\File Transfer\FX Trading\Midas\.env" user@runpod-host:/workspace/Midas/.env
```

On the cloud machine, verify and lock down the `.env`:

```bash
cd /workspace/Midas
chmod 600 .env
ls -la market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s/   # 17 parquet shards expected
ls -la midas-kb-mnq/                                     # skills/, knowledge/, proposer/ folders
```

### API key convention

The `.env` should contain:

```
DATABENTO_API_KEY=db-...
ANTHROPIC_MIDAS_API_KEY=sk-ant-api03-...
```

Project-namespaced `ANTHROPIC_MIDAS_API_KEY` is preferred over the
generic `ANTHROPIC_API_KEY` so it's clearly the Midas project key, not
the Claude Code CLI's key (which you may also want on the box).
`scripts/run_discovery.py` reads the namespaced one first.

UTF-8 IO matters; on Linux it's default but harmless to be explicit:

```bash
echo 'export PYTHONIOENCODING=utf-8' >> ~/.bashrc
echo 'export PYTHONUTF8=1'           >> ~/.bashrc
```

## 6. Pending code changes (for the new strategic direction)

The framework currently scores microstructure signals against **1-min-scale**
forward returns. To support the new direction, the following changes are
queued — they're small but should be done together as the first task on the
cloud machine:

1. **`midas/adapters/panel_loader.py`** — replace `_DEFAULT_HORIZON_MAP`
   so `ret_1h..ret_48h` columns map to **5 / 15 / 30 / 60 / 120 minutes**
   forward instead of **1 / 4 / 8 / 24 / 48 minutes**. (One dict edit.)
2. **`midas/evaluator.py`** — add a gated-evaluation mode: when a `gate`
   Series is supplied alongside `feature`, compute IC only on rows where
   gate==True; also report `trigger_frequency`, `mean_return_when_triggered`,
   `hit_rate`. (~30 lines.)
3. **`midas/loops.py`** — pass the gate through `OfflineCompoundLoop.run`
   so the LLM can propose "feature + gate" pairs. (~10 lines.)
4. **`scripts/run_discovery.py`** — new `DEFAULT_GOALS` list aimed at
   post-microstructure regimes:
   - "Mean reversion in the 5min after sharp L1 depth depletion"
   - "Directional drift in the 15min after a sustained VPIN spike"
   - "Spread-normalisation after high-intensity 1-min windows"
   - "Reversion after large absolute OFI accumulation"
   - "Drift in the 30min after queue-intensity peak"
   - "Microprice mean-reversion after >2-tick deviation"
5. **`midas-kb-mnq/knowledge/thresholds.json`** — re-tune for longer
   horizons (where ICs are smaller but per-trade returns are larger and
   transaction costs less punishing).

Reserve a ~2-3 hour window for these edits + a validation discovery run.

## 7. Running discovery on the cloud

```bash
cd /workspace/Midas
source .venv/bin/activate

python -u scripts/run_discovery.py \
  --features-dir "market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s" \
  --kb           "./midas-kb-mnq" \
  --provider     anthropic \
  --iters        5 \
  --goals        6
```

Typical session: 12-15 min wall-clock, ~$0.50-2 in Anthropic API cost.

Outputs:
- `midas-kb-mnq/knowledge/learnings/offline/*.md` — one per goal
- `midas-kb-mnq/knowledge/features/candidates/*.md` — accepted expressions
- Console log with rankIC / IR / OOS_IC / composite per candidate

## 8. Re-pulling raw MBO data on the cloud (only if you need to re-extract)

If a future extractor change requires re-running across the raw events:

```bash
# Pull MBO + ohlcv to the cloud machine directly (~10-30 min vs 80 min locally)
python scripts/pull_week.py \
  --symbol MNQ.c.0 \
  --start 2026-04-27 \
  --end   2026-05-15 \
  --schemas ohlcv-1s mbo \
  --root "market_data"

# Re-extract (currently single-threaded — ~80 min for 17 days)
python scripts/extract_mbo_features.py \
  --in  "market_data/databento/GLBX.MDP3/MNQ.c.0/mbo" \
  --out "market_data/features/GLBX.MDP3/MNQ.c.0/bar_1s"
```

**Parallelization opportunity**: `extract_mbo_features.py` processes one
day at a time on one core. Wrapping `MboFeatureExtractor.extract_from_dbn`
calls in a `multiprocessing.Pool(processes=N_CORES)` drops the runtime
from ~80 min to ~10 min on a 16-vCPU box. Worth doing early.

## 9. Storage budget (if you ever want to grow the corpus)

Per-symbol disk usage, observed and projected:

| Window | MBO `.dbn.zst` | Features parquet | Total per symbol |
|---|---|---|---|
| 17 days (current) | ~10 GB | ~56 MB | ~10 GB |
| 1 month | ~25-35 GB | ~150 MB | ~25-35 GB |
| 3 months | ~70-110 GB | ~450 MB | ~70-110 GB |
| 1 year | ~280-440 GB | ~1.8 GB | ~280-440 GB |

If you keep raw MBO indefinitely (recommended — extraction is lossy),
1 year of one symbol = ~280-440 GB. Resize the SSD if you grow past
3 months.

## 10. Current status (2026-05-23)

- **MNQ corpus**: 17 days × 14-col parquet, ready to use.
- **MNQ discovery (v3)**: 0 accepted candidates *at the wrong horizon*.
  Best robust signal at 1-min: `queue_intensity_ofi_product_ranked`
  (rankIC 0.0047, IR 0.57, OOS 0.0065). The horizon now considered
  unfit-for-purpose; this candidate may or may not survive at minute-scale.
- **Framework fragilities fixed in v3**: refine prompt now includes DSL
  skill doc (no more hallucinated operators); IR floor prevents 1e6
  explosions; thresholds re-tuned for microstructure scale; comparison
  operators (gt/lt/eq/etc.) added to DSL.
- **NQ pull**: DEFERRED — at minute horizons MNQ data is sufficient.
- **Nautilus backtests**: NEVER RUN — that's the workstream after a
  longer-horizon candidate survives discovery.

## 11. Suggested first-day-on-the-cloud sequence

1. Set up VM, install per §4, scp data per §5 (~30 min).
2. Apply the six code edits in §6 (~2 hr).
3. Run discovery v4 with new horizons + goals (~15 min).
4. Inspect candidates; if anything passes thresholds, save to candidates/.
5. If candidate found → wire into Nautilus for a backtest (~half-day).
6. Iterate on goals / thresholds / gates based on what you observe.

## 12. Known issues / improvements

- **MBO extractor is single-threaded** — parallelize across days for ~8× speedup.
- **Nesting cap of 5 in DSL** rejects ~60% of refinement attempts.
  Consider bumping to 7 (validator + skill doc).
- **Composite-score formula** in `evaluator.py::_composite` is opaque;
  worth auditing — a candidate passing IR + OOS_IC + turnover individually
  can still fail composite. Tune weights for longer horizons.
- **No Nautilus backtests yet** — wired adapter at
  `midas/adapters/nautilus_data.py` exists but has never been used.
- **Memory bank** at `~/.claude/projects/D--File-Transfer-FX-Trading-Midas/memory/`
  contains conversation-portable notes (key naming convention, etc.). Not
  required on the cloud machine, but useful if you continue with Claude Code there.
