# glori — Generate LOFAR Radio Images

`glori` implements a **latent diffusion model (LDM)** that generates realistic, synthetic
radio-continuum survey maps — simulated LOFAR skies — with explicit control over the
positions, brightness, and size of the sources on the map. It also implements **SWIIT
sampling**, a sliding-window technique that stitches many LDM patches together (via
inpainting) into maps of arbitrary size, seamlessly.

This is the reference implementation for:

> T. Vicánek Martínez and M. Brüggen (2026), *"Generating radio continuum survey maps of
> arbitrary size with latent diffusion models"*, A&A, arXiv:[2609.06549](https://arxiv.org/abs/2609.06549)

If you use this code or the pretrained models, please cite that paper.

## How it works (short version)

1. A **VQ-VAE** (`glori.models.vae.vqvae.VQVAE`) compresses 256px image patches of LOFAR
   sky maps into a small discrete latent space (downsampling factor 4).
2. A **diffusion denoiser** (`glori.models.diffusion.denoiser.Denoiser`, a U-Net) is
   trained to generate new samples *in that latent space*, conditioned on an optional
   **catalog context vector** — a small image encoding source positions, total flux,
   peak flux, and size, derived from a real or synthetic source catalog.
3. **SWIIT sampling** (`glori.inference.swiit_sampler.SWIITSampler`) generates maps
   larger than a single patch by sliding a window across the target area, sampling each
   new patch conditioned on the already-sampled neighboring latents (via inpainting), and
   blending the decoded, overlapping patches together.

See Sections 2–5 of the paper for the full methodology. A worked example is in
[`tutorials/tutorial_getting_started.ipynb`](tutorials/tutorial_getting_started.ipynb), and a
conceptual walkthrough of what actually happens during sampling (noise → denoising loop →
decode) is in [`tutorials/how_sampling_works.ipynb`](tutorials/how_sampling_works.ipynb).

## Repository layout

```
src/glori/
  config/       preset-loading machinery (JSON configs -> model/dataset kwargs)
  data/         dataset classes, transforms, scalers, catalog/mosaic loading
  inference/    DMSampler (low-level), LDMSampler (single patch), SWIITSampler (arbitrary size)
  models/       VAE / VQ-VAE, diffusion denoiser (U-Net, DiT), model loading utilities
  settings/     paths.py — all storage locations (see "Storage locations" below)
  train/        Lightning training entry points (train.denoiser, train.vae, train.flow)
  plotting/     plotting helpers for maps, metrics, training curves
  analysis/     PyBDSF-based source finding and result analysis
  maps/         legacy T-RECS + telescope-simulation pipeline (see "What's not covered" below)
configs/
  model_presets/    JSON configs for VAE / VQ-VAE / Denoiser architectures & training
  dataset_presets/  JSON configs for dataset construction (crops, context, scalers, ...)
notebooks/      exploratory notebooks (mostly research scratchpads, not documentation)
tutorials/      polished, documented example notebooks -- start here
scripts/        one-off training/analysis scripts, of varying levels of maintenance
scripts/release/  tooling for preparing pretrained models for public release
```

## What's *not* covered by this README

`glori.maps.map_maker` + `glori.maps.telescope_simulator` implement an older, separate
pipeline that builds a sky model from a T-RECS source catalog and simulates a full
radio-interferometric observation of it (calibration, noise, PSF) by driving external
LOFAR software (DP3, DDF, `ddf-pipeline`, `losito`). It is **not** the pipeline described
in the paper above.

The first half — T-RECS catalog generation + assembling a clean, noise-free sky model
(`MapMaker`, no external LOFAR software needed) — **is** demonstrated in
[`tutorials/tutorial_legacy_mapmaker.ipynb`](tutorials/tutorial_legacy_mapmaker.ipynb).
The second half — `TelescopeSimulator`, converting that clean sky model into a realistic
simulated observation — needs a cluster environment with the external LOFAR software
installed and is not documented here; read `src/glori/maps/telescope_simulator.py`
directly, or ask a group member familiar with the LOFAR imaging stack.

This older pipeline is the one described in the group's *previous* paper:

> T. Vicánek Martínez, H. W. Edler, and M. Brüggen (2025), *"Simulating realistic radio
> continuum survey maps with diffusion models"*, A&A, arXiv:[2506.11715](https://arxiv.org/abs/2506.11715)

There, individual source images (sampled from a separate, size-conditioned diffusion
model — `model_results/Prototypes_Model_SizeCond`, for extended sources — or simple 2D
Gaussians for compact ones) are assembled onto a T-RECS catalog's source positions to
build a "sky model", which is then run through a full simulated LOFAR observation
(predict visibilities, add noise, CLEAN). This is a fundamentally different use of a
"catalog" than the catalog *context vector* used elsewhere in this repo (see "How it
works" above) — here the catalog drives literal per-source image placement and a real
interferometric simulation, rather than conditioning a single generative model that
produces a whole map directly.

## Installation

The package targets Python 3.11 and is installed in editable mode from `environment.yml`.
Standard routine with your own [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
(or `mamba`/`conda`):

```bash
git clone https://github.com/tmartinezML/glori.git
cd glori
git switch tutorial   # the tutorials and the install fixes live on this branch

micromamba env create -n glori -f environment.yml
micromamba activate glori
pip install -e .
```

### Alternative: uv

With [uv](https://docs.astral.sh/uv/) all dependencies come from `pyproject.toml`:

```bash
git clone https://github.com/tmartinezML/glori.git
cd glori
git switch tutorial   # the tutorials and the install fixes live on this branch

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[notebooks]"
```

The `notebooks` extra adds `ipykernel`, which the tutorial notebooks need; leave it out
if you only use the package from scripts. Python must be 3.11 (the pinned `numpy` and
`torch` have no wheels for newer versions).

An editable install (`-e`) is recommended, but a regular one works too: the presets from
`configs/` are then shipped inside the package. Either run `uv pip install ".[notebooks]"`
in the clone (on the `tutorial` branch), or install straight from GitHub without cloning.
The part after the last `@` selects the branch:

```bash
uv pip install "glori[notebooks] @ git+https://github.com/tmartinezML/glori.git@tutorial"
```

### Set the storage directory

On a fresh machine you have to point `glori` to a storage directory of your own.
Either set the environment variable `GLORI_STORAGE_PARENT` before importing `glori`:

```bash
export GLORI_STORAGE_PARENT=$HOME/glori_storage
```

or open `src/glori/settings/paths.py` and change `STORAGE_PARENT` there (only possible
with an editable install). Every path this package uses (model checkpoints, cache, image
data, ...) is derived from it, and the whole directory tree is created automatically the
first time you import `glori` — but only under whatever `STORAGE_PARENT` currently
points to.

## Storage locations

All data/model paths are centralized in `src/glori/settings/paths.py`, keyed off a single
`STORAGE_PARENT` constant.

Importing `glori.settings.paths`
creates the directory tree under `STORAGE_PARENT` automatically if it doesn't exist yet.

**You don't need any of this machinery just to run inference**, though. The easiest
option for inference is `glori.hub.download_pretrained(...)` (see Quickstart), which
fetches straight from the public Hugging Face repo and doesn't use `STORAGE_PARENT`
for anything — it still creates the (empty, harmless) `STORAGE_PARENT` directory tree.

## Pretrained models

Two checkpoints are needed to reproduce the paper's main results (patch size 512px /
latent size 128px):

| Component | Checkpoint | Notes |
|---|---|---|
| VQ-VAE | `VQ-VAE-256-DR3opt-FT` | image <-> latent compression, downsampling factor 4 |
| Denoiser | `LDM-Denoiser-WnetCC-v5` | catalog-context-conditioned diffusion U-Net |

Public download: **[huggingface.co/astrokevin/glori-ldm-swiit](https://huggingface.co/astrokevin/glori-ldm-swiit)**.
(Stripped-down, inference-only versions of these checkpoints — weights only, no optimizer
state — are prepared with `scripts/release/strip_checkpoint.py`.)

**License:** the pretrained weights are released under **CC BY-SA 4.0**, separately from
this repo's MIT-licensed code. Both models were trained on [LoTSS-DR2](https://lofar-surveys.org/dr2_release.html)
data, which is itself CC BY-SA 3.0 — the share-alike clause carries over to derivative
works like these weights, so if you build on them (fine-tune, redistribute, etc.), your
result needs to stay under a compatible CC BY-SA license too. If you use the weights,
please cite both the paper above and Shimwell et al. (2022), *"The LOFAR Two-metre Sky
Survey. V. Second data release"*, A&A, [10.1051/0004-6361/202142484](https://doi.org/10.1051/0004-6361/202142484).

For inference, you don't need to place checkpoints under `STORAGE_PARENT` at all — every
sampler accepts an explicit file path in place of a symbolic checkpoint name. See the
Quickstart below.

## Quickstart

Minimal unconditional sampling — generates one 512x512px synthetic sky patch from pure
noise, no source catalog needed. Checkpoints download automatically from the public
Hugging Face repo the first time this runs (and are cached locally afterwards) — no
cluster access or manual download required:

```python
import torch
from glori.config.swiit_config import SWIITSamplerConfig
from glori.inference.swiit_sampler import SWIITSampler
from glori.data.trf.scalers import LOFARScaler
from glori.hub import download_pretrained

DENOISER_CKPT = download_pretrained("denoiser")
VAE_CKPT = download_pretrained("vae")
SCALER_PATH = download_pretrained("scaler")

config = SWIITSamplerConfig(
    denoiser="unused",  # ignored: an explicit "/" path in *_ckpt below bypasses name lookup
    denoiser_ckpt=DENOISER_CKPT,
    vae="unused",
    vae_ckpt=VAE_CKPT,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
sampler = SWIITSampler(config=config)
sampler.scaler = LOFARScaler(**torch.load(SCALER_PATH, weights_only=False))

map_image, latent_map = sampler.sample(sampling_steps=(2, 2), timesteps=25)
physical_map = sampler.scaler.inverse_scale(map_image)  # back to Jy/beam-like units
```

For the full walkthrough — including catalog-conditioned generation (controlling where
sources appear), sampling larger maps, and understanding the sampling parameters — see
[`tutorials/tutorial_getting_started.ipynb`](tutorials/tutorial_getting_started.ipynb).

## Known issues / gotchas

- **`STORAGE_PARENT`** in `settings/paths.py` is hardcoded to the original author's
  cluster path. Change it for your own setup or set `GLORI_STORAGE_PARENT` (see
  "Installation"/"Storage locations" above), or use the explicit-checkpoint-path pattern above to bypass it entirely for
  inference. The directory tree under it is created automatically on import, so you
  don't need to `mkdir` anything yourself once it points somewhere valid — but on a
  fresh machine you do need to change it *before* the first `import glori...`, or the
  auto-creation will target the original hardcoded path instead.
- **`glori.models.load.load_model()` is dead code** — it unconditionally raises
  `NotImplementedError`. The real loading path (used everywhere in this codebase) is
  `SomeLightningClass.load_from_checkpoint(parse_lightning_ckpt(...))`.
- **Old checkpoints can fail to load** with a `TypeError` about an unexpected `distance_fn`
  keyword argument in the VQ-VAE's vector-quantization layer. This happens for checkpoints
  saved before the `vqtorch` dependency was pinned to a newer commit (see commit
  `412cf25`). Fix: `del ckpt["hyper_parameters"]["vq_kwargs"]["distance_fn"]` before
  loading, or re-save the checkpoint after removing that key.
- **Several loose scripts under `scripts/`** (e.g. `sample_from_model.py`,
  `sample_intermediates.py`, `map_simulation_streamline.py`) still use a pre-refactor flat
  import layout (`models.*`, `data.*`, `utils.*` instead of `glori.models.*`, etc.) and
  will fail to import as-is. `scripts/ldm/*.py` and
  `tutorials/tutorial_getting_started.ipynb` are kept working; treat everything else under
  `scripts/` as reference material rather than ready-to-run code.
- `tests/unit_tests.py` only checks that every module *imports* without error — it does
  not exercise any sampling/training behavior.

## License

Code: MIT (see `pyproject.toml`). Pretrained model weights: **CC BY-SA 4.0**, see
"Pretrained models" above — this is a separate, stricter license required by the
LoTSS-DR2 training data's own share-alike terms.
