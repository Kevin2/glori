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
| Denoiser (noise-conditioned) | `LDM-Denoiser-WnetCC-rms-v1` | as v5, plus a local-noise context channel; used by the noise-band tools below (`download_pretrained("denoiser-rms")`) |

Public download: **[huggingface.co/astrokevin/glori-ldm-swiit](https://huggingface.co/astrokevin/glori-ldm-swiit)**.
(Stripped-down, inference-only versions of these checkpoints — weights only, no optimizer
state — are prepared with `scripts/release/strip_checkpoint.py`.)

**License:** the pretrained weights are released under **CC BY-SA 4.0**, separately from
this repo's MIT-licensed code. The models were trained on [LoTSS-DR3](https://lofar-surveys.org/dr3.html)
data, which is itself [CC BY-SA 3.0](https://vo.astron.nl/lotss_dr3/q/src_cone/howtocite) — the share-alike clause carries over to derivative
works like these weights, so if you build on them (fine-tune, redistribute, etc.), your
result needs to stay under a compatible CC BY-SA license too. If you use the weights,
please cite both the paper above and Shimwell et al. (2026), *"The LOFAR Two-metre Sky
Survey. VII. Third data release"*, A&A 707, A198, [10.1051/0004-6361/202557749](https://doi.org/10.1051/0004-6361/202557749).

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

## Test images in a noise band (for source finders)

`scripts/generate_noise_images.py` generates LOFAR-like 512x512 px images (1.5" px,
6" beam, 144 MHz, Jy/beam) whose noise lies in a requested band, together with the
catalog of the sources put into them:

```bash
python scripts/generate_noise_images.py --noise 40-55 --n 100 --out noise_40-55 --gpu 0
```

or from Python:

```python
from glori.inference.noise_sampler import NoiseBandSampler

NoiseBandSampler(device="cuda:0").sample(40, 55, n=100, out_dir="noise_40-55")
```

**Requirements.** One GPU, a working PyBDSF (`bdsf`) installation and internet access.
On first use, the noise-conditioned denoiser (`LDM-Denoiser-WnetCC-rms-v1`), the VQ-VAE,
the scaler and three small LoTSS-DR3 derived files (source properties for the synthetic
sources, a list of field centres, a list of image-sized tiles with their depth) are downloaded from
[huggingface.co/astrokevin/glori-ldm-swiit](https://huggingface.co/astrokevin/glori-ldm-swiit)
and cached; set `GLORI_PRETRAINED_DIR` to a local copy of that repository to use it instead
(on the group cluster: `/hs/fs08/data/group-brueggen/kevin/glori_paper_release`). Source
catalogs are queried from the public [LoTSS-DR3 cone search](https://vo.astron.nl/lotss_dr3/q/src_cone/info)
for every image, a few at a time.

**Noise** is the median of the PyBDSF rms map with the LoTSS settings of
`glori.analysis.bdsf_analysis.DEFAULT_KWARGS`. Every written image has a noise inside
the band (stored as `NOISE` in the FITS header), drawn log-uniformly over it.

**Valid range: 38.9–222 µJy/beam.** Requests outside it are refused. Real 6" DR3 cutouts
span ~55–200 µJy/beam (5–95 %).

**How an image is made.** The model (`LDM-Denoiser-WnetCC-rms-v1`) reads the depth of
an image mostly from its input catalog, so each image gets a catalog consistent with
its noise. There are two modes, recorded as `mode` in `summary.csv` and `MODE` in the
FITS header:

- **Real depth, below 65 µJy/beam:** the unchanged catalog of a real DR3 tile (12.8', the
  size of an image) whose own depth gives the requested noise. There are no synthetic
  sources. Bright sources occur about as often as in the real sky, and their artefacts
  follow the real `Isl_rms` of the catalog.
- **Depth matched, from 65 µJy/beam:** a catalog built for a depth T from a random DR3
  field of native noise <= 90 µJy/beam: its real sources above the completeness limit
  (8x its native noise) and at least 3 T, plus synthetic faint sources with DR3-like SNR,
  sizes and total/peak ratios down to T, in DR3-like numbers. The noise channel is T / 1.07
  plus, in quadrature, the real noise excess around bright sources, which keeps their artefacts.

The tile depth or T comes from the calibration so that the generated noise is close to the
requested one. The image is then measured with PyBDSF and rescaled to the requested noise
exactly. PyBDSF results scale linearly with the image, and the model scales sources and
noise together, so the rescale (0.83–1.20 for 90 % of the images, at most x1.3) changes the flux
scale only, not the LOFAR noise texture. Images that would need a larger rescale, whose
noise texture falls outside the range of real DR3 cutouts, or that miss bright input
sources are dropped and regenerated (`report.json` counts them): 6–12 % of the images in the
test runs (800 images kept, 869 generated).

**Calibration** (`configs/noise_calibration/rms-v1.json`). Real depth: 48 tiles at each of 7
noise levels, each with the real DR3 cutout of the same tile. Generated noise is 0.99–1.10
of the real noise, and source fluxes are within 10 %. Noise texture passes as often as for
the real cutouts (0.83–1.00 against 0.80–0.96). Above 65 µJy/beam this mode misses the flux
limit (0.83–0.84), which is why the modes switch there. Depth matched: 69 cutouts of native
noise <= 90 µJy/beam x 12 depths, 3000 real DR3 cutouts as reference: the noise rises with
the depth, the texture looks like real DR3 cutouts of that noise, bright sources are
recovered, and fluxes of the kept images stay consistent within 10 %. Above ~222 µJy/beam
the noise still rises, but bright sources are missed more often than the criterion allows.

**Output** in `--out`:

| File | Content |
|---|---|
| `images/img_#####.fits` | image [Jy/beam]; header: `NOISE` [µJy/beam], `RESCALE`, `DEPTH` (T; in real-depth mode the noise of the tile's catalog, 1.07 x its median `Isl_rms`), `MODE`, `FLUXFAC`, `FIELDRA`/`FIELDDEC` (DR3 field of the real sources), `CODEVER` (git commit of the code) |
| `truth_catalog.csv` | all input sources per `image_id`: `x_px`, `y_px` (0-based, x = column), `peak_flux`, `total_flux` [Jy, as in the image], `peak_flux_input`, `total_flux_input` [Jy, as in the catalog], `maj_arcsec`, `synthetic` (always false in real-depth mode) |
| `pybdsf_catalog.csv` | PyBDSF sources of each image (same pixel convention) |
| `summary.csv`, `report.json` | one row per image (noise, rescale, mode, `flux_factor`, `flux_factor_from`, texture metrics), and the run: request, rejections, runtime |

**Known limitations**
- Positions are on the 4 px grid of the catalog context (6"), and generated sources sit
  ~2–3 px (3–4.5") off their catalog position; match with a radius of ~12".
- **The truth catalog holds the flux in the image, not the flux in the sky:**
  `peak_flux = peak_flux_input x image flux factor x brightness factor`, and `total_flux` likewise
  with its own brightness factor. Compare your source finder with these columns.
- The **image flux factor** is the median measured / expected peak flux of the image's own sources
  (at least 5 with input peak >= 20 sigma), else the calibrated value (`flux_factor_from`). On random
  fields, per image (5–95 %): real depth 0.82–1.16, from its own sources in 92 % of the images; depth
  matched 0.60–1.05 from its own sources (37 % of the images), 0.65–0.93 calibrated.
- The **brightness factor** exists because **the denoiser compresses bright sources**, so they
  come out fainter than in the real sky; the VQ-VAE alone does not (0.87–0.97 of the real peak at all
  brightnesses). The compression follows the input SNR (input peak / noise of the image), so the same source
  loses more at low noise. Maps have their own curve, as they lose more above SNR ~5000
  (`brightness` in the calibration files). Below SNR 20 it is not measured and returns to 1 at SNR 10;
  above the last value it stays constant:

  | Input SNR | ~26 | ~147 | ~465 | ~830 | ~1480 | ~2650 | ~4600 | ~8300 | ~15000 and above |
  |---|---|---|---|---|---|---|---|---|---|
  | Images, peak | 1.04 | 0.93 | 0.81 | 0.76 | 0.75 | 0.63 | 0.54 | 0.42 | 0.31–0.26 |
  | Maps, peak | 1.05 | 0.92 | 0.81 | 0.72 | 0.71 | 0.57 | 0.49 | 0.25 | 0.22 |

  A 1 Jy source at 50 µJy/beam (SNR 20 000) is about 0.27 Jy in the image, and the truth catalog says so.
  The images are not corrected, so do not use them to study the bright end of the flux distribution.
  Detection is not affected (bright sources are recovered).
- **Check of the truth catalog on data not used for the fits:** 600 images and 16 maps on fields with a
  source >= 150 mJy, from sky stripes not used for the targeted fit runs (9 % of the sources >= 200 mJy also occur
  in the random fit runs). Median PyBDSF peak flux / `peak_flux`:

  | Input peak [mJy] | 0.5–5 | 5–10 | 10–20 | 20–50 | 50–100 | 100–200 | 200–500 | 500–1000 | > 1000 |
  |---|---|---|---|---|---|---|---|---|---|
  | Real depth | 1.00 | 0.99 | 1.00 | 1.00 | 0.91 | 0.98 | 0.97 (n=181) | 1.06 (n=36) | 0.97 (n=10) |
  | Depth matched | 1.00 | 1.00 | 0.97 | 0.98 | 0.99 | 0.99 | 1.02 (n=174) | 1.08 (n=36) | 1.39 (n=23) |
  | Maps | 1.01 | 1.00 | 0.96 | 1.00 | 0.92 | 0.90 | 1.01 (n=35) | 1.01 (n=10) | 1.14 (n=9) |

  This is correct on average, not per source. The 16–84 % range of single sources is 0.89–1.12 below 5 mJy, 0.76–1.21 at
  20–50 mJy, 0.69–1.44 at 200–500 mJy and 0.66–1.80 at 500–1000 mJy. Total flux is within 10 % up to 500 mJy except
  real depth at 50–100 mJy (0.85, n=37). Above 500 mJy samples are small, and total flux is less reliable (up to 1.51 in depth
  matched images and 1.69 in maps above 1 Jy, n=23 and n=9).
- **Images with bright sources are dropped more often at low noise, so low-noise sets hold fewer of them than
  the real sky.** On fields with a source >= 150 mJy, 49 % of the images at 40–65 µJy/beam are dropped, mostly by the
  texture check, and 13 % at 65–222 µJy/beam, against 6–12 % on random fields. At 40–65 µJy/beam, kept images hold a
  source above 1.2 Jy about 4 times less often than the fields on offer (1.7 % against 7.0 %). In the random-field
  runs below 55 µJy/beam, 3 % of the images hold a source >= 200 mJy, against 6–8 % of real tiles of that noise; at 40–45
  µJy/beam, 16 % hold a source >= 50 mJy, against 26 %. How many real images with such sources would fail the same
  check was not measured.
- At the switch to depth matching, images have about 15 % fewer sources than in real-depth
  mode at the same noise.
- Faint emission the model adds on its own (below the catalog depth) is not listed.

**Resources.** One GPU (`--gpu`) and `--workers` single-core PyBDSF processes at low
priority (default: CPU count - 2, at most 24). About 1 min per 16 images on an A100.

### Large maps in a noise band

`scripts/generate_noise_maps.py` makes survey-like maps of any size (default 2048x2048 px
= 51'; 512 + k x 256 px, at least 1024) with SWIIT, the tiled sampler, from a depth-matched
catalog as described above (real DR3 tiles of the right depth are not used for maps):

```bash
python scripts/generate_noise_maps.py --noise 60-80 --n 8 --size 2048 --out maps_60-80 --gpu 0
```

```python
from glori.inference.noise_map_sampler import NoiseBandMapSampler

NoiseBandMapSampler(device="cuda:0").sample(60, 80, n=8, out_dir="maps_60-80", map_px=2048)
```

**Noise of a map** is the median of its PyBDSF rms map; each map is rescaled as a whole
so that this median lies in the band.

**Flattening.** SWIIT maps drift in noise over several arcminutes, more than real DR3
fields. By default each map is therefore multiplied by a smooth factor, median / local
background rms (512 px blocks, bright-source areas masked, at most x1.3), and measured again.
Sources and noise are scaled together, so the noise peaks around bright sources stay, and so
does each source's local SNR. The background spread of 256 px blocks (p95/p5, bright-source
areas masked) falls from 1.36–1.72 to 1.14–1.25; real DR3 has 1.20 (1.51 above 110 µJy/beam). `--no-flatten` keeps the
drift and saves one PyBDSF run per map. The factor is in the truth catalog (`flatten_factor`),
and `FLATTEN` in the header says whether it was applied.

The noise still rises around bright sources, as in real DR3, so the PyBDSF rms map is
delivered with every map; use it for local SNR. Noise between neighbouring 64 px blocks
changes about as gradually as in real DR3 (p99 ratio 1.24–1.44 against 1.30–1.36; largest step
2.43 against 2.76), so the maps show no hard borders from the noise itself.

**Valid range: 50–224 µJy/beam** for the map median; requests outside it are refused.
The calibration (`configs/noise_calibration/rms-v1-swiit.json`, 8 DR3 fields of native depth
37–87 µJy/beam x 7 depths, 2048 px) found the map noise rising with the depth in every
field, no tile seams, ~99 % of bright input sources recovered and 91–97 % of the 512 px
tiles with a DR3-like noise texture. Below ~50 µJy/beam only ~77 % of the tiles pass
(striped, too anisotropic noise), so that part is excluded.

**How a map is made.** Real sources come from a random DR3 field with native depth
<= 90 µJy/beam and full survey coverage over the map and a 256 px margin (the margin is
the context SWIIT sees beyond the map edge); the catalog is made consistent with depth T
as for depth-matched single images. Maps that would need a rescale above x1.3, whose 512 px
tiles have a DR3-like texture in fewer than 75 % of cases, or that miss more than 20 % of the
bright input sources are dropped and regenerated.

**Artefacts around bright sources.** Spokes and noise peaks are present but somewhat weaker
than in real DR3. Same 10 sources above 200 mJy at T = 55, flattened, against real DR3, at
24–60" / 1–2' / 2–4': 164/64/28 against 169/78/32 µJy/beam. This rests on those 10 sources.

**Fluxes.** SWIIT maps come out with source fluxes below the catalog, depending on the
field: the factor is 0.56–0.96 per 2048 px map (`FLUXFAC` in the header). The truth catalog is
therefore calibrated per map: `peak_flux`/`total_flux` are the input fluxes times the flattening
factor at the source, the brightness factor of the maps (see the single images, and the table there) and the
median measured / expected peak flux of the map's own sources with input peak >= 20 T (the calibrated factor of
the depth is used if a map has fewer than 10 such sources). Checked by re-running PyBDSF on 16 written maps:
sources at 20–100 sigma come out at 0.95–1.05 of the truth flux per map. The check by input peak is in the table
for single images. As there, the maps themselves are not corrected, so bright sources in them are fainter
than in the real sky.

**Output** in `--out`:

| File | Content |
|---|---|
| `maps/map_#####.fits` | map [Jy/beam]; header: `NOISE` (median rms) [µJy/beam], `RESCALE`, `DEPTH` (T), `FLUXFAC`, `FLATTEN`, `FIELDRA`/`FIELDDEC` (DR3 field of the real sources), `CODEVER` |
| `maps/map_#####_rms.fits` | PyBDSF rms map of the map [Jy/beam] |
| `truth_catalog.csv` | input sources per `map_id`: as for single images (`peak_flux`, `total_flux` include the flattening), plus `detected` (PyBDSF source within 12") and `flatten_factor` |
| `pybdsf_catalog.csv` | PyBDSF sources of each map |
| `tiles.csv` | per 512 px tile: noise, texture metrics, `texture_ok` |
| `summary.csv`, `report.json` | one row per map (noise, rescale, `flux_factor`, `flux_factor_from`, rms spread before and after flattening, texture fraction, bright-source recovery), and the run |

The real sources can come from anywhere in DR3, including areas whose catalogs were seen
in training; the maps themselves are newly generated.

**Resources.** Requirements as for single images; one cone search per map (checked complete
up to 3° radius, i.e. maps up to 8192 px). One GPU and `--workers` single-core PyBDSF processes at low priority.
Sampling 8 maps of 2048 px together (`--batch 8`) takes ~26 min on an A100 (~3.3 min per
map; ~4.3 min per map in batches of 4); time grows with the map area. PyBDSF takes ~2–3 min
per 2048 px map and runs while the next batch is sampled.

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
LoTSS-DR3 training data's own share-alike terms.
