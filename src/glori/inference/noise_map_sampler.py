"""
Generate large LOFAR-like maps (e.g. 2048x2048 px = 51') whose noise lies in a
requested band, for testing source finders on survey-like areas.

Maps are sampled with SWIIT (overlapping 512 px tiles, sequential inpainting)
from a depth-matched catalog, as the single images of noise_sampler.py:
  - real LoTSS-DR3 sources (public cone search) of a random DR3 field (native
    depth <= 90 uJy/beam, survey coverage over the map and its context margin)
    with peak >= 8x the field's native noise and >= 3 T
  - synthetic DR3-like sources below that, peak = SNR x T
  - rms channel = T / 1.07 plus, in quadrature, the real noise excess around
    bright sources
The noise of a map is the median of its PyBDSF rms map. Per map a noise s is
drawn log-uniformly from the band, T is taken from the large-map calibration,
and the map is rescaled by s / measured noise as a whole, so its median noise is
exactly s; the noise still varies across the map (the rms map is delivered with
it). Maps are dropped and replaced if they need a rescale beyond `max_rescale`,
if too few of their 512 px tiles have a DR3-like noise texture, or if bright
input sources are missing.

SWIIT maps also drift in noise over several arcminutes, more than real DR3 fields
(p95/p5 of 256 px blocks, bright-source areas masked: ~1.4 vs. ~1.16). By default
each map is therefore flattened: it is multiplied by a smooth factor, median /
local background rms (512 px blocks, bright-source areas masked, at most x1.3),
and measured again. Sources and noise are scaled together, so local peaks around
bright sources stay, and so does each source's local SNR.

SWIIT maps come out with source fluxes below the catalog, depending on the
field (factor 0.56-0.96 per 2048 px map in the test runs). The truth catalog is therefore calibrated per map: its fluxes are
the input fluxes times the flattening factor at the source, a brightness factor
(the denoiser compresses sources by their SNR, see noise_sampler.py; more than in
single images above SNR ~5000, so maps have their own curve) and the median
measured / expected peak flux of the map's own bright sources (input peak >= 20
T), or the calibrated factor of the depth if a map has fewer than 10 of them.

Calibration: configs/noise_calibration/rms-v1-swiit.json (built by
scripts/ldm/build_noise_map_calibration.py). Requests outside its valid range
are refused.
"""

import contextlib
import json
import os
import shutil
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import glori.infra.logging as my_logging
import glori.settings.paths as paths
from glori.inference.noise_sampler import (
    CTX_PX,
    MATCH_PX,
    SAMPLE_KW,
    TEXTURE,
    NoiseBandSampler,
    _anisotropy,
    _autocorr_halfwidth,
    code_version,
    depth_matched_context,
    dr3_fields,
    fetch_field_context,
    load_scaler,
    snr_pool,
    start_bdsf_pool,
    write_fits,
)

NOISE_MAP_CALIBRATION = paths.CONFIG_PARENT / "noise_calibration" / "rms-v1-swiit.json"
TILE_PX = 512  # texture is checked per 512 px tile, the size of the real DR3 reference
PAD = 64  # context margin around the map for SWIIT's extended context [cells]
STRIDE_PX = 256  # SWIIT tile stride: map sizes are 512 + k * 256 px
FLATTEN_BLOCK_PX = 512  # scale of the background rms trend that flattening removes (13')
FLATTEN_MAX = 1.3  # largest local correction, like the global rescale
BRIGHT_MASK = [(100, 80), (30, 40)]  # (min peak SNR, radius [px]): ignored in the background


# ---- DR3 fields ------------------------------------------------------------------
def field_ok(ctx, fields):
    """Native depth [uJy/beam] of a field context, or None if the field is not usable."""
    src = ctx[0] > 0
    b = fields["block_cells"]
    nb = ctx.shape[1] // b
    count = src[: nb * b, : nb * b].reshape(nb, b, nb, b).sum(axis=(1, 3))
    if count.min() < fields["min_sources_per_block"]:
        return None
    native = float(np.median(ctx[4][src] / ((ctx[0][src] + 1) / 2)))
    return native if native <= fields["max_native_uJy"] else None


def map_truth_catalog(ctxt, synthetic):
    """Sources of a map context (without margin): pixel position, fluxes [Jy], Maj."""
    rows, cols = np.nonzero(ctxt[0] > 0)
    return pd.DataFrame(
        dict(
            x_px=cols * CTX_PX + (CTX_PX - 1) / 2,
            y_px=rows * CTX_PX + (CTX_PX - 1) / 2,
            peak_flux_input=ctxt[2][rows, cols] * 1e-3,
            total_flux_input=ctxt[1][rows, cols] * 1e-3,
            maj_arcsec=ctxt[3][rows, cols],
            synthetic=synthetic[rows, cols],
        )
    )


# ---- measurement (runs in PyBDSF worker processes) -----------------------------
def measure_map(img, tmpdir):
    """PyBDSF noise and rms map, texture metrics per 512 px tile and source catalog of one map."""
    from astropy.wcs import WCS

    from glori.analysis.bdsf_analysis import (
        DEFAULT_KWARGS,
        bdsf_on_image,
        catalogs_from_bdsf,
        make_header_dict,
    )

    tmp = Path(tempfile.mkdtemp(dir=tmpdir))
    kw = dict(DEFAULT_KWARGS, quiet=True, debug=False, ncores=1)
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            res = bdsf_on_image(img.astype(np.float64), tmpdir=tmp, **kw)
            srl = catalogs_from_bdsf(res, tmpdir=tmp)["srl"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # on NFS, files PyBDSF still holds stay until it lets go
    rms_map = np.asarray(res.rms_arr).squeeze().T
    resid = np.asarray(res.resid_gaus_arr).squeeze().T
    sigma = float(np.median(rms_map))
    z = resid / rms_map

    tiles = []
    for y0 in range(0, img.shape[0] - TILE_PX + 1, TILE_PX):
        for x0 in range(0, img.shape[1] - TILE_PX + 1, TILE_PX):
            sl = np.s_[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
            s = float(np.median(rms_map[sl]))
            tiles.append(
                dict(
                    x0_px=x0,
                    y0_px=y0,
                    noise_uJy=1e6 * s,
                    grain_px=float(_autocorr_halfwidth(img[sl], s)),
                    neg5=float(np.mean(z[sl] < -5)),
                    pos5=float(np.mean(z[sl] > 5)),
                    aniso=_anisotropy(z[sl]),
                )
            )
    n = img.shape[0] // STRIDE_PX
    blocks = np.median(
        rms_map[: n * STRIDE_PX, : n * STRIDE_PX].reshape(n, STRIDE_PX, n, STRIDE_PX), axis=(1, 3)
    )
    m = dict(
        noise_uJy=1e6 * sigma,
        rms_spread=float(np.percentile(blocks, 95) / np.percentile(blocks, 5)),
    )

    cols = ["RA", "DEC", "Peak_flux", "E_Peak_flux", "Total_flux", "E_Total_flux", "Maj", "Min", "PA"]
    srl = srl[[c for c in cols if c in srl]].reset_index(drop=True)
    if len(srl):
        x, y = WCS(make_header_dict()).celestial.all_world2pix(srl.RA.values, srl.DEC.values, 0)
        srl.insert(0, "y_px", y)
        srl.insert(0, "x_px", x)
    else:
        srl = pd.DataFrame(columns=["x_px", "y_px"] + cols)
    return m, tiles, srl, rms_map.astype(np.float32)


def bright_mask(shape, x, y, snr):
    """Pixels around bright sources (see BRIGHT_MASK)."""
    mask = np.zeros(shape, bool)
    for thr, rad in BRIGHT_MASK:
        for xi, yi in zip(x[snr >= thr], y[snr >= thr]):
            y0, x0 = max(int(yi) - rad - 1, 0), max(int(xi) - rad - 1, 0)
            yy, xx = np.ogrid[y0 : int(yi) + rad + 2, x0 : int(xi) + rad + 2]
            yy, xx = yy[yy < shape[0]][:, None], xx[xx < shape[1]][None, :]
            mask[y0 : y0 + yy.size, x0 : x0 + xx.size] |= (xx - xi) ** 2 + (yy - yi) ** 2 < rad**2
    return mask


def flatten_factor(rms_map, x, y, snr, block=FLATTEN_BLOCK_PX, max_factor=FLATTEN_MAX):
    """
    Smooth factor that flattens the background of an rms map: median / local
    background, the background being the median rms in blocks of `block` px
    outside the bright-source areas, interpolated with a spline in log rms.
    x, y [px], snr: PyBDSF sources of the map.
    """
    from scipy.interpolate import RectBivariateSpline

    mask = bright_mask(rms_map.shape, x, y, snr)
    n = rms_map.shape[0] // block
    bg = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            sl = np.s_[i * block : (i + 1) * block, j * block : (j + 1) * block]
            free = ~mask[sl]
            if free.mean() >= 0.3:
                bg[i, j] = np.median(rms_map[sl][free])
    if np.isnan(bg).all():
        return np.ones(rms_map.shape, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty neighbourhoods
        while np.isnan(bg).any():  # blocks dominated by bright sources: mean of their neighbours
            nb = [[np.nanmean(bg[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]) for j in range(n)] for i in range(n)]
            bg = np.where(np.isnan(bg), nb, bg)
    centres = (np.arange(n) + 0.5) * block - 0.5
    k = min(3, n - 1)
    spline = RectBivariateSpline(centres, centres, np.log(bg), kx=k, ky=k)
    px = np.arange(rms_map.shape[0])
    bg_map = np.exp(spline(px, px))
    return np.clip(np.median(bg_map) / bg_map, 1 / max_factor, max_factor).astype(np.float32)


def _measure_map_job(img, tmpdir, flatten):
    """measure_map of the map, with flatten: of the flattened map. Also returns the factor (or None)."""
    try:
        m, tiles, srl, rms_map = measure_map(img, tmpdir)
        factor = None
        if flatten:
            snr = srl.Peak_flux.values.astype(float) * 1e6 / m["noise_uJy"]
            factor = flatten_factor(rms_map, srl.x_px.values.astype(float), srl.y_px.values.astype(float), snr)
            before = m
            m, tiles, srl, rms_map = measure_map(img * factor, tmpdir)
            m["rms_spread_unflattened"] = before["rms_spread"]
            m["flatten_min"], m["flatten_max"] = float(factor.min()), float(factor.max())
        return (m, tiles, srl, rms_map, factor), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---- sampler ---------------------------------------------------------------------
class NoiseBandMapSampler(NoiseBandSampler):
    """
    sampler = NoiseBandMapSampler()
    report = sampler.sample(60, 80, n=4, out_dir="maps_60-80", map_px=2048)
    """

    def __init__(
        self,
        calibration=NOISE_MAP_CALIBRATION,
        denoiser_ckpt=None,
        vae_ckpt=None,
        device="cuda:0",
        batch_size=8,
        bdsf_workers=None,
        bdsf_nice=10,
    ):
        super().__init__(
            calibration=calibration,
            denoiser_ckpt=denoiser_ckpt,
            vae_ckpt=vae_ckpt,
            device=device,
            batch_size=batch_size,
            bdsf_workers=bdsf_workers,
            bdsf_nice=bdsf_nice,
        )
        self.logger = my_logging.get_logger(__name__)

    @staticmethod
    def check_size(map_px):
        if map_px < 1024 or (map_px - 512) % STRIDE_PX:
            raise ValueError(f"Map size {map_px} px: must be >= 1024 and 512 + k x {STRIDE_PX} px.")

    # -- lazy resources ---------------------------------------------------------
    def _load_sampler(self):
        if self._sampler is None:
            from glori.config.swiit_config import SWIITSamplerConfig
            from glori.inference.swiit_sampler import SWIITSampler

            config = SWIITSamplerConfig(
                denoiser="rms-v1",
                denoiser_ckpt=str(self.checkpoint("denoiser")),
                vae="VQ-VAE",
                vae_ckpt=str(self.checkpoint("vae")),
                device=self.device,
            )
            self._sampler = SWIITSampler(config=config)
            self._sampler.scaler = load_scaler()
        return self._sampler

    def _load_base(self):
        """Candidate DR3 fields covered for a 2048 px map and within the calibrated native depth."""
        if self._base is None:
            f = dr3_fields()
            self._base = f[f.map_ok & (f.native_uJy <= self.calib["fields"]["max_native_uJy"])].reset_index(drop=True)
            self._pool = snr_pool()
        return self._base

    def random_field(self, rng, n_cells, max_tries=50):
        """Centre, padded context and native depth of a random usable DR3 field."""
        fields = self._load_base()
        for _ in range(max_tries):
            f = fields.iloc[int(rng.integers(len(fields)))]
            ctx = fetch_field_context(float(f.ra), float(f.dec), n_cells + 2 * PAD)
            native = field_ok(ctx, self.calib["fields"])  # coverage of maps other than 2048 px
            if native is not None:
                return float(f.ra), float(f.dec), ctx, native
        raise RuntimeError(f"No usable DR3 field found in {max_tries} tries.")

    # -- generation ---------------------------------------------------------------
    def _generate(self, n, band, rng, map_px):
        """n maps, noise drawn log-uniformly over the band -> maps, contexts, info."""
        cells = map_px // CTX_PX
        method = self.calib["method"]
        exts, ctxs, syns, info = [], [], [], []
        for _ in range(n):
            ra, dec, full, native = self.random_field(rng, cells)
            noise = float(np.exp(rng.uniform(*np.log(band))))
            depth = float(self.depth_for_noise(noise))
            c, s = depth_matched_context(full, depth, self._pool, rng, method)
            sl = np.s_[..., PAD : PAD + cells, PAD : PAD + cells]
            exts.append(c)
            ctxs.append(c[sl])
            syns.append(s[sl])
            info.append(
                dict(field_ra=ra, field_dec=dec, native_uJy=native, noise_request_uJy=noise, depth_uJy=depth)
            )
        sampler = self._load_sampler()
        g = torch.Generator().manual_seed(int(rng.integers(2**63)))
        ext = torch.from_numpy(np.stack(exts)).float()
        imgs, _ = sampler.sample(
            ctxt_map=torch.from_numpy(np.stack(ctxs)).float(),
            ext_ctxt_map=ext,
            seed_noise_map=torch.randn((n, 3, cells, cells), generator=g),
            catalog_mode="combined",
            **SAMPLE_KW,
        )
        imgs = sampler.scaler.inverse_scale(torch.as_tensor(imgs))
        imgs = imgs.cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
        return imgs.reshape(n, map_px, map_px).astype(np.float32), ctxs, syns, info

    def _match(self, truth, srl):
        """Index of the nearest PyBDSF source of each truth source and whether it is within 12"."""
        if not len(srl):
            return np.zeros(len(truth), int), np.zeros(len(truth), bool)
        from scipy.spatial import cKDTree

        d, j = cKDTree(srl[["x_px", "y_px"]].values).query(truth[["x_px", "y_px"]].values)
        return j, d < MATCH_PX

    def _flux_factor_map(self, truth, srl, j, hit, depth, noise, g_peak):
        """Measured / expected peak flux of the map from its own bright sources, else calibrated."""
        crit = self.calib["criteria"]
        sel = hit & (truth.peak_flux_input.values * 1e6 >= crit["flux_snr"] * depth)
        if sel.sum() >= crit["min_flux_sources"]:
            expected = truth.peak_flux_input.values[sel] * truth.flatten_factor.values[sel] * g_peak[sel]
            r = srl.Peak_flux.values[j[sel]] / expected
            return float(np.median(r)), int(sel.sum()), "map"
        return float(self.flux_factor(depth, noise)), int(sel.sum()), "calibration"

    def _judge_map(self, m, tiles, truth, hit, max_rescale):
        """Reason for rejection, or None if the map is kept (tiles get a texture_ok flag)."""
        crit = self.calib["criteria"]
        for t in tiles:
            tb = self.texture_band(t["noise_uJy"])
            t["texture_ok"] = all(tb[k][0] <= t[k] <= tb[k][1] for k in TEXTURE)
        m["texture_ok_frac"] = float(np.mean([t["texture_ok"] for t in tiles]))
        bright = truth.peak_flux.values * 1e6 >= 10 * m["noise_uJy"]
        m["n_bright"] = int(bright.sum())
        m["bright_recovered"] = float(hit[bright].mean()) if bright.any() else None
        if not 1 / max_rescale <= m["rescale"] <= max_rescale:
            return "rescale too large"
        if m["texture_ok_frac"] < crit["min_texture_frac"]:
            return "noise texture unlike real DR3"
        if bright.any() and m["bright_recovered"] < crit["min_bright_recovery"]:
            return "bright sources missing"
        return None

    def sample(
        self,
        noise_min,
        noise_max,
        n,
        out_dir,
        map_px=2048,
        seed=0,
        max_rescale=None,
        max_attempts=None,
        flatten=True,
    ):
        """
        Generate `n` maps of `map_px` x `map_px` px whose median PyBDSF noise is in
        [noise_min, noise_max] uJy/beam and that pass the quality checks. Writes
        FITS maps and rms maps, the truth and PyBDSF catalogs, a tile table and a
        report to `out_dir`. flatten: remove the large-scale noise trend of each
        map (a second PyBDSF run per map).
        """
        self.check_band(noise_min, noise_max)
        self.check_size(map_px)
        band = (float(noise_min), float(noise_max))
        max_rescale = max_rescale or self.calib["criteria"]["max_rescale"]
        max_attempts = max_attempts or 2 * n + self.batch_size
        pool = start_bdsf_pool(self.bdsf_workers, self.bdsf_nice)
        out = Path(out_dir)
        (out / "maps").mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp(prefix="bdsf_", dir=out))
        rng = np.random.default_rng(seed)
        self.logger.info(
            f"Generating {n} maps of {map_px} px with median noise {band[0]}-{band[1]} uJy/beam "
            f"(catalog depth {self.depth_for_noise(band[0]):.0f}-{self.depth_for_noise(band[1]):.0f})"
        )

        kept, truths, srls, tile_rows, rejected = [], [], [], [], {}
        n_tried, t0 = 0, time.time()
        pending = {}
        try:
            while len(kept) < n:
                # keep the GPU busy while PyBDSF works on earlier batches
                need = n - len(kept) - len(pending)  # replacements follow rejections
                if n_tried < max_attempts and need > 0 and len(pending) < 2 * self.batch_size:
                    nb = min(self.batch_size, need, max_attempts - n_tried)
                    t_gen = time.time()
                    imgs, ctxs, syns, infos = self._generate(nb, band, rng, map_px)
                    self.logger.info(f"Sampled {nb} maps in {time.time() - t_gen:.0f}s")
                    for img, c, s, inf in zip(imgs, ctxs, syns, infos):
                        pending[pool.submit(_measure_map_job, img, tmpdir, flatten)] = (img, c, s, inf)
                    n_tried += nb
                    continue
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    img, c, s, inf = pending.pop(f)
                    res, err = f.result()
                    if err is not None:
                        self.logger.warning(f"PyBDSF failed: {err}")
                        rejected["PyBDSF failed"] = rejected.get("PyBDSF failed", 0) + 1
                        continue
                    m, tiles, srl, rms_map, factor = res
                    if factor is not None:
                        img = img * factor
                    # PyBDSF is scale-invariant: rescaling the map scales its rms
                    # map and catalog fluxes by the same factor, detections stay
                    m["noise_generated_uJy"] = m["noise_uJy"]
                    m["rescale"] = inf["noise_request_uJy"] / m["noise_uJy"]
                    m["noise_uJy"] = inf["noise_request_uJy"]
                    img = img * np.float32(m["rescale"])
                    rms_map = rms_map * np.float32(m["rescale"])
                    for t in tiles:
                        t["noise_uJy"] *= m["rescale"]
                    for col in ("Peak_flux", "E_Peak_flux", "Total_flux", "E_Total_flux"):
                        if col in srl:
                            srl[col] = srl[col] * m["rescale"]
                    truth = map_truth_catalog(c, s)
                    truth["flatten_factor"] = (
                        factor[truth.y_px.astype(int), truth.x_px.astype(int)] if factor is not None else 1.0
                    )
                    j, hit = self._match(truth, srl)
                    g_peak, g_total = self.brightness_factors(truth, m["noise_uJy"])
                    ff, n_ff, ff_from = self._flux_factor_map(
                        truth, srl, j, hit, inf["depth_uJy"], m["noise_uJy"], g_peak
                    )
                    truth["peak_flux"] = truth.peak_flux_input * ff * truth.flatten_factor * g_peak
                    truth["total_flux"] = truth.total_flux_input * ff * truth.flatten_factor * g_total
                    truth["detected"] = hit
                    why = self._judge_map(m, tiles, truth, hit, max_rescale)
                    if why is not None or len(kept) >= n:
                        rejected[why or "surplus"] = rejected.get(why or "surplus", 0) + 1
                        self.logger.info(f"Map rejected: {why or 'surplus'}")
                        continue
                    k = len(kept)
                    for df in (truth, srl):
                        df.insert(0, "map_id", k)
                    truths.append(truth)
                    srls.append(srl)
                    tile_rows += [dict(map_id=k, **t) for t in tiles]
                    kept.append(
                        dict(
                            map_id=k,
                            file=f"maps/map_{k:05d}.fits",
                            rms_file=f"maps/map_{k:05d}_rms.fits",
                            map_px=map_px,
                            **inf,
                            flux_factor=ff,
                            flux_factor_n=n_ff,
                            flux_factor_from=ff_from,
                            **m,
                        )
                    )
                    cards = dict(
                        NOISE=(round(m["noise_uJy"], 3), "median of PyBDSF rms map [uJy/beam]"),
                        RESCALE=(round(m["rescale"], 4), "factor applied to the generated map"),
                        DEPTH=(round(inf["depth_uJy"], 3), "catalog depth T [uJy/beam]"),
                        FLUXFAC=(round(ff, 4), f"flux factor of the map ({ff_from})"),
                        FLATTEN=(factor is not None, "large-scale noise trend removed"),
                        FIELDRA=(round(inf["field_ra"], 4), "DR3 field of the real sources [deg]"),
                        FIELDDEC=(round(inf["field_dec"], 4), "DR3 field of the real sources [deg]"),
                        MODEL=(self.checkpoint("denoiser").stem, "denoiser, SWIIT"),
                        CALIB=(self.calibration_file.name, "noise calibration"),
                        CODEVER=(code_version(), "glori code version"),
                    )
                    write_fits(img, out / kept[-1]["file"], **cards)
                    write_fits(rms_map, out / kept[-1]["rms_file"], **cards)
                    self.logger.info(
                        f"{len(kept)}/{n} kept (noise {m['noise_uJy']:.1f}, rescale {m['rescale']:.3f}, "
                        f"flux factor {ff:.3f} from {ff_from}), {n_tried} generated, {time.time() - t0:.0f}s"
                    )
        finally:
            for f in pending:
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            shutil.rmtree(tmpdir, ignore_errors=True)

        summary = pd.DataFrame(kept)
        summary.to_csv(out / "summary.csv", index=False)
        if truths:
            pd.concat(truths, ignore_index=True).to_csv(out / "truth_catalog.csv", index=False)
            pd.concat(srls, ignore_index=True).to_csv(out / "pybdsf_catalog.csv", index=False)
            pd.DataFrame(tile_rows).to_csv(out / "tiles.csv", index=False)
        report = dict(
            request=dict(
                noise_min_uJy=band[0], noise_max_uJy=band[1], n=n, map_px=map_px, seed=seed, flatten=flatten
            ),
            n_kept=len(kept),
            n_generated=n_tried,
            rejected=rejected,
            noise_kept_uJy=dict(
                zip(["min", "p50", "max"], np.percentile(summary.noise_uJy, [0, 50, 100]).round(2).tolist())
            )
            if len(kept)
            else None,
            denoiser=str(self.checkpoint("denoiser")),
            calibration=str(self.calibration_file),
            code_version=code_version(),
            valid_range_uJy=list(self.valid_range),
            seconds=round(time.time() - t0),
        )
        (out / "report.json").write_text(json.dumps(report, indent=1))
        if len(kept) < n:
            self.logger.warning(
                f"Only {len(kept)}/{n} maps in the band after {n_tried} attempts, see report.json"
            )
        return report
