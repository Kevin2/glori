"""
CPU smoke tests of the noise-band tools (no GPU, network or group storage):
calibration lookups, request checks and the catalog/context construction.

  pytest tests/test_noise_tools.py
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

main_folder = os.path.join(os.path.dirname(__file__), "..", "src")
if main_folder not in sys.path:
    sys.path.insert(0, main_folder)

from glori.inference import noise_map_sampler as nm  # noqa: E402
from glori.inference import noise_sampler as ns  # noqa: E402


@pytest.fixture(scope="module")
def samplers():
    return ns.NoiseBandSampler(device="cpu"), nm.NoiseBandMapSampler(device="cpu")


def test_valid_ranges(samplers):
    single, maps = samplers
    assert single.valid_range == pytest.approx((38.9, 222.1), abs=0.1)
    assert maps.valid_range == (50.0, 224.0)


@pytest.mark.parametrize("band", [(30, 50), (100, 300), (60, 50), (50, 50)])
def test_refused_bands(samplers, band):
    with pytest.raises(ValueError):
        samplers[1].check_band(*band)


@pytest.mark.parametrize("px,ok", [(512, False), (1000, False), (1024, True), (2048, True), (4096, True)])
def test_map_sizes(px, ok):
    if ok:
        nm.NoiseBandMapSampler.check_size(px)
    else:
        with pytest.raises(ValueError):
            nm.NoiseBandMapSampler.check_size(px)


def test_depth_for_noise_monotone(samplers):
    for s in samplers:
        c = s.calib["calibration"]["noise_p50"]
        t = s.depth_for_noise(np.geomspace(max(c[0], s.valid_range[0]), min(c[-1], s.valid_range[1]), 20))
        assert np.all(np.diff(t) > 0)
        assert s.calib["valid_range"]["target"][0] - 1e-6 <= t.min()
        assert t.max() <= s.calib["valid_range"]["target"][1] + 1e-6


def toy_catalog(ra=150.0, dec=50.0, n=400, seed=0):
    """Random sources around (ra, dec) in the column convention of the DR3 catalog (mJy, arcsec)."""
    rng = np.random.default_rng(seed)
    d = dict(
        RA=ra + rng.uniform(-0.2, 0.2, n) / np.cos(np.radians(dec)),
        DEC=dec + rng.uniform(-0.2, 0.2, n),
        Total_flux=rng.uniform(0.2, 20, n),
        Maj=rng.uniform(6, 20, n),
        Isl_rms=np.full(n, 0.06),
    )
    d["Peak_flux"] = d["Total_flux"] * 0.8
    order = np.argsort(d["DEC"])
    return {k: v[order] for k, v in d.items()}


def test_field_context_centre():
    cat = {k: np.array([v]) for k, v in dict(RA=150.0, DEC=50.0, Total_flux=5.0, Peak_flux=4.0, Maj=8.0,
                                             Isl_rms=0.06).items()}
    c = ns.field_context(cat, 150.0, 50.0, ns.CTX_SIZE)
    assert c.shape == (5, ns.CTX_SIZE, ns.CTX_SIZE)
    rows, cols = np.nonzero(c[0] > 0)
    assert len(rows) == 1 and abs(rows[0] - ns.CTX_SIZE / 2) <= 1 and abs(cols[0] - ns.CTX_SIZE / 2) <= 1
    assert c[1:, rows[0], cols[0]] == pytest.approx([5.0, 4.0, 8.0, 60.0])
    assert (c[0] == -1).sum() == ns.CTX_SIZE**2 - 1


def test_depth_matched_context(samplers):
    method = samplers[0].calib["method"]
    c = ns.field_context(toy_catalog(), 150.0, 50.0, ns.CTX_SIZE)
    pool = np.c_[np.geomspace(1, 1000, 5000), np.ones(5000), np.full(5000, 8.0)].astype(np.float32)
    depth = 50.0
    d, syn = ns.depth_matched_context(c, depth, pool, np.random.default_rng(1), method)
    src = d[0] > 0
    real = src & ~syn
    native = 60.0  # Isl_rms of the toy catalog
    assert np.all(d[2][real] * 1e3 >= method["keep_native"] * native)
    assert np.all(d[2][syn] * 1e3 < method["keep_native"] * native)
    assert syn.sum() > 0 and real.sum() > 0
    # no noise excess in the toy catalog: only the depth, summed over the sources of a cell
    assert d[4][src] == pytest.approx(depth / method["img_to_isl"] * (d[0][src] + 1) / 2)
    truth = ns.truth_catalog(d, syn)
    assert truth.x_px.between(0, ns.IMAGE_SIZE).all() and truth.y_px.between(0, ns.IMAGE_SIZE).all()


@pytest.mark.parametrize("depth", [50.0, 150.0])
def test_rms_channel_keeps_noise_excess(samplers, depth):
    method = samplers[0].calib["method"]
    full = -np.ones((5, 64, 64), np.float32)
    full[1:] = 0
    for (y, x), (count, isl) in {(10, 10): (1, 600.0), (30, 30): (2, 60.0), (50, 50): (1, 60.0),
                                 (50, 40): (1, 60.0), (40, 50): (1, 60.0)}.items():
        full[0, y, x], full[4, y, x] = 2 * count - 1, count * isl

    # kept: the source with the noise excess and the cell of two; synthetic: one near it, one far from all
    c = -np.ones_like(full)
    c[1:] = 0
    for y, x in [(10, 10), (30, 30)]:
        c[0, y, x] = full[0, y, x]
    syn = np.zeros((64, 64), bool)
    syn[10, 15] = syn[10, 60] = True
    c[0][syn] = 1

    ch = ns.rms_channel(full, c, syn, depth, method)
    base = depth / method["img_to_isl"]
    bright = np.sqrt(base**2 + 600.0**2 - 60.0**2)
    assert ch[10, 10] == pytest.approx(bright)
    assert ch[30, 30] == pytest.approx(2 * base)
    assert ch[10, 15] == pytest.approx(bright)
    assert ch[10, 60] == pytest.approx(base)
    assert (ch[c[0] <= 0] == 0).all()


def test_pick_field_modes(samplers):
    s = ns.NoiseBandSampler(device="cpu")
    rd = s.calib["real_depth"]
    s._tiles = pd.DataFrame(dict(ra=np.arange(200.0), dec=np.full(200, 50.0), isl_uJy=np.linspace(30, 95, 200)))
    s._base = pd.DataFrame(dict(ra=[1.0], dec=[60.0], native_uJy=[70.0]))
    rng = np.random.default_rng(0)
    for noise in (rd["min_noise_uJy"], 60.0, rd["max_noise_uJy"] - 1):
        ra, _, mode, depth = s._pick_field(noise, rng)
        isl = np.exp(np.interp(np.log(noise), np.log(rd["noise_p50"]), np.log(rd["isl_uJy"])))
        assert mode == "real_depth"
        assert abs(s._tiles.isl_uJy[int(ra)] / isl - 1) <= ns.TILE_WINDOW - 1 + 1e-6
        assert depth == pytest.approx(s._tiles.isl_uJy[int(ra)] * s.calib["method"]["img_to_isl"])
    ra, dec, mode, depth = s._pick_field(rd["max_noise_uJy"], rng)
    assert (ra, dec, mode) == (1.0, 60.0, "depth_matched")
    assert depth == pytest.approx(s.depth_for_noise(rd["max_noise_uJy"]))


@pytest.mark.parametrize("mode", ["real_depth", "depth_matched"])
def test_flux_factor_image(samplers, mode):
    s = samplers[0]
    rd = s.calib["real_depth"]
    m = dict(noise_uJy=50.0, noise_generated_uJy=55.0, rescale=50.0 / 55.0)
    inf = dict(mode=mode, depth_uJy=100.0)
    n = rd["min_flux_sources"]
    truth = pd.DataFrame(dict(x_px=10.0 + 30 * np.arange(n), y_px=np.full(n, 100.0), peak_flux_input=np.full(n, 1e-2)))
    g = s.brightness_factors(truth, m["noise_uJy"])[0]
    srl = pd.DataFrame(dict(x_px=truth.x_px + 1, y_px=truth.y_px, Peak_flux=0.8 * 1e-2 * g))
    assert s._flux_factor_image(truth, srl, m, g, inf) == (pytest.approx(0.8), "image")  # the image's own sources
    fallback = (m["rescale"] * np.interp(55.0, rd["noise_p50"], rd["flux"]) if mode == "real_depth"
                else s.flux_factor(100.0, 50.0))
    for few in (srl.iloc[:-1], srl.iloc[:0]):  # too few of them
        assert s._flux_factor_image(truth, few, m, g, inf) == (pytest.approx(fallback), "calibration")


def test_brightness_factors(samplers):
    for s in samplers:
        b = s.calib["brightness"]
        snr = np.geomspace(1, 1e6, 200)
        truth = pd.DataFrame(dict(peak_flux_input=snr * 50e-6))
        peak, total = s.brightness_factors(truth, 50.0)
        cal = snr >= b["snr"][1]  # from the first measured bin on: brighter sources are compressed at least as much
        assert np.all(np.diff(peak[cal]) <= 1e-9) and np.all(np.diff(total[cal]) <= 1e-9)
        assert peak[0] == total[0] == pytest.approx(1.0)  # no change for faint sources
        assert peak[-1] == pytest.approx(b["peak"][-1])  # stays at the last value, no extrapolation
        assert peak[-1] < 0.6 and np.all(total >= peak - 1e-9)


def test_field_ok(samplers):
    fields = samplers[1].calib["fields"]
    c = ns.field_context(toy_catalog(n=20000), 150.0, 50.0, 256)
    assert nm.field_ok(c, fields) == pytest.approx(60.0)
    c[0, :64, :64] = -1  # an empty 6.4' block: survey gap
    assert nm.field_ok(c, fields) is None


def test_map_truth_catalog():
    c = -np.ones((5, 8, 8), np.float32)
    c[1:] = 0
    c[:, 2, 5] = [1, 3.0, 2.0, 9.0, 60.0]
    t = nm.map_truth_catalog(c, np.zeros((8, 8), bool))
    assert (t.x_px.item(), t.y_px.item()) == (5 * ns.CTX_PX + 1.5, 2 * ns.CTX_PX + 1.5)
    assert (t.peak_flux_input.item(), t.total_flux_input.item()) == pytest.approx((2e-3, 3e-3))


def test_flatten_factor_removes_trend_keeps_bright_peak():
    n = 1024
    yy, xx = np.indices((n, n))
    trend = 1 + 0.2 * np.sin(2 * np.pi * xx / n)  # slow drift of +-20 %
    # a local peak of twice the noise around a bright source at (800, 300)
    peak = 1 + np.exp(-((xx - 800) ** 2 + (yy - 300) ** 2) / (2 * 30**2))
    x, y, snr = np.array([800.0]), np.array([300.0]), np.array([500.0])

    f = nm.flatten_factor(trend * peak, x, y, snr, block=256)
    flat = trend * peak * f
    away = np.hypot(xx - 800, yy - 300) > 100
    # at least halved: the factor is held constant beyond the outermost block centres
    assert np.ptp(np.percentile(flat[away], [5, 95])) < 0.5 * np.ptp(np.percentile(trend[away], [5, 95]))
    # the factor doesn't follow the peak: it stays smooth and the peak survives
    assert flat[300, 800] / np.median(flat[away]) > 1.7
    assert np.abs(np.diff(f, axis=1)).max() < 0.01


def test_flatten_factor_limited():
    n = 1024
    rms = np.ones((n, n))
    rms[:, : n // 2] = 4.0
    no_sources = np.array([], float)
    f = nm.flatten_factor(rms, no_sources, no_sources, no_sources, block=256, max_factor=1.3)
    assert f.min() >= 1 / 1.3 - 1e-6 and f.max() <= 1.3 + 1e-6


@pytest.mark.parametrize("tool", ["single", "map"])
def test_script_without_main_guard_stops_before_sampling(tool, tmp_path):
    cls = {"single": "noise_sampler.NoiseBandSampler", "map": "noise_map_sampler.NoiseBandMapSampler"}[tool]
    out = tmp_path / "out"
    script = tmp_path / "unguarded.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import glori.inference.{cls.split(".")[0]} as m
            m.{cls.split(".")[1]}(device="cpu", bdsf_workers=2).sample(60, 80, n=1, out_dir={str(out)!r})
            """
        )
    )
    env = {**os.environ, "PYTHONPATH": os.path.abspath(main_folder)}
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode != 0
    assert "__main__" in r.stderr
    assert not out.exists()
