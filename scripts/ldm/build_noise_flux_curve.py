"""
Add the brightness curves of the truth catalogs to the calibration files of the
noise-band tools, from end-to-end runs of the tools:

  python scripts/ldm/build_noise_flux_curve.py --runs IMAGE_RUN [IMAGE_RUN ...] \
      --calibration configs/noise_calibration/rms-v1.json
  python scripts/ldm/build_noise_flux_curve.py --runs MAP_RUN [MAP_RUN ...] \
      --calibration configs/noise_calibration/rms-v1-swiit.json

The denoiser compresses bright sources by their SNR, alike for real-depth and
depth-matched images and more in maps above SNR ~5000, so each file gets its own
curve. Random fields hold too few bright sources for the top of the curve (bins of
15-30 sources overcorrected by ~30 % in a held-out check): add runs on fields that
hold a bright source, selected on the input catalog only.
Per image, sources with input peak >=
FLUX_SNR x noise are matched to the PyBDSF catalog and their measured / input peak
and total flux are divided by the image's flux factor; the curves are the medians
in bins of input SNR, made monotone. The image's own factor depends on the curve,
so the two are iterated like the tools compute them.

Run this after scripts/ldm/build_noise_calibration.py, which rewrites the file.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MATCH_PX = 8.0
FLUX_SNR = 20.0
MIN_SOURCES = dict(image_id=5, map_id=10)
BIN_DEX, MIN_PER_BIN = 0.25, 15
ANCHOR_SNR = 10.0  # below the calibrated SNR, the curves return to 1 here
N_ITER = 3


def load_run(d):
    """Matched sources of one run: image key, SNR, measured / input fluxes, fallback factor."""
    d = Path(d)
    s = pd.read_csv(d / "summary.csv")
    t = pd.read_csv(d / "truth_catalog.csv")
    c = pd.read_csv(d / "pybdsf_catalog.csv")
    key = "map_id" if "map_id" in s else "image_id"
    rows = []
    for r in s.itertuples():
        ti, ci = t[t[key] == getattr(r, key)], c[c[key] == getattr(r, key)]
        if not len(ti) or not len(ci):
            continue
        dist, j = cKDTree(ci[["x_px", "y_px"]].values).query(ti[["x_px", "y_px"]].values)
        flat = ti.flatten_factor.values if "flatten_factor" in ti else 1.0
        snr = ti.peak_flux_input.values * 1e6 / r.noise_uJy
        sel = (dist < MATCH_PX) & (snr >= FLUX_SNR)
        rows.append(pd.DataFrame(dict(
            image=f"{d}_{getattr(r, key)}", min_sources=MIN_SOURCES[key], fallback=r.flux_factor, snr=snr[sel],
            peak=(ci.Peak_flux.values[j] / (ti.peak_flux_input.values * flat))[sel],
            total=(ci.Total_flux.values[j] / (ti.total_flux_input.values * flat))[sel],
        )))
    return pd.concat(rows, ignore_index=True)


def monotone(y, w):
    """Weighted non-increasing fit (pool adjacent violators)."""
    blocks = []
    for v, n in zip(y, w):
        blocks.append([v, n, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            b = blocks.pop()
            a = blocks[-1]
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            a[1] += b[1]
            a[2] += b[2]
    return np.concatenate([np.full(b[2], b[0]) for b in blocks])


def binned(x, y):
    """Monotone median of log y in bins of log x. Sparse bins merge with the next one."""
    lx, ly = np.log10(x), np.log(y)
    edges = np.arange(np.log10(FLUX_SNR), lx.max() + BIN_DEX, BIN_DEX)
    cx, cy, n = [], [], []
    lo = edges[0]
    for hi in edges[1:]:
        sel = (lx >= lo) & (lx < hi)
        if sel.sum() >= MIN_PER_BIN:
            cx.append(lx[sel].mean())
            cy.append(np.median(ly[sel]))
            n.append(int(sel.sum()))
            lo = hi
    return np.array(cx), monotone(np.array(cy), np.array(n)), n


def fit(p):
    """Curves and image factors, iterated. p: output of load_run, all runs."""
    x = np.log10(p.snr.values)
    g = np.ones(len(p))
    for _ in range(N_ITER):
        per = p.assign(r=p.peak / g).groupby("image").agg(n=("r", "size"), med=("r", "median"), m=("min_sources", "first"), f=("fallback", "first"))
        per["ff"] = np.where(per.n >= per.m, per.med, per.f)
        ff = p.image.map(per.ff).values
        cx, cpeak, n = binned(p.snr.values, p.peak.values / ff)
        g = np.exp(np.interp(x, cx, cpeak))
    _, ctotal, _ = binned(p.snr.values, p.total.values / ff)
    return dict(snr=[ANCHOR_SNR] + (10**cx).round(1).tolist(),
                peak=[1.0] + np.exp(cpeak).round(4).tolist(),
                total=[1.0] + np.exp(ctotal).round(4).tolist(),
                n=[0] + n), ff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--calibration", nargs="+", required=True)
    args = ap.parse_args()
    p = pd.concat([load_run(d) for d in args.runs], ignore_index=True)
    p = p[(p.peak > 0) & (p.total > 0)].reset_index(drop=True)
    curve, _ = fit(p)
    curve["runs"] = [Path(d).name for d in args.runs]
    curve["n_sources"], curve["n_images"] = len(p), int(p.image.nunique())
    print(pd.DataFrame({k: curve[k] for k in ["snr", "peak", "total", "n"]}).to_string(index=False))
    for f in args.calibration:
        c = json.loads(Path(f).read_text())
        c["brightness"] = curve
        Path(f).write_text(json.dumps(c, indent=1))
        print("written", f)


if __name__ == "__main__":
    main()
