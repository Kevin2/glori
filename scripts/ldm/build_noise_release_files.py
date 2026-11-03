"""
Build the LoTSS-DR3 derived data files of the noise-band tools, released in the
Hugging Face repository under noise/ (see glori.hub):

  dr3_snr_pool.npz  (SNR, total/peak, Maj) of DR3 sources from typical-depth
                    regions (Isl_rms 40-140 uJy/beam), drawn for the synthetic
                    faint sources
  dr3_fields.csv    field centres drawn uniformly on the sky (Dec > 0) and kept
                    where DR3 covers a single image (1024 px context), with the
                    native depth and whether a 2048 px map with its margin is
                    covered as well
  dr3_depth_tiles.csv tiles of the single-image size on a regular sky grid, with
                    their own depth (median Isl_rms of the sources in the tile),
                    for the images generated from unchanged real catalogs

  python scripts/ldm/build_noise_release_files.py --out <release dir>/noise [--tiles-only]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glori.inference.noise_map_sampler import PAD
from glori.inference.noise_sampler import CELL_DEG, CTX_PX, CTX_SIZE, IMAGE_SIZE, field_context, load_dr3_catalog

N_POOL = 200_000
N_FIELDS = 12_000  # uniform draws; outside the footprint they are dropped
BLOCK = 64  # coverage: no empty 64 x 64 cell (6.4') block
MAP_CELLS = 2048 // CTX_PX + 2 * PAD
TILE_DEG = IMAGE_SIZE // CTX_PX * CELL_DEG
TILE_ISL = (30.0, 95.0)  # uJy/beam, what the real-depth mode can ask for
TILE_MIN_SOURCES = 30
TILE_MIN_DEC = 20.0  # the beam is coarser further south
TILE_PER_UJY = 1000  # at most this many tiles per 1 uJy/beam of depth


def empty_block(ctx):
    src = ctx[0] > 0
    nb = src.shape[0] // BLOCK
    return bool((src[: nb * BLOCK, : nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).sum(axis=(1, 3)) == 0).any())


def depth_tiles(cat, rng):
    """Tiles of the single-image size with their own depth; the context around them has DR3 coverage."""
    iy = np.floor(cat["DEC"] / TILE_DEG).astype(int)
    ix = np.floor(cat["RA"] * np.cos(np.radians((iy + 0.5) * TILE_DEG)) / TILE_DEG).astype(int)
    t = pd.DataFrame(dict(iy=iy, ix=ix, isl=cat["Isl_rms"] * 1e3)).groupby(["iy", "ix"]).isl.agg(["median", "size"]).reset_index()
    t["dec"] = (t.iy + 0.5) * TILE_DEG
    t["ra"] = (t.ix + 0.5) * TILE_DEG / np.cos(np.radians(t.dec))
    t = t[(t["size"] >= TILE_MIN_SOURCES) & (t.dec >= TILE_MIN_DEC) & t["median"].between(*TILE_ISL)]
    t = t.sample(frac=1, random_state=np.random.RandomState(rng.integers(2**31)))
    t = t.groupby(t["median"].astype(int)).head(TILE_PER_UJY)
    keep = [not empty_block(field_context(cat, r.ra, r.dec, CTX_SIZE)) for r in t.itertuples()]
    t = t[keep]
    return pd.DataFrame(dict(ra=t.ra.round(5), dec=t.dec.round(5), isl_uJy=t["median"].round(2))).sort_values("isl_uJy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiles-only", action="store_true", help="only build dr3_depth_tiles.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cat = load_dr3_catalog()
    order = np.argsort(cat["DEC"])  # field_context expects this order
    cat = {k: v[order] for k, v in cat.items()}

    tiles = depth_tiles(cat, np.random.default_rng(args.seed))
    tiles.to_csv(out / "dr3_depth_tiles.csv", index=False)
    print(f"dr3_depth_tiles.csv: {len(tiles)} tiles, depth {tiles.isl_uJy.min()}-{tiles.isl_uJy.max()} uJy/beam")
    if args.tiles_only:
        return

    # source properties for the synthetic sources
    ok = (cat["Isl_rms"] >= 0.040) & (cat["Isl_rms"] < 0.140) & (cat["Peak_flux"] > 0)
    i = rng.choice(np.flatnonzero(ok), size=N_POOL, replace=False)
    pool = np.c_[cat["Peak_flux"][i] / cat["Isl_rms"][i], cat["Total_flux"][i] / cat["Peak_flux"][i], cat["Maj"][i]]
    np.savez(out / "dr3_snr_pool.npz", pool=pool.astype(np.float32))
    print(f"dr3_snr_pool.npz: {len(pool)} sources")

    # candidate fields: the map context (2048 px + margin) contains the single-image context
    rows = []
    lo = (MAP_CELLS - CTX_SIZE) // 2
    ras = rng.uniform(0, 360, N_FIELDS)
    decs = np.degrees(np.arcsin(rng.uniform(0, 1, N_FIELDS)))
    for k, (ra, dec) in enumerate(zip(ras, decs)):
        ctx = field_context(cat, ra, dec, MAP_CELLS)
        src = ctx[0] > 0
        if not src.any():
            continue
        rows.append(dict(
            ra=round(float(ra), 5), dec=round(float(dec), 5),
            native_uJy=round(float(np.median(ctx[4][src] / ((ctx[0][src] + 1) / 2))), 2),
            single_ok=not empty_block(ctx[:, lo:lo + CTX_SIZE, lo:lo + CTX_SIZE]),
            map_ok=not empty_block(ctx),
        ))
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{N_FIELDS} fields", flush=True)
    f = pd.DataFrame(rows)
    f = f[f.single_ok].reset_index(drop=True)
    f.to_csv(out / "dr3_fields.csv", index=False)
    print(f"dr3_fields.csv: {len(f)} of {N_FIELDS} uniform draws with single-image coverage, {int(f.map_ok.sum())} with map coverage, "
          f"{int((f.map_ok & (f.native_uJy <= 90)).sum())} usable for maps (native <= 90 uJy/beam)")


if __name__ == "__main__":
    main()
