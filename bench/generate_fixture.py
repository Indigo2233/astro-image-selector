"""Generate deterministic 2048x2048 FITS frames for Rust performance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def create_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = width = 2048
    image = rng.normal(1000.0, 8.0, (height, width)).astype(np.float32)
    rows, cols = np.indices(image.shape)
    star_positions = (
        (128, 192),
        (384, 512),
        (768, 320),
        (1024, 896),
        (1450, 430),
        (1760, 1520),
        (640, 1660),
        (1200, 1240),
        (900, 1800),
        (1850, 720),
    )
    for row, col in star_positions:
        image += 4200.0 * np.exp(
            -((rows - row) ** 2 / 12.0 + (cols - col) ** 2 / 9.0)
        ).astype(np.float32)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9153)
    parser.add_argument("--filter", default="H-ALPHA")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(args.count):
        data = create_frame(args.seed + index)
        header = fits.Header(
            {
                "FILTER": args.filter,
                "EXPTIME": 60.0,
                "OBJECT": "PerformanceFixture",
                "IMAGETYP": "LIGHT",
            }
        )
        path = args.output / f"bench_{index:03d}.fits"
        fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
        paths.append(str(path))
    print(json.dumps({"files": paths, "shape": [2048, 2048]}, indent=2))


if __name__ == "__main__":
    main()
