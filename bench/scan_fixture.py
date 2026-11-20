"""Generate many tiny FITS files to isolate directory scanning overhead."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    template = args.output / "template.fits"
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.uint16)).writeto(template)
    data = template.read_bytes()
    template.unlink()
    for index in range(args.count):
        (args.output / f"scan_{index:04d}.fits").write_bytes(data)


if __name__ == "__main__":
    main()
