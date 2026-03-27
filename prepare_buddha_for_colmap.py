#!/usr/bin/env python3
"""
Prepare AliceVision dataset_buddha buddha/ frames for vggt/demo_colmap.py.

demo_colmap expects:
  <scene_dir>/
    images/
      00.png, 01.png, ...   # RGB PNGs (same layout as vggt/examples/kitchen/images)

Source buddha folder contains color renders as 00001._c.png, 00002._c.png, ...
alongside other files; only *_c.png frames are collected and renamed.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _natural_c_pngs(src: Path) -> list[Path]:
    """Sorted list of *._c.png paths by leading frame number."""
    pat = re.compile(r"^(\d+)\._c\.png$", re.IGNORECASE)
    pairs: list[tuple[int, Path]] = []
    for p in src.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy buddha *._c.png frames into scene_dir/images/ as 00.png, 01.png, ..."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/app/dataset_buddha/buddha"),
        help="Directory containing 00001._c.png, ... (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Scene root for demo_colmap.py (creates output_dir/images/)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of copy (saves disk; less portable)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned actions without writing files",
    )
    args = parser.parse_args()

    src_dir = args.input_dir.resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {src_dir}")

    frames = _natural_c_pngs(src_dir)
    if not frames:
        raise SystemExit(
            f"No *._c.png files found under {src_dir} "
            "(expected names like 00001._c.png)"
        )

    n = len(frames)
    width = max(2, len(str(n - 1)))
    out_images = args.output_dir.resolve() / "images"

    print(f"Found {n} color frames under {src_dir}")
    print(f"{'Would create' if args.dry_run else 'Creating'}: {out_images}/")
    print(f"Naming: {width}-digit zero-padded .png (e.g. {0:0{width}d}.png)")

    if not args.dry_run:
        out_images.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(frames):
        name = f"{i:0{width}d}.png"
        dst = out_images / name
        if args.dry_run:
            print(f"  {src.name} -> images/{name}")
            continue
        if args.symlink:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

    if not args.dry_run:
        print(f"Done. Run demo_colmap.py with:\n  --scene_dir {args.output_dir.resolve()}")
    else:
        print("Dry run finished; omit --dry_run to write files.")


if __name__ == "__main__":
    main()
