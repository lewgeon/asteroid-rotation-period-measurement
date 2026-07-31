"""Generate a body-centred ellipsoid OBJ/PLY in metres."""

import argparse

from asteroid_radar.ellipsoid import save_ellipsoid


parser = argparse.ArgumentParser()
parser.add_argument("--axes", nargs=3, type=float, default=[70.0, 50.0, 40.0])
parser.add_argument("--subdivisions", type=int, default=3)
parser.add_argument("--output", default="models/ellipsoid.obj")
args = parser.parse_args()

save_ellipsoid(args.output, args.axes, args.subdivisions)
print(
    f"saved {args.output}: axes={args.axes} m, "
    f"subdivisions={args.subdivisions}"
)
