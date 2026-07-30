"""Generate a continuous-wave echo dataset."""

import argparse
import json
import time
from pathlib import Path

from asteroid_radar.data import save_echo
from asteroid_radar.simulation import simulate


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/simulation/cw_ellipsoid.json")
parser.add_argument("--output", default="outputs/simulation/cw_ellipsoid")
args = parser.parse_args()

config = json.loads(Path(args.config).read_text(encoding="utf-8"))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()
echo = simulate(config)
runtime = time.perf_counter() - start
save_echo(output / "echo.npz", echo)

summary = {
    "runtime_s": runtime,
    "sample_count": len(echo.iq),
    **echo.metadata,
}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
