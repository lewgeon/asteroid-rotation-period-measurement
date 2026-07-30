"""Query ephemerides and solve transmit/receive pointing."""

import argparse
import json
from pathlib import Path

from astropy.time import Time

from asteroid_radar.pointing.ephemeris import (
    HorizonsStateProvider,
    TerrestrialStation,
)
from asteroid_radar.pointing.solver import LightTimePointingSolver


def station(data, earth_ephemeris):
    return TerrestrialStation.from_geodetic(
        data["name"],
        data["longitude_deg"],
        data["latitude_deg"],
        data["height_m"],
        earth_ephemeris,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pointing/eros.json")
parser.add_argument("--output", default="outputs/pointing/eros.json")
args = parser.parse_args()

config = json.loads(Path(args.config).read_text(encoding="utf-8"))
tx = station(config["transmit_station"], config["earth_ephemeris"])
rx = station(
    config.get("receive_station", config["transmit_station"]),
    config["earth_ephemeris"],
)
provider = HorizonsStateProvider(
    config["target_id_type"],
    cache_directory=config.get("horizons_cache"),
)
solver = LightTimePointingSolver(
    provider,
    tx,
    rx,
    config["tolerance_s"],
    config["max_iterations"],
)
solution = solver.solve(
    config["target"],
    Time(config["observation_time_utc"], scale="utc"),
    config["time_role"],
)

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(solution.to_dict(), ensure_ascii=False, indent=2)
output.write_text(text, encoding="utf-8")
print(text)
