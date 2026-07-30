"""Command-line entry point for light-time-corrected radar pointing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astropy.time import Time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_pointing.config import load_pointing_config
from asteroid_pointing.ephemeris import HorizonsStateProvider, TerrestrialStation
from asteroid_pointing.solver import LightTimePointingSolver


def _station(data: dict, ephemeris: str) -> TerrestrialStation:
    return TerrestrialStation.from_geodetic(
        name=data["name"],
        longitude_deg=data["longitude_deg"],
        latitude_deg=data["latitude_deg"],
        height_m=data["height_m"],
        solar_system_ephemeris=ephemeris,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve asteroid radar transmit/receive line of sight"
    )
    parser.add_argument("--config", required=True, help="Pointing JSON config")
    parser.add_argument("--target", help="Override target in the JSON config")
    parser.add_argument("--time", help="Override observation UTC time")
    parser.add_argument(
        "--time-role",
        choices=("transmit", "receive"),
        help="Interpret input time as transmit or receive epoch",
    )
    parser.add_argument("--output", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pointing_config(args.config)
    target = args.target or config["target"]["id"]
    observation_time = Time(
        args.time or config["observation"]["time_utc"],
        scale="utc",
    )
    time_role = args.time_role or config["observation"]["time_role"]
    ephemeris = config["ephemeris"]
    transmit_station = _station(config["transmit_station"], ephemeris["earth"])
    receive_data = config.get("receive_station", config["transmit_station"])
    receive_station = _station(receive_data, ephemeris["earth"])
    provider = HorizonsStateProvider(
        id_type=config["target"].get("id_type"),
        cache=ephemeris["cache"],
        cache_directory=(
            ROOT / ephemeris["cache_directory"]
            if ephemeris.get("cache_directory")
            else None
        ),
    )
    solver = LightTimePointingSolver(
        provider,
        transmit_station,
        receive_station,
        tolerance_s=config["solver"]["tolerance_s"],
        max_iterations=config["solver"]["max_iterations"],
    )
    solution = solver.solve(target, observation_time, time_role)
    output = {
        "input": {
            "config": str(Path(args.config).resolve()),
            "observation_time_utc": observation_time.utc.isot,
            "time_role": time_role,
            "target": target,
            "transmit_station": transmit_station.name,
            "receive_station": receive_station.name,
        },
        "solution": solution.to_dict(),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)
    output_path = args.output or config.get("output_json")
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
