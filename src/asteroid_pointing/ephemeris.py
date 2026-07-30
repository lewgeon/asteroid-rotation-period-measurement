"""Ephemeris providers for targets and terrestrial radar stations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
from astropy.time import Time
from astropy.utils import iers

from .models import CartesianState


iers.conf.auto_download = False
iers.conf.auto_max_age = None


class TargetStateProvider(Protocol):
    """Interface supplying geometric barycentric target states."""

    def state(self, target: str, epoch: Time) -> CartesianState:
        ...


@dataclass(frozen=True)
class TerrestrialStation:
    """A fixed ITRS station whose state is evaluated at arbitrary epochs."""

    name: str
    location: EarthLocation
    solar_system_ephemeris: str = "builtin"

    @classmethod
    def from_geodetic(
        cls,
        name: str,
        longitude_deg: float,
        latitude_deg: float,
        height_m: float,
        solar_system_ephemeris: str = "builtin",
    ) -> "TerrestrialStation":
        location = EarthLocation.from_geodetic(
            lon=longitude_deg * u.deg,
            lat=latitude_deg * u.deg,
            height=height_m * u.m,
        )
        return cls(name, location, solar_system_ephemeris)

    def state(self, epoch: Time) -> CartesianState:
        """Return station state in barycentric axes.

        Astropy's geocentric GCRS station vector is added to Earth's
        barycentric state. GCRS and ICRS share the same practical axis
        orientation for this pointing calculation.
        """

        earth_position, earth_velocity = get_body_barycentric_posvel(
            "earth", epoch, ephemeris=self.solar_system_ephemeris
        )
        site_position, site_velocity = self.location.get_gcrs_posvel(epoch)
        position = (
            earth_position.xyz.to_value(u.km)
            + site_position.xyz.to_value(u.km)
        )
        velocity = (
            earth_velocity.xyz.to_value(u.km / u.s)
            + site_velocity.xyz.to_value(u.km / u.s)
        )
        return CartesianState(position, velocity)


class HorizonsStateProvider:
    """Query geometric SSB target states from JPL Horizons."""

    def __init__(
        self,
        id_type: str | None = "smallbody",
        cache: bool = True,
        cache_directory: str | Path | None = None,
    ) -> None:
        self.id_type = id_type
        self.cache = cache
        self.cache_directory = (
            Path(cache_directory).resolve() if cache_directory else None
        )

    def state(self, target: str, epoch: Time) -> CartesianState:
        return self._state_cached(target, f"{epoch.tdb.jd:.12f}")

    @lru_cache(maxsize=256)
    def _state_cached(self, target: str, epoch_tdb_jd: str) -> CartesianState:
        try:
            from astroquery.jplhorizons import Horizons
        except ImportError as error:
            raise RuntimeError(
                "JPL Horizons querying requires the optional astroquery package"
            ) from error

        query = Horizons(
            id=target,
            id_type=self.id_type,
            location="@0",
            epochs=float(epoch_tdb_jd),
        )
        if self.cache_directory is not None:
            query.cache_location = self.cache_directory / "Horizons"
            query.cache_location.mkdir(parents=True, exist_ok=True)
        vectors = query.vectors(
            refplane="earth",
            aberrations="geometric",
            cache=self.cache,
        )
        if len(vectors) != 1:
            raise RuntimeError(
                f"Horizons returned {len(vectors)} rows for target {target!r}"
            )
        position = np.array(
            [vectors["x"][0], vectors["y"][0], vectors["z"][0]],
            dtype=np.float64,
        )
        velocity = np.array(
            [vectors["vx"][0], vectors["vy"][0], vectors["vz"][0]],
            dtype=np.float64,
        )
        return CartesianState(
            (position * u.au).to_value(u.km),
            (velocity * u.au / u.day).to_value(u.km / u.s),
        )
