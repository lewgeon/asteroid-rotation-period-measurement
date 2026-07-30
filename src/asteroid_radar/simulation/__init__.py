"""Echo simulation module.

The public interface is deliberately small: ``simulate(config)`` returns an
``EchoDataset`` that can be saved and passed to inversion.
"""

from .experiment import simulate
from .geometry import SurfaceMesh

__all__ = ["SurfaceMesh", "simulate"]
