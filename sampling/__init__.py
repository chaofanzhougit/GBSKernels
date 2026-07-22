"""Thin sampling orchestration on top of the kernels.

This package ships both standard (Aaronson-Arkhipov) boson sampling through the
permanent and Gaussian boson-sampling probability/sampler paths through the
hafnian, loop hafnian, and torontonian. They exercise the batched API on the
workload photonic sampling produces -- many independent medium-sized
matrix-function evaluations (docs/DESIGN.md §2.3) -- and host the Layer-4
statistical validation (docs/DESIGN.md §8).
"""

from . import gbs
from .boson_sampling import (
    output_configurations,
    probabilities,
    total_probability,
)

__all__ = ["output_configurations", "probabilities", "total_probability", "gbs"]
