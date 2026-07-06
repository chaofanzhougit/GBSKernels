"""Thin sampling orchestration on top of the kernels.

Phase 1 ships the *standard* (Aaronson-Arkhipov) boson-sampling layer, which uses
only the permanent and is therefore validatable now; the *Gaussian* boson
sampling chain (hafnian / torontonian) lands in Phase 3. Both exist to exercise
the batched API on the workload photonic sampling actually produces -- many
independent medium-sized matrix-function evaluations (docs/DESIGN.md §2.3) -- and to
host the Layer-4 statistical validation (docs/DESIGN.md §8).
"""

from . import gbs
from .boson_sampling import (
    output_configurations,
    probabilities,
    total_probability,
)

__all__ = ["output_configurations", "probabilities", "total_probability", "gbs"]
