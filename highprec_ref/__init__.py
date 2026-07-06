"""Arbitrary-precision ground truth (mpmath, 50+ digits, tiny sizes).

This is the *independent extended-precision reference* of docs/DESIGN.md §6: the
source of truth for the accuracy-vs-throughput curves. It shares no code with
The Walrus (so the precision characterization cannot merely echo the incumbent)
and no code with the FP64 ``cpu_ref`` path (different arithmetic entirely), yet
it consumes the exact same FP64 input matrix bit-for-bit, so relative-error
numbers are meaningful. Slow by design; intended for small ``n``.
"""

from .hafnian import hafnian_mp
from .loop_hafnian import loop_hafnian_mp
from .permanent import permanent_mp
from .torontonian import torontonian_mp

__all__ = ["permanent_mp", "hafnian_mp", "loop_hafnian_mp", "torontonian_mp"]
