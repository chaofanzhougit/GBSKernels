"""Independent CPU reference implementations (plain, readable, slow).

These deliberately share **no code** with The Walrus. They are the
differential-test reference and the algorithmic specification that the CUDA
kernels in ``core/`` must reproduce bit-for-bit (per precision tier). Clarity
and obvious correctness outrank speed here; the fast paths live on the GPU.
"""

from .certified import certified, certified_hafnian, certified_permanent
from .repeated import lhaf_repeated, sieve_term_count
from .tor_recursive import torontonian_recursive
from .hafnian import haf, hafnian_naive, hafnian_powertrace
from .loop_hafnian import lhaf, loop_hafnian_naive, loop_hafnian_powertrace
from .permanent import perm, permanent_glynn, permanent_naive, permanent_ryser
from .torontonian import tor, torontonian
from .diagnostics import (  # imports the four refs above; keep last
    cancellation_ratio,
    glynn_abs_term_sum,
    hafnian_abs_term_sum,
    loop_hafnian_abs_term_sum,
    summation_condition_number,
    torontonian_abs_term_sum,
)

__all__ = [
    "certified",
    "lhaf_repeated",
    "sieve_term_count",
    "torontonian_recursive",
    "certified_hafnian",
    "certified_permanent",
    "perm",
    "permanent_naive",
    "permanent_ryser",
    "permanent_glynn",
    "glynn_abs_term_sum",
    "hafnian_abs_term_sum",
    "loop_hafnian_abs_term_sum",
    "torontonian_abs_term_sum",
    "cancellation_ratio",
    "summation_condition_number",
    "haf",
    "hafnian_naive",
    "hafnian_powertrace",
    "lhaf",
    "loop_hafnian_naive",
    "loop_hafnian_powertrace",
    "tor",
    "torontonian",
]
