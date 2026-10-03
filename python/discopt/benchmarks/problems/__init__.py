"""Standalone benchmark problem definitions for discopt."""

from discopt.benchmarks.problems.gas_network_minlp import (
    build_gas_network_minlp,
    gas_network_reference_solution,
)
from discopt.benchmarks.problems.pounce_flash import (
    FULL_TEMPERATURES,
    SMOKE_TEMPERATURES,
    build_pounce_flash,
    solve_flash_temperature,
)

__all__ = [
    "FULL_TEMPERATURES",
    "SMOKE_TEMPERATURES",
    "build_gas_network_minlp",
    "build_pounce_flash",
    "gas_network_reference_solution",
    "solve_flash_temperature",
]
