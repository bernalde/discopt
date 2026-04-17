"""Standalone benchmark problem definitions for discopt."""

from discopt.benchmarks.problems.gas_network_minlp import (
    build_gas_network_minlp,
    gas_network_reference_solution,
)

__all__ = [
    "build_gas_network_minlp",
    "gas_network_reference_solution",
]
