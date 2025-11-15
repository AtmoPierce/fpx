fpx — Floating-Point eXtensions

fpx is a core-only, allocation-free Rust library for building accurate,
deterministic mathematical functions from first principles.
It focuses on minimax polynomial approximation, IEEE-754 correctness, and
precise boundary-value modelling—with no dependency on the Rust standard
library.

The crate provides:

Core-compatible (no_std) math kernels, suitable for embedded, kernels,
DSP, and flight-software environments.

A math-first approach to constructing function approximations:
explicit domain selection, boundary-value handling, and interval reduction.

Optimal minimax polynomial coefficients generated using Chebyshev and
Remez techniques.

Predictable accuracy, deterministic behavior, and static storage only
(const coefficients, no allocation).

IEEE-754 utilities for bit-level inspection, ULP analysis, and
reproducibility across architectures.

fpx is intended for systems where:

the standard math library is unavailable,

the default algorithms are too slow or too unpredictable,

or a tightly controlled numerical profile is required.

By grounding every approximation in well-defined polynomial construction and
explicit interval bounds, fpx offers a transparent, verifiable foundation for
sine, cosine, exponential, logarithmic, and other common functions—built
entirely on core Rust.