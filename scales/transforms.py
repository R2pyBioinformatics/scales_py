"""
Scale transformations for continuous data.

Python port of the R *scales* package transform system, covering:
  - ``R/transform.R``
  - ``R/transform-numeric.R``
  - ``R/transform-compose.R``
  - ``R/transform-date.R``

Each transform is a lightweight object that bundles a forward transform,
its inverse, optional derivatives, break-generation logic, and a label
formatter.  Pre-built transforms are available for common cases (log,
sqrt, reverse, Box--Cox, Yeo--Johnson, etc.) and new transforms can be
created with :func:`new_transform`.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from functools import reduce
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .breaks import breaks_extended
from .minor_breaks import regular_minor_breaks

__all__ = [
    # Core API
    "Transform",
    "new_transform",
    "is_transform",
    "as_transform",
    "trans_breaks",
    "trans_format",
    # Transform constructors
    "transform_identity",
    "transform_log",
    "transform_log10",
    "transform_log2",
    "transform_log1p",
    "transform_exp",
    "transform_sqrt",
    "transform_reverse",
    "transform_reciprocal",
    "transform_asinh",
    "transform_asn",
    "transform_atanh",
    "transform_boxcox",
    "transform_modulus",
    "transform_yj",
    "transform_pseudo_log",
    "transform_logit",
    "transform_probit",
    "transform_probability",
    "transform_date",
    "transform_time",
    "transform_timespan",
    "transform_compose",
    # Legacy aliases
    "trans_new",
    "identity_trans",
    "log_trans",
    "log10_trans",
    "log2_trans",
    "log1p_trans",
    "exp_trans",
    "sqrt_trans",
    "reverse_trans",
    "reciprocal_trans",
    "asinh_trans",
    "asn_trans",
    "atanh_trans",
    "boxcox_trans",
    "modulus_trans",
    "yj_trans",
    "pseudo_log_trans",
    "logit_trans",
    "probit_trans",
    "probability_trans",
    "date_trans",
    "time_trans",
    "timespan_trans",
    "transform_hms",
    "hms_trans",
    "compose_trans",
    "is_trans",
    "as_trans",
]


# ---------------------------------------------------------------------------
# Helper: lightweight pretty breaks (fallback when breaks modules absent)
# ---------------------------------------------------------------------------

def _pretty_breaks(n: int = 5) -> Callable:
    """Return a break-generator using Wilkinson's extended algorithm.

    Mirrors R's ``new_transform`` default
    ``breaks = extended_breaks()`` — i.e. ``labeling::extended`` in R,
    :func:`breaks_extended` here.  Pure-numpy; no matplotlib dependency.
    """
    extended = breaks_extended(n=n)

    def _breaks(limits: Tuple[float, float]) -> np.ndarray:
        return extended(np.asarray(limits, dtype=float))

    return _breaks


def _log_breaks(base: float = 10, n: int = 5) -> Callable:
    """Return a break-generator suitable for log-scaled data."""
    def _breaks(limits: Tuple[float, float]) -> np.ndarray:
        lo, hi = float(limits[0]), float(limits[1])
        if lo <= 0:
            lo = 1e-10
        if hi <= 0:
            hi = 1.0
        lo_exp = np.floor(np.log(lo) / np.log(base))
        hi_exp = np.ceil(np.log(hi) / np.log(base))
        by = max(1, np.round((hi_exp - lo_exp) / n))
        exponents = np.arange(lo_exp, hi_exp + by, by)
        return base ** exponents
    return _breaks


def _default_format() -> Callable:
    """Return a simple label formatter (converts to string)."""
    def _fmt(x: np.ndarray) -> list[str]:
        out: list[str] = []
        for v in np.asarray(x).flat:
            if np.isnan(v):
                out.append("NA")
            else:
                # Remove trailing zeros for cleanliness
                s = f"{v:g}"
                out.append(s)
        return out
    return _fmt


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------

class Transform:
    """A scale transformation bundling forward/inverse functions and metadata.

    Parameters
    ----------
    name : str
        Human-readable name for the transform.
    transform_func : callable
        Forward transform ``f(x) -> y``.
    inverse_func : callable
        Inverse transform ``g(y) -> x``.
    d_transform : callable or None
        Derivative of the forward transform.
    d_inverse : callable or None
        Derivative of the inverse transform.
    breaks_func : callable
        Function ``(limits) -> array`` that generates axis breaks.
    minor_breaks_func : callable or None
        Function for generating minor breaks.
    format_func : callable
        Label formatter ``(x) -> list[str]``.
    domain : tuple of float
        ``(min, max)`` valid input domain for the forward transform.
    """

    __slots__ = (
        "name",
        "transform_func",
        "inverse_func",
        "d_transform",
        "d_inverse",
        "breaks_func",
        "minor_breaks_func",
        "format_func",
        "domain",
    )

    def __init__(
        self,
        name: str,
        transform_func: Callable,
        inverse_func: Callable,
        d_transform: Optional[Callable] = None,
        d_inverse: Optional[Callable] = None,
        breaks_func: Optional[Callable] = None,
        minor_breaks_func: Optional[Callable] = None,
        format_func: Optional[Callable] = None,
        domain: Tuple[float, float] = (-np.inf, np.inf),
    ) -> None:
        self.name = name
        self.transform_func = transform_func
        self.inverse_func = inverse_func
        self.d_transform = d_transform
        self.d_inverse = d_inverse
        self.breaks_func = breaks_func if breaks_func is not None else _pretty_breaks(5)
        self.minor_breaks_func = minor_breaks_func
        self.format_func = format_func if format_func is not None else _default_format()
        self.domain = (float(domain[0]), float(domain[1]))

    # Convenience: call the forward / inverse directly
    def transform(self, x: ArrayLike) -> np.ndarray:
        """Apply the forward transformation."""
        arr = np.asarray(x)
        # Pass string / object / timedelta64 / datetime64 arrays through
        # unchanged — transforms like hms / date parse them themselves.
        if arr.dtype.kind in ("U", "S", "O", "m", "M"):
            return np.asarray(self.transform_func(arr))
        return np.asarray(self.transform_func(np.asarray(x, dtype=float)))

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """Apply the inverse transformation."""
        arr = np.asarray(x)
        if arr.dtype.kind in ("U", "S", "O", "m", "M"):
            return np.asarray(self.inverse_func(arr))
        return np.asarray(self.inverse_func(np.asarray(x, dtype=float)))

    def __repr__(self) -> str:
        return f"Transform: {self.name}"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def new_transform(
    name: str,
    transform: Callable,
    inverse: Callable,
    d_transform: Optional[Callable] = None,
    d_inverse: Optional[Callable] = None,
    breaks: Optional[Callable] = None,
    minor_breaks: Optional[Callable] = None,
    format: Optional[Callable] = None,
    domain: Tuple[float, float] = (-np.inf, np.inf),
) -> Transform:
    """Create a new :class:`Transform` object.

    Parameters
    ----------
    name : str
        Human-readable name.
    transform : callable
        Forward transform.
    inverse : callable
        Inverse transform.
    d_transform : callable, optional
        Derivative of forward transform.
    d_inverse : callable, optional
        Derivative of inverse transform.
    breaks : callable, optional
        Break-generation function ``(limits) -> array``.
    minor_breaks : callable, optional
        Minor-break-generation function.
    format : callable, optional
        Label formatter ``(x) -> list[str]``.
    domain : tuple of float, optional
        Valid input range for the forward transform.
        Default ``(-inf, inf)``.

    Returns
    -------
    Transform
    """
    return Transform(
        name=name,
        transform_func=transform,
        inverse_func=inverse,
        d_transform=d_transform,
        d_inverse=d_inverse,
        breaks_func=breaks,
        minor_breaks_func=minor_breaks,
        format_func=format,
        domain=domain,
    )


# Legacy alias
trans_new = new_transform


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def is_transform(x: Any) -> bool:
    """Return ``True`` if *x* is a :class:`Transform` instance."""
    return isinstance(x, Transform)


# Legacy alias
is_trans = is_transform


# ---------------------------------------------------------------------------
# Transform registry / coercion
# ---------------------------------------------------------------------------

# Populated lazily at first call to ``as_transform``.
_REGISTRY: Dict[str, Callable[[], Transform]] = {}


def _ensure_registry() -> None:
    """Populate the name -> factory mapping (once)."""
    if _REGISTRY:
        return
    _REGISTRY.update({
        "identity": transform_identity,
        "log": transform_log,
        "log10": transform_log10,
        "log2": transform_log2,
        "log1p": transform_log1p,
        "exp": transform_exp,
        "sqrt": transform_sqrt,
        "reverse": transform_reverse,
        "reciprocal": transform_reciprocal,
        "asinh": transform_asinh,
        "asn": transform_asn,
        "atanh": transform_atanh,
        "logit": transform_logit,
        "probit": transform_probit,
        "pseudo_log": transform_pseudo_log,
        "date": transform_date,
        "time": transform_time,
        "timespan": transform_timespan,
    })


def as_transform(x: Union[str, Transform]) -> Transform:
    """Coerce *x* to a :class:`Transform`.

    Parameters
    ----------
    x : str or Transform
        If a string, look up the corresponding built-in transform by
        name.  If already a :class:`Transform`, return as-is.

    Returns
    -------
    Transform

    Raises
    ------
    TypeError
        If *x* is neither a string nor a :class:`Transform`.
    ValueError
        If no built-in transform matches the given name.
    """
    if isinstance(x, Transform):
        return x
    if isinstance(x, str):
        _ensure_registry()
        # Try exact match, then with common suffixes stripped
        key = x.lower().replace("-", "_")
        if key in _REGISTRY:
            return _REGISTRY[key]()
        # Try stripping "_trans" / "transform_" suffixes/prefixes
        for prefix in ("transform_", ""):
            for suffix in ("_trans", "_transform", ""):
                candidate = key.removeprefix(prefix).removesuffix(suffix)
                if candidate in _REGISTRY:
                    return _REGISTRY[candidate]()
        raise ValueError(
            f"Unknown transform name {x!r}. Available: "
            f"{sorted(_REGISTRY.keys())}"
        )
    raise TypeError(
        f"Cannot coerce {type(x).__name__!r} to a Transform; "
        f"expected a string or Transform instance."
    )


# Legacy alias
as_trans = as_transform


# ---------------------------------------------------------------------------
# trans_breaks / trans_format
# ---------------------------------------------------------------------------

def trans_breaks(
    trans: Union[str, Transform],
    n: int = 5,
    offset: float = 0,
) -> Callable:
    """Generate breaks in transformed space then map back.

    Parameters
    ----------
    trans : str or Transform
        The transform to use.
    n : int, optional
        Desired number of breaks (default 5).
    offset : float, optional
        Additive offset applied after transforming limits and removed
        before inverse-transforming breaks (default 0).

    Returns
    -------
    callable
        A function ``(limits) -> np.ndarray`` that returns break
        positions in the original (data) space.
    """
    t = as_transform(trans)
    inner_breaks = _pretty_breaks(n)

    def _breaks(limits: Tuple[float, float]) -> np.ndarray:
        tlimits = t.transform(np.array(limits))
        breaks_t = inner_breaks((tlimits[0] + offset, tlimits[1] + offset))
        return t.inverse(np.asarray(breaks_t) - offset)

    return _breaks


def trans_format(
    trans: Union[str, Transform],
    format: Optional[Callable] = None,
) -> Callable:
    """Format labels using the inverse of *trans*.

    Parameters
    ----------
    trans : str or Transform
        The transform whose inverse maps breaks back to data space.
    format : callable, optional
        A formatter ``(x) -> list[str]``.  When *None*, the transform's
        own :attr:`format_func` is used.

    Returns
    -------
    callable
        A function ``(x) -> list[str]`` that first inverse-transforms *x*
        and then formats the result.
    """
    t = as_transform(trans)
    fmt = format if format is not None else t.format_func

    def _format(x: ArrayLike) -> list[str]:
        inv = t.inverse(np.asarray(x, dtype=float))
        return fmt(inv)

    return _format


# ===========================================================================
# Built-in transforms
# ===========================================================================

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def transform_identity() -> Transform:
    """No-op (identity) transform."""
    return new_transform(
        name="identity",
        transform=lambda x: x,
        inverse=lambda x: x,
        d_transform=lambda x: np.ones_like(x),
        d_inverse=lambda x: np.ones_like(x),
    )


identity_trans = transform_identity


# ---------------------------------------------------------------------------
# Log family
# ---------------------------------------------------------------------------

def transform_log(base: float = np.e) -> Transform:
    """Logarithmic transform with arbitrary *base*.

    Parameters
    ----------
    base : float, optional
        Logarithm base (default ``e``).
    """
    log_base = np.log(base)

    def _fwd(x: np.ndarray) -> np.ndarray:
        return np.log(x) / log_base

    def _inv(x: np.ndarray) -> np.ndarray:
        return base ** x

    def _d_fwd(x: np.ndarray) -> np.ndarray:
        return 1.0 / (x * log_base)

    def _d_inv(x: np.ndarray) -> np.ndarray:
        return base ** x * log_base

    name = "log" if base == np.e else f"log-{base:g}"
    return new_transform(
        name=name,
        transform=_fwd,
        inverse=_inv,
        d_transform=_d_fwd,
        d_inverse=_d_inv,
        breaks=_log_breaks(base=base),
        domain=(0, np.inf),
    )


log_trans = transform_log


def transform_log10() -> Transform:
    """Base-10 logarithmic transform."""
    return transform_log(base=10)


log10_trans = transform_log10


def transform_log2() -> Transform:
    """Base-2 logarithmic transform."""
    return transform_log(base=2)


log2_trans = transform_log2


def transform_log1p() -> Transform:
    """``log(1 + x)`` transform."""
    return new_transform(
        name="log1p",
        transform=np.log1p,
        inverse=np.expm1,
        d_transform=lambda x: 1.0 / (1.0 + x),
        d_inverse=lambda x: np.exp(x),
        domain=(-1, np.inf),
    )


log1p_trans = transform_log1p


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------

def transform_exp(base: float = np.e) -> Transform:
    """Exponential transform (inverse of :func:`transform_log`).

    Parameters
    ----------
    base : float, optional
        Exponent base (default ``e``).
    """
    log_base = np.log(base)

    def _fwd(x: np.ndarray) -> np.ndarray:
        return base ** x

    def _inv(x: np.ndarray) -> np.ndarray:
        return np.log(x) / log_base

    name = "exp" if base == np.e else f"power-{base:g}"
    return new_transform(
        name=name,
        transform=_fwd,
        inverse=_inv,
    )


exp_trans = transform_exp


# ---------------------------------------------------------------------------
# Power / root
# ---------------------------------------------------------------------------

def transform_sqrt() -> Transform:
    """Square-root transform (domain ``[0, inf]``)."""
    return new_transform(
        name="sqrt",
        transform=np.sqrt,
        inverse=np.square,
        d_transform=lambda x: 0.5 / np.sqrt(x),
        d_inverse=lambda x: 2.0 * x,
        domain=(0, np.inf),
    )


sqrt_trans = transform_sqrt


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------

def transform_reverse() -> Transform:
    """Negate values (reverse the axis direction).

    Matches R's ``transform_reverse``: uses
    ``regular_minor_breaks(reverse=True)`` so minor ticks extend toward
    the numerically *smaller* side of each major, suitable for reversed
    axes.
    """
    return new_transform(
        name="reverse",
        transform=lambda x: -x,
        inverse=lambda x: -x,
        d_transform=lambda x: np.full_like(np.asarray(x, dtype=float), -1.0),
        d_inverse=lambda x: np.full_like(np.asarray(x, dtype=float), -1.0),
        minor_breaks=regular_minor_breaks(reverse=True),
    )


reverse_trans = transform_reverse


# ---------------------------------------------------------------------------
# Reciprocal
# ---------------------------------------------------------------------------

def transform_reciprocal() -> Transform:
    """Reciprocal transform ``1 / x``."""
    return new_transform(
        name="reciprocal",
        transform=lambda x: 1.0 / x,
        inverse=lambda x: 1.0 / x,
        d_transform=lambda x: -1.0 / x ** 2,
        d_inverse=lambda x: -1.0 / x ** 2,
    )


reciprocal_trans = transform_reciprocal


# ---------------------------------------------------------------------------
# Hyperbolic / trigonometric
# ---------------------------------------------------------------------------

def transform_asinh() -> Transform:
    """Inverse hyperbolic sine transform."""
    return new_transform(
        name="asinh",
        transform=np.arcsinh,
        inverse=np.sinh,
        d_transform=lambda x: 1.0 / np.sqrt(x ** 2 + 1),
        d_inverse=lambda x: np.cosh(x),
    )


asinh_trans = transform_asinh


def transform_asn() -> Transform:
    """Arc-sine-square-root transform: ``asin(sqrt(x))``.

    Domain ``[0, 1]``.
    """
    def _fwd(x: np.ndarray) -> np.ndarray:
        return 2.0 * np.arcsin(np.sqrt(x))

    def _inv(x: np.ndarray) -> np.ndarray:
        return np.sin(x / 2.0) ** 2

    return new_transform(
        name="asn",
        transform=_fwd,
        inverse=_inv,
        domain=(0, 1),
    )


asn_trans = transform_asn


def transform_atanh() -> Transform:
    """Inverse hyperbolic tangent transform.

    Domain ``(-1, 1)``.
    """
    return new_transform(
        name="atanh",
        transform=np.arctanh,
        inverse=np.tanh,
        d_transform=lambda x: 1.0 / (1.0 - x ** 2),
        d_inverse=lambda x: 1.0 / np.cosh(x) ** 2,
        domain=(-1, 1),
    )


atanh_trans = transform_atanh


# ---------------------------------------------------------------------------
# Box-Cox
# ---------------------------------------------------------------------------

def transform_boxcox(p: float, offset: float = 0) -> Transform:
    """Box--Cox power transform.

    Parameters
    ----------
    p : float
        Transformation parameter (power).  When ``p == 0`` the transform
        reduces to ``log(x + offset)``.
    offset : float, optional
        Additive offset applied before transformation (default 0).

    Notes
    -----
    When ``p != 0``: ``((x + offset)^p - 1) / p``.
    When ``p == 0``: ``log(x + offset)``.
    Domain is ``[max(0, -offset), inf)``.
    """
    if p == 0:
        def _fwd(x: np.ndarray) -> np.ndarray:
            return np.log(x + offset)

        def _inv(x: np.ndarray) -> np.ndarray:
            return np.exp(x) - offset

        def _d_fwd(x: np.ndarray) -> np.ndarray:
            return 1.0 / (x + offset)

        def _d_inv(x: np.ndarray) -> np.ndarray:
            return np.exp(x)
    else:
        def _fwd(x: np.ndarray) -> np.ndarray:
            return ((x + offset) ** p - 1.0) / p

        def _inv(x: np.ndarray) -> np.ndarray:
            return (x * p + 1.0) ** (1.0 / p) - offset

        def _d_fwd(x: np.ndarray) -> np.ndarray:
            return (x + offset) ** (p - 1.0)

        def _d_inv(x: np.ndarray) -> np.ndarray:
            return (x * p + 1.0) ** (1.0 / p - 1.0)

    domain_lo = max(0.0, -offset)
    return new_transform(
        name=f"boxcox-{p:g}" if offset == 0 else f"boxcox-{p:g}-{offset:g}",
        transform=_fwd,
        inverse=_inv,
        d_transform=_d_fwd,
        d_inverse=_d_inv,
        domain=(domain_lo, np.inf),
    )


boxcox_trans = transform_boxcox


# ---------------------------------------------------------------------------
# Modulus (sign-preserving Box-Cox)
# ---------------------------------------------------------------------------

def transform_modulus(p: float, offset: float = 1) -> Transform:
    """Modulus transform (sign-preserving Box--Cox).

    Parameters
    ----------
    p : float
        Power parameter.
    offset : float, optional
        Offset applied inside the absolute value (default 1).

    Notes
    -----
    When ``p != 0``: ``sign(x) * ((|x| + offset)^p - 1) / p``.
    When ``p == 0``: ``sign(x) * log(|x| + offset)``.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative for modulus transform")

    if p == 0:
        def _fwd(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * np.log(np.abs(x) + offset)

        def _inv(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * (np.exp(np.abs(x)) - offset)
    else:
        def _fwd(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * ((np.abs(x) + offset) ** p - 1.0) / p

        def _inv(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * ((np.abs(x) * p + 1.0) ** (1.0 / p) - offset)

    return new_transform(
        name=f"modulus-{p:g}-{offset:g}",
        transform=_fwd,
        inverse=_inv,
    )


modulus_trans = transform_modulus


# ---------------------------------------------------------------------------
# Yeo-Johnson
# ---------------------------------------------------------------------------

def transform_yj(p: float) -> Transform:
    """Yeo--Johnson power transform.

    Parameters
    ----------
    p : float
        Transformation parameter.

    Notes
    -----
    For ``x >= 0``:
      - ``p != 0``: ``((x + 1)^p - 1) / p``
      - ``p == 0``: ``log(x + 1)``
    For ``x < 0``:
      - ``p != 2``: ``-((-x + 1)^(2 - p) - 1) / (2 - p)``
      - ``p == 2``: ``-log(-x + 1)``
    """
    def _fwd(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos

        if p != 0:
            out[pos] = ((x[pos] + 1.0) ** p - 1.0) / p
        else:
            out[pos] = np.log(x[pos] + 1.0)

        if p != 2:
            out[neg] = -((-x[neg] + 1.0) ** (2.0 - p) - 1.0) / (2.0 - p)
        else:
            out[neg] = -np.log(-x[neg] + 1.0)

        return out

    def _inv(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos

        if p != 0:
            out[pos] = (x[pos] * p + 1.0) ** (1.0 / p) - 1.0
        else:
            out[pos] = np.exp(x[pos]) - 1.0

        if p != 2:
            out[neg] = 1.0 - (-(2.0 - p) * x[neg] + 1.0) ** (1.0 / (2.0 - p))
        else:
            # Matches R's inv_neg(x) = 1 - exp(-x). For x < 0, exp(-x) > 1,
            # so the expression is already ≤ 0 — no sign patch needed.
            out[neg] = 1.0 - np.exp(-x[neg])

        return out

    return new_transform(
        name=f"yeo-johnson-{p:g}",
        transform=_fwd,
        inverse=_inv,
    )


yj_trans = transform_yj


# ---------------------------------------------------------------------------
# Pseudo-log
# ---------------------------------------------------------------------------

def transform_pseudo_log(
    sigma: float = 1, base: float = np.e
) -> Transform:
    """Pseudo-log transform (smooth transition around zero).

    Parameters
    ----------
    sigma : float, optional
        Scaling parameter controlling the linear region near zero
        (default 1).
    base : float, optional
        Logarithm base (default ``e``).

    Notes
    -----
    Forward: ``asinh(x / (2 * sigma)) / log(base)``.
    Inverse: ``2 * sigma * sinh(x * log(base))``.
    """
    log_base = np.log(base)

    def _fwd(x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x / (2.0 * sigma)) / log_base

    def _inv(x: np.ndarray) -> np.ndarray:
        return 2.0 * sigma * np.sinh(x * log_base)

    return new_transform(
        name="pseudo_log",
        transform=_fwd,
        inverse=_inv,
    )


pseudo_log_trans = transform_pseudo_log


# ---------------------------------------------------------------------------
# Probability transforms (logit, probit, general)
# ---------------------------------------------------------------------------

def transform_probability(
    distribution: str, *args: Any, **kwargs: Any
) -> Transform:
    """Probability transform using a ``scipy.stats`` distribution.

    Mirrors R's ``scales::transform_probability``: forward is the
    quantile function (``q<dist>`` / ``ppf``), inverse is the CDF
    (``p<dist>`` / ``cdf``).  ``d_transform`` is ``1 / pdf(ppf(x))`` and
    ``d_inverse`` is ``pdf(x)``.

    Parameters
    ----------
    distribution : str
        Name of a distribution in :mod:`scipy.stats` (e.g. ``"norm"``,
        ``"logistic"``).
    *args, **kwargs
        Extra arguments forwarded to the ``scipy.stats`` distribution
        constructor.

    Returns
    -------
    Transform
        Domain ``(0, 1)``.
    """
    try:
        import scipy.stats as st
    except ImportError as exc:
        raise ImportError(
            "transform_probability requires scipy. "
            "Install it with: pip install scipy"
        ) from exc

    dist = getattr(st, distribution)(*args, **kwargs)

    return new_transform(
        name=f"prob-{distribution}",
        transform=lambda x: dist.ppf(x),
        inverse=lambda x: dist.cdf(x),
        d_transform=lambda x: 1.0 / dist.pdf(dist.ppf(x)),
        d_inverse=lambda x: dist.pdf(x),
        domain=(0, 1),
    )


probability_trans = transform_probability


def transform_logit() -> Transform:
    """Logit transform: ``log(x / (1 - x))``.

    Domain ``(0, 1)``.
    """
    def _fwd(x: np.ndarray) -> np.ndarray:
        return np.log(x / (1.0 - x))

    def _inv(x: np.ndarray) -> np.ndarray:
        ex = np.exp(x)
        return ex / (1.0 + ex)

    def _d_fwd(x: np.ndarray) -> np.ndarray:
        return 1.0 / (x * (1.0 - x))

    def _d_inv(x: np.ndarray) -> np.ndarray:
        ex = np.exp(x)
        return ex / (1.0 + ex) ** 2

    return new_transform(
        name="logit",
        transform=_fwd,
        inverse=_inv,
        d_transform=_d_fwd,
        d_inverse=_d_inv,
        domain=(0, 1),
    )


logit_trans = transform_logit


def transform_probit() -> Transform:
    """Probit transform (normal quantile function).

    Domain ``(0, 1)``.  Requires ``scipy``.
    """
    try:
        from scipy.stats import norm as _norm
    except ImportError as exc:
        raise ImportError(
            "transform_probit requires scipy. "
            "Install it with: pip install scipy"
        ) from exc

    return new_transform(
        name="probit",
        transform=lambda x: _norm.ppf(x),
        inverse=lambda x: _norm.cdf(x),
        domain=(0, 1),
    )


probit_trans = transform_probit


# ---------------------------------------------------------------------------
# Date / time transforms
# ---------------------------------------------------------------------------

# Reference epoch for date conversions (days since 1970-01-01)
_EPOCH = np.datetime64("1970-01-01", "D")
_EPOCH_NS = np.datetime64("1970-01-01T00:00:00", "ns")


def transform_date() -> Transform:
    """Transform between :class:`datetime.date` / ``datetime64[D]`` and numeric.

    Forward maps dates to float (days since 1970-01-01).
    Inverse maps float back to ``numpy.datetime64[D]``.
    """
    def _fwd(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.datetime64):
            return (x - _EPOCH) / np.timedelta64(1, "D")
        # Already numeric
        return np.asarray(x, dtype=float)

    def _inv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return _EPOCH + (x * np.timedelta64(1, "D")).astype("timedelta64[D]")

    def _date_format(x: np.ndarray) -> list[str]:
        dates = _inv(np.asarray(x, dtype=float))
        return [str(d) for d in np.asarray(dates).flat]

    return Transform(
        name="date",
        transform_func=_fwd,
        inverse_func=_inv,
        format_func=_date_format,
        breaks_func=_pretty_breaks(5),
    )


date_trans = transform_date


def transform_time(tz: Optional[str] = None) -> Transform:
    """Transform between ``datetime64`` and numeric (seconds since epoch).

    Parameters
    ----------
    tz : str or None, optional
        Timezone name (informational; NumPy datetimes are always UTC).
    """
    def _fwd(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.datetime64):
            return (x - _EPOCH_NS) / np.timedelta64(1, "s")
        return np.asarray(x, dtype=float)

    def _inv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return _EPOCH_NS + (x * np.timedelta64(1, "s")).astype("timedelta64[ns]")

    def _time_format(x: np.ndarray) -> list[str]:
        dts = _inv(np.asarray(x, dtype=float))
        return [str(d) for d in np.asarray(dts).flat]

    return Transform(
        name="time",
        transform_func=_fwd,
        inverse_func=_inv,
        format_func=_time_format,
        breaks_func=_pretty_breaks(5),
    )


time_trans = transform_time


def transform_timespan(unit: str = "secs") -> Transform:
    """Transform between ``timedelta64`` and numeric.

    Parameters
    ----------
    unit : str, optional
        Time unit for the numeric representation.  One of ``"secs"``,
        ``"mins"``, ``"hours"``, ``"days"``, ``"weeks"`` (default
        ``"secs"``).
    """
    _unit_map = {
        "secs": ("s", 1.0),
        "mins": ("s", 60.0),
        "hours": ("s", 3600.0),
        "days": ("D", 1.0),
        "weeks": ("D", 7.0),
    }
    if unit not in _unit_map:
        raise ValueError(
            f"Unknown unit {unit!r}; choose from {sorted(_unit_map)}"
        )
    np_unit, divisor = _unit_map[unit]

    def _fwd(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.timedelta64):
            return x / np.timedelta64(1, np_unit) / divisor
        return np.asarray(x, dtype=float)

    def _inv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x * divisor * np.timedelta64(1, np_unit)).astype(
            f"timedelta64[{np_unit}]"
        )

    return Transform(
        name=f"timespan-{unit}",
        transform_func=_fwd,
        inverse_func=_inv,
        domain=(0, np.inf),
    )


timespan_trans = transform_timespan


def transform_hms() -> Transform:
    """Transform clock-time values to/from numeric seconds.

    Mirrors R's ``transform_hms``: forward drops to raw seconds, inverse
    converts back to an ``"HH:MM:SS"`` string (R uses the ``hms`` class;
    Python uses a plain string, which is the natural stdlib equivalent).

    Accepted forward inputs:

    * ``float`` / ``int`` — already seconds, returned unchanged.
    * ``numpy.timedelta64`` — converted to seconds.
    * ``datetime.time`` — converted via its hour/minute/second fields.
    * ``str`` in ``"HH:MM:SS"`` or ``"HH:MM:SS.fff"`` — parsed.

    Inverse always returns ``"HH:MM:SS"`` strings, wrapping past 24 h.
    """
    import re
    from datetime import time as _time

    _pat = re.compile(r"^(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d+)?))?$")

    def _to_seconds(val: Any) -> float:
        if val is None:
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        if isinstance(val, np.timedelta64):
            return val / np.timedelta64(1, "s")
        if isinstance(val, _time):
            return (
                val.hour * 3600
                + val.minute * 60
                + val.second
                + val.microsecond / 1_000_000
            )
        if isinstance(val, str):
            m = _pat.match(val.strip())
            if not m:
                raise ValueError(f"cannot parse {val!r} as HH:MM:SS")
            h = int(m.group(1))
            mm = int(m.group(2))
            ss = float(m.group(3) or 0.0)
            return h * 3600 + mm * 60 + ss
        # Last resort: try array
        return float(val)

    def _fwd(x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x)
        if np.issubdtype(arr.dtype, np.timedelta64):
            # timedelta64 arrays → seconds (float). Division by
            # np.timedelta64(1, "s") returns a float array even for
            # non-integer-second inputs (microsecond precision kept).
            return (arr / np.timedelta64(1, "s")).astype(float)
        if arr.dtype.kind in ("i", "u", "f"):
            return arr.astype(float)
        # Object / string arrays — convert element-wise.
        out = np.empty(arr.shape, dtype=float)
        for idx, v in np.ndenumerate(arr):
            out[idx] = _to_seconds(v)
        return out

    def _inv(x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        out: list[str] = []
        for v in arr.flat:
            if not np.isfinite(v):
                out.append("NA")
                continue
            total = float(v)
            sign = "-" if total < 0 else ""
            total = abs(total)
            h = int(total // 3600)
            rem = total - h * 3600
            m = int(rem // 60)
            s = rem - m * 60
            # Render as HH:MM:SS; seconds keep fractional part when non-zero.
            if abs(s - round(s)) < 1e-9:
                out.append(f"{sign}{h:02d}:{m:02d}:{int(round(s)):02d}")
            else:
                out.append(f"{sign}{h:02d}:{m:02d}:{s:06.3f}")
        return np.array(out, dtype=object).reshape(arr.shape)

    return Transform(
        name="hms",
        transform_func=_fwd,
        inverse_func=_inv,
        domain=(-np.inf, np.inf),
    )


hms_trans = transform_hms


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def transform_compose(*transforms: Union[str, Transform]) -> Transform:
    """Compose multiple transforms (applied left to right).

    Faithful port of R's ``scales::transform_compose``:

    * Resolves a conservative domain by pushing the first transform's
      domain forward through the sequence (intersecting with each
      transform's own domain at every step) to get the range, then
      pulling that range back through the inverses.
    * Composes ``d_transform`` and ``d_inverse`` via the chain rule when
      *every* transform exposes them; otherwise the composed derivatives
      are ``None``.
    * Uses the first transform's ``breaks_func`` for tick generation.

    Parameters
    ----------
    *transforms : str or Transform
        One or more transforms to compose.  Strings are resolved via
        :func:`as_transform`.

    Returns
    -------
    Transform
    """
    ts = [as_transform(t) for t in transforms]
    if len(ts) < 1:
        raise ValueError(
            "transform_compose must include at least 1 transformer to compose"
        )

    def _fwd(x: ArrayLike) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        for t in ts:
            v = t.transform(v)
        return v

    def _inv(x: ArrayLike) -> np.ndarray:
        v = np.asarray(x, dtype=float)
        for t in reversed(ts):
            v = t.inverse(v)
        return v

    # Domain resolution — matches R's algorithm exactly.
    t0 = ts[0]
    rng = np.asarray(
        t0.transform(np.asarray(t0.domain, dtype=float)), dtype=float
    )
    for t in ts[1:]:
        d_lo, d_hi = float(t.domain[0]), float(t.domain[1])
        r_lo, r_hi = float(np.min(rng)), float(np.max(rng))
        lower = max(d_lo, r_lo)
        upper = min(d_hi, r_hi)
        if lower <= upper:
            rng = np.asarray(
                t.transform(np.asarray([lower, upper], dtype=float)), dtype=float
            )
        else:
            rng = np.array([np.nan, np.nan])
            break

    # Push range back through inverses to derive composed domain.
    dom_vals = rng
    for t in reversed(ts):
        dom_vals = np.asarray(t.inverse(np.asarray(dom_vals, dtype=float)), dtype=float)

    if np.any(np.isnan(dom_vals)):
        raise ValueError("Sequence of transformations yields invalid domain")
    domain = (float(np.min(dom_vals)), float(np.max(dom_vals)))

    has_d_transform = all(t.d_transform is not None for t in ts)
    has_d_inverse = all(t.d_inverse is not None for t in ts)

    def _d_fwd(x: ArrayLike) -> np.ndarray:
        # Forward chain rule: derivative = prod_i f'_i(x_i) where x_i is
        # the value fed into the i-th transform.
        v = np.asarray(x, dtype=float)
        deriv = np.ones_like(v, dtype=float)
        for t in ts:
            deriv = np.asarray(t.d_transform(v), dtype=float) * deriv
            v = t.transform(v)
        return deriv

    def _d_inv(x: ArrayLike) -> np.ndarray:
        # Inverse chain rule: apply inverses in reverse order.
        v = np.asarray(x, dtype=float)
        deriv = np.ones_like(v, dtype=float)
        for t in reversed(ts):
            deriv = np.asarray(t.d_inverse(v), dtype=float) * deriv
            v = t.inverse(v)
        return deriv

    names = ",".join(t.name for t in ts)

    return new_transform(
        name=f"composition({names})",
        transform=_fwd,
        inverse=_inv,
        d_transform=_d_fwd if has_d_transform else None,
        d_inverse=_d_inv if has_d_inverse else None,
        breaks=t0.breaks_func,
        domain=domain,
    )


compose_trans = transform_compose
