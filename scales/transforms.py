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
    """Return a break-generator that picks *n* \"pretty\" breaks.

    Uses :func:`numpy.linspace` rounded to nice numbers via
    ``matplotlib.ticker`` if available, otherwise a simple ``linspace``.
    """
    def _breaks(limits: Tuple[float, float]) -> np.ndarray:
        lo, hi = float(limits[0]), float(limits[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([lo])
        try:
            from matplotlib.ticker import MaxNLocator
            locator = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10])
            return np.asarray(locator.tick_values(lo, hi))
        except ImportError:
            return np.linspace(lo, hi, n)
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
        return np.asarray(self.transform_func(np.asarray(x, dtype=float)))

    def inverse(self, x: ArrayLike) -> np.ndarray:
        """Apply the inverse transformation."""
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
    """Negate values (reverse the axis direction)."""
    return new_transform(
        name="reverse",
        transform=lambda x: -x,
        inverse=lambda x: -x,
        d_transform=lambda x: np.full_like(np.asarray(x, dtype=float), -1.0),
        d_inverse=lambda x: np.full_like(np.asarray(x, dtype=float), -1.0),
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
            out[neg] = 1.0 - np.exp(-x[neg])
            # Negate because for x < 0 the original is negative
            out[neg] = -np.abs(out[neg])

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
    """Probability-integral transform using a ``scipy.stats`` distribution.

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
        Forward = CDF, inverse = PPF (quantile function).
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
        name=f"probability-{distribution}",
        transform=lambda x: dist.cdf(x),
        inverse=lambda x: dist.ppf(x),
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

# hms_trans / transform_hms are aliases for transform_timespan.
# R: transform_hms wraps hms::as_hms; Python datetime handles this natively.
transform_hms = transform_timespan
hms_trans = transform_timespan


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def transform_compose(*transforms: Union[str, Transform]) -> Transform:
    """Compose multiple transforms (applied left to right).

    Parameters
    ----------
    *transforms : str or Transform
        Two or more transforms to compose.  Strings are resolved via
        :func:`as_transform`.

    Returns
    -------
    Transform
        A composite transform whose forward function applies each
        transform in order and whose inverse applies the inverses in
        reverse order.
    """
    ts = [as_transform(t) for t in transforms]
    if len(ts) < 1:
        raise ValueError("transform_compose requires at least one transform")

    names = " -> ".join(t.name for t in ts)

    def _fwd(x: np.ndarray) -> np.ndarray:
        return reduce(lambda v, t: t.transform(v), ts, np.asarray(x, dtype=float))

    def _inv(x: np.ndarray) -> np.ndarray:
        return reduce(
            lambda v, t: t.inverse(v), reversed(ts), np.asarray(x, dtype=float)
        )

    # Domain comes from the first transform
    domain = ts[0].domain

    return new_transform(
        name=f"compose({names})",
        transform=_fwd,
        inverse=_inv,
        domain=domain,
    )


compose_trans = transform_compose
