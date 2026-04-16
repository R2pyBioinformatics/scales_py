"""
Label formatting functions for scales.

Python port of label functions from the R scales package
(https://github.com/r-lib/scales). Each ``label_*`` function is a
closure factory: it returns a callable that accepts a sequence of
values and returns a ``list[str]`` of formatted labels.

Direct formatting functions (``number``, ``comma``, ``dollar``, etc.)
are also provided for one-shot use without creating a closure first.
"""

from __future__ import annotations

import math
import re
import textwrap
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from ._utils import precision as _precision_util

__all__ = [
    # Closure factories
    "label_number",
    "label_comma",
    "label_percent",
    "label_dollar",
    "label_currency",
    "label_scientific",
    "label_bytes",
    "label_ordinal",
    "label_pvalue",
    "label_date",
    "label_date_short",
    "label_time",
    "label_timespan",
    "label_wrap",
    "label_glue",
    "label_parse",
    "label_math",
    "label_log",
    "label_number_auto",
    "label_number_si",
    "label_dictionary",
    "compose_label",
    # Ordinal helpers
    "ordinal_english",
    "ordinal_french",
    "ordinal_spanish",
    # Direct formatting functions
    "number",
    "comma",
    "dollar",
    "percent",
    "scientific",
    "ordinal",
    "pvalue",
    # Core log formatting
    "format_log",
    # Scale cut helpers
    "cut_short_scale",
    "cut_long_scale",
    "cut_time_scale",
    "cut_si",
    # Date utilities
    "date_breaks",
    "date_format",
    "time_format",
    # Legacy aliases
    "comma_format",
    "dollar_format",
    "percent_format",
    "scientific_format",
    "ordinal_format",
    "pvalue_format",
    "number_format",
    "number_bytes_format",
    "number_bytes",
    "parse_format",
    "math_format",
    "wrap_format",
    "unit_format",
    "format_format",
    "number_options",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _precision(x: np.ndarray) -> float:
    """Auto-detect precision for a numeric array."""
    return _precision_util(x)


def _format_number(
    value: float,
    accuracy: float,
    big_mark: Optional[str],
    decimal_mark: str,
    trim: bool,
) -> str:
    """Format a single number with the given accuracy and marks."""
    if not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Inf" if value > 0 else "-Inf"

    # Number of decimal places from accuracy
    if accuracy >= 1:
        ndigits = 0
    else:
        ndigits = max(0, -int(math.floor(math.log10(accuracy) - 1e-12)))

    rounded = round(value, ndigits)

    # Format with fixed decimals
    formatted = f"{rounded:.{ndigits}f}"

    if trim and ndigits > 0:
        # Strip trailing zeros after decimal point
        formatted = formatted.rstrip("0").rstrip(".")

    # Apply big_mark (thousands separator)
    if big_mark:
        parts = formatted.split(".")
        int_part = parts[0]
        # Handle negative sign
        sign = ""
        if int_part.startswith("-"):
            sign = "-"
            int_part = int_part[1:]
        # Insert separators from right
        groups: list[str] = []
        while len(int_part) > 3:
            groups.append(int_part[-3:])
            int_part = int_part[:-3]
        groups.append(int_part)
        int_part = big_mark.join(reversed(groups))
        parts[0] = sign + int_part
        formatted = ".".join(parts)

    # Apply decimal_mark
    if decimal_mark and decimal_mark != ".":
        formatted = formatted.replace(".", decimal_mark, 1)

    return formatted


def _apply_style(
    formatted: str,
    value: float,
    style_positive: str,
    style_negative: str,
) -> str:
    """Apply positive/negative styling to a formatted number string."""
    if np.isnan(value):
        return formatted

    if value > 0:
        if style_positive == "plus":
            formatted = "+" + formatted
        elif style_positive == "space":
            formatted = " " + formatted
        # "none" → no change
    elif value < 0:
        # The formatted string already has a '-' from Python formatting.
        # We may need to replace it.
        if style_negative == "minus":
            formatted = formatted.replace("-", "\u2212", 1)
        elif style_negative == "parens":
            formatted = "(" + formatted.replace("-", "", 1) + ")"
        # "hyphen" → keep the ASCII hyphen-minus (default)

    return formatted


def _apply_scale_cut(
    values: np.ndarray,
    scale_cut: list[tuple[float, str]],
) -> tuple[np.ndarray, list[str]]:
    """
    Apply scale_cut to values. Returns (scaled_values, suffixes).

    Each entry in *scale_cut* is ``(threshold, suffix)``. Values are
    divided by the largest threshold they exceed.
    """
    # Sort scale_cut by threshold ascending
    sc = sorted(scale_cut, key=lambda t: t[0])
    scaled = values.copy()
    suffixes: list[str] = []

    for v_idx in range(len(values)):
        val = values[v_idx]
        chosen_suffix = ""
        chosen_divisor = 1.0
        for threshold, sfx in sc:
            if threshold == 0:
                chosen_suffix = sfx
                chosen_divisor = 1.0
            elif abs(val) >= threshold:
                chosen_suffix = sfx
                chosen_divisor = threshold
        scaled[v_idx] = val / chosen_divisor
        suffixes.append(chosen_suffix)

    return scaled, suffixes


# ---------------------------------------------------------------------------
# Scale-cut helpers
# ---------------------------------------------------------------------------


def cut_short_scale(space: bool = False) -> list[tuple[float, str]]:
    """
    Short scale suffixes: K, M, B, T.

    Parameters
    ----------
    space : bool, optional
        If ``True``, prepend a space before the suffix (default ``False``).

    Returns
    -------
    list of (float, str)
        Scale-cut specification.
    """
    sp = " " if space else ""
    return [
        (0, ""),
        (1e3, f"{sp}K"),
        (1e6, f"{sp}M"),
        (1e9, f"{sp}B"),
        (1e12, f"{sp}T"),
    ]


def cut_long_scale(space: bool = False) -> list[tuple[float, str]]:
    """
    Long scale suffixes: K, M, B, T at 10^3, 10^6, 10^12, 10^18.

    Parameters
    ----------
    space : bool, optional
        If ``True``, prepend a space before the suffix (default ``False``).

    Returns
    -------
    list of (float, str)
        Scale-cut specification.
    """
    sp = " " if space else ""
    return [
        (0, ""),
        (1e3, f"{sp}K"),
        (1e6, f"{sp}M"),
        (1e12, f"{sp}B"),
        (1e18, f"{sp}T"),
    ]


def cut_time_scale(space: bool = False) -> list[tuple[float, str]]:
    """
    Time scale suffixes: ns, us, ms, s, m, h, d, w.

    Values are assumed to be in seconds.

    Parameters
    ----------
    space : bool, optional
        If ``True``, prepend a space before the suffix (default ``False``).

    Returns
    -------
    list of (float, str)
        Scale-cut specification.
    """
    sp = " " if space else ""
    # R uses "\u03BCs" (Greek small mu + s) when UTF-8 is available —
    # Python assumes UTF-8 everywhere, so emit it unconditionally.
    return [
        (0, ""),
        (1e-9, f"{sp}ns"),
        (1e-6, f"{sp}\u03bcs"),
        (1e-3, f"{sp}ms"),
        (1, f"{sp}s"),
        (60, f"{sp}m"),
        (3600, f"{sp}h"),
        (86400, f"{sp}d"),
        (604800, f"{sp}w"),
    ]


def cut_si(unit: str) -> list[tuple[float, str]]:
    """
    Full SI prefix scale cuts from yocto (10^-24) to yotta (10^24).

    Parameters
    ----------
    unit : str
        Base unit string to append after the SI prefix.

    Returns
    -------
    list of (float, str)
        Scale-cut specification.
    """
    prefixes = [
        (1e-24, "y"),
        (1e-21, "z"),
        (1e-18, "a"),
        (1e-15, "f"),
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "\u00b5"),  # micro sign
        (1e-3, "m"),
        (1, ""),
        (1e3, "k"),
        (1e6, "M"),
        (1e9, "G"),
        (1e12, "T"),
        (1e15, "P"),
        (1e18, "E"),
        (1e21, "Z"),
        (1e24, "Y"),
    ]
    return [(0, "")] + [(val, f" {pfx}{unit}") for val, pfx in prefixes]


# ---------------------------------------------------------------------------
# Core direct formatting function: number()
# ---------------------------------------------------------------------------


def number(
    x: ArrayLike,
    accuracy: Optional[float] = None,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    big_mark: Optional[str] = None,
    decimal_mark: Optional[str] = None,
    style_positive: Optional[str] = None,
    style_negative: Optional[str] = None,
    scale_cut: Optional[list[tuple[float, str]]] = None,
    trim: bool = True,
) -> list[str]:
    """
    Format a numeric vector.

    Parameters
    ----------
    x : array-like
        Numeric values to format.
    accuracy : float, optional
        Rounding precision. ``None`` for auto-detect.
    scale : float, optional
        Multiplicative scaling factor (default 1).
    prefix : str, optional
        String prepended to each label.
    suffix : str, optional
        String appended to each label.
    big_mark : str, optional
        Thousands separator. ``None`` means no separator.
    decimal_mark : str, optional
        Decimal separator. ``None`` defaults to ``"."``.
    style_positive : str, optional
        Treatment of positive values: ``"none"``, ``"plus"``, or
        ``"space"`` (default ``"none"``).
    style_negative : str, optional
        Treatment of negative values: ``"hyphen"`` (ASCII ``-``),
        ``"minus"`` (Unicode minus ``\u2212``), or ``"parens"``
        (default ``"hyphen"``).
    scale_cut : list of (float, str), optional
        SI-style suffix specification (see :func:`cut_short_scale`).
    trim : bool, optional
        Strip trailing zeros (default ``True``).

    Returns
    -------
    list of str
        Formatted strings.
    """
    x_arr = np.asarray(x, dtype=float)
    x_scaled = x_arr * scale

    # Resolve defaults from the module-level option store
    # (`number_options()` — mirrors R's getOption("scales.*")). Python
    # keeps an empty big_mark default so label output round-trips
    # through float(); R uses " " instead. Users can opt in via
    # `number_options(big_mark=" ")`.
    if big_mark is None:
        big_mark = str(_NUMBER_OPTIONS.get("big_mark", ""))
    if decimal_mark is None:
        decimal_mark = str(_NUMBER_OPTIONS.get("decimal_mark", "."))
    if style_positive is None:
        style_positive = str(_NUMBER_OPTIONS.get("style_positive", "none"))
    if style_negative is None:
        style_negative = str(_NUMBER_OPTIONS.get("style_negative", "hyphen"))

    # Apply scale_cut
    per_value_suffix: Optional[list[str]] = None
    if scale_cut is not None:
        x_scaled, per_value_suffix = _apply_scale_cut(x_scaled, scale_cut)

    if accuracy is None:
        accuracy = _precision(x_scaled)

    results: list[str] = []
    for i, val in enumerate(x_scaled.flat):
        fmt = _format_number(val, accuracy, big_mark, decimal_mark, trim)
        fmt = _apply_style(fmt, val, style_positive, style_negative)
        sc_sfx = per_value_suffix[i] if per_value_suffix is not None else ""
        results.append(f"{prefix}{fmt}{sc_sfx}{suffix}")

    return results


# ---------------------------------------------------------------------------
# Closure factory: label_number
# ---------------------------------------------------------------------------


def label_number(
    accuracy: Optional[float] = None,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    big_mark: Optional[str] = None,
    decimal_mark: Optional[str] = None,
    style_positive: str = "none",
    style_negative: str = "hyphen",
    scale_cut: Optional[list[tuple[float, str]]] = None,
    trim: bool = True,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label numbers with flexible formatting.

    Returns a closure that formats numeric values according to the
    parameters captured at construction time.

    Parameters
    ----------
    accuracy : float, optional
        Rounding precision. ``None`` for auto-detect.
    scale : float, optional
        Multiplicative scaling factor (default 1).
    prefix : str, optional
        Prepended to each label.
    suffix : str, optional
        Appended to each label.
    big_mark : str, optional
        Thousands separator.
    decimal_mark : str, optional
        Decimal separator.
    style_positive : str, optional
        ``"none"``, ``"plus"``, or ``"space"``.
    style_negative : str, optional
        ``"hyphen"``, ``"minus"``, or ``"parens"``.
    scale_cut : list of (float, str), optional
        SI-style suffix specification.
    trim : bool, optional
        Strip trailing zeros.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return number(
            x,
            accuracy=accuracy,
            scale=scale,
            prefix=prefix,
            suffix=suffix,
            big_mark=big_mark,
            decimal_mark=decimal_mark,
            style_positive=style_positive,
            style_negative=style_negative,
            scale_cut=scale_cut,
            trim=trim,
        )

    return formatter


# ---------------------------------------------------------------------------
# label_comma / comma
# ---------------------------------------------------------------------------


def label_comma(**kwargs: Any) -> Callable[[ArrayLike], list[str]]:
    """
    Label numbers with comma as thousands separator.

    Parameters
    ----------
    **kwargs
        Passed to :func:`label_number`. ``big_mark`` defaults to ``","``.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    kwargs.setdefault("big_mark", ",")
    return label_number(**kwargs)


def comma(x: ArrayLike, **kwargs: Any) -> list[str]:
    """Format *x* with comma thousands separator."""
    kwargs.setdefault("big_mark", ",")
    return number(x, **kwargs)


# ---------------------------------------------------------------------------
# label_percent / percent
# ---------------------------------------------------------------------------


def label_percent(
    accuracy: Optional[float] = None,
    scale: float = 100,
    suffix: str = "%",
    **kwargs: Any,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label percentages.

    Parameters
    ----------
    accuracy : float, optional
        Rounding precision.
    scale : float, optional
        Multiplicative factor (default 100 converts proportions to %).
    suffix : str, optional
        Appended string (default ``"%"``).
    **kwargs
        Passed to :func:`label_number`.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    return label_number(accuracy=accuracy, scale=scale, suffix=suffix, **kwargs)


def percent(x: ArrayLike, accuracy: Optional[float] = None, scale: float = 100,
            suffix: str = "%", **kwargs: Any) -> list[str]:
    """Format *x* as percentages."""
    return number(x, accuracy=accuracy, scale=scale, suffix=suffix, **kwargs)


# ---------------------------------------------------------------------------
# label_dollar / label_currency / dollar
# ---------------------------------------------------------------------------


def _needs_cents(x: np.ndarray, threshold: float) -> bool:
    """Mirror R's `needs_cents`: decide whether an auto-accuracy pass
    should use 0.01 (cents) or 1 (whole units).

    * Empty / all-NaN   → False
    * Max |x| > threshold → False (values too large; skip fractional)
    * Otherwise         → True iff **any** finite value is non-integer.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return False
    if np.nanmax(np.abs(finite)) > threshold:
        return False
    return bool(np.any(finite != np.floor(finite)))


def dollar(
    x: ArrayLike,
    accuracy: Optional[float] = None,
    scale: float = 1,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    big_mark: Optional[str] = None,
    decimal_mark: Optional[str] = None,
    trim: bool = True,
    largest_with_cents: float = 100000,
    style_negative: Optional[str] = None,
    scale_cut: Optional[list[tuple[float, str]]] = None,
    **kwargs: Any,
) -> list[str]:
    """Format *x* as currency.

    Matches R's ``dollar``:

    * Currency-specific defaults fall through ``_NUMBER_OPTIONS``
      (``currency_prefix``, ``currency_suffix``, ``currency_big_mark``,
      ``currency_decimal_mark``).  The baked-in fallbacks are ``"$"``,
      ``""``, ``","``, ``"."`` respectively.
    * When ``accuracy`` is ``None`` *and* no ``scale_cut`` is given, the
      accuracy is chosen by the ``largest_with_cents`` heuristic: use
      ``0.01`` when ``max(|x * scale|) <= largest_with_cents`` *and* any
      input has a fractional part; otherwise ``1``.
    * When ``big_mark == decimal_mark == ","``, ``big_mark`` is swapped
      to a space to avoid ambiguity.
    """
    if prefix is None:
        prefix = str(_NUMBER_OPTIONS.get("currency_prefix", "$"))
    if suffix is None:
        suffix = str(_NUMBER_OPTIONS.get("currency_suffix", ""))
    if big_mark is None:
        big_mark = str(_NUMBER_OPTIONS.get("currency_big_mark", ","))
    if decimal_mark is None:
        decimal_mark = str(_NUMBER_OPTIONS.get("currency_decimal_mark", "."))

    x_arr = np.asarray(x, dtype=float)
    if x_arr.size == 0:
        return []

    if accuracy is None and scale_cut is None:
        if _needs_cents(x_arr * scale, largest_with_cents):
            accuracy = 0.01
        else:
            accuracy = 1

    if big_mark == "," and decimal_mark == ",":
        big_mark = " "

    return number(
        x_arr,
        accuracy=accuracy,
        scale=scale,
        prefix=prefix,
        suffix=suffix,
        big_mark=big_mark,
        decimal_mark=decimal_mark,
        trim=trim,
        style_negative=style_negative,
        scale_cut=scale_cut,
        **kwargs,
    )


def label_currency(
    accuracy: Optional[float] = None,
    scale: float = 1,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    big_mark: Optional[str] = None,
    decimal_mark: Optional[str] = None,
    trim: bool = True,
    largest_with_fractional: float = 100000,
    **kwargs: Any,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label currency values.

    Thin closure around :func:`dollar` — matches R's ``label_currency``
    wrapping ``dollar(..., largest_with_cents = largest_with_fractional)``.

    Parameters
    ----------
    accuracy : float, optional
        Fixed rounding precision.  When ``None`` (default), accuracy is
        auto-detected via the ``largest_with_fractional`` heuristic.
    largest_with_fractional : float, optional
        Threshold above which fractional accuracy is suppressed
        (default ``100000``).  See :func:`dollar`.
    """

    def formatter(x: ArrayLike) -> list[str]:
        return dollar(
            x,
            accuracy=accuracy,
            scale=scale,
            prefix=prefix,
            suffix=suffix,
            big_mark=big_mark,
            decimal_mark=decimal_mark,
            trim=trim,
            largest_with_cents=largest_with_fractional,
            **kwargs,
        )

    return formatter


def label_dollar(
    accuracy: Optional[float] = None,
    scale: float = 1,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    big_mark: Optional[str] = None,
    decimal_mark: Optional[str] = None,
    trim: bool = True,
    largest_with_cents: float = 100000,
    **kwargs: Any,
) -> Callable[[ArrayLike], list[str]]:
    """Label currency values (superseded alias of :func:`label_currency`)."""

    def formatter(x: ArrayLike) -> list[str]:
        return dollar(
            x,
            accuracy=accuracy,
            scale=scale,
            prefix=prefix,
            suffix=suffix,
            big_mark=big_mark,
            decimal_mark=decimal_mark,
            trim=trim,
            largest_with_cents=largest_with_cents,
            **kwargs,
        )

    return formatter


# ---------------------------------------------------------------------------
# label_scientific / scientific
# ---------------------------------------------------------------------------


def _format_scientific_single(
    value: float,
    digits: int,
    decimal_mark: str,
    trim: bool,
) -> str:
    """Format a single value in scientific notation."""
    if not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Inf" if value > 0 else "-Inf"

    if value == 0:
        if trim:
            return "0"
        return f"0.{'0' * (digits - 1)}e+00"

    exp = int(math.floor(math.log10(abs(value))))
    coeff = value / (10.0 ** exp)
    # Round coefficient
    coeff = round(coeff, digits - 1)

    # Format coefficient
    ndecimals = max(0, digits - 1)
    coeff_str = f"{coeff:.{ndecimals}f}"

    if trim and ndecimals > 0:
        coeff_str = coeff_str.rstrip("0").rstrip(".")

    if decimal_mark != ".":
        coeff_str = coeff_str.replace(".", decimal_mark, 1)

    # Format exponent
    exp_sign = "+" if exp >= 0 else "-"
    exp_str = f"{abs(exp):02d}"

    return f"{coeff_str}e{exp_sign}{exp_str}"


def scientific(
    x: ArrayLike,
    digits: int = 3,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    decimal_mark: Optional[str] = None,
    trim: bool = True,
) -> list[str]:
    """
    Format *x* in scientific notation.

    Parameters
    ----------
    x : array-like
        Numeric values.
    digits : int, optional
        Significant digits (default 3).
    scale : float, optional
        Multiplicative factor.
    prefix : str, optional
        Prepended string.
    suffix : str, optional
        Appended string.
    decimal_mark : str, optional
        Decimal separator.
    trim : bool, optional
        Strip trailing zeros.

    Returns
    -------
    list of str
    """
    x_arr = np.asarray(x, dtype=float) * scale
    if decimal_mark is None:
        decimal_mark = "."

    results: list[str] = []
    for val in x_arr.flat:
        fmt = _format_scientific_single(val, digits, decimal_mark, trim)
        results.append(f"{prefix}{fmt}{suffix}")
    return results


def label_scientific(
    digits: int = 3,
    scale: float = 1,
    prefix: str = "",
    suffix: str = "",
    decimal_mark: Optional[str] = None,
    trim: bool = True,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label numbers in scientific notation.

    Parameters
    ----------
    digits : int, optional
        Significant digits (default 3).
    scale : float, optional
        Multiplicative factor.
    prefix : str, optional
        Prepended string.
    suffix : str, optional
        Appended string.
    decimal_mark : str, optional
        Decimal separator.
    trim : bool, optional
        Strip trailing zeros.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return scientific(
            x, digits=digits, scale=scale, prefix=prefix,
            suffix=suffix, decimal_mark=decimal_mark, trim=trim,
        )

    return formatter


# ---------------------------------------------------------------------------
# label_bytes
# ---------------------------------------------------------------------------

_SI_BYTE_UNITS = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
_BINARY_BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]


def label_bytes(
    units: str = "auto_si",
    accuracy: float = 1,
    scale: float = 1,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label byte sizes (e.g. 1 kB, 2 MiB).

    Parameters
    ----------
    units : str, optional
        ``"auto_si"`` (powers of 1000), ``"auto_binary"`` (powers of 1024),
        or an explicit unit name like ``"kB"`` or ``"MiB"``.
    accuracy : float, optional
        Rounding precision (default 1).
    scale : float, optional
        Multiplicative factor applied before formatting.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x, dtype=float) * scale
        results: list[str] = []

        for val in x_arr.flat:
            if not np.isfinite(val):
                results.append("NaN" if np.isnan(val) else ("Inf" if val > 0 else "-Inf"))
                continue

            if units == "auto_si":
                base = 1000
                unit_list = _SI_BYTE_UNITS
                idx = 0
                abs_val = abs(val)
                while abs_val >= base and idx < len(unit_list) - 1:
                    abs_val /= base
                    idx += 1
                divisor = base ** idx
                scaled_val = val / divisor if divisor else val
                unit_str = unit_list[idx]
            elif units == "auto_binary":
                base = 1024
                unit_list = _BINARY_BYTE_UNITS
                idx = 0
                abs_val = abs(val)
                while abs_val >= base and idx < len(unit_list) - 1:
                    abs_val /= base
                    idx += 1
                divisor = base ** idx
                scaled_val = val / divisor if divisor else val
                unit_str = unit_list[idx]
            else:
                # Explicit unit
                unit_str = units
                if units in _SI_BYTE_UNITS:
                    idx = _SI_BYTE_UNITS.index(units)
                    divisor = 1000 ** idx
                elif units in _BINARY_BYTE_UNITS:
                    idx = _BINARY_BYTE_UNITS.index(units)
                    divisor = 1024 ** idx
                else:
                    divisor = 1
                scaled_val = val / divisor if divisor else val

            fmt = _format_number(scaled_val, accuracy, None, ".", True)
            results.append(f"{fmt} {unit_str}")

        return results

    return formatter


# ---------------------------------------------------------------------------
# Ordinal helpers
# ---------------------------------------------------------------------------


class OrdinalRules(list):
    """R-style ordinal rule set: an ordered list of ``(suffix, regex)``.

    Iterating yields ``(suffix, pattern)`` pairs in priority order — this
    mirrors R's ``ordinal_english()`` return value (a named list of
    regex strings where the **name** is the suffix and the **value** is
    the pattern).

    For backwards compatibility, instances are *also callable*: calling
    with an integer returns the first matching suffix (Python's
    historical API).
    """

    __slots__ = ()

    def __call__(self, n: Any) -> str:
        import re as _re
        s = str(int(n))
        for suffix, pattern in self:
            if _re.search(pattern, s):
                return suffix
        return ""


def ordinal_english() -> OrdinalRules:
    """
    Return the English ordinal rule set.

    Mirrors R's ``ordinal_english``: a list of ``(suffix, regex)`` pairs
    applied in order, first match wins.  Handles the 11/12/13 quirk via
    lookbehind assertions.

    Returns
    -------
    OrdinalRules
        Priority-ordered ``[(suffix, pattern), ...]``, also callable as
        ``rules(n) -> suffix``.
    """
    return OrdinalRules([
        ("st", r"(?<!1)1$"),
        ("nd", r"(?<!1)2$"),
        ("rd", r"(?<!1)3$"),
        ("th", r"(?<=1)[123]$"),
        ("th", r"[0456789]$"),
        ("th", r"."),
    ])


def ordinal_french(
    gender: str = "masculin",
    plural: bool = False,
) -> OrdinalRules:
    """
    Return the French ordinal rule set.

    Mirrors R's ``ordinal_french``: only ``1`` gets the masculine/feminine
    first-position suffix (``"er"`` / ``"re"``); everything else gets
    ``"e"``.  When ``plural`` is ``True``, both suffixes receive a
    trailing ``"s"``.

    Parameters
    ----------
    gender : str, optional
        ``"masculin"`` or ``"feminin"`` (default ``"masculin"``).
    plural : bool, optional
        Use plural forms (default ``False``).

    Returns
    -------
    OrdinalRules
    """
    if gender not in ("masculin", "feminin"):
        raise ValueError("gender must be 'masculin' or 'feminin'")
    first = "er" if gender == "masculin" else "re"
    rest = "e"
    if plural:
        first += "s"
        rest += "s"
    return OrdinalRules([(first, r"^1$"), (rest, r".")])


def ordinal_spanish() -> OrdinalRules:
    """
    Return the Spanish ordinal rule set.

    Mirrors R's ``ordinal_spanish``: a single rule with suffix ``".º"``
    matching every number.
    """
    return OrdinalRules([(".\u00ba", r".")])


# ---------------------------------------------------------------------------
# label_ordinal / ordinal
# ---------------------------------------------------------------------------


def ordinal(
    x: ArrayLike,
    prefix: str = "",
    suffix: str = "",
    big_mark: Optional[str] = None,
    rules: Optional[Any] = None,
) -> list[str]:
    """
    Format *x* as ordinals (1st, 2nd, 3rd, ...).

    Parameters
    ----------
    x : array-like
        Numeric values (rounded to integers before formatting).
    prefix : str, optional
        Prepended string.
    suffix : str, optional
        Appended string.
    big_mark : str, optional
        Thousands separator.
    rules : :class:`OrdinalRules` or callable, optional
        Suffix rule set.  May be:

        * An :class:`OrdinalRules` instance (R-style list of
          ``(suffix, regex)``) — first match wins.
        * Any iterable of ``(suffix, regex)`` pairs — same behaviour.
        * A plain callable ``(int) -> str``.
        * ``None`` — defaults to :func:`ordinal_english` as in R.

    Returns
    -------
    list of str
    """
    import re as _re

    if rules is None:
        rules = ordinal_english()

    # Normalise the rule set into a callable that maps int -> suffix.
    if callable(rules) and not isinstance(rules, (list, tuple)):
        rule_fn = rules
    else:
        rule_list = list(rules)

        def rule_fn(n: int) -> str:
            s = str(int(n))
            for sfx, pat in rule_list:
                if _re.search(pat, s):
                    return sfx
            return ""

    x_arr = np.asarray(x, dtype=float)
    results: list[str] = []

    for val in x_arr.flat:
        if not np.isfinite(val):
            results.append("NaN" if np.isnan(val) else ("Inf" if val > 0 else "-Inf"))
            continue
        int_val = int(round(val))
        num_str = _format_number(float(int_val), 1, big_mark, ".", True)
        ord_suffix = rule_fn(int_val)
        results.append(f"{prefix}{num_str}{ord_suffix}{suffix}")

    return results


def label_ordinal(
    prefix: str = "",
    suffix: str = "",
    big_mark: Optional[str] = None,
    rules: Optional[Callable[[int], str]] = None,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label numbers as ordinals.

    Parameters
    ----------
    prefix : str, optional
        Prepended string.
    suffix : str, optional
        Appended string.
    big_mark : str, optional
        Thousands separator.
    rules : callable, optional
        Ordinal suffix function (default :func:`ordinal_english`).

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return ordinal(x, prefix=prefix, suffix=suffix,
                        big_mark=big_mark, rules=rules)

    return formatter


# ---------------------------------------------------------------------------
# label_pvalue / pvalue
# ---------------------------------------------------------------------------


def pvalue(
    x: ArrayLike,
    accuracy: float = 0.001,
    decimal_mark: Optional[str] = None,
    prefix: Optional[list[str]] = None,
    add_p: bool = False,
) -> list[str]:
    """
    Format p-values.

    Parameters
    ----------
    x : array-like
        P-values to format.
    accuracy : float, optional
        Smallest displayable value (default 0.001).
    decimal_mark : str, optional
        Decimal separator.
    prefix : list of str, optional
        Length-3 list ``[less_than, normal, greater_than]``.
        Defaults to ``["<", "", ">"]``.
    add_p : bool, optional
        If ``True``, prepend ``"p"`` or ``"p "`` to each label.

    Returns
    -------
    list of str
    """
    # Mirrors R's pvalue: prefix default is ("p<","p=","p>") if add_p else
    # ("<","",">"). The prefix is prepended verbatim — *no* added spaces.
    if prefix is None:
        prefix_list = ["p<", "p=", "p>"] if add_p else ["<", "", ">"]
    else:
        prefix_list = list(prefix)
        if len(prefix_list) != 3:
            raise ValueError("prefix must be a length-3 sequence")

    if decimal_mark is None:
        decimal_mark = str(_NUMBER_OPTIONS.get("decimal_mark", "."))

    x_arr = np.asarray(x, dtype=float)
    results: list[str] = []

    for val in x_arr.flat:
        if not np.isfinite(val):
            results.append("NaN" if np.isnan(val) else str(val))
            continue

        if val < accuracy:
            fmt = _format_number(accuracy, accuracy, None, decimal_mark, True)
            s = f"{prefix_list[0]}{fmt}"
        elif val > 1 - accuracy:
            fmt = _format_number(1 - accuracy, accuracy, None, decimal_mark, True)
            s = f"{prefix_list[2]}{fmt}"
        else:
            fmt = _format_number(val, accuracy, None, decimal_mark, False)
            s = f"{prefix_list[1]}{fmt}"

        results.append(s)

    return results


def label_pvalue(
    accuracy: float = 0.001,
    decimal_mark: Optional[str] = None,
    prefix: Optional[list[str]] = None,
    add_p: bool = False,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label p-values.

    Parameters
    ----------
    accuracy : float, optional
        Smallest displayable value (default 0.001).
    decimal_mark : str, optional
        Decimal separator.
    prefix : list of str, optional
        Length-3 list ``[less_than, normal, greater_than]``.
    add_p : bool, optional
        Prepend ``"p"`` to labels.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return pvalue(x, accuracy=accuracy, decimal_mark=decimal_mark,
                       prefix=prefix, add_p=add_p)

    return formatter


# ---------------------------------------------------------------------------
# Date / time labels
# ---------------------------------------------------------------------------


def _to_datetime(val: Any, tz_obj: Any) -> Optional[datetime]:
    """Convert a value to a datetime, handling numpy datetime64 etc."""
    if isinstance(val, datetime):
        return val.astimezone(tz_obj) if val.tzinfo else val.replace(tzinfo=tz_obj)
    if isinstance(val, np.datetime64):
        # Convert to Python datetime via timestamp
        ts = (val - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(float(ts), tz=tz_obj)
    if isinstance(val, (int, float)):
        if not np.isfinite(val):
            return None
        return datetime.fromtimestamp(float(val), tz=tz_obj)
    return None


def _make_tz(tz: str) -> timezone:
    """Create a timezone from a string. Supports 'UTC' and offset forms."""
    if tz.upper() == "UTC":
        return timezone.utc
    # Simple offset parsing for common cases
    m = re.match(r"^UTC([+-])(\d{1,2}):?(\d{2})?$", tz, re.IGNORECASE)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        minutes = int(m.group(3) or 0)
        return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
    # Fallback: try as UTC
    return timezone.utc


def label_date(
    format: str = "%Y-%m-%d",
    tz: str = "UTC",
) -> Callable[[ArrayLike], list[str]]:
    """
    Label dates.

    Parameters
    ----------
    format : str, optional
        ``strftime``-compatible format string (default ``"%Y-%m-%d"``).
    tz : str, optional
        Timezone (default ``"UTC"``).

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    tz_obj = _make_tz(tz)

    def formatter(x: ArrayLike) -> list[str]:
        results: list[str] = []
        x_arr = np.asarray(x)
        for val in x_arr.flat:
            dt = _to_datetime(val, tz_obj)
            if dt is None:
                results.append("NA")
            else:
                results.append(dt.strftime(format))
        return results

    return formatter


def label_date_short(
    format: Optional[Sequence[Optional[str]]] = None,
    sep: str = "\n",
    leading: str = "0",
    tz: str = "UTC",
    locale: Optional[str] = None,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label dates compactly, showing each component only when it changes.

    Faithful port of R's ``label_date_short``:

    * ``format`` is a length-4 vector of ``strftime`` codes for
      ``(year, month, day, hour)``.  Default
      ``("%Y", "%b", "%d", "%H:%M")``.
    * A component is rendered for a given date only if that component
      (or any *larger* component) differs from the previous date — i.e.
      ``cumsum(changed(component)) >= 1`` per R.
    * Components that are *always* zero / first-of-period across the
      whole input are trimmed from the smallest up (e.g. the hour line
      is dropped when every value is at 00:00).
    * ``leading`` controls the character replacing a leading ``"0"`` in
      each rendered component.  ``"0"`` keeps the zero (default);
      ``""`` removes it; ``"\\u2007"`` is a typographic figure space.

    Parameters
    ----------
    format : sequence of 4 str, optional
        ``(year_fmt, month_fmt, day_fmt, hour_fmt)``.
    sep : str, optional
        Separator between rendered components (default newline).
    leading : str, optional
        Replacement for each component's leading ``"0"`` digit
        (default ``"0"``, i.e. no replacement).
    tz : str, optional
        Timezone name (default ``"UTC"``).
    locale : str, optional
        Locale name.  Not implemented for month/day names (Python's
        ``strftime`` respects ``LC_TIME`` at the OS level); accepted for
        API parity.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    _ = locale  # accepted for API parity; OS-level locale is used
    tz_obj = _make_tz(tz)
    default_format = ["%Y", "%b", "%d", "%H:%M"]
    fmt = list(default_format) if format is None else list(format)
    if len(fmt) != 4:
        raise ValueError("format must be length 4 (year, month, day, hour)")

    def formatter(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x)
        dts: list[Optional[datetime]] = [
            _to_datetime(v, tz_obj) for v in x_arr.flat
        ]

        n = len(dts)
        if n == 0:
            return []

        # Extract year/month/day/hour/minute arrays; None for NA.
        year = [d.year if d is not None else None for d in dts]
        month = [d.month if d is not None else None for d in dts]
        day = [d.day if d is not None else None for d in dts]
        hour = [d.hour if d is not None else None for d in dts]
        minute = [d.minute if d is not None else None for d in dts]

        # changed[i]: True if i-th value differs from (i-1)-th, always
        # True at i==0 or when either neighbour is NA. Matches R's
        # `changed <- function(x) c(TRUE, is.na(x[-1]) | x[-1] != x[-1])`.
        def _changed(vals: list[Any]) -> list[bool]:
            out = [True] * n
            for i in range(1, n):
                if vals[i] is None or vals[i - 1] is None:
                    out[i] = True
                else:
                    out[i] = vals[i] != vals[i - 1]
            return out

        ch_year = _changed(year)
        ch_month = _changed(month)
        ch_day = _changed(day)

        # R's `cumsum(changes) >= 1` ensures that once a larger unit
        # changes, all smaller units are re-shown for that row.
        cum_year = [c for c in ch_year]
        cum_month = [(cy or cm) for cy, cm in zip(cum_year, ch_month)]
        cum_day = [(cym or cd) for cym, cd in zip(cum_month, ch_day)]

        # Decide which *positions* are worth ever showing (the "firsts"
        # trim).  Matches R's nested if-block.
        show_hour = not all(
            (h == 0 and m == 0) for h, m in zip(hour, minute) if h is not None
        )
        show_day = show_hour or not all(
            d == 1 for d in day if d is not None
        )
        show_month = show_day or not all(
            mo == 1 for mo in month if mo is not None
        )

        fmt_year = fmt[0]
        fmt_month = fmt[1] if show_month else None
        fmt_day = fmt[2] if show_day else None
        fmt_hour = fmt[3] if show_hour else None

        def _rstrip_leading(s: str) -> str:
            if leading == "0":
                return s
            # Replace a leading "0" digit in the whole string, and any
            # "0" directly after a separator (matches R's gsub).
            out = re.sub(r"^0", leading, s)
            if sep:
                out = out.replace(sep + "0", sep + leading)
            return out

        results: list[str] = []
        for i, dt in enumerate(dts):
            if dt is None:
                results.append("NA")
                continue

            parts: list[str] = []
            if fmt_hour is not None:
                parts.append(dt.strftime(fmt_hour))
            if fmt_day is not None and cum_day[i]:
                parts.append(dt.strftime(fmt_day))
            if fmt_month is not None and cum_month[i]:
                parts.append(dt.strftime(fmt_month))
            if cum_year[i]:
                parts.append(dt.strftime(fmt_year))

            # R builds the matrix with smallest-first then reverses when
            # joining — i.e. largest unit first visually.  We built
            # smallest-first too, so reverse here.
            parts = list(reversed(parts))
            results.append(_rstrip_leading(sep.join(parts)))

        return results

    return formatter


def label_time(
    format: str = "%H:%M:%S",
    tz: str = "UTC",
) -> Callable[[ArrayLike], list[str]]:
    """
    Label times.

    Parameters
    ----------
    format : str, optional
        ``strftime``-compatible format string (default ``"%H:%M:%S"``).
    tz : str, optional
        Timezone (default ``"UTC"``).

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    tz_obj = _make_tz(tz)

    def formatter(x: ArrayLike) -> list[str]:
        results: list[str] = []
        x_arr = np.asarray(x)
        for val in x_arr.flat:
            dt = _to_datetime(val, tz_obj)
            if dt is None:
                results.append("NA")
            else:
                results.append(dt.strftime(format))
        return results

    return formatter


def label_timespan(
    unit: str = "secs",
    space: bool = False,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label timespans with human-friendly units.

    Parameters
    ----------
    unit : str, optional
        Input unit: ``"secs"``, ``"mins"``, ``"hours"``, ``"days"``,
        ``"weeks"`` (default ``"secs"``).
    space : bool, optional
        Insert space between number and unit (default ``False``).

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    unit_seconds: dict[str, float] = {
        "secs": 1,
        "mins": 60,
        "hours": 3600,
        "days": 86400,
        "weeks": 604800,
    }

    # Unicode \u03bc mirrors R's cut_time_scale when UTF-8 is active.
    _thresholds = [
        (604800, "w"),
        (86400, "d"),
        (3600, "h"),
        (60, "m"),
        (1, "s"),
        (1e-3, "ms"),
        (1e-6, "\u03bcs"),
        (1e-9, "ns"),
    ]

    def formatter(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x, dtype=float)
        scale_factor = unit_seconds.get(unit, 1)
        x_secs = x_arr * scale_factor
        sp = " " if space else ""

        results: list[str] = []
        for val in x_secs.flat:
            if not np.isfinite(val):
                results.append("NaN" if np.isnan(val) else str(val))
                continue

            if val == 0:
                results.append(f"0{sp}s")
                continue

            abs_val = abs(val)
            chosen_div = 1.0
            chosen_unit = "s"
            for threshold, u in _thresholds:
                if abs_val >= threshold:
                    chosen_div = threshold
                    chosen_unit = u
                    break
            else:
                # Smaller than 1 ns
                chosen_div = 1e-9
                chosen_unit = "ns"

            scaled = val / chosen_div
            fmt = _format_number(scaled, _precision(np.array([scaled])), None, ".", True)
            results.append(f"{fmt}{sp}{chosen_unit}")

        return results

    return formatter


# ---------------------------------------------------------------------------
# label_wrap
# ---------------------------------------------------------------------------


def label_wrap(width: int) -> Callable[[ArrayLike], list[str]]:
    """
    Wrap label text at *width* characters.

    Parameters
    ----------
    width : int
        Maximum line width.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        if isinstance(x, str):
            x = [x]
        return [textwrap.fill(str(v), width=width) for v in np.asarray(x).flat]

    return formatter


# ---------------------------------------------------------------------------
# label_glue
# ---------------------------------------------------------------------------


def label_glue(
    pattern: str = "{x}",
) -> Callable[[ArrayLike], list[str]]:
    """
    Label with ``str.format``-style patterns.

    Parameters
    ----------
    pattern : str, optional
        Format pattern where ``{x}`` is replaced with the value
        (default ``"{x}"``).

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return [pattern.format(x=v) for v in np.asarray(x).flat]

    return formatter


# ---------------------------------------------------------------------------
# label_parse / label_math
# ---------------------------------------------------------------------------


def label_parse() -> Callable[[ArrayLike], list[str]]:
    """
    Return labels as-is (identity formatter).

    In R this parses plotmath expressions; in Python it simply converts
    values to strings.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        return [str(v) for v in np.asarray(x).flat]

    return formatter


def label_math(
    expr: Optional[str] = None,
    format_func: Optional[Callable[[ArrayLike], list[str]]] = None,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label with mathematical formatting.

    In R this wraps with plotmath expressions. In Python it applies
    *format_func* first and then optionally wraps each result with *expr*.

    Parameters
    ----------
    expr : str, optional
        A pattern containing ``{x}`` to wrap each label.
    format_func : callable, optional
        Pre-formatter applied to values before wrapping.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        if format_func is not None:
            labels = format_func(x)
        else:
            labels = [str(v) for v in np.asarray(x).flat]
        if expr is not None:
            labels = [expr.format(x=lbl) for lbl in labels]
        return labels

    return formatter


# ---------------------------------------------------------------------------
# label_log / format_log
# ---------------------------------------------------------------------------


def format_log(
    x: ArrayLike,
    base: float = 10,
    signed: Optional[bool] = None,
    digits: int = 3,
) -> list[str]:
    """
    Format values as log expressions (e.g. ``"10^3"``).

    Mirrors R's ``scales::format_log``: accepts **raw values** ``x`` and
    internally computes ``log(x, base)`` to obtain the exponent.

    Parameters
    ----------
    x : array-like
        Raw numeric values (not already-logged).
    base : float, optional
        Logarithmic base (default 10).
    signed : bool, optional
        If ``None`` (default), sign prefixes are shown when any finite
        value is ``<= 0``.  ``True`` forces signed formatting; ``False``
        disables it.
    digits : int, optional
        Significant digits for the exponent (default 3).

    Returns
    -------
    list of str
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size == 0:
        return []

    n = x_arr.size
    flat = x_arr.flatten()
    prefix = [""] * n

    finite = flat[np.isfinite(flat)]
    if signed is None:
        signed = bool(np.any(finite <= 0))

    signs = np.zeros(n, dtype=int)
    if signed:
        # sign(NaN) = 0 for our purposes; don't overwrite NaNs
        for i, v in enumerate(flat):
            if np.isnan(v):
                continue
            if v > 0:
                signs[i] = 1
                prefix[i] = "+"
            elif v < 0:
                signs[i] = -1
                prefix[i] = "-"
            else:
                signs[i] = 0
        flat = np.abs(flat)
        flat = np.where(flat == 0, 1.0, flat)

    base_str = str(int(base)) if float(base).is_integer() else str(base)
    log_base = math.log(base)

    def _zapsmall(v: float, tol: float = 1e-10) -> float:
        # Match R's zapsmall: values close to zero become exactly zero.
        return 0.0 if abs(v) < tol else v

    def _fmt_exponent(v: float) -> str:
        v = _zapsmall(v)
        if float(v).is_integer():
            return str(int(v))
        # R's format(x, digits=3) uses significant digits.
        return f"{v:.{max(digits - 1, 0)}g}" if digits > 0 else f"{v:g}"

    results: list[str] = []
    for i, v in enumerate(flat):
        if np.isnan(v):
            results.append("NaN")
            continue
        if np.isinf(v):
            results.append(str(v))
            continue
        exponent = math.log(v) / log_base
        exponent_str = _fmt_exponent(exponent)
        text = f"{prefix[i]}{base_str}^{exponent_str}"
        if signed and signs[i] == 0:
            text = "0"
        results.append(text)
    return results


def label_log(
    base: float = 10,
    digits: int = 3,
    signed: Optional[bool] = None,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label values on a log scale (e.g. ``"10^3"``).

    Parameters
    ----------
    base : float, optional
        Logarithmic base (default 10).
    digits : int, optional
        Significant digits for non-integer exponents (default 3).
    signed : bool, optional
        Show sign on exponents.  When ``None`` (default), signs appear
        whenever any finite input is ``<= 0``.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x, dtype=float)
        text = format_log(x_arr, base=base, signed=signed, digits=digits)
        # Restore NaN labels like R's label_log (ret[is.na(x)] <- NA).
        for i, v in enumerate(x_arr.flatten()):
            if np.isnan(v):
                text[i] = "NaN"
        return text

    return formatter


# ---------------------------------------------------------------------------
# label_number_auto
# ---------------------------------------------------------------------------


def label_number_auto() -> Callable[[ArrayLike], list[str]]:
    """
    Automatically choose between regular and scientific notation.

    Switches to scientific notation when values are very large or very small.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x, dtype=float)
        finite = x_arr[np.isfinite(x_arr)]
        if len(finite) == 0:
            return number(x_arr)

        abs_max = np.max(np.abs(finite)) if len(finite) else 0
        abs_min_nonzero = np.min(np.abs(finite[finite != 0])) if np.any(finite != 0) else 1

        # Use scientific if range spans many orders of magnitude or extreme values
        if abs_max >= 1e9 or (abs_min_nonzero > 0 and abs_min_nonzero < 1e-3):
            return scientific(x_arr)
        return number(x_arr)

    return formatter


# ---------------------------------------------------------------------------
# label_number_si (deprecated)
# ---------------------------------------------------------------------------


def label_number_si(
    unit: str = "",
    accuracy: Optional[float] = None,
    scale: float = 1,
    suffix: str = "",
) -> Callable[[ArrayLike], list[str]]:
    """
    Label numbers with SI prefixes (deprecated).

    Use :func:`label_number` with ``scale_cut=cut_si(unit)`` instead.

    Parameters
    ----------
    unit : str, optional
        Base unit.
    accuracy : float, optional
        Rounding precision.
    scale : float, optional
        Multiplicative factor.
    suffix : str, optional
        Extra suffix after SI unit.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    import warnings
    warnings.warn(
        "label_number_si() is deprecated. Use label_number(scale_cut=cut_si(unit)) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return label_number(
        accuracy=accuracy,
        scale=scale,
        suffix=suffix,
        scale_cut=cut_si(unit),
    )


# ---------------------------------------------------------------------------
# label_dictionary
# ---------------------------------------------------------------------------


def label_dictionary(
    dictionary: Optional[Dict[Any, str]] = None,
    nomatch: Optional[str] = None,
) -> Callable[[ArrayLike], list[str]]:
    """
    Label values by dictionary lookup.

    Parameters
    ----------
    dictionary : dict, optional
        Mapping from data values to label strings.
    nomatch : str, optional
        Fallback string when a value is not in the dictionary.
        ``None`` means the original value is converted to string.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    if dictionary is None:
        dictionary = {}

    def formatter(x: ArrayLike) -> list[str]:
        results: list[str] = []
        for val in np.asarray(x).flat:
            # Try various key types
            key = val.item() if hasattr(val, "item") else val
            if key in dictionary:
                results.append(str(dictionary[key]))
            elif nomatch is not None:
                results.append(nomatch)
            else:
                results.append(str(key))
        return results

    return formatter


# ---------------------------------------------------------------------------
# compose_label
# ---------------------------------------------------------------------------


def compose_label(
    *formatters: Callable[[ArrayLike], list[str]],
) -> Callable[[ArrayLike], list[str]]:
    """
    Compose multiple label formatters, applying them in sequence.

    Each formatter receives the output of the previous one.

    Parameters
    ----------
    *formatters : callable
        Label functions to compose.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """

    def formatter(x: ArrayLike) -> list[str]:
        result = x
        for f in formatters:
            result = f(result)
        return result  # type: ignore[return-value]

    return formatter


# ---------------------------------------------------------------------------
# unit_format
# ---------------------------------------------------------------------------


def unit_format(
    unit: str = "m",
    scale: float = 1,
    sep: str = " ",
    **kwargs: Any,
) -> Callable[[ArrayLike], list[str]]:
    """
    Append a unit to formatted numbers.

    Parameters
    ----------
    unit : str, optional
        Unit string (default ``"m"``).
    scale : float, optional
        Multiplicative factor.
    sep : str, optional
        Separator between number and unit (default ``" "``).
    **kwargs
        Passed to :func:`label_number`.

    Returns
    -------
    callable
        ``(x) -> list[str]``
    """
    return label_number(scale=scale, suffix=f"{sep}{unit}", **kwargs)


# ---------------------------------------------------------------------------
# Date utility aliases
# ---------------------------------------------------------------------------


def date_breaks(width: str) -> Callable:
    """
    Return a break function for date axes.

    Parameters
    ----------
    width : str
        Break width specification (e.g. ``"1 month"``). Delegates to
        ``breaks_width``.

    Returns
    -------
    callable
    """
    from .breaks import breaks_width
    return breaks_width(width)


def date_format(
    format: str = "%Y-%m-%d",
    tz: str = "UTC",
) -> Callable[[ArrayLike], list[str]]:
    """Alias for :func:`label_date`."""
    return label_date(format=format, tz=tz)


def time_format(
    format: str = "%H:%M:%S",
    tz: str = "UTC",
) -> Callable[[ArrayLike], list[str]]:
    """Alias for :func:`label_time`."""
    return label_time(format=format, tz=tz)


# ---------------------------------------------------------------------------
# Legacy aliases
# ---------------------------------------------------------------------------

comma_format = label_comma
dollar_format = label_dollar
percent_format = label_percent
scientific_format = label_scientific
ordinal_format = label_ordinal
pvalue_format = label_pvalue
number_format = label_number
number_bytes_format = label_bytes
number_bytes = label_bytes
parse_format = label_parse
math_format = label_math
wrap_format = label_wrap
format_format = label_glue


# ---------------------------------------------------------------------------
# Global number-formatting options
# ---------------------------------------------------------------------------

# Module-level option store – mirrors R's `options(scales.*)` mechanism.
# NOTE on `big_mark`: the default is an empty string rather than R's
# figure space `" "`. This is an intentional Python divergence decided
# 2026-04-16 so that label output round-trips through float(). Users
# wanting R's visual style call `number_options(big_mark=" ")`.
_NUMBER_OPTIONS: dict[str, object] = {
    "decimal_mark": ".",
    "big_mark": "",
    "style_positive": "none",
    "style_negative": "hyphen",
    "currency_prefix": "$",
    "currency_suffix": "",
    "currency_decimal_mark": ".",
    "currency_big_mark": ",",
}


def number_options(
    decimal_mark: str = ".",
    big_mark: str = "",
    style_positive: str = "none",
    style_negative: str = "hyphen",
    currency_prefix: str = "$",
    currency_suffix: str = "",
    currency_decimal_mark: str | None = None,
    currency_big_mark: str | None = None,
) -> dict[str, object]:
    """Set global default options for number formatting.

    In R this sets ``options(scales.*)``.  In Python the values are
    stored in the module-level ``_NUMBER_OPTIONS`` dict and can be
    read by label functions that wish to honour defaults.

    Calling with no arguments resets all options to their defaults.

    Parameters
    ----------
    decimal_mark : str
        Default decimal separator.
    big_mark : str
        Default thousands separator.
    style_positive : str
        How to display positive numbers: ``"none"``, ``"plus"``, or ``"space"``.
    style_negative : str
        How to display negative numbers: ``"hyphen"``, ``"minus"``, or ``"parens"``.
    currency_prefix : str
        Default currency prefix.
    currency_suffix : str
        Default currency suffix.
    currency_decimal_mark : str or None
        Decimal mark for currency (defaults to *decimal_mark*).
    currency_big_mark : str or None
        Big mark for currency (defaults to ``","`` if *currency_decimal_mark*
        is ``"."``, else ``"."``).

    Returns
    -------
    dict
        Previous option values (before this call changed them).
    """
    prev = dict(_NUMBER_OPTIONS)

    if currency_decimal_mark is None:
        currency_decimal_mark = decimal_mark
    if currency_big_mark is None:
        currency_big_mark = "," if currency_decimal_mark == "." else "."

    _NUMBER_OPTIONS.update(
        decimal_mark=decimal_mark,
        big_mark=big_mark,
        style_positive=style_positive,
        style_negative=style_negative,
        currency_prefix=currency_prefix,
        currency_suffix=currency_suffix,
        currency_decimal_mark=currency_decimal_mark,
        currency_big_mark=currency_big_mark,
    )
    return prev
