"""
Palette functions for the scales package.

Python port of the R scales palette system, covering:
  - R/pal-.R (core palette classes)
  - R/pal-brewer.R (ColorBrewer palettes)
  - R/pal-hue.R (HCL hue palettes)
  - R/pal-viridis.R (viridis family)
  - R/pal-gradient.R (gradient palettes)
  - R/pal-grey.R (grey palettes)
  - R/pal-area.R (area scaling)
  - R/pal-shape.r (shape palettes)
  - R/pal-linetype.R (linetype palettes)
  - R/pal-identity.R (identity palette)
  - R/pal-manual.R (manual palettes)
  - R/pal-rescale.R (rescale palettes)
  - R/pal-dichromat.R (colorblind-safe palettes)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    # Core classes
    "ContinuousPalette",
    "DiscretePalette",
    # Constructors
    "new_continuous_palette",
    "new_discrete_palette",
    # Testing / getters
    "is_pal",
    "is_continuous_pal",
    "is_discrete_pal",
    "is_colour_pal",
    "is_numeric_pal",
    "palette_nlevels",
    "palette_na_safe",
    "palette_type",
    # Coercion
    "as_discrete_pal",
    "as_continuous_pal",
    # Registry (port of palette-registry.R)
    "register_palette",
    "get_palette",
    "palette_names",
    "reset_palettes",
    # Discrete palette factories
    "pal_brewer",
    "pal_hue",
    "pal_viridis",
    "pal_grey",
    "pal_shape",
    "pal_linetype",
    "pal_identity",
    "pal_manual",
    "pal_dichromat",
    # Continuous palette factories
    "pal_gradient_n",
    "pal_div_gradient",
    "pal_seq_gradient",
    "pal_area",
    "pal_rescale",
    "abs_area",
    # Legacy aliases
    "brewer_pal",
    "hue_pal",
    "viridis_pal",
    "grey_pal",
    "shape_pal",
    "linetype_pal",
    "identity_pal",
    "manual_pal",
    "dichromat_pal",
    "gradient_n_pal",
    "div_gradient_pal",
    "seq_gradient_pal",
    "area_pal",
    "rescale_pal",
]


# ---------------------------------------------------------------------------
# HCL -> RGB helper
# ---------------------------------------------------------------------------

def _hcl_to_hex(h: float, c: float, l: float) -> str:
    """
    Convert a single HCL (CIE-LCh(uv) / polarLUV) colour to a hex string.

    Mirrors R ``grDevices::hcl()`` which uses ``polarLUV()`` — i.e.
    the **CIE L\\*C\\*h(uv)** cylindrical representation of **CIE L\\*u\\*v\\***,
    not L\\*C\\*h(ab) / L\\*a\\*b\\*.  The two spaces differ: using Lab for R's
    hue_pal() produces over-saturated colours (e.g. ``#FF0076`` instead
    of R's ``#F8766D``).

    Parameters
    ----------
    h : float
        Hue angle in degrees (0--360).
    c : float
        Chroma (0--100+).
    l : float
        Luminance (0--100).

    Returns
    -------
    str
        Hex colour string, e.g. ``"#F8766D"``.
    """
    # --- polarLUV -> LUV --------------------------------------------------
    h_rad = np.radians(h % 360)
    u = c * np.cos(h_rad)
    v = c * np.sin(h_rad)

    # --- LUV -> XYZ (D65 white point) ------------------------------------
    # Reference white (D65): X_n=0.95047, Y_n=1.00000, Z_n=1.08883
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    denom_n = Xn + 15.0 * Yn + 3.0 * Zn
    un_prime = 4.0 * Xn / denom_n       # ≈ 0.19783
    vn_prime = 9.0 * Yn / denom_n       # ≈ 0.46830

    # L* -> Y: piecewise (CIE L* definition)
    eps = 216.0 / 24389.0          # 0.008856
    kappa = 24389.0 / 27.0         # 903.2962962...
    if l > kappa * eps:            # l > 8
        y_rel = ((l + 16.0) / 116.0) ** 3
    else:
        y_rel = l / kappa
    y = y_rel * Yn

    # u, v -> u', v' (only valid when L* > 0)
    if l <= 0.0:
        x = y_val = z = 0.0
    else:
        u_prime = u / (13.0 * l) + un_prime
        v_prime = v / (13.0 * l) + vn_prime
        if v_prime == 0.0:
            x = y_val = z = 0.0
        else:
            x = y * (9.0 * u_prime) / (4.0 * v_prime)
            z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime)
            y_val = y

    # --- XYZ -> linear sRGB ----------------------------------------------
    rl = 3.2404542 * x - 1.5371385 * y_val - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y_val + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y_val + 1.0572252 * z

    # Gamma companding (sRGB)
    def _gamma(val: float) -> float:
        if val <= 0.0031308:
            return 12.92 * val
        return 1.055 * (val ** (1.0 / 2.4)) - 0.055

    r = np.clip(_gamma(rl), 0.0, 1.0)
    g = np.clip(_gamma(gl), 0.0, 1.0)
    b_val = np.clip(_gamma(bl), 0.0, 1.0)

    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b_val * 255)),
    )


# ---------------------------------------------------------------------------
# Embedded dichromat colour schemes
#
# Exact port of R's ``dichromat::colorschemes`` (all 17 palettes) — see
# the canonical source:
#   https://github.com/cran/dichromat/blob/master/R/colorschemes.R
# Reference: Light A, Bartlein PJ (2004) "The End of the Rainbow?",
# EOS Trans. AGU 85(40):385; data via Scott Waichler / Univ. of Oregon
# (https://geography.uoregon.edu/datagraphics/color_scales.htm).
# ---------------------------------------------------------------------------

_DICHROMAT_SCHEMES: dict[str, list[str]] = {
    "BrowntoBlue.10": [
        "#663000", "#996136", "#CC9B7A", "#D9AF98", "#F2DACE",
        "#CCFDFF", "#99F8FF", "#66F0FF", "#33E4FF", "#00AACC",
    ],
    "BrowntoBlue.12": [
        "#331A00", "#663000", "#996136", "#CC9B7A", "#D9AF98",
        "#F2DACE", "#CCFDFF", "#99F8FF", "#66F0FF", "#33E4FF",
        "#00AACC", "#007A99",
    ],
    "BluetoDarkOrange.12": [
        "#1F8F99", "#52C4CC", "#99FAFF", "#B2FCFF", "#CCFEFF",
        "#E6FFFF", "#FFE6CC", "#FFCA99", "#FFAD66", "#FF8F33",
        "#CC5800", "#994000",
    ],
    "BluetoDarkOrange.18": [
        "#006666", "#009999", "#00CCCC", "#00FFFF", "#33FFFF",
        "#66FFFF", "#99FFFF", "#B2FFFF", "#CCFFFF", "#E6FFFF",
        "#FFE6CC", "#FFCA99", "#FFAD66", "#FF8F33", "#FF6E00",
        "#CC5500", "#993D00", "#662700",
    ],
    "DarkRedtoBlue.12": [
        "#2A0BD9", "#264EFF", "#40A1FF", "#73DAFF", "#ABF8FF",
        "#E0FFFF", "#FFFFBF", "#FFE099", "#FFAD73", "#F76E5E",
        "#D92632", "#A60021",
    ],
    "DarkRedtoBlue.18": [
        "#2400D9", "#191DF7", "#2957FF", "#3D87FF", "#57B0FF",
        "#75D3FF", "#99EBFF", "#BDF9FF", "#EBFFFF", "#FFFFEB",
        "#FFF2BD", "#FFD699", "#FFAC75", "#FF7857", "#FF3D3D",
        "#F72836", "#D91630", "#A60021",
    ],
    "BluetoGreen.14": [
        "#0000FF", "#3333FF", "#6666FF", "#9999FF", "#B2B2FF",
        "#CCCCFF", "#E6E6FF", "#E6FFE6", "#CCFFCC", "#B2FFB2",
        "#99FF99", "#66FF66", "#33FF33", "#00FF00",
    ],
    "BluetoGray.8": [
        "#0099CC", "#66E6FF", "#99FFFF", "#CCFFFF", "#E6E6E6",
        "#999999", "#666666", "#333333",
    ],
    "BluetoOrangeRed.14": [
        "#085AFF", "#3377FF", "#5991FF", "#8CB2FF", "#BFD4FF",
        "#E6EEFF", "#F7FAFF", "#FFFFCC", "#FFFF99", "#FFFF00",
        "#FFCC00", "#FF9900", "#FF6600", "#FF0000",
    ],
    "BluetoOrange.10": [
        "#0055FF", "#3399FF", "#66CCFF", "#99EEFF", "#CCFFFF",
        "#FFFFCC", "#FFEE99", "#FFCC66", "#FF9933", "#FF5500",
    ],
    "BluetoOrange.12": [
        "#002BFF", "#1A66FF", "#3399FF", "#66CCFF", "#99EEFF",
        "#CCFFFF", "#FFFFCC", "#FFEE99", "#FFCC66", "#FF9933",
        "#FF661A", "#FF2B00",
    ],
    "BluetoOrange.8": [
        "#0080FF", "#4CC4FF", "#99EEFF", "#CCFFFF", "#FFFFCC",
        "#FFEE99", "#FFC44C", "#FF8000",
    ],
    "LightBluetoDarkBlue.10": [
        "#E6FFFF", "#CCFBFF", "#B2F2FF", "#99E6FF", "#80D4FF",
        "#66BFFF", "#4CA6FF", "#3388FF", "#1A66FF", "#0040FF",
    ],
    "LightBluetoDarkBlue.7": [
        "#FFFFFF", "#CCFDFF", "#99F8FF", "#66F0FF", "#33E4FF",
        "#00AACC", "#007A99",
    ],
    "Categorical.12": [
        "#FFBF80", "#FF8000", "#FFFF99", "#FFFF33", "#B2FF8C",
        "#33FF00", "#A6EDFF", "#1AB2FF", "#CCBFFF", "#664CFF",
        "#FF99BF", "#E61A33",
    ],
    "GreentoMagenta.16": [
        "#005100", "#008600", "#00BC00", "#00F100", "#51FF51",
        "#86FF86", "#BCFFBC", "#FFFFFF", "#FFF1FF", "#FFBCFF",
        "#FF86FF", "#FF51FF", "#F100F1", "#BC00BC", "#860086",
        "#510051",
    ],
    "SteppedSequential.5": [
        "#990F0F", "#B22D2D", "#CC5252", "#E67E7E", "#FFB2B2",
        "#99700F", "#B28B2D", "#CCA852", "#E6C77E", "#FFE8B2",
        "#1F990F", "#3CB22D", "#60CC52", "#8AE67E", "#BCFFB2",
        "#710F99", "#8B2DB2", "#A852CC", "#C77EE6", "#E9B2FF",
        "#990F20", "#B22D3C", "#CC5260", "#E67E8A", "#FFB2BC",
    ],
}

# ---------------------------------------------------------------------------
# Linetype values (matching R linetype names)
# ---------------------------------------------------------------------------

_LINETYPES: list[str] = [
    "solid",
    "dashed",
    "dotted",
    "dotdash",
    "longdash",
    "twodash",
]

# ---------------------------------------------------------------------------
# Viridis option -> matplotlib cmap name
# ---------------------------------------------------------------------------

_VIRIDIS_OPTIONS: dict[str, str] = {
    "A": "magma",
    "B": "inferno",
    "C": "plasma",
    "D": "viridis",
    "E": "cividis",
    "F": "rocket",
    "G": "mako",
    "H": "turbo",
    "magma": "magma",
    "inferno": "inferno",
    "plasma": "plasma",
    "viridis": "viridis",
    "cividis": "cividis",
    "rocket": "rocket",
    "mako": "mako",
    "turbo": "turbo",
}


# ===================================================================
# Core palette classes
# ===================================================================

class ContinuousPalette:
    """
    A palette backed by a function mapping ``[0, 1]`` to colours/values.

    Parameters
    ----------
    fun : callable
        A function that accepts an array of values in ``[0, 1]`` and returns
        a corresponding array of colours or numeric values.
    type : str
        Either ``"colour"`` or ``"numeric"``.
    na_safe : bool or None, optional
        Whether the palette function handles ``NaN`` values gracefully.
    """

    def __init__(
        self,
        fun: Callable[..., Any],
        type: str,
        na_safe: Optional[bool] = None,
    ) -> None:
        self._fun = fun
        self.type = type
        self.na_safe = na_safe

    def __call__(self, x: Any) -> Any:
        """Evaluate the palette at *x* (values in ``[0, 1]``)."""
        return self._fun(x)

    def __repr__(self) -> str:
        return (
            f"ContinuousPalette(type={self.type!r}, na_safe={self.na_safe!r})"
        )


class DiscretePalette:
    """
    A palette backed by a function mapping an integer *n* to *n* values.

    Parameters
    ----------
    fun : callable
        A function that accepts a positive integer *n* and returns a list or
        array of *n* colours or values.
    type : str
        Either ``"colour"`` or ``"numeric"``.
    nlevels : int or None, optional
        Maximum number of levels the palette supports.
    """

    def __init__(
        self,
        fun: Callable[..., Any],
        type: str,
        nlevels: Optional[int] = None,
    ) -> None:
        self._fun = fun
        self.type = type
        self.nlevels = nlevels

    def __call__(self, n: int) -> Any:
        """Return *n* palette values."""
        return self._fun(n)

    def __repr__(self) -> str:
        return (
            f"DiscretePalette(type={self.type!r}, nlevels={self.nlevels!r})"
        )


# ===================================================================
# Constructors
# ===================================================================

def new_continuous_palette(
    fun: Callable[..., Any],
    type: str,
    na_safe: Optional[bool] = None,
) -> ContinuousPalette:
    """
    Create a new continuous palette.

    Parameters
    ----------
    fun : callable
        Function mapping values in ``[0, 1]`` to colours or numbers.
    type : str
        ``"colour"`` or ``"numeric"``.
    na_safe : bool or None, optional
        Whether *fun* handles ``NaN`` gracefully.

    Returns
    -------
    ContinuousPalette
    """
    return ContinuousPalette(fun, type, na_safe=na_safe)


def new_discrete_palette(
    fun: Callable[..., Any],
    type: str,
    nlevels: Optional[int] = None,
) -> DiscretePalette:
    """
    Create a new discrete palette.

    Parameters
    ----------
    fun : callable
        Function mapping an integer *n* to a sequence of *n* values.
    type : str
        ``"colour"`` or ``"numeric"``.
    nlevels : int or None, optional
        Maximum number of levels the palette supports.

    Returns
    -------
    DiscretePalette
    """
    return DiscretePalette(fun, type, nlevels=nlevels)


# ===================================================================
# Testing / getters
# ===================================================================

def is_pal(x: Any) -> bool:
    """Return ``True`` if *x* is a palette object."""
    return isinstance(x, (ContinuousPalette, DiscretePalette))


def is_continuous_pal(x: Any) -> bool:
    """Return ``True`` if *x* is a continuous palette."""
    return isinstance(x, ContinuousPalette)


def is_discrete_pal(x: Any) -> bool:
    """Return ``True`` if *x* is a discrete palette."""
    return isinstance(x, DiscretePalette)


def is_colour_pal(x: Any) -> bool:
    """Return ``True`` if *x* is a colour palette."""
    return is_pal(x) and getattr(x, "type", None) == "colour"


def is_numeric_pal(x: Any) -> bool:
    """Return ``True`` if *x* is a numeric palette."""
    return is_pal(x) and getattr(x, "type", None) == "numeric"


def palette_nlevels(pal: DiscretePalette) -> Optional[int]:
    """
    Return the maximum number of levels for a discrete palette.

    Parameters
    ----------
    pal : DiscretePalette
        A discrete palette.

    Returns
    -------
    int or None
    """
    return getattr(pal, "nlevels", None)


def palette_na_safe(pal: ContinuousPalette) -> Optional[bool]:
    """
    Return whether a continuous palette handles ``NaN`` safely.

    Parameters
    ----------
    pal : ContinuousPalette
        A continuous palette.

    Returns
    -------
    bool or None
    """
    return getattr(pal, "na_safe", None)


def palette_type(pal: Union[ContinuousPalette, DiscretePalette]) -> str:
    """
    Return the type of a palette (``"colour"`` or ``"numeric"``).

    Parameters
    ----------
    pal : ContinuousPalette or DiscretePalette
        A palette object.

    Returns
    -------
    str
    """
    return getattr(pal, "type", "unknown")


# ===================================================================
# Coercion
# ===================================================================

def as_discrete_pal(x: Any) -> DiscretePalette:
    """
    Coerce *x* to a discrete palette.

    Parameters
    ----------
    x : DiscretePalette, ContinuousPalette, or str
        If already a :class:`DiscretePalette`, returned as-is.
        If a :class:`ContinuousPalette`, samples *n* evenly-spaced values.
        If a ``str``, looks up by name (currently supports ``"viridis"``,
        ``"brewer"`` family, and dichromat scheme names).

    Returns
    -------
    DiscretePalette
    """
    if isinstance(x, DiscretePalette):
        return x

    if isinstance(x, ContinuousPalette):
        pal_type = x.type

        def _sampler(n: int, _cont=x) -> Any:
            if n == 1:
                points = np.array([0.5])
            else:
                points = np.linspace(0.0, 1.0, n)
            return _cont(points)

        return DiscretePalette(_sampler, type=pal_type)

    if isinstance(x, str):
        # Consult the global registry (mirrors R's get_palette path).
        try:
            pal = get_palette(x)
        except KeyError as err:
            raise ValueError(f"Unknown palette name: {x!r}") from err
        if isinstance(pal, DiscretePalette):
            return pal
        if isinstance(pal, ContinuousPalette):
            # Recurse through the ContinuousPalette branch to derive a
            # discrete sampler.
            return as_discrete_pal(pal)
        raise ValueError(f"Unknown palette name: {x!r}")

    raise TypeError(f"Cannot coerce {type(x).__name__} to DiscretePalette")


def as_continuous_pal(x: Any) -> ContinuousPalette:
    """
    Coerce *x* to a continuous palette.

    Parameters
    ----------
    x : ContinuousPalette, DiscretePalette, or str
        If already a :class:`ContinuousPalette`, returned as-is.
        If a :class:`DiscretePalette`, samples a set of colours and
        interpolates between them.
        If a ``str``, looks up by name.

    Returns
    -------
    ContinuousPalette
    """
    if isinstance(x, ContinuousPalette):
        return x

    if isinstance(x, DiscretePalette):
        # Sample enough colours and build a gradient through them
        n_sample = max(palette_nlevels(x) or 7, 7)
        colours = x(n_sample)
        return pal_gradient_n(colours)

    if isinstance(x, str):
        # Consult the registry first — it may hold a continuous palette
        # (viridis variants) or a discrete one (Brewer).
        try:
            pal = get_palette(x)
        except KeyError as err:
            raise ValueError(f"Unknown palette name: {x!r}") from err
        if isinstance(pal, ContinuousPalette):
            return pal
        if isinstance(pal, DiscretePalette):
            return as_continuous_pal(pal)
        raise ValueError(f"Unknown palette name: {x!r}")

    raise TypeError(f"Cannot coerce {type(x).__name__} to ContinuousPalette")


# ===================================================================
# Discrete palette factories
# ===================================================================

def pal_brewer(
    type: str = "seq",
    palette: Union[int, str] = 1,
    direction: int = 1,
) -> DiscretePalette:
    """
    ColorBrewer palettes.

    Parameters
    ----------
    type : str, optional
        One of ``"seq"`` (sequential), ``"div"`` (diverging), or ``"qual"``
        (qualitative). Default ``"seq"``.
    palette : int or str, optional
        Palette index (1-based) or name (e.g. ``"Blues"``, ``"Set1"``).
        Default ``1``.
    direction : int, optional
        ``1`` for normal order, ``-1`` to reverse. Default ``1``.

    Returns
    -------
    DiscretePalette
    """
    from ._palettes_data import BREWER, BREWER_MAXCOLORS, BREWER_TYPES

    _TYPE_PALETTES: dict[str, list[str]] = {
        t: [k for k, v in sorted(BREWER_TYPES.items()) if v == t]
        for t in ("seq", "div", "qual")
    }

    if isinstance(palette, int):
        names = _TYPE_PALETTES.get(type, _TYPE_PALETTES["seq"])
        idx = max(0, min(palette - 1, len(names) - 1))
        cmap_name = names[idx]
    else:
        cmap_name = palette

    max_n = BREWER_MAXCOLORS.get(cmap_name, 9)
    pal_data = BREWER.get(cmap_name, BREWER["Greens"])

    def _brewer_fun(n: int) -> list[str]:
        # R: brewer.pal(n, pal) returns the pre-designed n-colour subset.
        # If n < 3, R returns the 3-colour subset (suppresses warning).
        # If n > maxcolors, R returns maxcolors colours.
        n_lookup = max(3, min(n, max_n))

        # pal_data is {n: [colours]} — exact R brewer.pal(n, name) output
        colours = list(pal_data.get(n_lookup, pal_data[max_n]))

        # Take first n colours (for n < 3 case)
        colours = colours[:min(n, max_n)]

        # Pad with None (NA) if n > maxcolors
        while len(colours) < n:
            colours.append(None)

        if direction == -1:
            colours = colours[::-1]
        return colours

    return DiscretePalette(_brewer_fun, type="colour", nlevels=max_n)


def pal_hue(
    h: Tuple[float, float] = (15, 375),
    c: float = 100,
    l: float = 65,
    h_start: float = 0,
    direction: int = 1,
) -> DiscretePalette:
    """
    Evenly-spaced HCL hue palette.

    Divides the hue range into *n* equally-spaced segments and converts each
    HCL triplet to an sRGB hex colour.

    Parameters
    ----------
    h : tuple of float, optional
        Hue range in degrees. Default ``(15, 375)``.
    c : float, optional
        Chroma. Default ``100``.
    l : float, optional
        Luminance. Default ``65``.
    h_start : float, optional
        Starting hue offset in degrees. Default ``0``.
    direction : int, optional
        ``1`` for increasing hue, ``-1`` for decreasing. Default ``1``.

    Returns
    -------
    DiscretePalette
    """

    def _hue_fun(n: int) -> list[str]:
        if n == 0:
            raise ValueError(
                "Must request at least one colour from a hue palette."
            )
        # Mirror R: only collapse the endpoint when the hue range is a
        # full 360° circle (i.e. `diff(h) %% 360 < 1`) — otherwise the
        # endpoint is a genuine boundary and must be included.
        h0, h1 = float(h[0]), float(h[1])
        if ((h1 - h0) % 360) < 1:
            h1_used = h1 - 360.0 / n
            hues = np.linspace(h0, h1_used, n)
        else:
            hues = np.linspace(h0, h1, n)
        hues = (hues + h_start) % 360
        result = [_hcl_to_hex(hue, c, l) for hue in hues]
        if direction == -1:
            result = result[::-1]
        return result

    return DiscretePalette(_hue_fun, type="colour", nlevels=255)


def pal_viridis(
    alpha: float = 1,
    begin: float = 0,
    end: float = 1,
    direction: int = 1,
    option: str = "D",
) -> DiscretePalette:
    """
    Viridis family colour palettes.

    Parameters
    ----------
    alpha : float, optional
        Opacity (0--1). Default ``1``.
    begin : float, optional
        Start of colour map range (0--1). Default ``0``.
    end : float, optional
        End of colour map range (0--1). Default ``1``.
    direction : int, optional
        ``1`` for normal, ``-1`` for reversed. Default ``1``.
    option : str, optional
        Colour map variant. One of ``"A"``/``"magma"``, ``"B"``/``"inferno"``,
        ``"C"``/``"plasma"``, ``"D"``/``"viridis"`` (default),
        ``"E"``/``"cividis"``, ``"H"``/``"turbo"``.

    Returns
    -------
    DiscretePalette
    """
    from ._palettes_data import VIRIDIS
    from ._colors import to_hex as _to_hex

    cmap_name = _VIRIDIS_OPTIONS.get(option, "viridis")
    cmap_data = VIRIDIS.get(cmap_name, VIRIDIS["viridis"])  # 256 hex colors
    n_cmap = len(cmap_data)

    def _viridis_fun(n: int) -> list[str]:
        if n == 0:
            return []

        if direction == -1:
            positions = np.linspace(end, begin, n)
        else:
            positions = np.linspace(begin, end, n)

        colours: list[str] = []
        for pos in positions:
            idx = min(int(round(pos * (n_cmap - 1))), n_cmap - 1)
            idx = max(0, idx)
            hex_col = cmap_data[idx]
            if alpha < 1:
                # Parse hex and add alpha
                from ._colors import to_rgba as _to_rgba
                r, g, b, _ = _to_rgba(hex_col)
                colours.append(_to_hex((r, g, b, alpha), keep_alpha=True))
            else:
                colours.append(hex_col.lower())
        return colours

    return DiscretePalette(_viridis_fun, type="colour")


def pal_grey(
    start: float = 0.2,
    end: float = 0.8,
) -> DiscretePalette:
    """
    Grey palette.

    Parameters
    ----------
    start : float, optional
        Starting grey level (0 = black, 1 = white). Default ``0.2``.
    end : float, optional
        Ending grey level. Default ``0.8``.

    Returns
    -------
    DiscretePalette
    """

    def _grey_fun(n: int) -> list[str]:
        if n == 0:
            return []
        if n == 1:
            levels = [(start + end) / 2.0]
        else:
            levels = np.linspace(start, end, n).tolist()
        return [
            "#{0:02X}{0:02X}{0:02X}".format(int(round(lv * 255)))
            for lv in levels
        ]

    return DiscretePalette(_grey_fun, type="colour")


def pal_shape(solid: bool = True) -> DiscretePalette:
    """
    Shape palette.

    Returns R plotting-code integers (compatible with ggplot2's shape
    scale).  Codes mirror R's ``pal_shape``:

    * ``solid=True``  → ``[16, 17, 15, 3, 7, 8]``
    * ``solid=False`` → ``[1, 2, 0, 3, 7, 8]``

    Parameters
    ----------
    solid : bool, optional
        Use solid (filled) shapes if ``True`` (default) else open shapes.

    Returns
    -------
    DiscretePalette
    """
    # Per R's pal-shape.r:
    #   solid=TRUE  -> c(16, 17, 15, 3, 7, 8)
    #   solid=FALSE -> c( 1,  2,  0, 3, 7, 8)
    shapes = [16, 17, 15, 3, 7, 8] if solid else [1, 2, 0, 3, 7, 8]
    max_n = len(shapes)

    def _shape_fun(n: int) -> list[Optional[int]]:
        if n > max_n:
            # R: cli::cli_warn(...) — warn but do not abort; positions
            # beyond max_n come back as NA (None in Python).
            warnings.warn(
                "The shape palette can deal with a maximum of 6 discrete "
                f"values because more than 6 becomes difficult to "
                f"discriminate; you have requested {n} values. Consider "
                "specifying shapes manually if you need that many.",
                stacklevel=2,
            )
            return [*shapes, *([None] * (n - max_n))]
        return shapes[:n]

    return DiscretePalette(_shape_fun, type="shape", nlevels=max_n)


def pal_linetype() -> DiscretePalette:
    """
    Linetype palette.

    Returns linetype name strings compatible with matplotlib linestyle
    specification.

    Returns
    -------
    DiscretePalette
    """
    max_n = len(_LINETYPES)

    def _linetype_fun(n: int) -> list[Optional[str]]:
        if n > max_n:
            # Per R's pal_manual (which pal_linetype wraps): warn rather
            # than abort; positions past max_n come back as NA / None.
            warnings.warn(
                f"This manual palette can handle a maximum of {max_n} "
                f"values. You have supplied {n}.",
                stacklevel=2,
            )
            return [*_LINETYPES, *([None] * (n - max_n))]
        return _LINETYPES[:n]

    return DiscretePalette(_linetype_fun, type="linetype", nlevels=max_n)


def pal_identity() -> DiscretePalette:
    """
    Identity palette.

    Returns the input values unchanged. Mirrors R's
    ``pal_identity <- function() function(x) x``: the returned palette
    is a pass-through that echoes whatever is passed to it.

    Returns
    -------
    DiscretePalette
    """

    def _identity_fun(x: Any) -> Any:
        return x

    return DiscretePalette(_identity_fun, type="numeric")


def pal_manual(
    values: Union[list[Any], dict[str, Any]],
    type: str = "colour",
) -> DiscretePalette:
    """
    Manual palette from user-supplied values.

    Parameters
    ----------
    values : list or dict
        Palette values. If a list, the first *n* entries are returned.
        If a dict, values are returned in insertion order.
    type : str, optional
        ``"colour"`` (default) or ``"numeric"``.

    Returns
    -------
    DiscretePalette
    """
    if isinstance(values, dict):
        vals = list(values.values())
    else:
        vals = list(values)

    max_n = len(vals)

    def _manual_fun(n: int) -> list[Any]:
        # Mirrors R's pal_manual: warn (don't abort) when n exceeds the
        # palette size, and pad tail positions with None (R's NA).
        if n > max_n:
            warnings.warn(
                f"This manual palette can handle a maximum of {max_n} "
                f"values. You have supplied {n}.",
                stacklevel=2,
            )
            return [*vals, *([None] * (n - max_n))]
        return vals[:n]

    return DiscretePalette(_manual_fun, type=type, nlevels=max_n)


def pal_dichromat(name: str = "Categorical.12") -> DiscretePalette:
    """
    Colorblind-safe palette from the dichromat colour schemes (Light &
    Bartlein 2004).

    Faithful port of R's ``pal_dichromat``: delegates to
    :func:`pal_manual` with ``type="colour"`` — so requesting more
    colours than the scheme provides produces a :class:`UserWarning`
    and pads the tail with ``None`` (R pads with ``NA``), rather than
    aborting.

    Parameters
    ----------
    name : str, optional
        Scheme name.  All 17 schemes from R's
        ``dichromat::colorschemes`` are supported:

        ``BrowntoBlue.10``, ``BrowntoBlue.12``,
        ``BluetoDarkOrange.12``, ``BluetoDarkOrange.18``,
        ``DarkRedtoBlue.12``, ``DarkRedtoBlue.18``,
        ``BluetoGreen.14``, ``BluetoGray.8``,
        ``BluetoOrangeRed.14``,
        ``BluetoOrange.8``, ``BluetoOrange.10``, ``BluetoOrange.12``,
        ``LightBluetoDarkBlue.7``, ``LightBluetoDarkBlue.10``,
        ``Categorical.12``, ``GreentoMagenta.16``,
        ``SteppedSequential.5``.

        Default ``"Categorical.12"``.

    Returns
    -------
    DiscretePalette
    """
    if name not in _DICHROMAT_SCHEMES:
        available = ", ".join(sorted(_DICHROMAT_SCHEMES))
        raise ValueError(
            f"Unknown dichromat scheme {name!r}. Available: {available}"
        )

    colours = list(_DICHROMAT_SCHEMES[name])
    # R: pal_dichromat <- pal_manual(pal, type="colour"). pal_manual's
    # `n > length(values)` branch warns and pads with NA.
    return pal_manual(colours, type="colour")


# ===================================================================
# Continuous palette factories
# ===================================================================

def pal_gradient_n(
    colours: Sequence[str],
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
) -> ContinuousPalette:
    """
    Gradient through *n* colours, interpolated in CIELAB colour space.

    Parameters
    ----------
    colours : sequence of str
        Colour strings defining the gradient stops.
    values : sequence of float or None, optional
        Positions of each colour in ``[0, 1]``. If ``None``, colours are
        evenly spaced.
    space : str, optional
        Colour interpolation space.  Must be ``"Lab"`` — other values
        are deprecated (matching R scales >= 0.3.0).

    Returns
    -------
    ContinuousPalette
    """
    from .colour_ramp import colour_ramp

    ramp = colour_ramp(colours)

    if values is not None:
        values_arr = np.asarray(values, dtype=float)
        if len(values_arr) != len(colours):
            raise ValueError(
                f"Length of values ({len(values_arr)}) must match "
                f"length of colours ({len(colours)})"
            )

    def _gradient_fun(x: ArrayLike) -> list[str]:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.size == 0:
            return []
        if values is not None:
            # Remap x through the custom value positions.
            # R uses approxfun(values, xs) which returns NA for
            # extrapolation (rule=1).  np.interp clamps, so we
            # must manually set out-of-range values to NaN.
            xs = np.linspace(0.0, 1.0, len(values_arr))
            lo, hi = values_arr[0], values_arr[-1]
            x_arr = np.where(
                (x_arr < lo) | (x_arr > hi), np.nan,
                np.interp(x_arr, values_arr, xs),
            )
        return ramp(x_arr)

    return ContinuousPalette(_gradient_fun, type="colour", na_safe=False)


def pal_div_gradient(
    low: str = "#2B6788",
    mid: str = "#CBCBCB",
    high: str = "#90503F",
    space: str = "Lab",
) -> ContinuousPalette:
    """
    Diverging gradient palette.

    Parameters
    ----------
    low : str, optional
        Colour for the low end. Default ``"#2B6788"``.
    mid : str, optional
        Colour for the midpoint. Default ``"#CBCBCB"``.
    high : str, optional
        Colour for the high end. Default ``"#90503F"``.
    space : str, optional
        Interpolation colour space. Default ``"Lab"``.

    Returns
    -------
    ContinuousPalette
    """
    return pal_gradient_n([low, mid, high], values=[0.0, 0.5, 1.0], space=space)


def pal_seq_gradient(
    low: str = "#2B6788",
    high: str = "#90503F",
    space: str = "Lab",
) -> ContinuousPalette:
    """
    Sequential gradient palette.

    Parameters
    ----------
    low : str, optional
        Colour for the low end. Default ``"#2B6788"``.
    high : str, optional
        Colour for the high end. Default ``"#90503F"``.
    space : str, optional
        Interpolation colour space. Default ``"Lab"``.

    Returns
    -------
    ContinuousPalette
    """
    return pal_gradient_n([low, high], space=space)


def pal_area(
    range: Tuple[float, float] = (1, 6),
) -> ContinuousPalette:
    """
    Area-scaling palette (numeric, not colour).

    Maps values in ``[0, 1]`` to ``sqrt``-scaled sizes in *range*.

    Parameters
    ----------
    range : tuple of float, optional
        ``(min_size, max_size)``. Default ``(1, 6)``.

    Returns
    -------
    ContinuousPalette
    """

    def _area_fun(x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return range[0] + np.sqrt(x) * (range[1] - range[0])

    return ContinuousPalette(_area_fun, type="numeric", na_safe=False)


def pal_rescale(
    range: Tuple[float, float] = (0.1, 1),
) -> ContinuousPalette:
    """
    Rescale palette (numeric).

    Linearly maps ``[0, 1]`` into *range*.

    Parameters
    ----------
    range : tuple of float, optional
        Target range. Default ``(0.1, 1)``.

    Returns
    -------
    ContinuousPalette
    """

    def _rescale_fun(x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return range[0] + x * (range[1] - range[0])

    return ContinuousPalette(_rescale_fun, type="numeric", na_safe=False)


def abs_area(max_val: float) -> ContinuousPalette:
    """
    Absolute-area palette.

    Maps the absolute value of input into ``[0, max_val]`` via square-root
    scaling, so that ``0`` always maps to size ``0``.

    Parameters
    ----------
    max_val : float
        Maximum output size.

    Returns
    -------
    ContinuousPalette
    """

    def _abs_area_fun(x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sqrt(np.abs(x)) * max_val

    return ContinuousPalette(_abs_area_fun, type="numeric", na_safe=False)


# ===================================================================
# Palette registry — port of R/palette-registry.R
# ===================================================================
#
# R's `as_continuous_pal("viridis")` / `as_discrete_pal("Set1")` resolve
# names via a global environment (`.known_palettes`) populated at load
# time with viridis / Brewer / dichromat / HCL palettes plus the two
# factory entries `"hue"` and `"grey"`. Python mirrors that with a
# lowercase-keyed module dict populated by `_init_palettes()`.

_KNOWN_PALETTES: dict[str, Any] = {}


def register_palette(
    name: str,
    palette: Any,
    warn_conflict: bool = True,
) -> None:
    """Register a palette under *name* in the global registry.

    Mirrors R's ``set_palette``.  Accepts any of:

    * A :class:`ContinuousPalette` / :class:`DiscretePalette`.
    * A zero-argument **factory** that returns a palette (e.g.
      :func:`pal_hue`).
    * A sequence of colour strings (wrapped via :func:`pal_manual`).

    Parameters
    ----------
    name : str
        Registry key (case-insensitive).
    palette : palette, callable, or sequence
        Value to store.
    warn_conflict : bool, optional
        Emit :class:`UserWarning` when overwriting (default ``True``).
    """
    key = name.lower()
    if warn_conflict and key in _KNOWN_PALETTES:
        warnings.warn(f"Overwriting pre-existing {name!r} palette.", stacklevel=2)
    _KNOWN_PALETTES[key] = palette


def palette_names() -> list[str]:
    """Return the sorted list of registered palette names."""
    return sorted(_KNOWN_PALETTES)


def get_palette(name: str, *args: Any, **kwargs: Any) -> Union[
    ContinuousPalette, DiscretePalette
]:
    """Look up a palette by name.

    Mirrors R's ``get_palette``:

    * If the registered value is already a palette, return it.
    * If it is a **factory** (callable but not a palette), call it with
      ``*args, **kwargs`` and return the result.
    * If it is a sequence of colour strings, wrap with :func:`pal_manual`.
    """
    key = name.lower()
    if key not in _KNOWN_PALETTES:
        raise KeyError(f"Unknown palette: {name!r}")
    val = _KNOWN_PALETTES[key]

    if is_pal(val):
        return val
    if callable(val):
        try:
            made = val(*args, **kwargs)
        except TypeError:
            # Factory rejected extra args; try with no args.
            made = val()
        if is_pal(made):
            return made
        if isinstance(made, (list, tuple)):
            return pal_manual(list(made), type="colour")
        raise ValueError(
            f"Factory for palette {name!r} did not return a palette."
        )
    if isinstance(val, (list, tuple)):
        return pal_manual(list(val), type="colour")

    raise ValueError(f"Cannot interpret registered entry for {name!r}.")


def reset_palettes() -> None:
    """Clear the registry and re-seed with the built-in R palettes."""
    _KNOWN_PALETTES.clear()
    _init_palettes()


def _init_palettes() -> None:
    """Seed the registry with R's built-in palette set.

    Mirrors R's ``init_palettes``: registers viridis, Brewer, and
    dichromat entries plus the factories ``"hue"`` and ``"grey"``.
    HCL ramps are not ported (they depend on grDevices internals).
    """
    from ._palettes_data import BREWER, VIRIDIS

    for name in VIRIDIS:
        register_palette(name, pal_viridis(option=name), warn_conflict=False)

    # R's viridis lets users pass either the full name or the letter
    # code ("D" for "viridis", "A" for "magma", ...).  Mirror that
    # convenience by registering each letter alias.
    _VIRIDIS_LETTER_ALIASES = {
        "A": "magma", "B": "inferno", "C": "plasma", "D": "viridis",
        "E": "cividis", "F": "rocket", "G": "mako", "H": "turbo",
    }
    for letter, full_name in _VIRIDIS_LETTER_ALIASES.items():
        if full_name in VIRIDIS:
            register_palette(
                letter, pal_viridis(option=full_name), warn_conflict=False
            )

    for name in BREWER:
        register_palette(name, pal_brewer(palette=name), warn_conflict=False)

    for name in _DICHROMAT_SCHEMES:
        register_palette(
            name, pal_dichromat(name=name), warn_conflict=False
        )

    # R registers pal_hue / pal_grey as factories so users can override
    # parameters via `get_palette("hue", h=c(0,90))`.
    register_palette("hue", pal_hue, warn_conflict=False)
    register_palette("grey", pal_grey, warn_conflict=False)


_init_palettes()


# ===================================================================
# Legacy aliases
# ===================================================================

brewer_pal = pal_brewer
hue_pal = pal_hue
viridis_pal = pal_viridis
grey_pal = pal_grey
shape_pal = pal_shape
linetype_pal = pal_linetype
identity_pal = pal_identity
manual_pal = pal_manual
dichromat_pal = pal_dichromat
gradient_n_pal = pal_gradient_n
div_gradient_pal = pal_div_gradient
seq_gradient_pal = pal_seq_gradient
area_pal = pal_area
rescale_pal = pal_rescale
