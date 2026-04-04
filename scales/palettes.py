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
    Convert a single HCL (CIE-LCh(uv)) colour to a hex string.

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
        Hex colour string, e.g. ``"#3A7CBF"``.
    """
    # LCh(ab) -> Lab
    h_rad = np.radians(h % 360)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)

    # Lab -> XYZ  (D65 white point)
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    xr = fx ** 3 if fx ** 3 > eps else (116.0 * fx - 16.0) / kappa
    yr = ((l + 16.0) / 116.0) ** 3 if l > kappa * eps else l / kappa
    zr = fz ** 3 if fz ** 3 > eps else (116.0 * fz - 16.0) / kappa

    # D65 reference white
    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883

    # XYZ -> linear sRGB
    rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # Gamma companding
    def _gamma(v: float) -> float:
        if v <= 0.0031308:
            return 12.92 * v
        return 1.055 * (v ** (1.0 / 2.4)) - 0.055

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
# ---------------------------------------------------------------------------

_DICHROMAT_SCHEMES: dict[str, list[str]] = {
    "Categorical.12": [
        "#4477AA", "#332288", "#6699CC", "#88CCEE",
        "#44AA99", "#117733", "#999933", "#DDCC77",
        "#661100", "#CC6677", "#AA4466", "#882255",
    ],
    "BluetoOrange.8": [
        "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
        "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
    ],
    "BluetoOrange.12": [
        "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
        "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D",
        "#B2182B", "#67001F", "#053061", "#313695",
    ],
    "DarkRedtoBlue.12": [
        "#67001F", "#B2182B", "#D6604D", "#F4A582",
        "#FDDBC7", "#F7F7F7", "#D1E5F0", "#92C5DE",
        "#4393C3", "#2166AC", "#053061", "#313695",
    ],
    "LightBluetoDarkBlue.7": [
        "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1",
        "#6BAED6", "#3182BD", "#08519C",
    ],
    "SteppedSequential.5": [
        "#990F26", "#E03531", "#FC7747", "#FABA6F", "#FEEAA1",
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
        # Try dichromat schemes first
        if x in _DICHROMAT_SCHEMES:
            return pal_dichromat(name=x)
        # Try as viridis option
        if x in _VIRIDIS_OPTIONS:
            return pal_viridis(option=x)
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
        disc = as_discrete_pal(x)
        return as_continuous_pal(disc)

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
    import matplotlib.pyplot as plt

    _TYPE_PALETTES: dict[str, list[str]] = {
        "seq": [
            "Blues", "Greens", "Greys", "Oranges", "Purples", "Reds",
            "BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuBuGn",
            "PuRd", "RdPu", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
        ],
        "div": [
            "BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy",
            "RdYlBu", "RdYlGn", "Spectral",
        ],
        "qual": [
            "Accent", "Dark2", "Paired", "Pastel1", "Pastel2",
            "Set1", "Set2", "Set3",
        ],
    }

    if isinstance(palette, int):
        names = _TYPE_PALETTES.get(type, _TYPE_PALETTES["seq"])
        idx = max(0, min(palette - 1, len(names) - 1))
        cmap_name = names[idx]
    else:
        cmap_name = palette

    def _brewer_fun(n: int) -> list[str]:
        try:
            cmap = plt.get_cmap(cmap_name, n)
        except ValueError:
            cmap = plt.get_cmap(cmap_name)
        colours = [
            "#{:02X}{:02X}{:02X}".format(
                int(round(c[0] * 255)),
                int(round(c[1] * 255)),
                int(round(c[2] * 255)),
            )
            for c in [cmap(i / max(n - 1, 1)) for i in range(n)]
        ]
        if direction == -1:
            colours = colours[::-1]
        return colours

    return DiscretePalette(_brewer_fun, type="colour")


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
            return []
        hue_range = h[1] - h[0]
        # Do not use the full endpoint so colours don't wrap around
        hues = np.linspace(h[0], h[1], n, endpoint=False)
        if direction == -1:
            hues = hues[::-1]
        hues = (hues + h_start) % 360
        return [_hcl_to_hex(hue, c, l) for hue in hues]

    return DiscretePalette(_hue_fun, type="colour")


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
    import matplotlib.pyplot as plt

    cmap_name = _VIRIDIS_OPTIONS.get(option, "viridis")

    def _viridis_fun(n: int) -> list[str]:
        if n == 0:
            return []
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError:
            cmap = plt.get_cmap("viridis")

        if direction == -1:
            positions = np.linspace(end, begin, n)
        else:
            positions = np.linspace(begin, end, n)

        colours: list[str] = []
        for pos in positions:
            rgba = cmap(pos)
            r, g, b = rgba[0], rgba[1], rgba[2]
            a = alpha
            if a < 1:
                colours.append(
                    "#{:02X}{:02X}{:02X}{:02X}".format(
                        int(round(r * 255)),
                        int(round(g * 255)),
                        int(round(b * 255)),
                        int(round(a * 255)),
                    )
                )
            else:
                colours.append(
                    "#{:02X}{:02X}{:02X}".format(
                        int(round(r * 255)),
                        int(round(g * 255)),
                        int(round(b * 255)),
                    )
                )
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

    Returns integer codes suitable for matplotlib marker selection.

    Parameters
    ----------
    solid : bool, optional
        If ``True`` (default), return solid marker codes ``[0..5]``.
        If ``False``, return a mix of open and solid codes.

    Returns
    -------
    DiscretePalette
    """
    if solid:
        shapes = [0, 1, 2, 3, 4, 5]
    else:
        shapes = [0, 1, 2, 15, 16, 17, 18]

    max_n = len(shapes)

    def _shape_fun(n: int) -> list[int]:
        if n > max_n:
            raise ValueError(
                f"Shape palette supports at most {max_n} levels, got {n}"
            )
        return shapes[:n]

    return DiscretePalette(_shape_fun, type="numeric", nlevels=max_n)


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

    def _linetype_fun(n: int) -> list[str]:
        if n > max_n:
            raise ValueError(
                f"Linetype palette supports at most {max_n} levels, got {n}"
            )
        return _LINETYPES[:n]

    return DiscretePalette(_linetype_fun, type="numeric", nlevels=max_n)


def pal_identity() -> DiscretePalette:
    """
    Identity palette.

    Returns the input values unchanged.

    Returns
    -------
    DiscretePalette
    """

    def _identity_fun(n: int) -> list[int]:
        return list(range(n))

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
        if n > max_n:
            raise ValueError(
                f"Manual palette has {max_n} values but {n} were requested"
            )
        return vals[:n]

    return DiscretePalette(_manual_fun, type=type, nlevels=max_n)


def pal_dichromat(name: str = "Categorical.12") -> DiscretePalette:
    """
    Colorblind-safe palette from embedded dichromat colour schemes.

    Parameters
    ----------
    name : str, optional
        Scheme name. Available schemes: ``"Categorical.12"``,
        ``"BluetoOrange.8"``, ``"BluetoOrange.12"``,
        ``"DarkRedtoBlue.12"``, ``"LightBluetoDarkBlue.7"``,
        ``"SteppedSequential.5"``.

    Returns
    -------
    DiscretePalette
    """
    if name not in _DICHROMAT_SCHEMES:
        available = ", ".join(sorted(_DICHROMAT_SCHEMES))
        raise ValueError(
            f"Unknown dichromat scheme {name!r}. Available: {available}"
        )

    colours = _DICHROMAT_SCHEMES[name]
    max_n = len(colours)

    def _dichromat_fun(n: int) -> list[str]:
        if n > max_n:
            raise ValueError(
                f"Dichromat palette {name!r} has {max_n} colours "
                f"but {n} were requested"
            )
        return colours[:n]

    return DiscretePalette(_dichromat_fun, type="colour", nlevels=max_n)


# ===================================================================
# Continuous palette factories
# ===================================================================

def pal_gradient_n(
    colours: Sequence[str],
    values: Optional[Sequence[float]] = None,
    space: str = "Lab",
) -> ContinuousPalette:
    """
    Gradient through *n* colours.

    Parameters
    ----------
    colours : sequence of str
        Hex colour strings defining the gradient stops.
    values : sequence of float or None, optional
        Positions of each colour in ``[0, 1]``. If ``None``, colours are
        evenly spaced.
    space : str, optional
        Colour interpolation space (default ``"Lab"``). Currently uses
        matplotlib's ``LinearSegmentedColormap`` which interpolates in RGB;
        ``"Lab"`` is accepted for API compatibility.

    Returns
    -------
    ContinuousPalette
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    n_colours = len(colours)
    if n_colours < 2:
        raise ValueError("pal_gradient_n requires at least 2 colours")

    if values is None:
        values_arr = np.linspace(0.0, 1.0, n_colours)
    else:
        values_arr = np.asarray(values, dtype=float)
        if len(values_arr) != n_colours:
            raise ValueError(
                f"Length of values ({len(values_arr)}) must match "
                f"length of colours ({n_colours})"
            )

    rgba_list = [to_rgba(c) for c in colours]
    cmap = LinearSegmentedColormap.from_list(
        "custom_gradient",
        list(zip(values_arr, rgba_list)),
    )

    def _gradient_fun(x: ArrayLike) -> list[str]:
        x = np.asarray(x, dtype=float)
        result: list[str] = []
        for val in x.ravel():
            if np.isnan(val):
                result.append(None)  # type: ignore[arg-type]
            else:
                rgba = cmap(np.clip(val, 0.0, 1.0))
                result.append(
                    "#{:02X}{:02X}{:02X}".format(
                        int(round(rgba[0] * 255)),
                        int(round(rgba[1] * 255)),
                        int(round(rgba[2] * 255)),
                    )
                )
        return result

    return ContinuousPalette(_gradient_fun, type="colour", na_safe=True)


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
        return np.sqrt(np.abs(x) / max_val) * max_val

    return ContinuousPalette(_abs_area_fun, type="numeric", na_safe=False)


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
