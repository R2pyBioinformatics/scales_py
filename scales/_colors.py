"""Pure-Python colour parsing — replaces matplotlib.colors dependency.

Provides to_rgba, to_hex, to_rgb with the same API as matplotlib.colors
but without requiring matplotlib to be installed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union


# CSS4 named colours (148 entries, matching matplotlib.colors.CSS4_COLORS)
_CSS4_COLORS: dict[str, str] = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blanchedalmond": "#ffebcd",
    "blue": "#0000ff",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "cyan": "#00ffff",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgreen": "#006400",
    "darkgrey": "#a9a9a9",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "floralwhite": "#fffaf0",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#adff2f",
    "grey": "#808080",
    "honeydew": "#f0fff0",
    "hotpink": "#ff69b4",
    "indianred": "#cd5c5c",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead",
    "navy": "#000080",
    "oldlace": "#fdf5e6",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#ff0000",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "seashell": "#fff5ee",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "whitesmoke": "#f5f5f5",
    "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}

# Also accept single-letter aliases
_BASE_COLORS: dict[str, str] = {
    "b": "#0000ff", "g": "#008000", "r": "#ff0000",
    "c": "#00bfbf", "m": "#bf00bf", "y": "#bfbf00",
    "k": "#000000", "w": "#ffffff",
}

# Merge into single lookup
_NAMED_COLORS: dict[str, str] = {**_CSS4_COLORS, **_BASE_COLORS}

ColorLike = Union[str, Tuple[float, ...], Sequence[float]]


def _parse_hex(s: str) -> Tuple[float, float, float, float]:
    """Parse #RGB, #RRGGBB, or #RRGGBBAA → (r, g, b, a) in [0, 1]."""
    s = s.lstrip("#")
    if len(s) == 3:
        r = int(s[0] * 2, 16) / 255.0
        g = int(s[1] * 2, 16) / 255.0
        b = int(s[2] * 2, 16) / 255.0
        return (r, g, b, 1.0)
    elif len(s) == 6:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return (r, g, b, 1.0)
    elif len(s) == 8:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        a = int(s[6:8], 16) / 255.0
        return (r, g, b, a)
    else:
        raise ValueError(f"Invalid hex colour: #{s}")


def to_rgba(c: ColorLike) -> Tuple[float, float, float, float]:
    """
    Convert a colour specification to an (r, g, b, a) tuple in [0, 1].

    Accepts hex strings (#RGB, #RRGGBB, #RRGGBBAA), CSS4 colour names,
    single-letter aliases (r/g/b/c/m/y/k/w), or numeric tuples.
    """
    if isinstance(c, str):
        c = c.strip().lower()
        if c.startswith("#"):
            return _parse_hex(c)
        if c in _NAMED_COLORS:
            return _parse_hex(_NAMED_COLORS[c])
        # Try "none" / "transparent"
        if c in ("none", "transparent"):
            return (0.0, 0.0, 0.0, 0.0)
        raise ValueError(f"Unknown colour: {c!r}")

    # Tuple / list / sequence
    t = tuple(float(x) for x in c)
    if len(t) == 3:
        return (t[0], t[1], t[2], 1.0)
    elif len(t) == 4:
        return t  # type: ignore[return-value]
    else:
        raise ValueError(f"Colour tuple must have 3 or 4 elements, got {len(t)}")


def to_rgb(c: ColorLike) -> Tuple[float, float, float]:
    """Convert a colour specification to an (r, g, b) tuple in [0, 1]."""
    r, g, b, _a = to_rgba(c)
    return (r, g, b)


def to_hex(c: ColorLike, keep_alpha: bool = True) -> str:
    """
    Convert a colour specification to a hex string.

    Parameters
    ----------
    c : colour-like
        Colour to convert.
    keep_alpha : bool, default True
        If True and alpha != 1.0, return #RRGGBBAA.
        If False, always return #RRGGBB.
    """
    if isinstance(c, str) and c.startswith("#"):
        rgba = _parse_hex(c)
    elif isinstance(c, str):
        rgba = to_rgba(c)
    else:
        t = tuple(float(x) for x in c)
        if len(t) == 3:
            rgba = (t[0], t[1], t[2], 1.0)
        elif len(t) == 4:
            rgba = t
        else:
            raise ValueError(f"Expected 3 or 4 values, got {len(t)}")

    r = int(round(rgba[0] * 255))
    g = int(round(rgba[1] * 255))
    b = int(round(rgba[2] * 255))
    a = int(round(rgba[3] * 255))

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    a = max(0, min(255, a))

    if keep_alpha and a < 255:
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    else:
        return f"#{r:02x}{g:02x}{b:02x}"
