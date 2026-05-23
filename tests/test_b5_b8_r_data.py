"""Regression tests for B5 (R named-colour table) + B8 (viridis F/G/H tables).

B5 — R has 657 named colours via ``colors()``; the Python port previously
only had 156 (CSS4 + matplotlib base), so e.g. ``"purple3"`` /
``"darkslategray2"`` / ``"lightslategrey"`` lookups failed. ``_R_COLORS``
embeds the full 657-entry table extracted from R's ``col2rgb(colors())``.

B8 — R's ``viridis_pal(option=...)`` supports options A–H (viridis,
magma, inferno, plasma, cividis, rocket, mako, turbo). The Python port
previously only had A–E; F/G/H silently fell back to viridis via the
palette registry. ``VIRIDIS`` now contains all 8 keys, with the F/G/H
data extracted from R ``viridis_pal(option=...)(256)``.
"""

from __future__ import annotations

import pytest


class TestB5_R_NamedColors:
    def test_all_657_R_colors_present(self):
        from scales._colors import _R_COLORS
        assert len(_R_COLORS) == 657

    def test_named_colors_merge_total(self):
        from scales._colors import _CSS4_COLORS, _BASE_COLORS, _R_COLORS, _NAMED_COLORS
        # R takes precedence on overlaps (gray, blue, etc.); merge is
        # CSS4 ∪ base ∪ R, deduped by key.
        expected_unique = len({*_CSS4_COLORS, *_BASE_COLORS, *_R_COLORS})
        assert len(_NAMED_COLORS) == expected_unique

    @pytest.mark.parametrize("name,hex_", [
        # X11-extended (previously missing in Py)
        ("purple3", "#7d26cd"),
        ("darkslategray2", "#8deeee"),
        ("burlywood3", "#cdaa7d"),
        ("lightslategrey", "#778899"),
        ("gray42", "#6b6b6b"),
        ("grey42", "#6b6b6b"),     # R supports both spellings → same hex
        ("mediumorchid1", "#e066ff"),
        ("palevioletred3", "#cd6889"),
        ("navajowhite4", "#8b795e"),
        ("sandybrown", "#f4a460"),
        # Common ones (sanity)
        ("steelblue", "#4682b4"),
        ("red", "#ff0000"),
        ("white", "#ffffff"),
        ("black", "#000000"),
        # R-vs-CSS4 collision: R wins per merge order
        ("gray", "#bebebe"),
    ])
    def test_R_color_hex(self, name, hex_):
        from scales._colors import _NAMED_COLORS
        assert _NAMED_COLORS.get(name) == hex_

    def test_color_resolvable_via_public_api(self):
        """to_hex() / to_rgba() should resolve any R color name."""
        from scales._colors import to_hex
        # Spot-check via public API
        assert to_hex("purple3") == "#7d26cd"
        assert to_hex("darkslategray2") == "#8deeee"


class TestB8_Viridis_FGH:
    @pytest.mark.parametrize("name", ["rocket", "mako", "turbo"])
    def test_256_entries(self, name):
        from scales._palettes_data import VIRIDIS
        assert name in VIRIDIS
        assert len(VIRIDIS[name]) == 256

    def test_R_endpoints_byte_exact(self):
        """First and last 3 hex of each F/G/H table must byte-match
        what R's ``viridis_pal(option=...)(256)`` produces."""
        from scales._palettes_data import VIRIDIS
        # Extracted via:
        #   Rscript -e 'library(scales); substr(viridis_pal(option="F")(256), 1, 7)'
        cases = {
            "rocket": (["#03051A", "#04051A", "#05061B"],
                       ["#FAE8D8", "#FAE9DA", "#FAEBDD"]),
            "mako":   (["#0B0405", "#0D0406", "#0E0508"],
                       ["#DAF3E1", "#DCF4E3", "#DEF5E5"]),
            "turbo":  (["#30123B", "#321543", "#33184A"],
                       ["#810602", "#7E0502", "#7A0403"]),
        }
        for name, (head, tail) in cases.items():
            t = VIRIDIS[name]
            assert t[:3] == head, f"{name} head mismatch"
            assert t[-3:] == tail, f"{name} tail mismatch"

    def test_F_G_H_distinct_from_viridis(self):
        """Previously F/G/H silently fell back to viridis."""
        from scales._palettes_data import VIRIDIS
        viridis_first = VIRIDIS["viridis"][0]
        for name in ("rocket", "mako", "turbo"):
            assert VIRIDIS[name][0] != viridis_first, (
                f"{name}[0] equals viridis[0] — fallback regression"
            )

    def test_F_option_via_pal_viridis(self):
        """pal_viridis(option='F')(N) should now return rocket data."""
        from scales import viridis_pal
        from scales._palettes_data import VIRIDIS
        # At N=256 the palette is the full table
        out = [c.upper() for c in viridis_pal(option="F")(256)]
        assert out == VIRIDIS["rocket"]
