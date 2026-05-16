---
name: use-scales
description: Axis transforms, breaks, palettes, and label formatters — the visual-styling utilities that ggplot2-python builds on top of.
---

# scales-python

Python port of [r-lib/scales](https://github.com/r-lib/scales) (v1.4.0.9000). The toolbox of small, composable callables that turn a numeric vector into "this is what the axis looks like."

## Mental model — four kinds of objects

1. **Transform** — a bijective pair `(transform, inverse)` between data space and visual space (`log_trans`, `sqrt_trans`, `boxcox_trans`, ...). Carries the breaks/format defaults that suit the visual space.

2. **Palette** — a *factory*: configure it once (`pal_viridis(option="D")`), then call the result with a count or a position to get hex colors. The two-step pattern is the source of most palette bugs.

3. **Range** — a small mutable accumulator (`ContinuousRange`, `DiscreteRange`) that ggplot2_py uses internally to merge cross-layer data extents. Users rarely touch it directly.

4. **Formatter** — a callable that turns tick values into strings (`label_number`, `label_percent`, `label_dollar`, `label_log`, `label_bytes`). Compose these instead of writing your own `f"{x:.2f}"`.

## The cardinal idiom — the three-piece kit

When customizing a non-default axis, supply *all three* of: `trans=`, `breaks=`, `labels=`. Each piece without the others is incoherent.

```python
gg.scale_x_continuous(
    trans  = log10_trans(),
    breaks = breaks_log(n=5, base=10),
    labels = label_log(base=10),
)
```

## When NOT to reach for scales

- The user is happy with default linear axes — don't add complexity.
- You're outside ggplot2_py entirely (working in matplotlib) — use matplotlib's `ticker` instead.
- The user wants a discrete categorical axis — that's `scale_x_discrete` / `scale_color_discrete`, no scales primitives needed.

## Quick reference

```python
from scales import (
    log10_trans, sqrt_trans, pseudo_log_trans,
    breaks_log, breaks_extended,
    label_percent, label_dollar, label_number, cut_si,
    pal_viridis, brewer_pal, hue_pal, pal_seq_gradient,
)

# Axis transforms + breaks
trans = log10_trans()
ticks = breaks_log(n=5, base=10)([1, 1e6])     # → [1, 10, 100, ..., 1e6]
strs  = label_percent(accuracy=1)([0.0, 0.5, 1.0])  # → ["0%", "50%", "100%"]

# Palettes
viridis8 = pal_viridis()(8)                    # 8 viridis hex codes
rdylbu   = brewer_pal("RdYlBu")(11)            # 11 ColorBrewer codes
```

For more: `biobabel.list_idioms(package="scales")` and `biobabel.describe_concept("scales.Transform")`.
