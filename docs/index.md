# scales

Python port of the R **scales** package — scale functions for visualization.

Graphical scales map data to aesthetics, and provide methods for automatically determining breaks and labels for axes and legends.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
import scales

# Format numbers
scales.comma(1234567)        # ['1,234,567']
scales.percent(0.15)         # ['15%']
scales.dollar(42.5)          # ['$42.50']

# Generate breaks for axis ticks
brk = scales.breaks_extended(n=5)
brk(np.array([0, 100]))     # array([0, 25, 50, 75, 100])

# Rescale data to [0, 1]
scales.rescale(np.array([1.0, 5.0, 10.0]))  # array([0., 0.44, 1.])

# Color palettes
pal = scales.pal_viridis()
pal(5)                       # 5 viridis colors as hex strings

# Transforms
t = scales.transform_log10()
t.transform_func(np.array([1, 10, 100, 1000]))  # array([0, 1, 2, 3])

# Label formatting
labeler = scales.label_number(prefix="$", big_mark=",")
labeler(np.array([1234, 5678]))  # ['$1,234', '$5,678']
```

## Features

- **Transforms**: log, sqrt, reverse, Box-Cox, Yeo-Johnson, pseudo-log, date/time, and more
- **Break generators**: extended Wilkinson, pretty, log, fixed-width, exponential
- **Label formatters**: numbers, currency, percent, scientific, bytes, ordinal, p-values, dates
- **Color palettes**: Brewer, viridis, HCL hue, gradient, manual, and more
- **Color manipulation**: alpha, muted, lighten, darken, saturate, shift, mix
- **Color mapping**: continuous, binned, quantile, factor
- **Rescaling**: linear, midpoint, max, out-of-bounds handling
- **Range training**: continuous and discrete range accumulation

## Origin

This is a Python port of the R [scales](https://scales.r-lib.org/) package (v1.4.0.9000) by Hadley Wickham, Thomas Lin Pedersen, and Dana Seidel.
