"""Recipe: log10 transform + log breaks + log labels on a numeric vector.

Demonstrates the canonical three-piece kit (idiom `scales.transform_breaks_labels_combo`)
*without* requiring ggplot2_py — just the scales primitives.
"""

from __future__ import annotations

import numpy as np

from scales import breaks_log, label_log, log10_trans


def main() -> dict[str, object]:
    rng = np.random.default_rng(0)
    raw = rng.uniform(1, 1e6, size=200)

    trans = log10_trans()
    transformed = trans.transform(raw)            # data → visual space
    back = trans.inverse(transformed)             # round-trip
    assert np.allclose(back, raw, rtol=1e-9), "log10_trans must be invertible"

    breaks_fn = breaks_log(n=5, base=10)
    breaks = breaks_fn([raw.min(), raw.max()])    # axis tick positions

    labels_fn = label_log(base=10)
    labels = labels_fn(breaks)                    # tick label strings

    return {"breaks": list(breaks), "labels": list(labels)}


if __name__ == "__main__":
    result = main()
    for b, l in zip(result["breaks"], result["labels"]):
        print(f"{b:>14}  →  {l}")
