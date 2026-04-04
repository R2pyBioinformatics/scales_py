"""Validation result recorder — controls results_<name>.csv schema.

This module is the **single source of truth** for the results CSV format.
Validation scripts must use ``ResultRecorder`` to write results; never
construct the CSV manually.

Usage::

    from _recorder import ResultRecorder

    rec = ResultRecorder("seurat_steps")
    rec.record("get_expressed_genes", "jaccard", value=1.0, threshold=0.95, tier=1)
    rec.log("get_expressed_genes", "n_R=1922, n_Python=1922")
    rec.save()
"""

import csv
from pathlib import Path

__all__ = ["ResultRecorder", "FIELDNAMES"]

# Canonical CSV columns — shared with update_port_status.py parser.
FIELDNAMES = ["tutorial", "function", "tier", "metric", "value", "threshold", "pass"]


class ResultRecorder:
    """Accumulate validation results and write them to a CSV file.

    Parameters
    ----------
    tutorial : str
        Tutorial name (e.g. ``"seurat_steps"``).  Used as the ``tutorial``
        column value and to derive the output filename
        ``results_<tutorial>.csv``.
    output_dir : str or Path
        Directory for the output CSV.  Defaults to ``"validation"``.
    """

    def __init__(self, tutorial: str, output_dir: str | Path = "validation"):
        if not tutorial:
            raise ValueError("tutorial name is required (e.g. 'seurat_steps')")
        self.tutorial = tutorial
        self._rows: list[dict[str, str]] = []
        self._path = Path(output_dir) / f"results_{tutorial}.csv"

    def record(
        self,
        function: str,
        metric: str,
        *,
        value: float,
        threshold: float,
        tier: int = 1,
        passed: bool | None = None,
    ) -> None:
        """Record a single validation check.

        Parameters
        ----------
        function : str
            What was tested (e.g. ``"get_expressed_genes"``).
        metric : str
            Measurement type (e.g. ``"pearson_r"``, ``"jaccard"``).
        value : float
            Measured value.
        threshold : float
            Pass/fail threshold.
        tier : int
            Validation tier (1, 2, or 3).
        passed : bool or None
            Explicit pass/fail.  If ``None``, auto-determined as
            ``value >= threshold``.
        """
        if passed is None:
            passed = float(value) >= float(threshold)
        self._rows.append({
            "tutorial": self.tutorial,
            "function": function,
            "tier": str(tier),
            "metric": metric,
            "value": f"{value:.6f}" if isinstance(value, float) else str(value),
            "threshold": str(threshold),
            "pass": str(passed).lower(),
        })
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {function}: {metric}={value} (threshold={threshold})")

    def log(self, function: str, message: str) -> None:
        """Print informational output (not written to CSV)."""
        print(f"  [INFO] {function}: {message}")

    def save(self) -> None:
        """Write accumulated results to the CSV file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)
        n_pass = sum(1 for r in self._rows if r["pass"] == "true")
        n_total = len(self._rows)
        print(f"\nResults: {n_pass}/{n_total} passed -> {self._path}")

    @property
    def path(self) -> Path:
        """Output file path."""
        return self._path
