#!/usr/bin/env python3
"""Plot selected 2D observables versus 1/L_eff without fitting.

This is the 2D counterpart of code_fit_1d.py.  It recursively searches for
numbered VMC summary files dat01.csv, dat02.csv, ... and reads only the file
with the largest numeric suffix in each directory.  Rows are grouped by
(jx, jn, jp) and, by default, by aspect ratio Lx/Ly.

The default finite-size coordinate is

    L_eff = sqrt(N) = sqrt(Lx * Ly),

which equals L for an L x L square.  Exact-diagonalization data are overlaid
from ../2d_ed_python/dat_2d_summary.csv.  The expected ED columns are

    L,N,hilbert_dimension,energy,energy_density,sigma_x_projected,
    sigma_z,n,gap,S_00_connected,S_pipi_connected

The ED energy_density column is used for the energy-density plot and sigma_z
for the longitudinal-magnetization plot.  Since the ED file has no V-score,
the V-score plot contains VMC data only.

Three PDF plots are produced for every coupling/shape group:

  * energy density versus 1/L_eff
  * longitudinal magnetization versus 1/L_eff
  * V-score versus 1/L_eff

If a matching <observable>_error column exists and all retained errors are
finite and positive, vertical VMC error bars are shown.  No extrapolation or
fitting is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


OBSERVABLES = {
    "energy_density": (
        "energy_density",
        "energy density",
        "energy_per_site",
        "energy",
    ),
    "longitudinal_magnetization": (
        "longitudinal_magnetization",
        "longitudinal magnetization",
        "magnetization_z",
        "sigma_z",
        "mz",
    ),
    "v_score": (
        "v_score",
        "v-score",
        "vscore",
    ),
}

Y_LABELS = {
    "energy_density": "Energy density",
    "longitudinal_magnetization": "Longitudinal magnetization",
    "v_score": "V-score",
}

SIZE_LABELS = {
#    "sqrtN": r"$1/\sqrt{N}$",
    "sqrtN": r"$1/L$",
    "Lx": r"$1/L_x$",
    "Ly": r"$1/L_y$",
    "min": r"$1/\min(L_x,L_y)$",
    "max": r"$1/\max(L_x,L_y)$",
    "L": r"$1/L$",
}

# Publication-oriented typography, matching code_fit_1d.py.
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "axes.linewidth": 1.8,
        "lines.linewidth": 3.0,
        "lines.markersize": 14,
    }
)


def as_float(value: object) -> float | None:
    """Convert a value to a finite float; return None on failure."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_tag(value: object) -> str:
    """Convert a value to a file-system-safe compact string."""
    if isinstance(value, str):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return (
        format(float(value), ".12g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def read_latest_rows(root: Path, pattern: str) -> list[dict[str, str]]:
    """Read only the largest-numbered datNN.csv in each directory."""
    numbered_name = re.compile(r"^dat(\d{2,})\.csv$")
    latest_by_directory: dict[Path, tuple[int, Path]] = {}

    for file in root.glob(pattern):
        match = numbered_name.fullmatch(file.name)
        if match is None:
            continue
        number = int(match.group(1))
        current = latest_by_directory.get(file.parent)
        if current is None or number > current[0]:
            latest_by_directory[file.parent] = (number, file)

    selected_files = sorted(
        (entry[1] for entry in latest_by_directory.values()), key=str
    )
    if not selected_files:
        raise FileNotFoundError(
            f"No numbered CSV file (datNN.csv) matched {pattern!r} below {root}"
        )

    rows: list[dict[str, str]] = []
    for file in selected_files:
        with file.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(file)
                rows.append(row)
    return rows


def find_column(
    rows: list[dict[str, str]], aliases: tuple[str, ...]
) -> str | None:
    """Return the first available column matching one of the aliases."""
    columns = set().union(*(row.keys() for row in rows))
    lower_to_original = {
        column.strip().lower(): column for column in columns if column is not None
    }
    for alias in aliases:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    return None


def linear_size(row: dict[str, str], variable: str) -> float | None:
    """Return the requested effective linear size from a VMC row."""
    Lx = as_float(row.get("Lx"))
    Ly = as_float(row.get("Ly"))
    N = as_float(row.get("N"))
    legacy_L = as_float(row.get("L"))

    if N is None and Lx is not None and Ly is not None:
        N = Lx * Ly

    if variable == "sqrtN":
        if N is not None and N > 0.0:
            return math.sqrt(N)
        return legacy_L
    if variable == "L":
        return legacy_L
    if variable == "Lx":
        return Lx if Lx is not None else legacy_L
    if variable == "Ly":
        return Ly if Ly is not None else legacy_L
    if variable == "min":
        if Lx is not None and Ly is not None:
            return min(Lx, Ly)
        return legacy_L
    if variable == "max":
        if Lx is not None and Ly is not None:
            return max(Lx, Ly)
        return legacy_L
    raise ValueError(f"Unknown size variable: {variable}")


def aspect_ratio(row: dict[str, str]) -> float | None:
    """Return Lx/Ly; square legacy rows with only L are assigned ratio 1."""
    Lx = as_float(row.get("Lx"))
    Ly = as_float(row.get("Ly"))
    legacy_L = as_float(row.get("L"))
    if Lx is not None and Ly is not None and Ly != 0.0:
        return Lx / Ly
    if legacy_L is not None and legacy_L > 0.0:
        return 1.0
    return None


def read_ed_data(
    ed_file: Path,
    size_variable: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read compatible observables from the 2D ED summary CSV.

    Expected ED columns:
      L,N,hilbert_dimension,energy,energy_density,sigma_x_projected,
      sigma_z,n,gap,S_00_connected,S_pipi_connected

    The ED data describe square L x L clusters.  Thus sqrt(N) and L are
    equivalent.  For Lx, Ly, min, and max, the ED column L is used.
    """
    if not ed_file.is_file():
        print(f"warning: ED file not found: {ed_file}")
        return {}

    with ed_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required = {"L", "energy_density", "sigma_z"}
    available = set(rows[0].keys()) if rows else set()
    missing = sorted(required - available)
    if missing:
        print(
            "warning: ED file is missing required column(s): "
            + ", ".join(missing)
        )

    column_map = {
        "energy_density": "energy_density",
        "longitudinal_magnetization": "sigma_z",
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for canonical_name, column in column_map.items():
        data: list[tuple[float, float]] = []
        for row in rows:
            L = as_float(row.get("L"))
            N = as_float(row.get("N"))
            value = as_float(row.get(column))

            if size_variable == "sqrtN" and N is not None and N > 0.0:
                Leff = math.sqrt(N)
            else:
                Leff = L

            if Leff is not None and Leff > 0.0 and value is not None:
                data.append((1.0 / Leff, value))

        if data:
            data.sort(key=lambda item: item[0])
            result[canonical_name] = (
                np.asarray([item[0] for item in data], dtype=float),
                np.asarray([item[1] for item in data], dtype=float),
            )
    return result


def set_y_axis_format(ax: plt.Axes, canonical_name: str) -> None:
    """Apply robust formatting without hard-coding 1D-specific y limits."""
    if canonical_name == "v_score":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
    elif canonical_name == "energy_density":
        formatter = ScalarFormatter(useMathText=True, useOffset=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
    else:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)


def apply_padded_y_limits(
    ax: plt.Axes,
    values: np.ndarray,
    errors: np.ndarray | None,
    ed_values: np.ndarray | None,
    canonical_name: str,
) -> None:
    """Choose data-driven y limits, unlike the 1D-specific fixed ranges."""
    arrays = [values]
    if errors is not None:
        arrays.extend((values - errors, values + errors))
    if ed_values is not None:
        arrays.append(ed_values)
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if finite.size == 0:
        return

    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    span = ymax - ymin
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(ymin), abs(ymax), 1.0) * 0.05
    padding = 0.12 * span

    if canonical_name == "v_score" and ymin >= 0.0:
#        ax.set_ylim(0.0, ymax + padding)
        ax.set_ylim(-0.0025, 0.035)
    else:
        ax.set_ylim(ymin - padding, ymax + padding)


def plot_observable(
    rows: list[dict[str, str]],
    column: str,
    canonical_name: str,
    output_file: Path,
    ed_data: dict[str, tuple[np.ndarray, np.ndarray]],
    size_variable: str,
    x_max: float | None,
) -> bool:
    """Plot one observable against inverse effective size."""
    error_column = f"{column}_error"
    data: list[tuple[float, float, float | None]] = []

    for row in rows:
        Leff = linear_size(row, size_variable)
        value = as_float(row.get(column))
        error = as_float(row.get(error_column))
        if Leff is not None and Leff > 0.0 and value is not None:
            data.append((Leff, value, error))

    if not data:
        return False

    # Sort from small to large 1/L_eff, so the guide line runs left to right.
    data.sort(key=lambda item: 1.0 / item[0])
    inverse_size = np.asarray([1.0 / item[0] for item in data], dtype=float)
    values = np.asarray([item[1] for item in data], dtype=float)
    error_values = [item[2] for item in data]
    use_error_bars = all(
        error is not None and error > 0.0 for error in error_values
    )
    errors = (
        np.asarray(error_values, dtype=float) if use_error_bars else None
    )

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    if errors is not None:
        ax.errorbar(
            inverse_size,
            values,
            yerr=errors,
            fmt="o-",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=3.0,
            color="red",
            ecolor="red",
            capsize=7,
            capthick=2.0,
            elinewidth=3.0,
            label="VMC",
        )
    else:
        ax.plot(
            inverse_size,
            values,
            "o-",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=3.0,
            color="red",
            label="VMC",
        )

    ed_values_for_limits: np.ndarray | None = None
    if canonical_name in ed_data:
        ed_inverse_size, ed_values = ed_data[canonical_name]
        ax.plot(
            ed_inverse_size,
            ed_values,
            "s--",
            markersize=14,
            markerfacecolor="none",
            markeredgewidth=3.0,
            color="black",
            label="ED",
        )
        ed_values_for_limits = ed_values

    all_x = [inverse_size]
    if canonical_name in ed_data:
        all_x.append(ed_data[canonical_name][0])
    largest_x = max(float(np.max(x)) for x in all_x if x.size)
#    effective_x_max = x_max if x_max is not None else 1.08 * largest_x
    effective_x_max = x_max if x_max is not None else 1.1 * largest_x

    ax.set_xlabel(SIZE_LABELS[size_variable])
    ax.set_ylabel(Y_LABELS[canonical_name])
    ax.set_xlim(0.0, effective_x_max)
    set_y_axis_format(ax, canonical_name)
    apply_padded_y_limits(
        ax, values, errors, ed_values_for_limits, canonical_name
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=8,
        width=1.8,
        pad=8,
    )
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D energy density, longitudinal magnetization, and V-score "
            "against inverse effective size without fitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="root directory containing numbered datNN.csv files",
    )
    parser.add_argument(
        "--pattern",
        default="**/dat*.csv",
        help="recursive glob; only the largest datNN.csv in each directory is read",
    )
    parser.add_argument(
        "--ed-file",
        default="../2d_ed_python/dat_2d_summary.csv",
        help="2D exact-diagonalization summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="finite_size_plots_2d",
        help="directory in which PDF plots are saved",
    )
    parser.add_argument(
        "--size-variable",
        choices=("sqrtN", "L", "Lx", "Ly", "min", "max"),
        default="sqrtN",
        help="effective linear size used on the horizontal axis",
    )
    parser.add_argument(
        "--combine-aspect-ratios",
        action="store_true",
        help="combine different Lx/Ly shapes in the same plot",
    )
    parser.add_argument(
        "--aspect-ratio-tolerance",
        type=float,
        default=1.0e-10,
        help="tolerance used to group aspect ratios",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="optional upper limit of inverse-size axis; default is automatic",
    )
    args = parser.parse_args()

    rows = read_latest_rows(Path(args.root), args.pattern)
    ed_data = read_ed_data(Path(args.ed_file), args.size_variable)

    groups: dict[
        tuple[float, float, float, float | str | None],
        list[dict[str, str]],
    ] = {}
    skipped_sources: list[str] = []

    for row in rows:
        size = linear_size(row, args.size_variable)
        couplings = tuple(as_float(row.get(name)) for name in ("jx", "jn", "jp"))
        ratio = aspect_ratio(row)

        if size is None or size <= 0.0 or any(value is None for value in couplings):
            skipped_sources.append(row.get("_source", "unknown"))
            continue

        if args.combine_aspect_ratios:
            shape_key: float | str | None = None
        elif ratio is None:
            shape_key = "unknown"
        else:
            tolerance = args.aspect_ratio_tolerance
            shape_key = (
                round(ratio / tolerance) * tolerance if tolerance > 0.0 else ratio
            )

        key = (
            float(couplings[0]),
            float(couplings[1]),
            float(couplings[2]),
            shape_key,
        )
        groups.setdefault(key, []).append(row)

    if not groups:
        raise ValueError(
            "No rows with valid 2D size and jx, jn, jp values were found"
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    number_of_plots = 0

    for (jx, jn, jp, shape_key), group_rows in sorted(
        groups.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        if args.combine_aspect_ratios:
            shape_tag = "all_shapes"
        elif shape_key == "unknown":
            shape_tag = "shape_unknown"
        else:
            shape_tag = f"aspect{safe_tag(shape_key)}"

        tag = (
            f"jx{safe_tag(jx)}_jn{safe_tag(jn)}_jp{safe_tag(jp)}_"
            f"{shape_tag}"
        )
        group_output = output_root / tag
        group_output.mkdir(parents=True, exist_ok=True)

        for canonical_name, aliases in OBSERVABLES.items():
            column = find_column(group_rows, aliases)
            if column is None:
                print(f"warning: {canonical_name} column not found for {tag}")
                continue

            output_file = (
                group_output
                / f"{canonical_name}_vs_inverse_{args.size_variable}.pdf"
            )
            if plot_observable(
                group_rows,
                column,
                canonical_name,
                output_file,
                ed_data,
                args.size_variable,
                args.x_max,
            ):
                number_of_plots += 1
            else:
                print(f"warning: no valid {column} data for {tag}")

    print(f"read {len(rows)} rows in {len(groups)} coupling/shape group(s)")
    if skipped_sources:
        print(
            f"skipped {len(skipped_sources)} row(s) without usable size or couplings"
        )
    print(f"saved {number_of_plots} plot(s) under: {output_root}")


if __name__ == "__main__":
    main()
