#!/usr/bin/env python3
"""Plot selected 1D observables versus 1/L without extrapolation or fitting.

Exact-diagonalization (ED) data can be overlaid from
../1d_ed_python/dat_1d_summary.csv. The ED energy_density column is used for
the energy-density plot, and sigma_z is used for the longitudinal-
magnetization plot. Since the ED file has no V-score column, the V-score plot
contains VMC data only.

The script recursively searches for dat01.csv, dat02.csv, ... and reads only
the file with the largest numeric suffix in each directory. Data are grouped by
(jx, jn, jp), and three plots are produced for each coupling group:

  * energy density vs 1/L
  * longitudinal magnetization vs 1/L
  * V-score vs 1/L

If a matching <observable>_error column exists and contains finite positive
values, vertical error bars are shown. No fit or thermodynamic-limit
extrapolation is performed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np


# Canonical output name and accepted input-column aliases.
# The first matching column present in the CSV is used.
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
        "mz",
    ),
    "v_score": (
        "v_score",
        "v-score",
        "vscore",
    ),
}

# Publication-oriented typography, approximately three times Matplotlib's
# standard text size.
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

X_MIN = 0.0
X_MAX = 0.065
ENERGY_OFFSET = -0.60356
ENERGY_SCALE = 1.0e-6

Y_LABELS = {
    "energy_density": "Energy density",
    "longitudinal_magnetization": "Longitudinal\nmagnetization",
    "v_score": "V-score",
}


def as_float(value: object) -> float | None:
    """Convert a value to a finite float; return None if conversion fails."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_tag(value: float) -> str:
    """Convert a coupling value to a file-system-safe short string."""
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
        (entry[1] for entry in latest_by_directory.values()),
        key=str,
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


def find_column(rows: list[dict[str, str]], aliases: tuple[str, ...]) -> str | None:
    """Return the first available column matching one of the aliases."""
    columns = set().union(*(row.keys() for row in rows))
    lower_to_original = {column.strip().lower(): column for column in columns}

    for alias in aliases:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    return None


def read_ed_data(ed_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read exact-diagonalization data used for comparison.

    Expected columns are:
    L, hilbert_dimension, energy, energy_density, sigma_x, sigma_z, n, gap
    """
    if not ed_file.is_file():
        print(f"warning: ED file not found: {ed_file}")
        return {}

    with ed_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    column_map = {
        "energy_density": "energy_density",
        "longitudinal_magnetization": "sigma_z",
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for canonical_name, column in column_map.items():
        data = []
        for row in rows:
            length = as_float(row.get("L"))
            value = as_float(row.get(column))
            if length is not None and length > 0.0 and value is not None:
                inverse_length = 1.0 / length
                if X_MIN <= inverse_length < X_MAX:
                    data.append((inverse_length, value))
        if data:
            data.sort(key=lambda item: item[0])
            result[canonical_name] = (
                np.asarray([item[0] for item in data], dtype=float),
                np.asarray([item[1] for item in data], dtype=float),
            )
    return result


def plot_combined_observables(rows, columns, output_file, ed_data):
    """Create a compact three-panel figure with a shared 1/L axis."""
    names = ("energy_density", "longitudinal_magnetization", "v_score")
    letters = ("(a)", "(b)", "(c)")
    fig, axes = plt.subplots(
#        3, 1, figsize=(10.0, 8.5), sharex=True,
        3, 1, figsize=(13.0, 8.5), sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    for i, (ax, name, letter) in enumerate(zip(axes, names, letters)):
        column = columns[name]
        error_column = f"{column}_error"
        data = []
        for row in rows:
            length = as_float(row.get("L"))
            value = as_float(row.get(column))
            error = as_float(row.get(error_column))
            if length is not None and length > 0.0 and value is not None:
                data.append((length, value, error))
        if not data:
            plt.close(fig)
            return False
        data.sort(key=lambda item: 1.0 / item[0])
        x = np.asarray([1.0 / item[0] for item in data])
        y = np.asarray([item[1] for item in data])
        errors = [item[2] for item in data]
        if all(e is not None and e > 0.0 for e in errors):
            ax.errorbar(
                x, y, yerr=np.asarray(errors), fmt="o-", markersize=14,
                markerfacecolor="none", markeredgewidth=3.0, color="red",
                ecolor="red", capsize=7, capthick=2.0, elinewidth=3.0,
                label="VMC",
            )
        else:
            ax.plot(
                x, y, "o-", markersize=14, markerfacecolor="none",
                markeredgewidth=3.0, color="red", label="VMC",
            )
        if name in ed_data:
            ed_x, ed_y = ed_data[name]
            ax.plot(
                ed_x, ed_y, "s--", markersize=14, markerfacecolor="none",
                markeredgewidth=3.0, color="black", label="ED",
            )
        ax.set_ylabel(Y_LABELS[name])
        ax.set_xlim(X_MIN, X_MAX)
        if name == "energy_density":
            ax.set_ylim(
#                ENERGY_OFFSET - 2.25 * ENERGY_SCALE,
#                ENERGY_OFFSET - 0.25 * ENERGY_SCALE,
                ENERGY_OFFSET - 2.25 * ENERGY_SCALE,
                ENERGY_OFFSET + 0.25 * ENERGY_SCALE,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda value, position:
                f"{(value - ENERGY_OFFSET) / ENERGY_SCALE:g}"
            ))
            ax.text(
#                0.035, 0.955, r"$\times 10^{-6} - 0.60356$",
                0.035, 0.92, r"$\times 10^{-6} - 0.60356$",
                transform=ax.transAxes, ha="left", va="top",
            )
        elif name == "longitudinal_magnetization":
#            ax.set_ylim(0.598, 0.618)
            ax.set_ylim(0.598, 0.622)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.set_ylim(-0.5e-10, 10.5e-10)
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda value, position: f"{value / 1.0e-10:g}"
            ))
            ax.text(
#                0.035, 0.955, r"$\times 10^{-10}$",
                0.035, 0.92, r"$\times 10^{-10}$",
                transform=ax.transAxes, ha="left", va="top",
            )
        ax.text(
            0.035, 0.08, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="normal", fontsize=28,
        )
        ax.tick_params(
            axis="both", which="both", direction="in", top=True, right=True,
            length=8, width=1.8, pad=8, labelbottom=(i == 2),
        )
        ax.grid(alpha=0.25)
        if i == 1:
            ax.legend(
                frameon=True, facecolor="white", edgecolor="none",
                framealpha=1.0, borderpad=0.35, handletextpad=0.5,
                labelspacing=0.35, loc="best",
            )
    axes[-1].set_xlabel(r"$1/L$")
    fig.subplots_adjust(
        left=0.20, right=0.97, bottom=0.08, top=0.98, hspace=0.0
    )
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot three observables versus 1/L without extrapolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="root directory containing datNN.csv files",
    )
    parser.add_argument(
        "--pattern",
#        default="**/dat*.csv",
        default="../1d_vmc_cpp_issp_ohtaka_restart/dat/L*_jx1_jn0_jp0/dat*.csv",
        help="recursive glob for input files",
    )
    parser.add_argument(
        "--ed-file",
        default="../1d_ed_python/dat_1d_summary.csv",
        help="exact-diagonalization summary CSV overlaid on compatible plots",
    )
    parser.add_argument(
        "--output-dir",
#        default="finite_size_plots",
        default="finite_size_plots_1d",
        help="directory in which plots are saved",
    )
    args = parser.parse_args()

    rows = read_latest_rows(Path(args.root), args.pattern)
    ed_data = read_ed_data(Path(args.ed_file))

    groups: dict[tuple[float, float, float], list[dict[str, str]]] = {}
    for row in rows:
        length = as_float(row.get("L"))
        couplings = tuple(as_float(row.get(name)) for name in ("jx", "jn", "jp"))
        if length is None or length <= 0.0 or any(value is None for value in couplings):
            continue
        key = (float(couplings[0]), float(couplings[1]), float(couplings[2]))
        groups.setdefault(key, []).append(row)

    if not groups:
        raise ValueError("No rows with valid L, jx, jn, and jp values were found")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    number_of_plots = 0
    for (jx, jn, jp), group_rows in sorted(groups.items()):
        tag = f"jx{safe_tag(jx)}_jn{safe_tag(jn)}_jp{safe_tag(jp)}"
        group_output = output_root / tag
        group_output.mkdir(parents=True, exist_ok=True)
        columns = {}
        for canonical_name, aliases in OBSERVABLES.items():
            column = find_column(group_rows, aliases)
            if column is None:
                print(f"warning: {canonical_name} column not found for {tag}")
                break
            columns[canonical_name] = column
        else:
            output_file = group_output / "combined_observables_vs_inverse_L.pdf"
            if plot_combined_observables(
                group_rows, columns, output_file, ed_data
            ):
                number_of_plots += 1
            else:
                print(f"warning: incomplete observable data for {tag}")

    print(f"read {len(rows)} rows in {len(groups)} coupling group(s)")
    print(f"saved {number_of_plots} combined plot(s) under: {output_root}")


if __name__ == "__main__":
    main()
