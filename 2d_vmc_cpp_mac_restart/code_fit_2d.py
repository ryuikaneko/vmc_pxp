#!/usr/bin/env python3
"""Finite-size extrapolation of 2D square-lattice PXP/pBCS VMC CSVs.

The script recursively searches for one-row ``dat01.csv``, ``dat02.csv``, ...
summary files and, within each directory, reads only the file with the largest
numeric suffix.  Equal Hamiltonian couplings and equal aspect ratios are grouped
by default, preventing, for example, square and strip geometries from being
mixed in one fit.  The default linear scale is

    L_eff = sqrt(N) = sqrt(Lx * Ly),

which reduces to Lx=Ly for square clusters.  Alternative scales can be selected
with ``--size-variable``.  Candidate finite-size corrections are ranked by
AICc:

  constant, 1/L_eff, 1/L_eff^2, 1/L_eff + 1/L_eff^2,
  exp(-L_eff/xi), and 1/L_eff^p.

For variances and V-scores, corresponding zero-intercept models are also
ranked.  Matching ``<observable>_error`` columns are used as absolute standard
deviations when every retained point has a finite positive uncertainty.

Outputs for each coupling/shape group are model rankings, best extrapolations,
plots against 1/L_eff, the exact input points used, and a Markdown report.
"""
from __future__ import annotations
import argparse
import csv
import math
import re
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ERROR_SUFFIX = "_error"
IDENTITY_COLUMNS = {
    "L",
    "Lx",
    "Ly",
    "N",
    "jx",
    "jn",
    "jp",
    "z",
    "z_initializer",
    "parameters",
    "averaged_iterations",
    "blockade_dimension_transfer",
    "final_samples_used",
    "final_blocks",
    "elapsed_seconds",
    "frustration_free",
    "parameterization",
    "v_score_enabled",
    "ed_dimension",
    "mpi_ranks",
    "initializer_residual",
    "sign_kernel_sigma_min",
    "projective_Tx_residual",
    "projective_Ty_residual",
}


def safe_tag(x: float) -> str:
    return (
        format(float(x), ".12g").replace("-", "m").replace("+", "p").replace(".", "p")
    )


def read_rows(root: Path, pattern: str) -> list[dict]:
    """Read the highest-numbered ``datNN.csv`` in each directory.

    For example, if one directory contains ``dat01.csv``, ``dat02.csv``, and
    ``dat03.csv``, only ``dat03.csv`` is read.  Selection is performed
    independently in every directory below ``root``.  The numeric suffix may
    contain two or more digits, so names beyond ``dat99.csv`` also work.
    """
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
        (item[1] for item in latest_by_directory.values()),
        key=lambda file: str(file),
    )
    if not selected_files:
        raise FileNotFoundError(
            f"no numbered CSV (datNN.csv) matched {pattern!r} below {root}"
        )

    rows = []
    for file in selected_files:
        with file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(file)
                rows.append(row)
    return rows


def as_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def linear_size(row, variable):
    Lx = as_float(row.get("Lx"))
    Ly = as_float(row.get("Ly"))
    N = as_float(row.get("N"))
    legacy_L = as_float(row.get("L"))
    if N is None and Lx is not None and Ly is not None:
        N = Lx * Ly
    if variable == "sqrtN":
        return math.sqrt(N) if N is not None and N > 0 else legacy_L
    if variable == "Lx":
        return Lx
    if variable == "Ly":
        return Ly
    if variable == "min":
        return min(Lx, Ly) if Lx is not None and Ly is not None else legacy_L
    if variable == "max":
        return max(Lx, Ly) if Lx is not None and Ly is not None else legacy_L
    raise ValueError(variable)


def aspect_ratio(row):
    Lx = as_float(row.get("Lx"))
    Ly = as_float(row.get("Ly"))
    if Lx is None or Ly is None or Ly == 0:
        return None
    return Lx / Ly


def models(include_zero: bool):
    out = {
        # Return an array, not a scalar.  scipy.curve_fit accepts a scalar
        # through broadcasting, but matplotlib requires x and y to have the
        # same first dimension when the fitted curve is plotted.
        "constant": (
            lambda L, c: np.full_like(np.asarray(L, dtype=float), c),
            ["y_inf"],
            None,
        ),
        "inverse_L": (lambda L, c, a: c + a / L, ["y_inf", "a"], None),
        "inverse_L2": (lambda L, c, a: c + a / L**2, ["y_inf", "a"], None),
        "inverse_L_L2": (
            lambda L, c, a, b: c + a / L + b / L**2,
            ["y_inf", "a", "b"],
            None,
        ),
        "exponential": (
            lambda L, c, a, xi: c + a * np.exp(-L / xi),
            ["y_inf", "a", "xi"],
            ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
        ),
        "power_free": (
            lambda L, c, a, power: c + a / L**power,
            ["y_inf", "a", "power"],
            ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 8.0]),
        ),
    }
    if include_zero:
        out.update(
            {
                "zero_inverse_L": (lambda L, a: a / L, ["a"], None),
                "zero_inverse_L2": (lambda L, a: a / L**2, ["a"], None),
                "zero_exponential": (
                    lambda L, a, xi: a * np.exp(-L / xi),
                    ["a", "xi"],
                    ([-np.inf, 1e-6], [np.inf, np.inf]),
                ),
                "zero_power_free": (
                    lambda L, a, power: a / L**power,
                    ["a", "power"],
                    ([-np.inf, 0.05], [np.inf, 8.0]),
                ),
            }
        )
    return out


def initial_guess(name, y):
    span = float(y[0] - y[-1]) if len(y) else 1.0
    c = float(y[-1])
    if name == "constant":
        return [float(np.mean(y))]
    if name in ("inverse_L", "inverse_L2"):
        return [c, span]
    if name == "inverse_L_L2":
        return [c, span, span]
    if name == "exponential":
        return [c, span, 3.0]
    if name == "power_free":
        return [c, span, 1.0]
    if name in ("zero_inverse_L", "zero_inverse_L2"):
        return [float(y[0])]
    if name == "zero_exponential":
        return [float(y[0]), 3.0]
    if name == "zero_power_free":
        return [float(y[0]), 1.0]
    raise KeyError(name)


def fit_one(name, spec, L, y, sigma):
    fun, names, bounds = spec
    k = len(names)
    n = len(y)
    if n <= k:
        return None
    kwargs = dict(p0=initial_guess(name, y), maxfev=100000)
    if bounds is not None:
        kwargs["bounds"] = bounds
    if sigma is not None:
        kwargs.update(sigma=sigma, absolute_sigma=True)
    try:
        popt, pcov = curve_fit(fun, L, y, **kwargs)
    except Exception:
        return None
    pred = fun(L, *popt)
    residual = y - pred
    if sigma is None:
        rss = float(np.sum(residual**2))
        scale = max(rss / n, np.finfo(float).tiny)
        minus2loglike = n * math.log(scale)
        chi2 = float("nan")
    else:
        z = residual / sigma
        chi2 = float(np.sum(z * z))
        minus2loglike = chi2 + float(np.sum(np.log(2 * np.pi * sigma * sigma)))
        rss = float(np.sum(residual**2))
    aic = minus2loglike + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else float("inf")
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    params = dict(zip(names, map(float, popt)))
    errors = dict(zip((x + "_error" for x in names), map(float, perr)))
    return dict(
        model=name,
        n=n,
        k=k,
        rss=rss,
        chi2=chi2,
        aicc=aicc,
        params=params,
        errors=errors,
        function=fun,
    )


def observable_kind(name):
    lower = name.lower()
    return (
        "zero_expected"
        if ("variance" in lower or "v_score" in lower)
        else "finite_limit"
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument(
        "--pattern",
        default="**/dat*.csv",
        help=(
            "glob used to find numbered CSVs; only the largest datNN.csv "
            "suffix in each directory is read"
        ),
    )
    ap.add_argument("--output-dir", default="finite_size_extrapolation_2d")
    ap.add_argument(
        "--observables",
        nargs="*",
        default=None,
        help="explicit columns; default is every numeric non-metadata column",
    )
    ap.add_argument(
        "--size-variable",
        choices=("sqrtN", "Lx", "Ly", "min", "max"),
        default="sqrtN",
        help="linear scale used by the finite-size models",
    )
    ap.add_argument("--min-linear-size", type=float, default=0.0)
    ap.add_argument(
        "--min-L",
        type=float,
        default=None,
        help="deprecated alias for --min-linear-size",
    )
    ap.add_argument(
        "--combine-aspect-ratios",
        action="store_true",
        help="fit different Lx/Ly shapes together; off by default",
    )
    ap.add_argument("--aspect-ratio-tolerance", type=float, default=1.0e-10)
    args = ap.parse_args()
    if args.min_L is not None:
        args.min_linear_size = args.min_L

    root = Path(args.root)
    rows = read_rows(root, args.pattern)
    groups = {}
    skipped = []
    for row in rows:
        coupling = tuple(as_float(row.get(x)) for x in ("jx", "jn", "jp"))
        size = linear_size(row, args.size_variable)
        ratio = aspect_ratio(row)
        if None in coupling or size is None:
            skipped.append(row.get("_source", "unknown"))
            continue
        if args.combine_aspect_ratios:
            shape_key = None
        elif ratio is None:
            shape_key = "unknown"
        else:
            tol = args.aspect_ratio_tolerance
            shape_key = round(ratio / tol) * tol if tol > 0 else ratio
        groups.setdefault((*coupling, shape_key), []).append(row)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for (jx, jn, jp, shape_key), group in sorted(
        groups.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        group = sorted(group, key=lambda r: linear_size(r, args.size_variable))
        columns = sorted(set().union(*(r.keys() for r in group)))
        if args.observables is None:
            observables = []
            for col in columns:
                if (
                    col.startswith("_")
                    or col in IDENTITY_COLUMNS
                    or col.endswith(ERROR_SUFFIX)
                ):
                    continue
                vals = [as_float(r.get(col)) for r in group]
                if sum(v is not None for v in vals) >= 3:
                    observables.append(col)
        else:
            observables = args.observables

        shape_tag = (
            "all_shapes"
            if args.combine_aspect_ratios
            else (
                "shape_unknown"
                if shape_key == "unknown"
                else f"aspect{safe_tag(shape_key)}"
            )
        )
        tag = f"jx{safe_tag(jx)}_jn{safe_tag(jn)}_jp{safe_tag(jp)}_" f"{shape_tag}"
        out = out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        rankings = []
        best_rows = []
        used_points = []
        heading = f"# 2D finite-size extrapolation: jx={jx:g}, jn={jn:g}, jp={jp:g}"
        report = [
            heading,
            "",
            f"Linear scale: `{args.size_variable}`; aspect-ratio group: `{shape_key}`.",
            "",
        ]

        for obs in observables:
            data = []
            err_col = obs + ERROR_SUFFIX
            for r in group:
                Leff = linear_size(r, args.size_variable)
                y = as_float(r.get(obs))
                error = as_float(r.get(err_col))
                if Leff is not None and y is not None and Leff >= args.min_linear_size:
                    data.append((Leff, y, error, r))
            if len(data) < 3:
                continue
            Leff = np.asarray([x[0] for x in data], float)
            y = np.asarray([x[1] for x in data], float)
            error_values = [x[2] for x in data]
            sigma = (
                np.asarray(error_values, float)
                if all(x is not None and x > 0 for x in error_values)
                else None
            )
            include_zero = observable_kind(obs) == "zero_expected"
            fits = []
            for name, spec in models(include_zero).items():
                fit = fit_one(name, spec, Leff, y, sigma)
                if fit is not None and np.isfinite(fit["aicc"]):
                    fits.append(fit)
            if not fits:
                continue
            fits.sort(key=lambda z: z["aicc"])
            amin = fits[0]["aicc"]
            weights = np.exp(-0.5 * np.asarray([f["aicc"] - amin for f in fits]))
            weights /= weights.sum()
            for fit, weight in zip(fits, weights):
                record = dict(
                    observable=obs,
                    model=fit["model"],
                    size_variable=args.size_variable,
                    aspect_ratio=shape_key,
                    aicc=fit["aicc"],
                    delta_aicc=fit["aicc"] - amin,
                    akaike_weight=float(weight),
                    n=fit["n"],
                    k=fit["k"],
                    rss=fit["rss"],
                    chi2=fit["chi2"],
                )
                record.update(fit["params"])
                record.update(fit["errors"])
                rankings.append(record)
            best = fits[0]
            best_rows.append(
                dict(
                    observable=obs,
                    best_model=best["model"],
                    size_variable=args.size_variable,
                    aspect_ratio=shape_key,
                    y_inf=best["params"].get("y_inf", 0.0),
                    y_inf_error=best["errors"].get("y_inf_error", 0.0),
                    aicc=best["aicc"],
                    n=len(Leff),
                    weighted=bool(sigma is not None),
                )
            )
            report += [
                f"## {obs}",
                "",
                f"Best model: `{best['model']}`; AICc={best['aicc']:.6g}; "
                f"thermodynamic limit={best['params'].get('y_inf', 0.0):.12g} "
                f"+/- {best['errors'].get('y_inf_error', 0.0):.3g}.",
                "",
            ]

            for Leff_i, y_i, error_i, row in data:
                used_points.append(
                    dict(
                        observable=obs,
                        Lx=row.get("Lx", ""),
                        Ly=row.get("Ly", ""),
                        N=row.get("N", ""),
                        L_eff=Leff_i,
                        value=y_i,
                        error=("" if error_i is None else error_i),
                        source=row.get("_source", ""),
                    )
                )

            inverse_data = 1.0 / Leff
            positive_inverse = np.linspace(
                1.0 / (1.15 * Leff.max()), 1.0 / (0.8 * Leff.min()), 399
            )
            L_grid = 1.0 / positive_inverse
            inverse_grid = np.concatenate(([0.0], positive_inverse))
            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            if sigma is None:
                ax.plot(inverse_data, y, "o", label="data")
            else:
                ax.errorbar(
                    inverse_data, y, yerr=sigma, fmt="o", capsize=3, label="data"
                )
            model_specs = models(include_zero)
            for fit, weight in zip(fits[:3], weights[:3]):
                parameter_names = model_specs[fit["model"]][1]
                pars = [fit["params"][x] for x in parameter_names]
                finite_curve = np.asarray(fit["function"](L_grid, *pars), dtype=float)
                finite_curve = (
                    np.full_like(L_grid, float(finite_curve))
                    if finite_curve.ndim == 0
                    else np.broadcast_to(finite_curve, L_grid.shape)
                )
                thermodynamic_limit = fit["params"].get("y_inf", 0.0)
                fitted_curve = np.concatenate(([thermodynamic_limit], finite_curve))
                ax.plot(
                    inverse_grid, fitted_curve, label=f"{fit['model']} (w={weight:.2f})"
                )
            ax.axvline(0.0, color="0.75", linewidth=0.8)
            ax.set_xlim(left=0.0)
            ax.set_xlabel(f"1/{args.size_variable}")
            ax.set_ylabel(obs)

            if sigma is None:
                data_low, data_high = y.copy(), y.copy()
            else:
                data_low, data_high = y - sigma, y + sigma
            finite_data = np.concatenate(
                (data_low[np.isfinite(data_low)], data_high[np.isfinite(data_high)])
            )
            if finite_data.size:
                data_min, data_max = float(np.min(finite_data)), float(
                    np.max(finite_data)
                )
                data_span = data_max - data_min
                if not np.isfinite(data_span) or data_span <= 0.0:
                    data_span = max(abs(data_min), abs(data_max), 1.0) * 0.10
                guard_low, guard_high = (
                    data_min - 5.0 * data_span,
                    data_max + 5.0 * data_span,
                )
                visible_values = [finite_data]
                for line in ax.lines:
                    line_y = np.asarray(line.get_ydata(), dtype=float)
                    line_y = line_y[np.isfinite(line_y)]
                    line_y = line_y[(line_y >= guard_low) & (line_y <= guard_high)]
                    if line_y.size:
                        visible_values.append(line_y)
                visible = np.concatenate(visible_values)
                y_low, y_high = float(np.min(visible)), float(np.max(visible))
                span = y_high - y_low
                if not np.isfinite(span) or span <= 0.0:
                    span = data_span
                ax.set_ylim(y_low - 0.08 * span, y_high + 0.08 * span)
            ax.legend(fontsize=8)
            fig.tight_layout()
            safe_obs = re.sub(r"[^A-Za-z0-9_.-]+", "_", obs)
            fig.savefig(out / f"{safe_obs}.png", dpi=180)
            plt.close(fig)

        if rankings:
            fields = sorted(set().union(*(r.keys() for r in rankings)))
            with (out / "model_rankings.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rankings)
        if best_rows:
            fields = list(best_rows[0])
            with (out / "best_extrapolations.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(best_rows)
        if used_points:
            fields = list(used_points[0])
            with (out / "input_points.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(used_points)
        (out / "report.md").write_text("\n".join(report))
        reports += report + [""]

    (out_root / "all_groups_report.md").write_text("\n".join(reports))
    print(f"read {len(rows)} rows in {len(groups)} coupling/shape group(s)")
    if skipped:
        print(f"skipped {len(skipped)} row(s) without usable couplings or 2D size")
    print(f"saved extrapolations under: {out_root}")


if __name__ == "__main__":
    main()
