#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

DEFAULT_DIRECTORY_PATTERN = "L*_jx1_jn0_jp0"
DEFAULT_OUTPUT_FILENAME = "energy_vscore_fidelity_table.tex"
DATA_FILE_PATTERN = re.compile(r"^dat([0-9]+)[.]csv$")
ERROR_SIGNIFICANT_DIGITS = 2
ED_DECIMAL_DIGITS = 12
FIDELITY_DECIMAL_DIGITS = 12


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read the latest dat<number>.csv file for each system size L "
            "and generate a LaTeX table containing ED energy density, VMC energy density, "
            "V-score, and fidelity."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing directories such as L16_jx1_jn0_jp0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which the LaTeX table is written.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output filename. Default: {DEFAULT_OUTPUT_FILENAME}",
    )
    parser.add_argument(
        "--directory-pattern",
        default=DEFAULT_DIRECTORY_PATTERN,
        help=f"Glob pattern for system-size directories. Default: {DEFAULT_DIRECTORY_PATTERN}",
    )
    return parser.parse_args()


def extract_dat_number(path):
    """Extract the integer index from dat<number>.csv, or return None."""
    match = DATA_FILE_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def extract_directory_L(directory):
    """Extract L from the beginning of a directory name."""
    match = re.match(r"^L([0-9]+)_", directory.name)
    if match is None:
        raise ValueError(f"Could not extract L from directory name: {directory}")
    return int(match.group(1))


def clean_key(key):
    """Remove whitespace and line breaks from a CSV column name."""
    return None if key is None else re.sub(r"\s+", "", key)


def clean_value(value):
    """Remove leading and trailing whitespace from a CSV cell."""
    return value.strip() if isinstance(value, str) else value


def read_csv_rows(path):
    """Read all rows from a CSV file and normalize column names."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [
            {
                clean_key(key): clean_value(value)
                for key, value in raw_row.items()
                if key is not None
            }
            for raw_row in reader
        ]
    if not rows:
        raise ValueError(f"No data rows were found in {path}")
    return rows


def select_row_for_L(rows, directory_L, path):
    """Select the final row whose L value matches the directory system size."""
    matching_rows = []
    for row in rows:
        try:
            row_L = int(float(row["L"]))
        except (KeyError, TypeError, ValueError):
            continue
        if row_L == directory_L:
            matching_rows.append(row)
    if matching_rows:
        return matching_rows[-1]
    if len(rows) == 1:
        return rows[0]
    raise ValueError(f"No row with L={directory_L} was found in {path}")


def get_float(row, column, path):
    """Read a named CSV column as a floating-point value."""
    if column not in row:
        raise KeyError(
            f"Column '{column}' was not found in {path}\n"
            f"Available columns: {list(row.keys())}"
        )
    value = row[column]
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Could not convert column '{column}' value '{value}' to float in {path}"
        ) from error


def get_int(row, column, path):
    """Read a named CSV column as an integer."""
    return int(round(get_float(row, column, path)))


def trim_fixed_number(text):
    """Remove unnecessary trailing zeros from a fixed-point number."""
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "+0") else text


def format_fixed(value, digits):
    """Format a finite value in fixed-point notation."""
    if not math.isfinite(value):
        return r"\text{--}"
    return trim_fixed_number(f"{value:.{digits}f}")


def scientific_to_latex(text):
    """Convert a numeric string in scientific notation to LaTeX notation."""
    text = text.strip()
    lower_text = text.lower()
    if "e" not in lower_text:
        return text
    mantissa, exponent_text = lower_text.split("e", maxsplit=1)
    try:
        exponent = int(exponent_text)
    except ValueError:
        return text
    return rf"{mantissa} \times 10^{{{exponent}}}"


def round_uncertainty(error, significant_digits=2):
    """Round a positive uncertainty to the requested significant digits."""
    if not math.isfinite(error) or error <= 0.0:
        return None, None

    exponent = math.floor(math.log10(abs(error)))
    decimal_places = significant_digits - 1 - exponent
    rounded_error = round(error, decimal_places)

    # Rounding can change the order of magnitude, for example 9.99 -> 10.0.
    rounded_exponent = math.floor(math.log10(abs(rounded_error)))
    decimal_places = significant_digits - 1 - rounded_exponent
    rounded_error = round(error, decimal_places)

    return rounded_error, decimal_places


def format_parenthetical_error(value, error, significant_digits=2):
    """Format a value and uncertainty using parenthetical notation.

    Examples
    --------
    -9.656988893, 1.35e-7 -> -9.65698889(14)
    1.2345, 0.0067        -> 1.2345(67)
    """
    if not math.isfinite(value):
        return r"\text{--}"

    rounded_error, decimal_places = round_uncertainty(
        error,
        significant_digits,
    )

    if rounded_error is None:
        return format_fixed(value, 12)

    if decimal_places >= 0:
        value_text = f"{value:.{decimal_places}f}"
        error_digits = int(round(rounded_error * (10 ** decimal_places)))
    else:
        scale = 10 ** (-decimal_places)
        rounded_value = round(value / scale) * scale
        value_text = f"{rounded_value:.0f}"
        error_digits = int(round(rounded_error / scale))

    return f"{value_text}({error_digits})"


def format_scientific_parenthetical_error(
    value,
    error,
    significant_digits=2,
):
    """Format a value and uncertainty in scientific parenthetical notation.

    Example
    -------
    6.709868e-10, 3.139876e-12 -> 6.710(31) \\times 10^{-10}
    """
    if not math.isfinite(value):
        return r"\text{--}"

    if not math.isfinite(error) or error <= 0.0:
        return scientific_to_latex(f"{value:.6e}")

    if value == 0.0:
        return format_parenthetical_error(
            value,
            error,
            significant_digits,
        )

    exponent = math.floor(math.log10(abs(value)))
    scale = 10.0 ** exponent
    mantissa = value / scale
    mantissa_error = error / scale

    formatted_mantissa = format_parenthetical_error(
        mantissa,
        mantissa_error,
        significant_digits,
    )

    return rf"{formatted_mantissa} \times 10^{{{exponent}}}"


def find_latest_files(input_dir, directory_pattern):
    """Select the dat<number>.csv file with the largest number for each L."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory was not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"The input path is not a directory: {input_dir}")

    directories = sorted(
        path for path in input_dir.glob(directory_pattern) if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(
            "No matching system-size directories were found.\n"
            f"Input directory: {input_dir}\n"
            f"Directory pattern: {directory_pattern}"
        )

    latest_by_L = {}
    for directory in directories:
        directory_L = extract_directory_L(directory)
        valid_candidates = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            dat_number = extract_dat_number(path)
            if dat_number is not None:
                valid_candidates.append((dat_number, path))

        if not valid_candidates:
            print(
                f"Warning: no dat<number>.csv file was found for L={directory_L} "
                f"in {directory}"
            )
            continue

        dat_number, selected_path = max(valid_candidates, key=lambda item: item[0])
        previous = latest_by_L.get(directory_L)
        if previous is None or dat_number > previous[0]:
            latest_by_L[directory_L] = (dat_number, selected_path)

    if not latest_by_L:
        raise FileNotFoundError(
            "No valid dat<number>.csv files were found. Files such as "
            "dat01_checkpoint.csv are intentionally ignored."
        )
    return latest_by_L


def collect_data(input_dir, directory_pattern):
    """Read the quantities required for the LaTeX table."""
    latest_by_L = find_latest_files(input_dir, directory_pattern)
    required_columns = [
        "L",
        "energy_density",
        "energy_density_error",
        "v_score",
        "v_score_error",
        "ed_energy",
        "fidelity_ed",
    ]
    results = []

    for directory_L in sorted(latest_by_L):
        dat_number, path = latest_by_L[directory_L]
        rows = read_csv_rows(path)
        row = select_row_for_L(rows, directory_L, path)
        missing_columns = [column for column in required_columns if column not in row]
        if missing_columns:
            raise KeyError(
                f"Missing columns in {path}: {missing_columns}\n"
                f"Available columns: {list(row.keys())}"
            )

        csv_L = get_int(row, "L", path)
        if csv_L != directory_L:
            raise ValueError(
                f"L mismatch in {path}: the directory gives L={directory_L}, "
                f"but the CSV file gives L={csv_L}"
            )

        results.append(
            {
                "L": csv_L,
                "dat_number": dat_number,
                "path": path,
                "ed_energy_density": get_float(row, "ed_energy", path) / csv_L,
                "vmc_energy_density": get_float(row, "energy_density", path),
                "vmc_energy_density_error": get_float(row, "energy_density_error", path),
                "v_score": get_float(row, "v_score", path),
                "v_score_error": get_float(row, "v_score_error", path),
                "fidelity": get_float(row, "fidelity_ed", path),
            }
        )
    return results


def make_latex_table(results):
    """Generate a LaTeX table using the booktabs package."""
    lines = [
        r"\begin{table}[tb]",
        r"  \centering",
        (
            r"  \caption{Comparison of the ED and VMC energy densities. "
            r"The uncertainties indicate Monte Carlo statistical errors.}"
        ),
        r"  \label{tab:energy-vscore-fidelity}",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
        (
            r"    $L$"
            r" & $e_{\mathrm{ED}}$"
            r" & $e_{\mathrm{VMC}}$"
            r" & $V$-score"
            r" & Fidelity \\"
        ),
        r"    \midrule",
    ]

    for result in results:
        ed_energy_density = format_fixed(
            result["ed_energy_density"],
            ED_DECIMAL_DIGITS,
        )
        vmc_energy_density = format_parenthetical_error(
            result["vmc_energy_density"],
            result["vmc_energy_density_error"],
            ERROR_SIGNIFICANT_DIGITS,
        )
        v_score = format_scientific_parenthetical_error(
            result["v_score"],
            result["v_score_error"],
            ERROR_SIGNIFICANT_DIGITS,
        )
        fidelity = format_fixed(result["fidelity"], FIDELITY_DECIMAL_DIGITS)
        lines.append(
            rf"    {result['L']}"
            rf" & ${ed_energy_density}$"
            rf" & ${vmc_energy_density}$"
            rf" & ${v_score}$"
            rf" & ${fidelity}$ \\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    """Run the data-selection and LaTeX-generation workflow."""
    args = parse_arguments()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    results = collect_data(input_dir, args.directory_pattern)
    latex = make_latex_table(results)
    output_path.write_text(latex, encoding="utf-8")

    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_path}")
    print("\nSelected input files:")
    for result in results:
        print(
            f"  L={result['L']:>3d}: {result['path']} "
            f"(dat number = {result['dat_number']})"
        )
    print("\nGenerated LaTeX:\n")
    print(latex)


if __name__ == "__main__":
    main()
