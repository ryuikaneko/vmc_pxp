#!/usr/bin/env python3
"""Exact diagonalization of the one-dimensional PXP model.

Hamiltonian:
    H = coefficient * sum_i P_{i-1} X_i P_{i+1}

The Hilbert space is restricted by n_i n_{i+1} = 0.  Both open and
periodic boundary conditions are supported.  The script computes the
lowest eigenvalues and standard ground-state observables.

Examples
--------
python pxp_1d_ed.py --L 20 --boundary pbc --nev 4
python pxp_1d_ed.py --L 22 --boundary obc --correlation-max 10
python pxp_1d_ed.py --L 18 --boundary pbc --entropy --output result.json

Conventions
-----------
The default coefficient is +1, so E0/L approaches approximately
-0.60356 for the periodic chain.  If your Hamiltonian is
(Omega/2) * sum_i P X P, use --coefficient 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class PXPResult:
    L: int
    boundary: str
    coefficient: float
    hilbert_dimension: int
    eigenvalues: np.ndarray
    density: float
    local_density: np.ndarray
    correlations: np.ndarray
    structure_factor_0: float
    structure_factor_pi: float
    projected_flip: float
    entanglement_entropy: float | None


def is_allowed(state: int, L: int, periodic: bool) -> bool:
    """Return True when a bit string satisfies nearest-neighbor blockade."""
    if state & (state << 1):
        return False
    if periodic and L > 1:
        if (state & 1) and ((state >> (L - 1)) & 1):
            return False
    return True


def generate_basis(L: int, periodic: bool) -> np.ndarray:
    """Generate all allowed configurations as unsigned integers."""
    if L < 2:
        raise ValueError("L must be at least 2")
    if L > 63:
        raise ValueError("This bit representation supports L <= 63")
    states = [s for s in range(1 << L) if is_allowed(s, L, periodic)]
    return np.asarray(states, dtype=np.uint64)


def flip_is_allowed(state: int, site: int, L: int, periodic: bool) -> bool:
    """Check whether P_{i-1} X_i P_{i+1} acts nontrivially."""
    if periodic:
        left = (site - 1) % L
        right = (site + 1) % L
        return (((state >> left) & 1) == 0 and
                ((state >> right) & 1) == 0)

    left_ok = site == 0 or ((state >> (site - 1)) & 1) == 0
    right_ok = site == L - 1 or ((state >> (site + 1)) & 1) == 0
    return left_ok and right_ok


def build_hamiltonian(
    basis: np.ndarray,
    L: int,
    periodic: bool,
    coefficient: float,
) -> csr_matrix:
    """Construct the constrained PXP Hamiltonian as a sparse matrix."""
    index: Dict[int, int] = {int(s): i for i, s in enumerate(basis)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for row, state_u in enumerate(basis):
        state = int(state_u)
        for site in range(L):
            if flip_is_allowed(state, site, L, periodic):
                new_state = state ^ (1 << site)
                rows.append(row)
                cols.append(index[new_state])
                data.append(coefficient)

    dim = len(basis)
    hamiltonian = coo_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(dim, dim),
        dtype=np.float64,
    ).tocsr()

    # The construction should be exactly Hermitian.
    asymmetry = hamiltonian - hamiltonian.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1.0e-14:
        raise RuntimeError("Hamiltonian construction is not Hermitian")
    return hamiltonian


def lowest_states(
    hamiltonian: csr_matrix,
    nev: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the lowest algebraic eigenvalues and corresponding vectors."""
    dim = hamiltonian.shape[0]
    if dim <= 2:
        vals, vecs = np.linalg.eigh(hamiltonian.toarray())
        return vals, vecs

    k = min(max(1, nev), dim - 1)
    vals, vecs = eigsh(
        hamiltonian,
        k=k,
        which="SA",
        tol=tolerance,
        maxiter=max(10000, 20 * dim),
    )
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def occupation_matrix(basis: np.ndarray, L: int) -> np.ndarray:
    """Return occupations with shape (Hilbert dimension, L)."""
    sites = np.arange(L, dtype=np.uint64)
    return ((basis[:, None] >> sites[None, :]) & 1).astype(np.float64)


def density_correlations(
    probability: np.ndarray,
    occupations: np.ndarray,
    periodic: bool,
    r_max: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute local density and translationally averaged connected C(r)."""
    dim, L = occupations.shape
    if probability.shape != (dim,):
        raise ValueError("Probability vector has incompatible shape")

    local_density = probability @ occupations
    density = float(np.mean(local_density))
    r_max = min(r_max, L - 1)
    corr = np.zeros((r_max + 1, 3), dtype=np.float64)

    for r in range(r_max + 1):
        products = []
        disconnected = []
        for i in range(L):
            j = i + r
            if periodic:
                j %= L
            elif j >= L:
                continue
            products.append(float(probability @ (occupations[:, i] * occupations[:, j])))
            disconnected.append(float(local_density[i] * local_density[j]))
        raw = float(np.mean(products))
        connected = raw - float(np.mean(disconnected))
        corr[r] = (r, raw, connected)

    return density, local_density, corr


def structure_factor(
    probability: np.ndarray,
    occupations: np.ndarray,
    q: float,
) -> float:
    """Compute connected S(q) = L^-1 sum_ij exp[iq(i-j)] C_ij."""
    _, L = occupations.shape
    means = probability @ occupations
    weighted = occupations * np.sqrt(probability[:, None])
    nn = weighted.T @ weighted
    connected = nn - np.outer(means, means)
    positions = np.arange(L)
    phase = np.exp(1j * q * positions)
    value = np.vdot(phase, connected @ phase) / L
    return float(np.real_if_close(value))


def half_chain_entropy(
    psi: np.ndarray,
    basis: np.ndarray,
    L: int,
) -> float:
    """Compute half-chain von Neumann entropy by embedding in the spin basis.

    This routine allocates 2**L amplitudes and is therefore intended only for
    modest L.  Natural logarithms are used.
    """
    if L > 24:
        raise ValueError(
            "Entropy calculation embeds into the full 2**L space; use L <= 24"
        )
    left_sites = L // 2
    right_sites = L - left_sites
    matrix = np.zeros((1 << right_sites, 1 << left_sites), dtype=np.float64)
    mask = (1 << left_sites) - 1
    for amplitude, state_u in zip(psi, basis):
        state = int(state_u)
        left = state & mask
        right = state >> left_sites
        matrix[right, left] = amplitude

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    weights = singular_values**2
    weights = weights[weights > 1.0e-15]
    return float(-np.sum(weights * np.log(weights)))


def solve(args: argparse.Namespace) -> PXPResult:
    periodic = args.boundary == "pbc"
    basis = generate_basis(args.L, periodic)
    hamiltonian = build_hamiltonian(
        basis=basis,
        L=args.L,
        periodic=periodic,
        coefficient=args.coefficient,
    )
    eigenvalues, eigenvectors = lowest_states(
        hamiltonian, args.nev, args.tolerance
    )

    psi0 = eigenvectors[:, 0]
    probability = np.abs(psi0) ** 2
    probability /= np.sum(probability)
    occupations = occupation_matrix(basis, args.L)
    density, local_density, correlations = density_correlations(
        probability,
        occupations,
        periodic,
        args.correlation_max,
    )

    entropy = None
    if args.entropy:
        entropy = half_chain_entropy(psi0, basis, args.L)

    # Hellmann-Feynman: <sum_i PXP_i>/L = E0/(L*coefficient).
    if args.coefficient == 0.0:
        projected_flip = float("nan")
    else:
        projected_flip = float(eigenvalues[0] / (args.L * args.coefficient))

    return PXPResult(
        L=args.L,
        boundary=args.boundary,
        coefficient=args.coefficient,
        hilbert_dimension=len(basis),
        eigenvalues=eigenvalues,
        density=density,
        local_density=local_density,
        correlations=correlations,
        structure_factor_0=structure_factor(probability, occupations, 0.0),
        structure_factor_pi=structure_factor(probability, occupations, np.pi),
        projected_flip=projected_flip,
        entanglement_entropy=entropy,
    )


def result_as_dict(result: PXPResult) -> dict:
    gap = None
    if len(result.eigenvalues) >= 2:
        gap = float(result.eigenvalues[1] - result.eigenvalues[0])
    return {
        "L": result.L,
        "boundary": result.boundary,
        "coefficient": result.coefficient,
        "hilbert_dimension": result.hilbert_dimension,
        "eigenvalues": [float(x) for x in result.eigenvalues],
        "ground_state_energy": float(result.eigenvalues[0]),
        "energy_per_site": float(result.eigenvalues[0] / result.L),
        "gap": gap,
        "density": result.density,
        "sigma_z_expectation_per_site": 1.0 - 2.0 * result.density,
        "sigma_x_expectation_per_site": result.projected_flip,
        "local_density": result.local_density.tolist(),
        "projected_flip_expectation_per_site": result.projected_flip,
        "connected_structure_factor_q0": result.structure_factor_0,
        "connected_structure_factor_qpi": result.structure_factor_pi,
        "half_chain_von_neumann_entropy": result.entanglement_entropy,
        "correlations_columns": ["r", "raw", "connected"],
        "correlations": result.correlations.tolist(),
    }


def print_result(result: PXPResult) -> None:
    data = result_as_dict(result)
    print("\n=== 1D PXP exact diagonalization ===")
    print(f"L                         : {data['L']}")
    print(f"boundary                  : {data['boundary'].upper()}")
    print(f"Hamiltonian coefficient   : {data['coefficient']:.12g}")
    print(f"constrained Hilbert dim   : {data['hilbert_dimension']}")
    print(f"ground-state energy E0    : {data['ground_state_energy']:.12f}")
    print(f"energy per site E0/L      : {data['energy_per_site']:.12f}")
    if data["gap"] is not None:
        print(f"lowest gap E1-E0          : {data['gap']:.12f}")
    print(f"mean excitation density   : {data['density']:.12f}")
    print(f"<PXP> per site            : {data['projected_flip_expectation_per_site']:.12f}")
    print(f"connected S(0)            : {data['connected_structure_factor_q0']:.12f}")
    print(f"connected S(pi)           : {data['connected_structure_factor_qpi']:.12f}")
    if result.entanglement_entropy is not None:
        print(f"half-chain entropy        : {result.entanglement_entropy:.12f}")

    print("\n# r   <n_i n_{i+r}>       connected C(r)")
    for r, raw, connected in result.correlations:
        print(f"{int(r):3d}  {raw: .12e}  {connected: .12e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse exact diagonalization of the 1D PXP model"
    )
    parser.add_argument("--L", type=int, default=20, help="number of sites")
    parser.add_argument(
        "--boundary",
        choices=("obc", "pbc"),
        default="pbc",
        help="open or periodic boundary condition",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=1.0,
        help="coefficient multiplying sum_i P X P",
    )
    parser.add_argument(
        "--nev", type=int, default=4, help="number of lowest eigenvalues"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0e-11, help="Lanczos tolerance"
    )
    parser.add_argument(
        "--correlation-max",
        type=int,
        default=10,
        help="maximum separation for density correlations",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        help="compute half-chain entropy; recommended only for L <= 24",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--sizes",
        type=str,
#        default="4:24:2",
        default="4:36:2",
        help="inclusive start:stop:step or comma-separated sizes; used unless --L is explicitly given",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="run a size scan (default sizes: 4,6,...,24)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="directory for per-size JSON files and aggregate outputs",
    )
    parser.add_argument(
        "--summary-prefix",
        type=str,
        default="dat_1d_summary",
        help="basename for aggregate CSV and JSON files",
    )
    return parser.parse_args()


def parse_sizes(spec: str) -> List[int]:
    """Parse an inclusive start:stop:step range or a comma-separated list."""
    spec = spec.strip()
    if ":" in spec:
        fields = spec.split(":")
        if len(fields) != 3:
            raise ValueError("--sizes must be start:stop:step or a comma-separated list")
        start, stop, step = map(int, fields)
        if step == 0:
            raise ValueError("--sizes step must be nonzero")
        if (stop - start) * step < 0:
            raise ValueError("--sizes step has the wrong sign")
        sizes = list(range(start, stop + (1 if step > 0 else -1), step))
    else:
        sizes = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not sizes:
        raise ValueError("No system sizes were specified")
    if any(L < 2 or L > 63 for L in sizes):
        raise ValueError("Every L must satisfy 2 <= L <= 63")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Duplicate system sizes are not allowed")
    return sizes


def summary_as_dict(result: PXPResult) -> dict:
    """Return one row of the aggregate size-scan output."""
    data = result_as_dict(result)
    return {
        "L": result.L,
        "hilbert_dimension": result.hilbert_dimension,
        "energy": data["ground_state_energy"],
        "energy_density": data["energy_per_site"],
        "sigma_x": result.projected_flip,
        "sigma_z": 1.0 - 2.0 * result.density,
        "n": result.density,
        "gap": data["gap"],
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "L", "hilbert_dimension", "energy", "energy_density",
        "sigma_x", "sigma_z", "n", "gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def solve_one_size(args: argparse.Namespace, L: int) -> PXPResult:
    """Call the original solver for one L without changing its internals."""
    original_L = args.L
    args.L = L
    try:
        return solve(args)
    finally:
        args.L = original_L


def main() -> None:
    args = parse_args()
    if args.nev < 2:
        raise ValueError("--nev must be at least 2 because the gap is requested")
    if args.correlation_max < 0:
        raise ValueError("--correlation-max must be nonnegative")

    # Backward compatibility: --L without --scan keeps the old single-size mode.
    # Running without --L performs the requested default scan L=4,6,...,24.
    explicit_L = any(x == "--L" or x.startswith("--L=") for x in __import__("sys").argv[1:])
    scan_mode = args.scan or not explicit_L
    sizes = parse_sizes(args.sizes) if scan_mode else [args.L]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[dict] = []

    for L in sizes:
        print(f"\n>>> Solving L={L}")
        result = solve_one_size(args, L)
        print_result(result)
        detail = result_as_dict(result)

        if not scan_mode and args.output is not None:
            json_path = args.output
        else:
            json_path = args.output_dir / f"dat_1d_L{L:03d}_{args.boundary}.json"
        write_json(json_path, detail)
        print(f"Saved detailed JSON to: {json_path}")
        summary_rows.append(summary_as_dict(result))

    summary_csv = args.output_dir / f"{args.summary_prefix}.csv"
    summary_json = args.output_dir / f"{args.summary_prefix}.json"
    write_summary_csv(summary_csv, summary_rows)
    write_json(summary_json, summary_rows)
    print("\n=== Aggregate outputs ===")
    print(f"CSV  : {summary_csv}")
    print(f"JSON : {summary_json}")


if __name__ == "__main__":
    main()
