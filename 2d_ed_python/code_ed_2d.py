#!/usr/bin/env python3
"""Sparse exact diagonalization of the 2D PXP model on an L x L square lattice.

Hamiltonian
-----------
    H = coefficient * sum_i X_i prod_{j in NN(i)} P_j,
where P_j = |0><0| and the Hilbert space is restricted to configurations
with no simultaneously excited nearest-neighbor pair.

Both open (obc) and periodic (pbc) boundary conditions are supported in the
x and y directions.  The implementation is intended for L=3, 4, 5.

Examples
--------
python code_ed_2d.py --L 3 --boundary pbc --nev 4
python code_ed_2d.py --L 4 --boundary obc --nev 4 --output result.json
python code_ed_2d.py --scan --sizes 3,4,5 --boundary pbc --output-dir results
python code_ed_2d.py --L 5 --boundary obc --nev 2 --entropy

Site convention
---------------
site = x + L*y, with x,y = 0,...,L-1.  Bit 1 denotes an excitation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class PXP2DResult:
    L: int
    boundary: str
    coefficient: float
    hilbert_dimension: int
    eigenvalues: np.ndarray
    density: float
    local_density: np.ndarray
    correlations: np.ndarray
    structure_factor_00: float
    structure_factor_pipi: float
    projected_flip: float
    entanglement_entropy: float | None


def site_index(x: int, y: int, L: int) -> int:
    return x + L * y


def lattice_neighbors(L: int, periodic: bool) -> List[Tuple[int, ...]]:
    """Return unique nearest neighbors of every site."""
    neighbors: List[Tuple[int, ...]] = []
    for y in range(L):
        for x in range(L):
            values = set()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if periodic:
                    xx %= L
                    yy %= L
                elif not (0 <= xx < L and 0 <= yy < L):
                    continue
                j = site_index(xx, yy, L)
                i = site_index(x, y, L)
                if j != i:
                    values.add(j)
            neighbors.append(tuple(sorted(values)))
    return neighbors


def neighbor_masks(neighbors: Sequence[Sequence[int]]) -> np.ndarray:
    masks = np.zeros(len(neighbors), dtype=np.uint64)
    for i, nbs in enumerate(neighbors):
        mask = 0
        for j in nbs:
            mask |= 1 << j
        masks[i] = np.uint64(mask)
    return masks


def row_states(L: int, periodic_x: bool) -> List[int]:
    """Allowed bit patterns in one row."""
    result = []
    for row in range(1 << L):
        if row & (row << 1):
            continue
        if periodic_x and L > 2 and (row & 1) and ((row >> (L - 1)) & 1):
            continue
        result.append(row)
    return result


def generate_basis(L: int, periodic: bool) -> np.ndarray:
    """Generate all independent sets of the L x L square lattice row by row."""
#    if not 2 <= L <= 5:
#        raise ValueError("This implementation is intended for 2 <= L <= 5")
    if not 2 <= L <= 6:
        raise ValueError("This implementation is intended for 2 <= L <= 6")

    rows = row_states(L, periodic_x=periodic)
    compatible = {r: [s for s in rows if (r & s) == 0] for r in rows}
    states: List[int] = []

    def extend(y: int, first_row: int, previous_row: int, state: int) -> None:
        if y == L:
            if not periodic or (previous_row & first_row) == 0:
                states.append(state)
            return
        candidates = rows if y == 0 else compatible[previous_row]
        for row in candidates:
            if y == L - 1 and periodic and y > 0 and (row & first_row):
                continue
            new_first = row if y == 0 else first_row
            extend(y + 1, new_first, row, state | (row << (L * y)))

    extend(0, 0, 0, 0)
    return np.asarray(states, dtype=np.uint64)


def build_hamiltonian(
    basis: np.ndarray,
    masks: np.ndarray,
    coefficient: float,
) -> csr_matrix:
    """Construct H in the constrained basis."""
    index: Dict[int, int] = {int(s): i for i, s in enumerate(basis)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for row, state_u in enumerate(basis):
        state = int(state_u)
        for site, mask_u in enumerate(masks):
            if state & int(mask_u):
                continue
            new_state = state ^ (1 << site)
            col = index.get(new_state)
            if col is None:
                raise RuntimeError("A projected flip left the constrained basis")
            rows.append(row)
            cols.append(col)
            data.append(coefficient)

    dim = len(basis)
    hamiltonian = coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(dim, dim),
        dtype=np.float64,
    ).tocsr()
    hamiltonian.sum_duplicates()

    asymmetry = hamiltonian - hamiltonian.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1.0e-14:
        raise RuntimeError("Hamiltonian construction is not Hermitian")
    return hamiltonian


def lowest_states(
    hamiltonian: csr_matrix,
    nev: int,
    tolerance: float,
    maxiter: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = hamiltonian.shape[0]
    if dim <= 2 or nev >= dim:
        vals, vecs = np.linalg.eigh(hamiltonian.toarray())
        keep = min(nev, dim)
        return vals[:keep], vecs[:, :keep]

    k = min(max(1, nev), dim - 1)
    vals, vecs = eigsh(
        hamiltonian,
        k=k,
        which="SA",
        tol=tolerance,
        maxiter=maxiter if maxiter is not None else max(10000, 20 * dim),
    )
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def occupation_matrix(basis: np.ndarray, n_sites: int) -> np.ndarray:
    sites = np.arange(n_sites, dtype=np.uint64)
    return ((basis[:, None] >> sites[None, :]) & 1).astype(np.float64)


def displacement_correlations(
    probability: np.ndarray,
    occupations: np.ndarray,
    L: int,
    periodic: bool,
) -> np.ndarray:
    """Return translationally averaged density correlations by (dx,dy)."""
    local = probability @ occupations
    rows: List[Tuple[float, float, float, float, float]] = []
    max_d = L - 1
    for dy in range(max_d + 1):
        for dx in range(max_d + 1):
            raw_values: List[float] = []
            disconnected_values: List[float] = []
            for y in range(L):
                for x in range(L):
                    xx, yy = x + dx, y + dy
                    if periodic:
                        xx %= L
                        yy %= L
                    elif xx >= L or yy >= L:
                        continue
                    i = site_index(x, y, L)
                    j = site_index(xx, yy, L)
                    raw_values.append(
                        float(probability @ (occupations[:, i] * occupations[:, j]))
                    )
                    disconnected_values.append(float(local[i] * local[j]))
            raw = float(np.mean(raw_values))
            connected = raw - float(np.mean(disconnected_values))
            multiplicity = float(len(raw_values))
            rows.append((float(dx), float(dy), raw, connected, multiplicity))
    return np.asarray(rows, dtype=np.float64)


def structure_factor(
    probability: np.ndarray,
    occupations: np.ndarray,
    L: int,
    qx: float,
    qy: float,
) -> float:
    """Connected S(q)=N^-1 sum_ij exp[i q.(r_i-r_j)] C_ij."""
    means = probability @ occupations
    weighted = occupations * np.sqrt(probability[:, None])
    nn = weighted.T @ weighted
    connected = nn - np.outer(means, means)
    positions = np.asarray(
        [(x, y) for y in range(L) for x in range(L)], dtype=np.float64
    )
    phase = np.exp(1j * (qx * positions[:, 0] + qy * positions[:, 1]))
    value = np.vdot(phase, connected @ phase) / (L * L)
    return float(np.real_if_close(value))


def vertical_cut_entropy(psi: np.ndarray, basis: np.ndarray, L: int) -> float:
    """Von Neumann entropy across a vertical cut, using natural logarithms.

    The left subsystem contains columns x < L//2.  Only subsystem bit strings
    present in the constrained basis are retained, avoiding a full 2^(L^2)
    embedding.
    """
    left_columns = L // 2
    left_sites = [site_index(x, y, L) for y in range(L) for x in range(left_columns)]
    right_sites = [site_index(x, y, L) for y in range(L) for x in range(left_columns, L)]

    def compress(state: int, sites: Iterable[int]) -> int:
        out = 0
        for new_bit, old_bit in enumerate(sites):
            out |= ((state >> old_bit) & 1) << new_bit
        return out

    pairs = [(compress(int(s), left_sites), compress(int(s), right_sites)) for s in basis]
    left_values = sorted({a for a, _ in pairs})
    right_values = sorted({b for _, b in pairs})
    left_index = {value: i for i, value in enumerate(left_values)}
    right_index = {value: i for i, value in enumerate(right_values)}
    matrix = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    for amplitude, (left, right) in zip(psi, pairs):
        matrix[left_index[left], right_index[right]] = amplitude

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    weights = singular_values**2
    weights = weights[weights > 1.0e-15]
    return float(-np.sum(weights * np.log(weights)))


def solve(args: argparse.Namespace, L: int) -> PXP2DResult:
    periodic = args.boundary == "pbc"
    n_sites = L * L
    neighbors = lattice_neighbors(L, periodic)
    masks = neighbor_masks(neighbors)
    basis = generate_basis(L, periodic)
    hamiltonian = build_hamiltonian(basis, masks, args.coefficient)
    eigenvalues, eigenvectors = lowest_states(
        hamiltonian, args.nev, args.tolerance, args.maxiter
    )

    psi0 = eigenvectors[:, 0]
    probability = np.abs(psi0) ** 2
    probability /= probability.sum()
    occupations = occupation_matrix(basis, n_sites)
    local_density_flat = probability @ occupations
    density = float(np.mean(local_density_flat))
    correlations = displacement_correlations(
        probability, occupations, L, periodic
    )
    entropy = vertical_cut_entropy(psi0, basis, L) if args.entropy else None
    projected_flip = (
        float(eigenvalues[0] / (n_sites * args.coefficient))
        if args.coefficient != 0.0
        else float("nan")
    )

    return PXP2DResult(
        L=L,
        boundary=args.boundary,
        coefficient=args.coefficient,
        hilbert_dimension=len(basis),
        eigenvalues=eigenvalues,
        density=density,
        local_density=local_density_flat.reshape(L, L),
        correlations=correlations,
        structure_factor_00=structure_factor(probability, occupations, L, 0.0, 0.0),
        structure_factor_pipi=structure_factor(
            probability, occupations, L, np.pi, np.pi
        ),
        projected_flip=projected_flip,
        entanglement_entropy=entropy,
    )


def result_as_dict(result: PXP2DResult) -> dict:
    n_sites = result.L * result.L
    gap = (
        float(result.eigenvalues[1] - result.eigenvalues[0])
        if len(result.eigenvalues) >= 2
        else None
    )
    return {
        "L": result.L,
        "N": n_sites,
        "boundary": result.boundary,
        "coefficient": result.coefficient,
        "hilbert_dimension": result.hilbert_dimension,
        "eigenvalues": [float(x) for x in result.eigenvalues],
        "ground_state_energy": float(result.eigenvalues[0]),
        "energy_per_site": float(result.eigenvalues[0] / n_sites),
        "gap": gap,
        "density": result.density,
        "sigma_z_expectation_per_site": 1.0 - 2.0 * result.density,
        "projected_flip_expectation_per_site": result.projected_flip,
        "local_density_rows_y_columns_x": result.local_density.tolist(),
        "connected_structure_factor_q00": result.structure_factor_00,
        "connected_structure_factor_qpipi": result.structure_factor_pipi,
        "vertical_cut_von_neumann_entropy": result.entanglement_entropy,
        "correlations_columns": ["dx", "dy", "raw", "connected", "number_of_pairs"],
        "correlations": result.correlations.tolist(),
    }


def summary_as_dict(result: PXP2DResult) -> dict:
    data = result_as_dict(result)
    return {
        "L": result.L,
        "N": result.L * result.L,
        "hilbert_dimension": result.hilbert_dimension,
        "energy": data["ground_state_energy"],
        "energy_density": data["energy_per_site"],
        "sigma_x_projected": result.projected_flip,
        "sigma_z": 1.0 - 2.0 * result.density,
        "n": result.density,
        "gap": data["gap"],
        "S_00_connected": result.structure_factor_00,
        "S_pipi_connected": result.structure_factor_pipi,
    }


def print_result(result: PXP2DResult) -> None:
    data = result_as_dict(result)
    print("\n=== 2D PXP exact diagonalization ===")
    print(f"L x L                     : {result.L} x {result.L}")
    print(f"number of sites N         : {data['N']}")
    print(f"boundary                  : {result.boundary.upper()}")
    print(f"Hamiltonian coefficient   : {result.coefficient:.12g}")
    print(f"constrained Hilbert dim   : {result.hilbert_dimension}")
    print(f"ground-state energy E0    : {data['ground_state_energy']:.12f}")
    print(f"energy per site E0/N      : {data['energy_per_site']:.12f}")
    if data["gap"] is not None:
        print(f"lowest gap E1-E0          : {data['gap']:.12f}")
    print(f"mean excitation density   : {result.density:.12f}")
    print(f"<projected X> per site    : {result.projected_flip:.12f}")
    print(f"connected S(0,0)          : {result.structure_factor_00:.12f}")
    print(f"connected S(pi,pi)        : {result.structure_factor_pipi:.12f}")
    if result.entanglement_entropy is not None:
        print(f"vertical-cut entropy      : {result.entanglement_entropy:.12f}")
    print("\n# local density <n_(x,y)>; rows are y, columns are x")
    print(np.array2string(result.local_density, precision=10, suppress_small=False))
    print("\n# dx dy  <n_i n_(i+r)>       connected C(dx,dy)     number_of_pairs")
    for dx, dy, raw, connected, count in result.correlations:
        print(f"{int(dx):3d} {int(dy):3d}  {raw: .12e}  {connected: .12e}  {int(count):6d}")


def parse_sizes(spec: str) -> List[int]:
    values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("No sizes were specified")
#    if any(L < 2 or L > 5 for L in values):
#        raise ValueError("Every L must satisfy 2 <= L <= 5")
    if any(L < 2 or L > 6 for L in values):
        raise ValueError("Every L must satisfy 2 <= L <= 6")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate sizes are not allowed")
    return values


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "L", "N", "hilbert_dimension", "energy", "energy_density",
        "sigma_x_projected", "sigma_z", "n", "gap",
        "S_00_connected", "S_pipi_connected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse exact diagonalization of the 2D square-lattice PXP model"
    )
#    parser.add_argument("--L", type=int, default=3, help="linear size, 2 <= L <= 5")
    parser.add_argument("--L", type=int, default=3, help="linear size, 2 <= L <= 6")
    parser.add_argument(
        "--boundary", choices=("obc", "pbc"), default="pbc",
        help="open or periodic boundary conditions in both directions",
    )
    parser.add_argument(
        "--coefficient", type=float, default=1.0,
        help="coefficient multiplying sum_i X_i prod_(j in NN(i)) P_j",
    )
    parser.add_argument("--nev", type=int, default=4, help="number of lowest eigenvalues")
    parser.add_argument("--tolerance", type=float, default=1.0e-11, help="Lanczos tolerance")
    parser.add_argument(
        "--maxiter", type=int, default=None,
        help="maximum Lanczos iterations; default is max(10000,20*dim)",
    )
    parser.add_argument(
        "--entropy", action="store_true",
        help="compute vertical-cut von Neumann entropy",
    )
    parser.add_argument("--output", type=Path, default=None, help="single-size JSON path")
    parser.add_argument(
        "--scan", action="store_true", help="scan sizes supplied by --sizes"
    )
    parser.add_argument(
#        "--sizes", type=str, default="3,4,5", help="comma-separated sizes for --scan"
        "--sizes", type=str, default="3,4,5,6", help="comma-separated sizes for --scan"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./"),
        help="directory for detailed and aggregate outputs",
    )
    parser.add_argument(
        "--summary-prefix", type=str, default="dat_2d_summary",
        help="basename for aggregate CSV and JSON files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nev < 1:
        raise ValueError("--nev must be positive")
#    if not 2 <= args.L <= 5:
#        raise ValueError("--L must satisfy 2 <= L <= 5")
    if not 2 <= args.L <= 6:
        raise ValueError("--L must satisfy 2 <= L <= 6")

    explicit_L = any(x == "--L" or x.startswith("--L=") for x in sys.argv[1:])
    scan_mode = args.scan or not explicit_L
    sizes = parse_sizes(args.sizes) if scan_mode else [args.L]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    for L in sizes:
        print(f"\n>>> Solving L={L} ({L*L} sites)")
        result = solve(args, L)
        print_result(result)
        detail = result_as_dict(result)
        if not scan_mode and args.output is not None:
            json_path = args.output
        else:
            json_path = args.output_dir / f"dat_2d_L{L:02d}_{args.boundary}.json"
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
