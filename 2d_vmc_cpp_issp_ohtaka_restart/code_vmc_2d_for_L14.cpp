#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
struct MultiwordState {
  std::vector<std::uint64_t> words;
  MultiwordState() = default;
  explicit MultiwordState(int number_of_sites)
      : words((static_cast<std::size_t>(number_of_sites) + 63U) / 64U, 0ULL) {}
};
using State = MultiwordState;
namespace fs = std::filesystem;

struct Args {
  int Lx = 4;
  int Ly = 4;
  double jx = 1.0;
  double jn = 0.0;
  double jp = 0.0;
  double z = 0.65;
  int iterations = 200;
  int average_iterations = 100;
  int opt_samples = 10000;
  int final_samples = 100000;
  int opt_burn = 500;
  int final_burn = 2000;
  int thin = 1;
  double learning_rate = 0.03;
  double sr_diag = 1.0e-3;
  double max_step = 0.3;
  int pf_rebuild_interval = 256;
  int block_size = 500;
  std::uint64_t seed = 12345;
  int ed_max_sites = 16;
  std::string output_root = ".";
  std::string output_prefix = "dat";
  std::string restart_file = "";
  std::string checkpoint_file = "";
  int checkpoint_every = 1;
  double initial_noise = 0.0;
  std::uint64_t noise_seed = 67890;
  bool verbose = false;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string option_value(int &i, int argc, char **argv) {
  if (++i >= argc)
    fail("missing value for command-line option");
  return argv[i];
}

Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--Lx")
      a.Lx = std::stoi(option_value(i, argc, argv));
    else if (key == "--Ly")
      a.Ly = std::stoi(option_value(i, argc, argv));
    else if (key == "--jx")
      a.jx = std::stod(option_value(i, argc, argv));
    else if (key == "--jn")
      a.jn = std::stod(option_value(i, argc, argv));
    else if (key == "--jp")
      a.jp = std::stod(option_value(i, argc, argv));
    else if (key == "--z")
      a.z = std::stod(option_value(i, argc, argv));
    else if (key == "--iterations")
      a.iterations = std::stoi(option_value(i, argc, argv));
    else if (key == "--average-iterations")
      a.average_iterations = std::stoi(option_value(i, argc, argv));
    else if (key == "--opt-samples")
      a.opt_samples = std::stoi(option_value(i, argc, argv));
    else if (key == "--final-samples")
      a.final_samples = std::stoi(option_value(i, argc, argv));
    else if (key == "--opt-burn")
      a.opt_burn = std::stoi(option_value(i, argc, argv));
    else if (key == "--final-burn")
      a.final_burn = std::stoi(option_value(i, argc, argv));
    else if (key == "--thin")
      a.thin = std::stoi(option_value(i, argc, argv));
    else if (key == "--learning-rate")
      a.learning_rate = std::stod(option_value(i, argc, argv));
    else if (key == "--sr-diag")
      a.sr_diag = std::stod(option_value(i, argc, argv));
    else if (key == "--max-step")
      a.max_step = std::stod(option_value(i, argc, argv));
    else if (key == "--pf-rebuild-interval")
      a.pf_rebuild_interval = std::stoi(option_value(i, argc, argv));
    else if (key == "--block-size")
      a.block_size = std::stoi(option_value(i, argc, argv));
    else if (key == "--seed")
      a.seed = std::stoull(option_value(i, argc, argv));
    else if (key == "--ed-max-sites")
      a.ed_max_sites = std::stoi(option_value(i, argc, argv));
    else if (key == "--output-root")
      a.output_root = option_value(i, argc, argv);
    else if (key == "--output-prefix")
      a.output_prefix = option_value(i, argc, argv);
    else if (key == "--restart")
      a.restart_file = option_value(i, argc, argv);
    else if (key == "--checkpoint-file")
      a.checkpoint_file = option_value(i, argc, argv);
    else if (key == "--checkpoint-every")
      a.checkpoint_every = std::stoi(option_value(i, argc, argv));
    else if (key == "--initial-noise")
      a.initial_noise = std::stod(option_value(i, argc, argv));
    else if (key == "--noise-seed")
      a.noise_seed = std::stoull(option_value(i, argc, argv));
    else if (key == "--verbose")
      a.verbose = true;
    else if (key == "--help") {
      std::cout
          << "Kasteleyn/projective pBCS VMC options:\n"
          << "  --Lx INT --Ly INT --jx FLOAT --jn FLOAT --jp FLOAT --z FLOAT\n"
          << "  --iterations INT --average-iterations INT\n"
          << "  --opt-samples INT --final-samples INT\n"
          << "  --opt-burn INT --final-burn INT --thin INT\n"
          << "  --learning-rate FLOAT --sr-diag FLOAT --max-step FLOAT\n"
          << "  --pf-rebuild-interval INT --block-size INT --seed INT\n"
          << "  --ed-max-sites INT --output-root PATH --output-prefix NAME\n"
          << "  --restart FILE --checkpoint-file FILE --checkpoint-every INT\n"
          << "  --initial-noise FLOAT --noise-seed INT --verbose\n";
      std::exit(0);
    } else {
      fail("unknown option: " + key);
    }
  }

  const int N = a.Lx * a.Ly;
  if (a.Lx < 2 || a.Ly < 2)
    fail("Lx and Ly must be >= 2");
  if (a.Lx % 2 != 0)
    fail("the current Kasteleyn sector requires even Lx");
  if (N % 2 != 0)
    fail("Lx*Ly must be even");
  if (a.iterations < 1)
    fail("iterations must be positive");
  if (a.average_iterations < 1 || a.average_iterations > a.iterations)
    fail("average-iterations must satisfy 1 <= average-iterations <= "
         "iterations");
  if (a.opt_samples < 1 || a.final_samples < 1)
    fail("sample counts must be positive");
  if (a.block_size < 1)
    fail("block-size must be positive");
  if (a.checkpoint_every < 0)
    fail("checkpoint-every must be nonnegative");
  if (!(a.initial_noise >= 0.0) || !std::isfinite(a.initial_noise))
    fail("initial-noise must be finite and nonnegative");
  return a;
}

struct Lattice {
  int Lx;
  int Ly;
  int N;
  std::vector<std::vector<int>> neighbors;

  Lattice(int lx, int ly) : Lx(lx), Ly(ly), N(lx * ly), neighbors(N) {
    for (int y = 0; y < Ly; ++y) {
      for (int x = 0; x < Lx; ++x) {
        const int i = x + Lx * y;
        std::vector<int> n = {(x + Lx - 1) % Lx + Lx * y, (x + 1) % Lx + Lx * y,
                              x + Lx * ((y + Ly - 1) % Ly),
                              x + Lx * ((y + 1) % Ly)};
        std::sort(n.begin(), n.end());
        n.erase(std::unique(n.begin(), n.end()), n.end());
        n.erase(std::remove(n.begin(), n.end(), i), n.end());
        neighbors[i] = std::move(n);
      }
    }
  }
};

inline bool occupied(const State &state, int site) {
  const std::size_t word = static_cast<std::size_t>(site) >> 6;
  const unsigned offset = static_cast<unsigned>(site) & 63U;
  return ((state.words[word] >> offset) & 1ULL) != 0ULL;
}
inline void flip_site(State &state, int site) {
  const std::size_t word = static_cast<std::size_t>(site) >> 6;
  const unsigned offset = static_cast<unsigned>(site) & 63U;
  state.words[word] ^= (1ULL << offset);
}
int particle_count(const State &state) {
  int result = 0;
  for (std::uint64_t value : state.words) {
#if defined(__GNUC__) || defined(__clang__)
    result += __builtin_popcountll(value);
#else
    while (value != 0ULL) {
      value &= value - 1ULL;
      ++result;
    }
#endif
  }
  return result;
}

bool is_blockaded(const State &state, const Lattice &lattice) {
  for (int i = 0; i < lattice.N; ++i) {
    if (!occupied(state, i))
      continue;
    for (int j : lattice.neighbors[i]) {
      if (occupied(state, j))
        return false;
    }
  }
  return true;
}

bool can_flip(const State &state, int site, const Lattice &lattice) {
  if (occupied(state, site))
    return true;
  for (int j : lattice.neighbors[site]) {
    if (occupied(state, j))
      return false;
  }
  return true;
}

int empty_star_count(const State &state, const Lattice &lattice) {
  int count = 0;
  for (int i = 0; i < lattice.N; ++i) {
    if (!occupied(state, i) && can_flip(state, i, lattice))
      ++count;
  }
  return count;
}

double diagonal_energy(const State &state, const Lattice &lattice, double jn,
                       double jp) {
  return jn * particle_count(state) + jp * empty_star_count(state, lattice);
}

MatrixXd kasteleyn_sign_kernel(int Lx, int Ly, int twist_x = -1,
                               int twist_y = -1) {
  const int N = Lx * Ly;
  MatrixXd S = MatrixXd::Zero(N, N);
  auto add_edge = [&S](int i, int j, double sign) {
    S(i, j) += sign;
    S(j, i) -= sign;
  };

  for (int y = 0; y < Ly; ++y) {
    for (int x = 0; x < Lx; ++x) {
      const int i = x + Lx * y;
      const int jx = (x + 1) % Lx + Lx * y;
      const double sx = (x == Lx - 1) ? twist_x : 1.0;
      add_edge(i, jx, sx);

      const int jy = x + Lx * ((y + 1) % Ly);
      double sy = (x & 1) ? -1.0 : 1.0;
      if (y == Ly - 1)
        sy *= twist_y;
      add_edge(i, jy, sy);
    }
  }
  return 0.5 * (S - S.transpose());
}

std::pair<MatrixXd, MatrixXd>
kasteleyn_projective_translations(int Lx, int Ly, int twist_x = -1,
                                  int twist_y = -1) {
  const int N = Lx * Ly;
  MatrixXd Tx = MatrixXd::Zero(N, N);
  MatrixXd Ty = MatrixXd::Zero(N, N);
  for (int y = 0; y < Ly; ++y) {
    for (int x = 0; x < Lx; ++x) {
      const int src = x + Lx * y;
      const int dst_x = (x + 1) % Lx + Lx * y;
      double gx = (y & 1) ? -1.0 : 1.0;
      if (x == Lx - 1)
        gx *= twist_x;
      Tx(dst_x, src) = gx;

      const int dst_y = x + Lx * ((y + 1) % Ly);
      const double gy = (y == Ly - 1) ? twist_y : 1.0;
      Ty(dst_y, src) = gy;
    }
  }
  return {Tx, Ty};
}

std::vector<MatrixXd> matrix_powers(const MatrixXd &U, int count) {
  std::vector<MatrixXd> result;
  result.reserve(count);
  result.push_back(MatrixXd::Identity(U.rows(), U.cols()));
  for (int i = 1; i < count; ++i)
    result.push_back(result.back() * U);
  return result;
}

struct ProjectiveBasis {
  std::vector<MatrixXd> matrices;
  std::vector<std::string> labels;
  MatrixXd Tx_site;
  MatrixXd Ty_site;
  MatrixXd sign_kernel;
};

ProjectiveBasis build_kasteleyn_translation_basis(int Lx, int Ly,
                                                  double tolerance = 1.0e-12) {
  const int N = Lx * Ly;
  auto [Tx, Ty] = kasteleyn_projective_translations(Lx, Ly);
  const auto px = matrix_powers(Tx, Lx);
  const auto py = matrix_powers(Ty, Ly);

  std::vector<MatrixXd> translations;
  translations.reserve(N);
  for (int y = 0; y < Ly; ++y)
    for (int x = 0; x < Lx; ++x)
      translations.push_back(px[x] * py[y]);

  ProjectiveBasis output;
  output.Tx_site = Tx;
  output.Ty_site = Ty;
  output.sign_kernel = kasteleyn_sign_kernel(Lx, Ly);
  std::vector<VectorXd> orthonormal;

  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int dy = 0; dy < Ly; ++dy) {
        for (int dx = 0; dx < Lx; ++dx) {
          const int origin = 0;
          const int target = (dx % Lx) + Lx * (dy % Ly);
          const int ia = a * N + origin;
          const int jb = b * N + target;
          if (ia == jb)
            continue;

          MatrixXd B = MatrixXd::Zero(2 * N, 2 * N);

          // Tsite is a signed permutation matrix, while the original seed has
          // only seed(ia,jb)=+1 and seed(jb,ia)=-1.  Thus the conjugation
          // Torb*seed*Torb^T is obtained by translating the two nonzero
          // indices and multiplying their signs; no dense matrix product is
          // required.
          for (const MatrixXd &Tsite : translations) {
            auto signed_image = [&](int orbital_index) {
              const int orbital = orbital_index / N;
              const int site = orbital_index % N;
              for (int translated_site = 0; translated_site < N;
                   ++translated_site) {
                const double value = Tsite(translated_site, site);
                if (std::abs(value) > tolerance)
                  return std::make_pair(orbital * N + translated_site, value);
              }
              fail("projective translation is not a signed permutation");
              return std::make_pair(-1, 0.0); // Unreachable.
            };

            const auto image_i = signed_image(ia);
            const auto image_j = signed_image(jb);
            const double translated_sign = image_i.second * image_j.second;
            B(image_i.first, image_j.first) += translated_sign;
            B(image_j.first, image_i.first) -= translated_sign;
          }
          B /= static_cast<double>(translations.size());
          B = (0.5 * (B - B.transpose())).eval();
          const double norm = B.norm();
          if (norm < tolerance)
            continue;
          B /= norm;

          VectorXd v = Eigen::Map<VectorXd>(B.data(), B.size());
          for (const VectorXd &q : orthonormal)
            v.noalias() -= q.dot(v) * q;
          const double residual = v.norm();
          if (residual < tolerance)
            continue;
          orthonormal.push_back(v / residual);
          output.matrices.push_back(B);
          output.labels.push_back(
              "a=" + std::to_string(a) + ",b=" + std::to_string(b) +
              ",dx=" + std::to_string(dx) + ",dy=" + std::to_string(dy));
        }
      }
    }
  }
  return output;
}

MatrixXd exact_projected_product_pairing(int Lx, int Ly, double z) {
  const int N = Lx * Ly;
  VectorXd spinor(2);
  spinor << 1.0, -z;
  spinor.normalize();
  const MatrixXd S = kasteleyn_sign_kernel(Lx, Ly);
  MatrixXd F(2 * N, 2 * N);
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      F.block(a * N, b * N, N, N) = spinor(a) * spinor(b) * S;
  return (0.5 * (F - F.transpose())).eval();
}

VectorXd project_to_basis(const MatrixXd &target,
                          const std::vector<MatrixXd> &basis,
                          double &max_residual) {
  MatrixXd X(target.size(), static_cast<int>(basis.size()));
  for (int p = 0; p < static_cast<int>(basis.size()); ++p)
    X.col(p) = Eigen::Map<const VectorXd>(basis[p].data(), basis[p].size());
  const VectorXd y = Eigen::Map<const VectorXd>(target.data(), target.size());
  const VectorXd theta = X.colPivHouseholderQr().solve(y);
  const VectorXd difference = X * theta - y;
  max_residual = difference.cwiseAbs().maxCoeff();
  return theta;
}

MatrixXd make_pairing_matrix(const VectorXd &theta,
                             const std::vector<MatrixXd> &basis) {
  MatrixXd F = MatrixXd::Zero(basis.front().rows(), basis.front().cols());
  for (int p = 0; p < theta.size(); ++p)
    F.noalias() += theta(p) * basis[p];
  return (0.5 * (F - F.transpose())).eval();
}

std::pair<double, double>
projective_symmetry_residual(const MatrixXd &F, const ProjectiveBasis &basis,
                             int N) {
  MatrixXd Tx = MatrixXd::Zero(2 * N, 2 * N);
  MatrixXd Ty = MatrixXd::Zero(2 * N, 2 * N);
  Tx.topLeftCorner(N, N) = basis.Tx_site;
  Tx.bottomRightCorner(N, N) = basis.Tx_site;
  Ty.topLeftCorner(N, N) = basis.Ty_site;
  Ty.bottomRightCorner(N, N) = basis.Ty_site;
  const double rx = (Tx * F * Tx.transpose() - F).cwiseAbs().maxCoeff();
  const double ry = (Ty * F * Ty.transpose() - F).cwiseAbs().maxCoeff();
  return {rx, ry};
}

MatrixXd scaled_skew_inverse(const MatrixXd &A) {
  const double scale = A.cwiseAbs().maxCoeff();
  if (!(scale > 0.0) || !std::isfinite(scale))
    fail("selected Pfaffian matrix has zero or invalid scale");
  const MatrixXd scaled = A / scale;
  Eigen::FullPivLU<MatrixXd> lu(scaled);
  if (!lu.isInvertible())
    fail("selected Pfaffian matrix is singular");
  MatrixXd inverse = lu.inverse() / scale;
  if (!inverse.allFinite())
    fail("selected inverse contains non-finite values");
  return inverse;
}

void build_selected_matrix(const MatrixXd &F, const State &state, int N,
                           std::vector<int> &q, MatrixXd &A) {
  for (int i = 0; i < N; ++i)
    q[i] = occupied(state, i) ? N + i : i;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      A(i, j) = F(q[i], q[j]);
}

void rebuild_selected(const MatrixXd &F, const State &state, int N,
                      std::vector<int> &q, MatrixXd &A, MatrixXd &Ainv) {
  build_selected_matrix(F, state, N, q, A);
  Ainv = scaled_skew_inverse(A);
}

VectorXd replacement_row(const MatrixXd &F, const std::vector<int> &q, int site,
                         int N) {
  const int new_orbital = (q[site] >= N) ? site : N + site;
  VectorXd row(N);
  for (int j = 0; j < N; ++j)
    row(j) = F(new_orbital, q[j]);
  row(site) = 0.0;
  return row;
}

double pfaffian_replace_ratio(const MatrixXd &Ainv, const VectorXd &row,
                              int site) {
  return row.dot(Ainv.col(site));
}

// O(N^2) skew-symmetric row/column replacement update.
bool inverse_replace_inplace(MatrixXd &A, MatrixXd &Ainv, const VectorXd &row,
                             int site, double tolerance = 1.0e-12) {
  const int N = A.rows();
  const VectorXd d = row - A.row(site).transpose();
  const VectorXd u = Ainv.col(site);
  const double ratio = 1.0 + d.dot(u);
  if (!std::isfinite(ratio) || std::abs(ratio) < tolerance)
    return false;

  const VectorXd x = (-Ainv * d).eval();
  const MatrixXd correction =
      ((u * x.transpose() - x * u.transpose()) / ratio).eval();
  Ainv -= correction;
  Ainv = (0.5 * (Ainv - Ainv.transpose())).eval();
  Ainv.diagonal().setZero();

  for (int j = 0; j < N; ++j) {
    A(site, j) = row(j);
    A(j, site) = -row(j);
  }
  A(site, site) = 0.0;
  return Ainv.allFinite();
}

VectorXd log_derivatives(const std::vector<int> &q, const MatrixXd &Ainv,
                         const std::vector<MatrixXd> &basis) {
  const int N = static_cast<int>(q.size());
  VectorXd result = VectorXd::Zero(static_cast<int>(basis.size()));
  for (int p = 0; p < static_cast<int>(basis.size()); ++p) {
    double value = 0.0;
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j)
        value += Ainv(i, j) * basis[p](q[j], q[i]);
    result(p) = value;
  }
  return result;
}

struct Samples {
  VectorXd energy;
  VectorXd mz;
  VectorXd mx;
  MatrixXd log_derivative;
  State state;
  double accepted = 0.0;
  double proposed = 0.0;
};

Samples sample_chain(const VectorXd &theta, const std::vector<MatrixXd> &basis,
                     const Lattice &lattice, const Args &args, int samples,
                     int burn, State initial_state, std::uint64_t seed,
                     bool need_log_derivative) {
  const int N = lattice.N;
  const MatrixXd F = make_pairing_matrix(theta, basis);
  State state =
      is_blockaded(initial_state, lattice) ? initial_state : State(lattice.N);
  std::vector<int> q(N);
  MatrixXd A(N, N), Ainv;
  rebuild_selected(F, state, N, q, A, Ainv);

  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::uniform_int_distribution<int> random_site(0, N - 1);

  Samples output;
  output.energy.resize(samples);
  output.mz.resize(samples);
  output.mx.resize(samples);
  if (need_log_derivative)
    output.log_derivative.resize(samples, static_cast<int>(basis.size()));

  int measured = 0;
  int since_rebuild = 0;
  const int total_sweeps = burn + samples * args.thin;
  for (int sweep = 0; sweep < total_sweeps; ++sweep) {
    for (int attempt = 0; attempt < N; ++attempt) {
      const int site = random_site(generator);
      output.proposed += 1.0;
      if (!can_flip(state, site, lattice))
        continue;

      const VectorXd row = replacement_row(F, q, site, N);
      const double ratio = pfaffian_replace_ratio(Ainv, row, site);
      const double ratio2 = ratio * ratio;
      if (!std::isfinite(ratio2) ||
          !(ratio2 >= 1.0 || uniform(generator) < ratio2))
        continue;

      const bool update_ok = inverse_replace_inplace(A, Ainv, row, site);
      flip_site(state, site);
      output.accepted += 1.0;
      ++since_rebuild;
      if (update_ok)
        q[site] = (q[site] >= N) ? site : N + site;
      if (!update_ok || (args.pf_rebuild_interval > 0 &&
                         since_rebuild >= args.pf_rebuild_interval)) {
        rebuild_selected(F, state, N, q, A, Ainv);
        since_rebuild = 0;
      }
    }

    if (sweep >= burn && (sweep - burn) % args.thin == 0) {
      double local_energy = diagonal_energy(state, lattice, args.jn, args.jp);
      double mx = 0.0;
      for (int site = 0; site < N; ++site) {
        if (!can_flip(state, site, lattice))
          continue;
        const VectorXd row = replacement_row(F, q, site, N);
        const double ratio = pfaffian_replace_ratio(Ainv, row, site);
        local_energy += args.jx * ratio;
        mx += ratio;
      }
      output.energy(measured) = local_energy;
      output.mz(measured) =
          (N - 2.0 * particle_count(state)) / static_cast<double>(N);
      output.mx(measured) = mx / static_cast<double>(N);
      if (need_log_derivative)
        output.log_derivative.row(measured) =
            log_derivatives(q, Ainv, basis).transpose();
      ++measured;
    }
  }
  output.state = state;
  return output;
}

void allreduce_vector(VectorXd &value) {
  VectorXd input = value;
  MPI_Allreduce(input.data(), value.data(), static_cast<int>(value.size()),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void allreduce_matrix(MatrixXd &value) {
  MatrixXd input = value;
  MPI_Allreduce(input.data(), value.data(), static_cast<int>(value.size()),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

struct HistoryRow {
  int iteration;
  double energy;
  double variance;
  double gradient_norm;
  double acceptance;
};

struct MeanError {
  double mean;
  double error;
};

MeanError binned_mean_error(const VectorXd &values, int block_size) {
  const int blocks = static_cast<int>(values.size()) / block_size;
  if (blocks < 2)
    return {values.mean(), std::numeric_limits<double>::quiet_NaN()};
  VectorXd means(blocks);
  for (int b = 0; b < blocks; ++b)
    means(b) = values.segment(b * block_size, block_size).mean();
  const double mean = means.mean();
  const double error = std::sqrt((means.array() - mean).square().sum() /
                                 (blocks * (blocks - 1.0)));
  return {mean, error};
}

struct VScoreResult {
  double value;
  double error;
};

VScoreResult conditional_v_score(const VectorXd &energy, int N, int block_size,
                                 bool enabled) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (!enabled)
    return {nan, nan};
  const double E = energy.mean();
  const double variance = (energy.array() - E).square().mean();
  if (!std::isfinite(E) || E == 0.0)
    return {nan, nan};
  const double value = N * variance / (E * E);
  const int blocks = static_cast<int>(energy.size()) / block_size;
  if (blocks < 2)
    return {value, nan};
  VectorXd scores(blocks);
  int valid = 0;
  for (int b = 0; b < blocks; ++b) {
    const VectorXd block = energy.segment(b * block_size, block_size);
    const double Eb = block.mean();
    if (!std::isfinite(Eb) || Eb == 0.0)
      continue;
    const double Vb = (block.array() - Eb).square().mean();
    scores(valid++) = N * Vb / (Eb * Eb);
  }
  if (valid < 2)
    return {value, nan};
  const double mean = scores.head(valid).mean();
  const double error =
      std::sqrt((scores.head(valid).array() - mean).square().sum() /
                (valid * (valid - 1.0)));
  return {value, error};
}

std::pair<double, int> exact_ground_state(const Lattice &lattice,
                                          const Args &args) {
  if (lattice.N > 24)
    return {std::numeric_limits<double>::quiet_NaN(), 0};
  using EDState = std::uint64_t;
  auto occ = [](EDState state, int site) {
    return ((state >> site) & 1ULL) != 0ULL;
  };
  auto blockaded = [&](EDState state) {
    for (int i = 0; i < lattice.N; ++i)
      if (occ(state, i))
        for (int j : lattice.neighbors[i])
          if (occ(state, j))
            return false;
    return true;
  };
  auto flip_allowed = [&](EDState state, int site) {
    if (occ(state, site))
      return true;
    for (int j : lattice.neighbors[site])
      if (occ(state, j))
        return false;
    return true;
  };
  auto particles = [](EDState state) { return __builtin_popcountll(state); };
  auto empty_stars = [&](EDState state) {
    int count = 0;
    for (int i = 0; i < lattice.N; ++i)
      if (!occ(state, i) && flip_allowed(state, i))
        ++count;
    return count;
  };
  std::vector<EDState> states;
  const EDState end_state = EDState{1} << lattice.N;
  for (EDState state = 0; state < end_state; ++state)
    if (blockaded(state))
      states.push_back(state);
  const int dimension = static_cast<int>(states.size());
  std::unordered_map<EDState, int> index;
  index.reserve(states.size());
  for (int i = 0; i < dimension; ++i)
    index[states[i]] = i;
  MatrixXd H = MatrixXd::Zero(dimension, dimension);
  for (int a = 0; a < dimension; ++a) {
    const EDState state = states[a];
    H(a, a) = args.jn * particles(state) + args.jp * empty_stars(state);
    for (int site = 0; site < lattice.N; ++site) {
      if (!flip_allowed(state, site))
        continue;
      const EDState flipped = state ^ (EDState{1} << site);
      H(a, index.at(flipped)) += args.jx;
    }
  }
  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(H, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success)
    fail("dense exact diagonalization failed");
  return {solver.eigenvalues()(0), dimension};
}

std::string parameter_tag(double value) {
  std::ostringstream out;
  out << std::setprecision(12) << value;
  std::string result = out.str();
  for (char &c : result) {
    if (c == '-')
      c = 'm';
    else if (c == '+')
      c = 'p';
    else if (c == '.')
      c = 'p';
  }
  return result;
}

struct RestartData {
  VectorXd theta;
  int completed_iterations = 0;
};

RestartData load_restart(const fs::path &path, int expected_Lx, int expected_Ly,
                         const ProjectiveBasis &basis) {
  std::ifstream input(path);
  if (!input)
    fail("cannot open restart file: " + path.string());
  RestartData result;
  result.theta = VectorXd::Zero(static_cast<int>(basis.matrices.size()));
  std::vector<bool> seen(static_cast<std::size_t>(result.theta.size()), false);
  int file_Lx = -1, file_Ly = -1;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("# Lx=", 0) == 0) {
      file_Lx = std::stoi(line.substr(5));
      continue;
    }
    if (line.rfind("# Ly=", 0) == 0) {
      file_Ly = std::stoi(line.substr(5));
      continue;
    }
    if (line.rfind("# completed_iterations=", 0) == 0) {
      result.completed_iterations = std::stoi(line.substr(23));
      continue;
    }
    if (line.empty() || line[0] == '#' || line.rfind("index,", 0) == 0)
      continue;
    const std::size_t first = line.find(',');
    const std::size_t last = line.rfind(',');
    if (first == std::string::npos || last == first)
      fail("malformed restart row: " + line);
    const int index = std::stoi(line.substr(0, first));
    const double value = std::stod(line.substr(last + 1));
    if (index < 0 || index >= result.theta.size())
      fail("restart parameter index out of range");
    result.theta(index) = value;
    seen[static_cast<std::size_t>(index)] = true;
  }
  if ((file_Lx >= 0 && file_Lx != expected_Lx) ||
      (file_Ly >= 0 && file_Ly != expected_Ly))
    fail("restart lattice size does not match --Lx/--Ly");
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    fail("restart file does not contain every variational parameter");
  const double norm = result.theta.norm();
  if (!(norm > 0.0) || !std::isfinite(norm))
    fail("restart theta has invalid norm");
  result.theta /= norm;
  return result;
}

void add_gaussian_noise(VectorXd &theta, double sigma, std::uint64_t seed) {
  if (sigma == 0.0)
    return;
  std::mt19937_64 generator(seed);
  std::normal_distribution<double> normal(0.0, sigma);
  for (int p = 0; p < theta.size(); ++p)
    theta(p) += normal(generator);
  const double norm = theta.norm();
  if (!(norm > 0.0) || !std::isfinite(norm))
    fail("noisy initial theta has invalid norm");
  theta /= norm;
}

void save_checkpoint_atomic(const fs::path &path, int Lx, int Ly,
                            int completed_iterations, const VectorXd &theta,
                            const ProjectiveBasis &basis) {
  fs::path temporary = path;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
      fail("cannot write checkpoint: " + temporary.string());
    output << "# code_vmc_2d checkpoint\n"
           << "# Lx=" << Lx << "\n"
           << "# Ly=" << Ly << "\n"
           << "# completed_iterations=" << completed_iterations << "\n"
           << "index,label,theta\n";
    for (int p = 0; p < theta.size(); ++p)
      output << p << ",\"" << basis.labels[p] << "\"," << std::setprecision(17)
             << theta(p) << "\n";
    output.flush();
    if (!output)
      fail("failed while writing checkpoint: " + temporary.string());
  }
  std::error_code error;
  fs::rename(temporary, path, error);
  if (error) {
    fs::remove(path, error);
    error.clear();
    fs::rename(temporary, path, error);
  }
  if (error)
    fail("cannot install checkpoint: " + path.string() + ": " +
         error.message());
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    const Args args = parse_args(argc, argv);
    const auto start = std::chrono::steady_clock::now();
    const Lattice lattice(args.Lx, args.Ly);
    ProjectiveBasis basis = build_kasteleyn_translation_basis(args.Lx, args.Ly);

    double projection_residual = 0.0;
    VectorXd theta = project_to_basis(
        exact_projected_product_pairing(args.Lx, args.Ly, args.z),
        basis.matrices, projection_residual);
    theta.normalize();
    int completed_before = 0;
    if (!args.restart_file.empty()) {
      if (rank == 0) {
        const RestartData restart =
            load_restart(args.restart_file, args.Lx, args.Ly, basis);
        theta = restart.theta;
        completed_before = restart.completed_iterations;
      }
      MPI_Bcast(&completed_before, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(theta.data(), static_cast<int>(theta.size()), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    }
    if (rank == 0)
      add_gaussian_noise(theta, args.initial_noise, args.noise_seed);
    MPI_Bcast(theta.data(), static_cast<int>(theta.size()), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    const fs::path directory =
        fs::path(args.output_root) /
        ("Lx" + std::to_string(args.Lx) + "_Ly" + std::to_string(args.Ly) +
         "_jx" + parameter_tag(args.jx) + "_jn" + parameter_tag(args.jn) +
         "_jp" + parameter_tag(args.jp));
    const fs::path prefix = directory / fs::path(args.output_prefix).filename();
    const fs::path checkpoint =
        args.checkpoint_file.empty()
            ? fs::path(prefix.string() + "_checkpoint.csv")
            : fs::path(args.checkpoint_file);
    if (rank == 0) {
      fs::create_directories(directory);
      if (!checkpoint.parent_path().empty())
        fs::create_directories(checkpoint.parent_path());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MatrixXd F = make_pairing_matrix(theta, basis.matrices);
    auto [symmetry_x, symmetry_y] =
        projective_symmetry_residual(F, basis, lattice.N);

    Eigen::JacobiSVD<MatrixXd> svd(basis.sign_kernel);
    const double sigma_min = svd.singularValues().minCoeff();
    if (projection_residual > 1.0e-10)
      fail("initializer projection residual is too large");
    if (sigma_min < 1.0e-12)
      fail("Kasteleyn sign kernel is singular in the selected torus sector");

    if (rank == 0) {
      std::cout << "N=" << lattice.N
                << ", Kasteleyn/projective parameters=" << theta.size()
                << ", initializer residual=" << std::scientific
                << projection_residual
                << ", restart_completed=" << completed_before
                << ", initial_noise=" << args.initial_noise << "\n";
      std::cout << "Kasteleyn-kernel sigma_min=" << sigma_min
                << "; symmetry residuals Tx=" << symmetry_x
                << ", Ty=" << symmetry_y << "\n";
    }

    double ed_energy = std::numeric_limits<double>::quiet_NaN();
    int ed_dimension = 0;
    if (rank == 0 && lattice.N <= args.ed_max_sites) {
      std::tie(ed_energy, ed_dimension) = exact_ground_state(lattice, args);
      std::cout << "ED ground-state total energy=" << std::setprecision(12)
                << std::fixed << ed_energy
                << ", blockade dimension=" << ed_dimension << "\n";
    }

    State state(lattice.N);
    std::vector<HistoryRow> history;
    std::vector<VectorXd> theta_tail;
    for (int iteration = 0; iteration < args.iterations; ++iteration) {
      const int global_iteration = completed_before + iteration;
      const int local_samples =
          args.opt_samples / ranks + (rank < args.opt_samples % ranks ? 1 : 0);
      Samples samples = sample_chain(
          theta, basis.matrices, lattice, args, local_samples, args.opt_burn,
          state, args.seed + 1000003ULL * global_iteration + rank, true);
      state = samples.state;

      double local_scalars[5] = {
          static_cast<double>(local_samples), samples.energy.sum(),
          samples.energy.squaredNorm(), samples.accepted, samples.proposed};
      double global_scalars[5] = {};
      MPI_Allreduce(local_scalars, global_scalars, 5, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      VectorXd sum_O = samples.log_derivative.colwise().sum();
      VectorXd sum_OE = samples.log_derivative.transpose() * samples.energy;
      MatrixXd sum_OO =
          samples.log_derivative.transpose() * samples.log_derivative;
      allreduce_vector(sum_O);
      allreduce_vector(sum_OE);
      allreduce_matrix(sum_OO);

      const double count = global_scalars[0];
      const double energy = global_scalars[1] / count;
      const double variance = global_scalars[2] / count - energy * energy;
      const VectorXd mean_O = sum_O / count;
      MatrixXd metric = sum_OO / count - mean_O * mean_O.transpose();
      const VectorXd gradient = 2.0 * (sum_OE / count - mean_O * energy);
      const double regularizer =
          args.sr_diag * std::max(0.1, std::pow(0.95, global_iteration));
      for (int p = 0; p < metric.rows(); ++p)
        metric(p, p) += regularizer * std::max(metric(p, p), 1.0e-8) + 1.0e-10;

      VectorXd delta = metric.ldlt().solve(-gradient);
      if (!delta.allFinite())
        delta = metric.completeOrthogonalDecomposition().solve(-gradient);
      if (!delta.allFinite())
        fail("SR solve produced non-finite update");
      if (delta.norm() > args.max_step)
        delta *= args.max_step / delta.norm();
      theta += args.learning_rate * delta;
      theta.normalize();

      if (iteration >= args.iterations - args.average_iterations)
        theta_tail.push_back(theta);
      const double acceptance =
          global_scalars[3] / std::max(1.0, global_scalars[4]);
      history.push_back(
          {global_iteration, energy, variance, gradient.norm(), acceptance});
      if (rank == 0 && args.checkpoint_every > 0 &&
          ((iteration + 1) % args.checkpoint_every == 0 ||
           iteration + 1 == args.iterations))
        save_checkpoint_atomic(checkpoint, args.Lx, args.Ly,
                               global_iteration + 1, theta, basis);
      if (rank == 0 && args.verbose &&
          (iteration % 10 == 0 || iteration + 1 == args.iterations)) {
        std::cout << "SR " << std::setw(4) << global_iteration
                  << " E=" << std::showpos << std::setprecision(10)
                  << std::fixed << energy << std::noshowpos
                  << " var=" << std::scientific << variance
                  << " |g|=" << gradient.norm() << " acc=" << std::fixed
                  << std::setprecision(3) << acceptance << "\n";
      }
    }

    theta.setZero();
    for (const VectorXd &value : theta_tail)
      theta += value;
    theta /= static_cast<double>(theta_tail.size());
    theta.normalize();
    F = make_pairing_matrix(theta, basis.matrices);
    std::tie(symmetry_x, symmetry_y) =
        projective_symmetry_residual(F, basis, lattice.N);

    const int local_final = args.final_samples / ranks +
                            (rank < args.final_samples % ranks ? 1 : 0);
    Samples final_samples = sample_chain(theta, basis.matrices, lattice, args,
                                         local_final, args.final_burn, state,
                                         args.seed + 99999991ULL + rank, false);

    std::vector<int> counts(ranks), displacements(ranks);
    MPI_Gather(&local_final, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      for (int r = 1; r < ranks; ++r)
        displacements[r] = displacements[r - 1] + counts[r - 1];
    }
    VectorXd all_energy(rank == 0 ? args.final_samples : 0);
    VectorXd all_mz(rank == 0 ? args.final_samples : 0);
    VectorXd all_mx(rank == 0 ? args.final_samples : 0);
    MPI_Gatherv(final_samples.energy.data(), local_final, MPI_DOUBLE,
                all_energy.data(), counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(final_samples.mz.data(), local_final, MPI_DOUBLE, all_mz.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(final_samples.mx.data(), local_final, MPI_DOUBLE, all_mx.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
      const MeanError energy_stats =
          binned_mean_error(all_energy, args.block_size);
      const MeanError mz_stats = binned_mean_error(all_mz, args.block_size);
      const MeanError mx_stats = binned_mean_error(all_mx, args.block_size);
      const double variance =
          (all_energy.array() - energy_stats.mean).square().mean();
      const bool vscore_enabled =
          std::abs(args.jn) < 1.0e-14 && std::abs(args.jp) < 1.0e-14;
      const VScoreResult vscore = conditional_v_score(
          all_energy, lattice.N, args.block_size, vscore_enabled);
      const double Einf =
          vscore_enabled ? 0.0 : std::numeric_limits<double>::quiet_NaN();
      const double acceptance =
          final_samples.accepted / std::max(1.0, final_samples.proposed);
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();

      fs::create_directories(directory);

      std::ofstream summary(prefix.string() + ".csv");
      summary << "Lx,Ly,N,jx,jn,jp,z,parameters,averaged_iterations,"
                 "parameterization,initializer_residual,sign_kernel_sigma_min,"
                 "projective_Tx_residual,projective_Ty_residual,"
                 "infinite_temperature_energy,v_score_enabled,v_score,"
                 "v_score_error,total_energy,total_energy_error,energy_density,"
                 "energy_density_error,total_energy_variance,"
                 "energy_variance_density,longitudinal_magnetization,"
                 "longitudinal_magnetization_error,transverse_magnetization,"
                 "transverse_magnetization_error,acceptance,ed_total_energy,"
                 "ed_dimension,mpi_ranks,elapsed_seconds,restart_completed,"
                 "initial_noise,noise_seed\n";
      summary << std::setprecision(17) << args.Lx << ',' << args.Ly << ','
              << lattice.N << ',' << args.jx << ',' << args.jn << ',' << args.jp
              << ',' << args.z << ',' << theta.size() << ','
              << theta_tail.size() << ','
              << "kasteleyn_2x1_projective_translation_twist_mm" << ','
              << projection_residual << ',' << sigma_min << ',' << symmetry_x
              << ',' << symmetry_y << ',' << Einf << ','
              << (vscore_enabled ? 1 : 0) << ',' << vscore.value << ','
              << vscore.error << ',' << energy_stats.mean << ','
              << energy_stats.error << ',' << energy_stats.mean / lattice.N
              << ',' << energy_stats.error / lattice.N << ',' << variance << ','
              << variance / lattice.N << ',' << mz_stats.mean << ','
              << mz_stats.error << ',' << mx_stats.mean << ',' << mx_stats.error
              << ',' << acceptance << ',' << ed_energy << ',' << ed_dimension
              << ',' << ranks << ',' << elapsed << ',' << completed_before
              << ',' << args.initial_noise << ',' << args.noise_seed << "\n";

      std::ofstream parameters(prefix.string() + "_parameters.csv");
      parameters << "index,label,theta\n";
      for (int p = 0; p < theta.size(); ++p)
        parameters << p << ",\"" << basis.labels[p] << "\","
                   << std::setprecision(17) << theta(p) << "\n";

      std::ofstream pairing(prefix.string() + "_pairing_matrix.csv");
      for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < F.cols(); ++j) {
          if (j)
            pairing << ',';
          pairing << std::setprecision(17) << F(i, j);
        }
        pairing << '\n';
      }

      std::ofstream history_file(prefix.string() + "_sr_history.csv");
      history_file << "iteration,energy,variance,grad_norm,acceptance\n";
      for (const HistoryRow &row : history)
        history_file << row.iteration << ',' << std::setprecision(17)
                     << row.energy << ',' << row.variance << ','
                     << row.gradient_norm << ',' << row.acceptance << "\n";

      std::cout << "VMC total energy=" << std::showpos << std::setprecision(12)
                << std::fixed << energy_stats.mean << std::noshowpos << " +/- "
                << std::scientific << energy_stats.error
                << "; energy density=" << energy_stats.mean / lattice.N
                << "; acc=" << std::fixed << std::setprecision(3) << acceptance
                << "\n";
      if (vscore_enabled)
        std::cout << "V-score=" << std::scientific << vscore.value << " +/- "
                  << vscore.error << " (E_inf=0)\n";
      else
        std::cout << "V-score not computed: only jn=jp=0 is supported, "
                     "where E_inf=0.\n";
      std::cout << "final projective symmetry residuals: Tx=" << std::scientific
                << symmetry_x << ", Ty=" << symmetry_y << "\n";
      std::cout << "saved under: " << directory << "\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
