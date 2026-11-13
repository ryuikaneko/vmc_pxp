#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mpi.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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
  int L = 16, iterations = 80, opt_samples = 5000, final_samples = 40000;
  int opt_burn = 400, final_burn = 1000, thin = 1, block_size = 500;
  int print_every = 10, pf_rebuild_interval = 256, average_iterations = 100;
  std::uint64_t seed = 12345;
  double jx = 1.0, jn = std::numeric_limits<double>::quiet_NaN();
  double jp = std::numeric_limits<double>::quiet_NaN(), z = 1.0;
  double learning_rate = 0.05, sr_diag = 1e-3, max_step = 0.5;
  std::string output_root = ".", output_prefix = "dat";
  std::string restart_file = "", checkpoint_file = "";
  int checkpoint_every = 1;
  double initial_noise = 0.0;
  std::uint64_t noise_seed = 67890;
  bool verbose = false;
};

[[noreturn]] void die(const std::string &s) { throw std::runtime_error(s); }
std::string need(int &i, int argc, char **argv) {
  if (++i >= argc)
    die("missing value");
  return argv[i];
}
Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--L")
      a.L = std::stoi(need(i, argc, argv));
    else if (k == "--iterations")
      a.iterations = std::stoi(need(i, argc, argv));
    else if (k == "--average-iterations")
      a.average_iterations = std::stoi(need(i, argc, argv));
    else if (k == "--opt-samples")
      a.opt_samples = std::stoi(need(i, argc, argv));
    else if (k == "--final-samples")
      a.final_samples = std::stoi(need(i, argc, argv));
    else if (k == "--opt-burn")
      a.opt_burn = std::stoi(need(i, argc, argv));
    else if (k == "--final-burn")
      a.final_burn = std::stoi(need(i, argc, argv));
    else if (k == "--thin")
      a.thin = std::stoi(need(i, argc, argv));
    else if (k == "--block-size")
      a.block_size = std::stoi(need(i, argc, argv));
    else if (k == "--print-every")
      a.print_every = std::stoi(need(i, argc, argv));
    else if (k == "--pf-rebuild-interval")
      a.pf_rebuild_interval = std::stoi(need(i, argc, argv));
    else if (k == "--seed")
      a.seed = std::stoull(need(i, argc, argv));
    else if (k == "--jx")
      a.jx = std::stod(need(i, argc, argv));
    else if (k == "--jn")
      a.jn = std::stod(need(i, argc, argv));
    else if (k == "--jp")
      a.jp = std::stod(need(i, argc, argv));
    else if (k == "--z")
      a.z = std::stod(need(i, argc, argv));
    else if (k == "--learning-rate")
      a.learning_rate = std::stod(need(i, argc, argv));
    else if (k == "--sr-diag")
      a.sr_diag = std::stod(need(i, argc, argv));
    else if (k == "--max-step")
      a.max_step = std::stod(need(i, argc, argv));
    else if (k == "--output-root")
      a.output_root = need(i, argc, argv);
    else if (k == "--output-prefix")
      a.output_prefix = need(i, argc, argv);
    else if (k == "--restart")
      a.restart_file = need(i, argc, argv);
    else if (k == "--checkpoint-file")
      a.checkpoint_file = need(i, argc, argv);
    else if (k == "--checkpoint-every")
      a.checkpoint_every = std::stoi(need(i, argc, argv));
    else if (k == "--initial-noise")
      a.initial_noise = std::stod(need(i, argc, argv));
    else if (k == "--noise-seed")
      a.noise_seed = std::stoull(need(i, argc, argv));
    else if (k == "--verbose")
      a.verbose = true;
    else if (k == "--help") {
      std::cout << "MPI pBCS VMC options: --L --jx --jn --jp --z --iterations "
                << "--average-iterations --opt-samples --final-samples "
                   "--output-root --output-prefix --restart --checkpoint-file "
                   "--checkpoint-every --initial-noise --noise-seed\n";
      std::exit(0);
    } else
      die("unknown option: " + k);
  }
  if (std::isnan(a.jn))
    a.jn = a.jx / a.z;
  if (std::isnan(a.jp))
    a.jp = a.jx * a.z;
  if (a.L < 2 || a.L % 2)
    die("L must be even and >=2");
  if (a.iterations < 1)
    die("iterations must be >= 1");
  if (a.average_iterations < 1)
    a.average_iterations = 1;
  if (a.checkpoint_every < 0)
    die("checkpoint-every must be >= 0");
  if (!(a.initial_noise >= 0.0) || !std::isfinite(a.initial_noise))
    die("initial-noise must be finite and >= 0");
  a.average_iterations = std::min(a.average_iterations, a.iterations);
  return a;
}

inline bool bit(const State &s, int i) {
  const std::size_t word = static_cast<std::size_t>(i) >> 6;
  const unsigned offset = static_cast<unsigned>(i) & 63U;
  return ((s.words[word] >> offset) & 1ULL) != 0ULL;
}
inline void flip_site(State &s, int i) {
  const std::size_t word = static_cast<std::size_t>(i) >> 6;
  const unsigned offset = static_cast<unsigned>(i) & 63U;
  s.words[word] ^= (1ULL << offset);
}
bool is_blockaded(const State &s, int L) {
  for (int i = 0; i < L; ++i)
    if (bit(s, i) && bit(s, (i + 1) % L))
      return false;
  return true;
}
bool can_flip(const State &s, int i, int L) {
  return bit(s, i) || (!bit(s, (i + L - 1) % L) && !bit(s, (i + 1) % L));
}
int count_particles(const State &s, int L) {
  (void)L;
  int n = 0;
  for (std::uint64_t value : s.words) {
#if defined(__GNUC__) || defined(__clang__)
    n += __builtin_popcountll(value);
#else
    while (value != 0ULL) {
      value &= value - 1ULL;
      ++n;
    }
#endif
  }
  return n;
}
int count_ppp(const State &s, int L) {
  int n = 0;
  for (int i = 0; i < L; ++i)
    if (!bit(s, (i + L - 1) % L) && !bit(s, i) && !bit(s, (i + 1) % L))
      ++n;
  return n;
}
double diagonal_energy(const State &s, int L, double jn, double jp) {
  return jn * count_particles(s, L) + jp * count_ppp(s, L);
}

struct Basis {
  std::vector<MatrixXd> B;
  std::vector<std::string> labels;
  Eigen::MatrixXi pindex;
  MatrixXd pvalue;
};
Basis build_basis(int L) {
  using Key = std::tuple<int, int, int>;
  std::vector<std::tuple<int, int, Key, double>> pairs;
  std::map<Key, int> keys;
  for (int a = 0; a < 2 * L; ++a) {
    int i = a % L, si = a / L;
    for (int b = a + 1; b < 2 * L; ++b) {
      int j = b % L, sj = b / L, d = std::abs(j - i);
      Key key{si, sj, std::min(d, L - d)};
      double sg = j < i ? -1 : 1;
      pairs.push_back({a, b, key, sg});
      keys[key] = 0;
    }
  }
  Basis out;
  for (auto &kv : keys) {
    MatrixXd M = MatrixXd::Zero(2 * L, 2 * L);
    for (auto &p : pairs) {
      int a, b;
      Key k;
      double sg;
      std::tie(a, b, k, sg) = p;
      if (k == kv.first) {
        M(a, b) = sg;
        M(b, a) = -sg;
      }
    }
    out.B.push_back(M);
    auto [s0, s1, d] = kv.first;
    out.labels.push_back("s" + std::to_string(s0) + "s" + std::to_string(s1) +
                         "[d=" + std::to_string(d) + "]");
  }
  out.pindex = Eigen::MatrixXi::Constant(2 * L, 2 * L, -1);
  out.pvalue = MatrixXd::Zero(2 * L, 2 * L);
  for (int p = 0; p < (int)out.B.size(); ++p)
    for (int i = 0; i < 2 * L; ++i)
      for (int j = 0; j < 2 * L; ++j)
        if (out.B[p](i, j) != 0) {
          out.pindex(i, j) = p;
          out.pvalue(i, j) = out.B[p](i, j);
        }
  return out;
}
MatrixXd exact_F(int L, double z) {
  VectorXd v(2);
  v << 1, -z;
  v.normalize();
  MatrixXd S(L, L);
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      S(i, j) = (j > i) - (j < i);
  MatrixXd F(2 * L, 2 * L);
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      F.block(a * L, b * L, L, L) = v(a) * v(b) * S;
  return .5 * (F - F.transpose());
}
VectorXd project_basis(const MatrixXd &T, const Basis &b) {
  int n = 2 * T.size(), P = b.B.size();
  (void)n;
  MatrixXd X(T.size(), P);
  for (int p = 0; p < P; ++p)
    X.col(p) = Eigen::Map<const VectorXd>(b.B[p].data(), b.B[p].size());
  VectorXd y = Eigen::Map<const VectorXd>(T.data(), T.size());
  return X.colPivHouseholderQr().solve(y);
}

struct Wavefunction {
  int L;
  Basis basis;
  VectorXd theta;
  MatrixXd F;
  Wavefunction(int l, Basis b, VectorXd t)
      : L(l), basis(std::move(b)), theta(std::move(t)) {
    refresh();
  }
  void refresh() {
    F = MatrixXd::Zero(2 * L, 2 * L);
    for (int p = 0; p < theta.size(); ++p)
      F.noalias() += theta(p) * basis.B[p];
    F = .5 * (F - F.transpose());
  }
};

void selected(const MatrixXd &F, const State &s, int L, std::vector<int> &q,
              MatrixXd &A) {
  for (int i = 0; i < L; ++i)
    q[i] = bit(s, i) ? L + i : i;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      A(i, j) = F(q[i], q[j]);
}
MatrixXd scaled_inverse(const MatrixXd &A) {
  double scale = A.cwiseAbs().maxCoeff();
  if (!(scale > 0))
    die("singular selected Pfaffian matrix");
  MatrixXd B = A / scale;
  Eigen::FullPivLU<MatrixXd> lu(B);
  if (!lu.isInvertible())
    die("singular selected Pfaffian matrix");
  return lu.inverse() / scale;
}
VectorXd replacement_row(const MatrixXd &F, const std::vector<int> &q,
                         int site) {
  int L = q.size(), orb = q[site] >= L ? site : L + site;
  VectorXd r(L);
  for (int j = 0; j < L; ++j)
    r(j) = F(orb, q[j]);
  r(site) = 0;
  return r;
}
double ratio(const MatrixXd &Ainv, const VectorXd &r, int site) {
  return r.dot(Ainv.col(site));
}

// O(L^2) skew-symmetric rank-2 inverse update for replacing row/column
// `site` of A by `row` and -row, respectively.  If
//
//   d = row - A(site,:),  u = A^{-1} e_site,  x = -A^{-1} d,
//
// then the Pfaffian ratio is r = 1 + d^T u and
//
//   A_new^{-1} = A^{-1} - (u x^T - x u^T) / r.
//
// The update is rejected numerically (not physically) when r is too small or
// non-finite; the caller then rebuilds A and A^{-1} from the accepted state.
bool inverse_replace_inplace(MatrixXd &A, MatrixXd &Ainv, const VectorXd &row,
                             int site, double tol = 1.0e-12) {
  const int n = A.rows();
  VectorXd d = row - A.row(site).transpose();
  VectorXd u = Ainv.col(site);
  const double pf_ratio = 1.0 + d.dot(u);
  if (!std::isfinite(pf_ratio) || std::abs(pf_ratio) < tol)
    return false;

  // eval() is essential because Ainv is modified below.
  VectorXd x = (-Ainv * d).eval();
  MatrixXd correction =
      ((u * x.transpose() - x * u.transpose()) / pf_ratio).eval();
  Ainv -= correction;

  // Remove roundoff-generated symmetric components and reset the diagonal.
  Ainv = (0.5 * (Ainv - Ainv.transpose())).eval();
  Ainv.diagonal().setZero();

  for (int j = 0; j < n; ++j) {
    A(site, j) = row(j);
    A(j, site) = -row(j);
  }
  A(site, site) = 0.0;
  return Ainv.allFinite();
}

void rebuild(const MatrixXd &F, const State &s, int L, std::vector<int> &q,
             MatrixXd &A, MatrixXd &Ainv) {
  selected(F, s, L, q, A);
  Ainv = scaled_inverse(A);
}
VectorXd log_deriv(const Basis &b, const std::vector<int> &q,
                   const MatrixXd &Ainv) {
  VectorXd o = VectorXd::Zero(b.B.size());
  int L = q.size();
  for (int i = 0; i < L; ++i)
    for (int j = i + 1; j < L; ++j) {
      int p = b.pindex(q[j], q[i]);
      if (p >= 0)
        o(p) += Ainv(i, j) * b.pvalue(q[j], q[i]);
    }
  return o;
}

struct Samples {
  VectorXd E, Mz, Mx;
  MatrixXd O;
  State state;
  double accepted = 0, proposed = 0;
};
Samples sample_chain(const Wavefunction &w, const Args &a, int ns, int burn,
                     State initial, std::uint64_t seed, bool needO) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> u(0, 1);
  std::uniform_int_distribution<int> site(0, a.L - 1);
  Samples r;
  r.E.resize(ns);
  r.Mz.resize(ns);
  r.Mx.resize(ns);
  if (needO)
    r.O.resize(ns, w.theta.size());
  State s = is_blockaded(initial, a.L) ? initial : State(a.L);
  std::vector<int> q(a.L);
  MatrixXd A(a.L, a.L), Ai(a.L, a.L);
  rebuild(w.F, s, a.L, q, A, Ai);
  int m = 0, since = 0, total = burn + ns * a.thin;
  for (int sweep = 0; sweep < total; ++sweep) {
    for (int z = 0; z < a.L; ++z) {
      int i = site(gen);
      r.proposed++;
      if (!can_flip(s, i, a.L))
        continue;
      VectorXd row = replacement_row(w.F, q, i);
      double rr = ratio(Ai, row, i), r2 = rr * rr;
      if (std::isfinite(r2) && (r2 >= 1 || u(gen) < r2)) {
        // The physical move has been accepted with probability |Pf_new/Pf|^2.
        // Update A^{-1} in O(L^2); use an O(L^3) rebuild only as a numerical
        // fallback or at the requested periodic stabilization interval.
        const bool update_ok = inverse_replace_inplace(A, Ai, row, i);
        flip_site(s, i);
        r.accepted++;
        since++;
        if (update_ok) {
          q[i] = (q[i] >= a.L) ? i : (a.L + i);
        }
        if (!update_ok ||
            (a.pf_rebuild_interval > 0 && since >= a.pf_rebuild_interval)) {
          rebuild(w.F, s, a.L, q, A, Ai);
          since = 0;
        }
      }
    }
    if (sweep >= burn && (sweep - burn) % a.thin == 0) {
      double e = diagonal_energy(s, a.L, a.jn, a.jp), mx = 0;
      for (int i = 0; i < a.L; ++i)
        if (can_flip(s, i, a.L)) {
          double rr = ratio(Ai, replacement_row(w.F, q, i), i);
          e += a.jx * rr;
          mx += rr;
        }
      r.E(m) = e;
      r.Mz(m) = (a.L - 2.0 * count_particles(s, a.L)) / a.L;
      r.Mx(m) = mx / a.L;
      if (needO)
        r.O.row(m) = log_deriv(w.basis, q, Ai).transpose();
      ++m;
    }
  }
  r.state = s;
  return r;
}

void allreduce_vec(VectorXd &v) {
  VectorXd x = v;
  MPI_Allreduce(x.data(), v.data(), v.size(), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
}
void allreduce_mat(MatrixXd &m) {
  MatrixXd x = m;
  MPI_Allreduce(x.data(), m.data(), m.size(), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
}
struct Hist {
  int it;
  double E, var, g, acc;
};

std::tuple<double, long long, double, double> einf_tm(int L, double jn,
                                                      double jp) {
  if (L == 2)
    return {(jn + jp) * 2.0 / 3.0, 3, 2.0 / 3.0, 2.0 / 3.0};
  std::vector<std::pair<int, int>> ps{{0, 0}, {0, 1}, {1, 0}};
  MatrixXd M = MatrixXd::Zero(3, 3), X = M, Y = M;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 2; ++c) {
      auto [a, b] = ps[r];
      if (b && c)
        continue;
      int col = -1;
      for (int k = 0; k < 3; ++k)
        if (ps[k] == std::make_pair(b, c))
          col = k;
      M(r, col) = 1;
      if (c)
        X(r, col) = 1;
      if (!a && !b && !c)
        Y(r, col) = 1;
    }
  MatrixXd P = MatrixXd::Identity(3, 3), Dx = MatrixXd::Zero(3, 3), Dy = Dx;
  for (int k = 0; k < L; ++k) {
    Dx = Dx * M + P * X;
    Dy = Dy * M + P * Y;
    P = P * M;
  }
  double D = P.trace(), n1 = Dx.trace() / D, n0 = Dy.trace() / D;
  return {jn * n1 + jp * n0, (long long)std::llround(D), n1, n0};
}
std::pair<double, double> block_mean_error(const VectorXd &v, int bs) {
  int nb = v.size() / bs;
  if (nb < 2)
    return {v.mean(), NAN};
  VectorXd b(nb);
  for (int i = 0; i < nb; ++i)
    b(i) = v.segment(i * bs, bs).mean();
  double mean = b.mean(), var = (b.array() - mean).square().sum() / (nb - 1);
  return {mean, std::sqrt(var / nb)};
}

struct VScoreResult {
  double value;
  double error;
};

VScoreResult block_v_score(const VectorXd &energy, int L, double Einf,
                           int block_size) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double E = energy.mean();
  const double variance = (energy.array() - E).square().mean();
  const double denominator = E - Einf;
  if (!std::isfinite(denominator) || denominator == 0.0)
    return {nan, nan};

  const double value = L * variance / (denominator * denominator);
  const int blocks = static_cast<int>(energy.size()) / block_size;
  if (blocks < 2)
    return {value, nan};

  VectorXd scores(blocks);
  int valid = 0;
  for (int b = 0; b < blocks; ++b) {
    const VectorXd block = energy.segment(b * block_size, block_size);
    const double Eb = block.mean();
    const double block_denominator = Eb - Einf;
    if (!std::isfinite(block_denominator) || block_denominator == 0.0)
      continue;
    const double Vb = (block.array() - Eb).square().mean();
    const double score = L * Vb / (block_denominator * block_denominator);
    if (std::isfinite(score))
      scores(valid++) = score;
  }
  if (valid < 2)
    return {value, nan};

  const double mean = scores.head(valid).mean();
  const double error =
      std::sqrt((scores.head(valid).array() - mean).square().sum() /
                (valid * (valid - 1.0)));
  return {value, error};
}
std::string tag(double x) {
  std::ostringstream o;
  o << std::setprecision(12) << x;
  std::string s = o.str();
  for (char &c : s) {
    if (c == '-')
      c = 'm';
    else if (c == '.')
      c = 'p';
    else if (c == '+')
      c = 'p';
  }
  return s;
}

struct RestartData {
  VectorXd theta;
  int completed_iterations = 0;
};

RestartData load_restart(const fs::path &path, int expected_L,
                         const Basis &basis) {
  std::ifstream in(path);
  if (!in)
    die("cannot open restart file: " + path.string());
  RestartData out;
  out.theta = VectorXd::Zero(basis.B.size());
  std::vector<bool> seen(basis.B.size(), false);
  int file_L = -1;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("# L=", 0) == 0) {
      file_L = std::stoi(line.substr(4));
      continue;
    }
    if (line.rfind("# completed_iterations=", 0) == 0) {
      out.completed_iterations = std::stoi(line.substr(23));
      continue;
    }
    if (line.empty() || line[0] == '#' || line.rfind("index,", 0) == 0)
      continue;
    const std::size_t first = line.find(',');
    const std::size_t last = line.rfind(',');
    if (first == std::string::npos || last == first)
      die("malformed restart row: " + line);
    const int index = std::stoi(line.substr(0, first));
    const double value = std::stod(line.substr(last + 1));
    if (index < 0 || index >= out.theta.size())
      die("restart parameter index out of range");
    out.theta(index) = value;
    seen[index] = true;
  }
  if (file_L >= 0 && file_L != expected_L)
    die("restart L does not match --L");
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    die("restart file does not contain every variational parameter");
  const double norm = out.theta.norm();
  if (!(norm > 0.0) || !std::isfinite(norm))
    die("restart theta has invalid norm");
  out.theta /= norm;
  return out;
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
    die("noisy initial theta has invalid norm");
  theta /= norm;
}

void save_checkpoint_atomic(const fs::path &path, int L,
                            int completed_iterations, const Wavefunction &w) {
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out)
      die("cannot write checkpoint: " + tmp.string());
    out << "# code_vmc_1d checkpoint\n"
        << "# L=" << L << "\n"
        << "# completed_iterations=" << completed_iterations << "\n"
        << "index,label,theta\n";
    for (int i = 0; i < w.theta.size(); ++i)
      out << i << ",\"" << w.basis.labels[i] << "\"," << std::setprecision(17)
          << w.theta(i) << "\n";
    out.flush();
    if (!out)
      die("failed while writing checkpoint: " + tmp.string());
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(path, ec);
    ec.clear();
    fs::rename(tmp, path, ec);
  }
  if (ec)
    die("cannot install checkpoint: " + path.string() + ": " + ec.message());
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  try {
    Args a = parse_args(argc, argv);
    auto t0 = std::chrono::steady_clock::now();
    Basis b = build_basis(a.L);
    VectorXd theta = project_basis(exact_F(a.L, a.z), b);
    theta.normalize();
    int completed_before = 0;
    if (!a.restart_file.empty()) {
      if (rank == 0) {
        RestartData r = load_restart(a.restart_file, a.L, b);
        theta = r.theta;
        completed_before = r.completed_iterations;
      }
      MPI_Bcast(&completed_before, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(theta.data(), theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      add_gaussian_noise(theta, a.initial_noise, a.noise_seed);
    MPI_Bcast(theta.data(), theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Wavefunction w(a.L, std::move(b), theta);
    State state(a.L);
    fs::path dir = fs::path(a.output_root) /
                   ("L" + std::to_string(a.L) + "_jx" + tag(a.jx) + "_jn" +
                    tag(a.jn) + "_jp" + tag(a.jp));
    fs::path pre = dir / fs::path(a.output_prefix).filename();
    fs::path checkpoint = a.checkpoint_file.empty()
                              ? fs::path(pre.string() + "_checkpoint.csv")
                              : fs::path(a.checkpoint_file);
    if (rank == 0) {
      fs::create_directories(dir);
      if (!checkpoint.parent_path().empty())
        fs::create_directories(checkpoint.parent_path());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<Hist> hist;
    std::vector<VectorXd> tail;
    if (rank == 0)
      std::cout << "MPI ranks=" << size << " L=" << a.L
                << " parameters=" << theta.size()
                << " restart_completed=" << completed_before
                << " initial_noise=" << a.initial_noise
                << " noise_seed=" << a.noise_seed << "\n";
    for (int it = 0; it < a.iterations; ++it) {
      const int global_it = completed_before + it;
      int local_n = a.opt_samples / size + (rank < a.opt_samples % size);
      Samples s = sample_chain(w, a, local_n, a.opt_burn, state,
                               a.seed + 1000003ULL * global_it + rank, true);
      state = s.state;
      double n = local_n;
      double se = s.E.sum(), se2 = s.E.squaredNorm(), acc = s.accepted,
             prop = s.proposed;
      VectorXd so = s.O.colwise().sum(), soe = s.O.transpose() * s.E;
      MatrixXd soo = s.O.transpose() * s.O;
      double vals[5]{n, se, se2, acc, prop}, glob[5];
      MPI_Allreduce(vals, glob, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      allreduce_vec(so);
      allreduce_vec(soe);
      allreduce_mat(soo);
      double N = glob[0], E = glob[1] / N, var = glob[2] / N - E * E;
      VectorXd om = so / N;
      MatrixXd S = soo / N - om * om.transpose();
      VectorXd g = 2 * (soe / N - om * E);
      double reg = a.sr_diag * std::max(.1, std::pow(.95, global_it));
      MatrixXd metric = S;
      for (int p = 0; p < metric.rows(); ++p)
        metric(p, p) += reg * std::max(S(p, p), 1e-8) + 1e-10;
      VectorXd delta = metric.ldlt().solve(-g);
      if (delta.norm() > a.max_step)
        delta *= a.max_step / delta.norm();
      w.theta += a.learning_rate * delta;
      w.theta.normalize();
      w.refresh();
      if (it >= a.iterations - a.average_iterations)
        tail.push_back(w.theta);
      hist.push_back(
          {global_it, E, var, g.norm(), glob[3] / std::max(1.0, glob[4])});
      if (rank == 0 && a.checkpoint_every > 0 &&
          ((it + 1) % a.checkpoint_every == 0 || it + 1 == a.iterations))
        save_checkpoint_atomic(checkpoint, a.L, global_it + 1, w);
      if (rank == 0 && a.verbose &&
          (it % a.print_every == 0 || it + 1 == a.iterations))
        std::cout << "SR " << global_it << " E=" << std::setprecision(12) << E
                  << " var=" << var << " |g|=" << g.norm()
                  << " acc=" << hist.back().acc << "\n";
    }
    VectorXd avg = VectorXd::Zero(w.theta.size());
    for (auto &x : tail)
      avg += x;
    avg /= tail.size();
    avg.normalize();
    w.theta = avg;
    w.refresh();
    int local_n = a.final_samples / size + (rank < a.final_samples % size);
    Samples s = sample_chain(w, a, local_n, a.final_burn, state,
                             a.seed + 987654321ULL + rank, false);
    std::vector<int> counts(size), disp(size);
    int ln = local_n;
    MPI_Gather(&ln, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      for (int i = 1; i < size; ++i)
        disp[i] = disp[i - 1] + counts[i - 1];
    VectorXd Eall(rank == 0 ? a.final_samples : 0),
        Mzall(rank == 0 ? a.final_samples : 0),
        Mxall(rank == 0 ? a.final_samples : 0);
    MPI_Gatherv(s.E.data(), ln, MPI_DOUBLE, Eall.data(), counts.data(),
                disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(s.Mz.data(), ln, MPI_DOUBLE, Mzall.data(), counts.data(),
                disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(s.Mx.data(), ln, MPI_DOUBLE, Mxall.data(), counts.data(),
                disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      auto [Einf, D, n1, n000] = einf_tm(a.L, a.jn, a.jp);
      auto [Etot, Eerr] = block_mean_error(Eall, a.block_size);
      auto [mz, mze] = block_mean_error(Mzall, a.block_size);
      auto [mx, mxe] = block_mean_error(Mxall, a.block_size);
      double var = (Eall.array() - Etot).square().mean(), ed = Etot / a.L,
             vd = var / a.L;
      const VScoreResult vscore = block_v_score(Eall, a.L, Einf, a.block_size);
      fs::create_directories(dir);
      std::ofstream f(pre.string() + ".csv");
      f << "L,jx,jn,jp,z_initializer,parameters,averaged_iterations,infinite_"
           "temperature_energy,blockade_dimension_transfer,infinite_"
           "temperature_mean_n1,infinite_temperature_mean_n000,energy_density,"
           "energy_density_error,total_energy,total_energy_error,total_energy_"
           "variance,energy_variance_density,v_score,v_score_error,"
           "longitudinal_"
           "magnetization,longitudinal_magnetization_error,transverse_"
           "magnetization,transverse_magnetization_error,mpi_ranks,"
           "restart_completed,initial_noise,noise_seed\n"
        << std::setprecision(17) << a.L << ',' << a.jx << ',' << a.jn << ','
        << a.jp << ',' << a.z << ',' << w.theta.size() << ',' << tail.size()
        << ',' << Einf << ',' << D << ',' << n1 << ',' << n000 << ',' << ed
        << ',' << Eerr / a.L << ',' << Etot << ',' << Eerr << ',' << var << ','
        << vd << ',' << vscore.value << ',' << vscore.error << ',' << mz << ','
        << mze << ',' << mx << ',' << mxe << ',' << size << ','
        << completed_before << ',' << a.initial_noise << ',' << a.noise_seed
        << "\n";
      std::ofstream p(pre.string() + "_parameters.csv");
      p << "index,label,theta\n";
      for (int i = 0; i < w.theta.size(); ++i)
        p << i << ",\"" << w.basis.labels[i] << "\"," << std::setprecision(17)
          << w.theta(i) << "\n";
      std::ofstream h(pre.string() + "_sr_history.csv");
      h << "iteration,energy,variance,grad_norm,acceptance\n";
      for (auto &r : hist)
        h << r.it << ',' << std::setprecision(17) << r.E << ',' << r.var << ','
          << r.g << ',' << r.acc << "\n";
      std::cout << "E_inf=" << Einf << " energy density=" << ed << " +/- "
                << Eerr / a.L << " V-score=" << vscore.value << " +/- "
                << vscore.error << "\nsaved under: " << dir << "\n";
    }
  } catch (const std::exception &e) {
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    std::cerr << "rank " << r << ": " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}
