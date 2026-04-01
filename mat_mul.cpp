#include <cassert>
#include <chrono>
#include <cstddef>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <ostream>
#include <random>
#include <vector>

template <typename T>
auto transpose(std::vector<T> &mat, std::size_t n, std::size_t m) {
  auto mat_t = std::vector<T>(mat.size());
  hpx::experimental::for_loop(hpx::execution::par_unseq, 0, n, [&](auto i) {
    hpx::experimental::for_loop(
        0, m, [&](auto j) { mat_t[j * n + i] = mat[i * m + j]; });
  });
  return mat_t;
}

template <typename T>
auto hpx_mat_mul(std::vector<T> &a, std::vector<T> &b, std::size_t n,
                 std::size_t m, std::size_t p) {
  assert(a.size() == n * m);
  assert(b.size() == m * p);
  auto c = std::vector<T>(n * p);
  auto b_t = transpose(b, m, p);
  hpx::experimental::for_loop(hpx::execution::par_unseq, 0, n, [&](auto i) {
    hpx::experimental::for_loop(0, p, [&](auto k) {
      T count = T(0);
      hpx::experimental::for_loop(
          0, m, [&](auto j) { count += a[i * m + j] * b_t[k * m + j]; });
      c[i * p + k] = count;
    });
  });
  return c;
}

template <typename T>
auto std_mat_mul(std::vector<T> &a, std::vector<T> &b, std::size_t n,
                 std::size_t m, std::size_t p) {
  assert(a.size() == n * m);
  assert(b.size() == m * p);
  auto c = std::vector<T>(n * p);
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t k = 0; k < p; k++) {
      T count = T(0);
      for (std::size_t j = 0; j < m; j++) {
        count += a[i * m + j] * b[j * p + k];
      }
      c[i * p + k] = count;
    }
  }
  return c;
}

int main() {
  std::default_random_engine gen(42);
  std::uniform_int_distribution<> dist;
  // 1024 x 2048
  auto a = std::vector<int>(1024 * 2048);
  // 2048 x 4096
  auto b = std::vector<int>(2048*4096);

  for (auto &ele : a) {
    ele = dist(gen);
  }
  for (auto &ele : b) {
    ele = dist(gen);
  }
  auto _ = hpx_mat_mul(a, b, 1024, 2048, 4096);
  auto st = std::chrono::steady_clock::now();
  auto c = hpx_mat_mul(a, b, 1024, 2048, 4096);
  auto ck = std::chrono::steady_clock::now();
  auto std_c = std_mat_mul(a, b, 1024, 2048, 4096);
  auto en = std::chrono::steady_clock::now();

  assert(c == std_c);
  hpx::cout << "Speed up: "
            << static_cast<double>((en - ck).count()) /
                   static_cast<double>((ck - st).count())
            << std::flush;
}