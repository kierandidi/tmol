#include <tmol/numeric/bspline_compiled/bspline_kernel.hh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Independent cardinal-basis recurrence, deliberately unlike the kernel's
// expanded polynomials. Sum all periodic images, including supports wider
// than the table, so the oracle does not share the kernel's stencil indexing.
long double cardinal(int degree, long double x) {
  if (degree == 0) return x >= -0.5L && x < 0.5L ? 1 : 0;
  if (std::abs(x) >= (degree + 1) / 2.L) return 0;
  return ((x + (degree + 1) / 2.L) * cardinal(degree - 1, x + 0.5L)
          + ((degree + 1) / 2.L - x) * cardinal(degree - 1, x - 0.5L))
         / degree;
}

long double derivative(int degree, long double x) {
  return cardinal(degree - 1, x + 0.5L) - cardinal(degree - 1, x - 0.5L);
}

template <int Degree, class Real>
bool check() {
  using Kernel = tmol::numeric::bspline::
      ndspline_kernel<2, Degree, tmol::Device::CPU, Real, int>;
  Real coefficients[56];
  for (int i = 0; i < 56; ++i) coefficients[i] = ((i * 17 + 3) % 23) - 11;
  long double max_error = 0;
  for (int size : {2, 7}) {
    int64_t sizes[2] = {size, size + 1}, strides[2] = {size + 1, 1};
    tmol::TensorAccessor<Real, 2, tmol::Device::CPU> table(
        coefficients, sizes, strides);
    for (Real x :
         {Real(-0.75),
          Real(-0.5),
          Real(-0.125),
          Real(0),
          Real(0.25),
          Real(0.5),
          Real(0.75),
          Real(3.125)}) {
      tmol::portable::Vec<Real, 2> point{x, Real(1.375)};
      long double expected = 0, dx = 0, dy = 0;
      for (int i = -Degree - 5; i <= size + Degree + 5; ++i)
        for (int j = -Degree - 5; j <= size + Degree + 6; ++j) {
          Real coefficient = coefficients
              [((i % size + size) % size) * (size + 1)
               + (j % (size + 1) + size + 1) % (size + 1)];
          long double a = point[0] - static_cast<long double>(i);
          long double b = point[1] - static_cast<long double>(j);
          expected += coefficient * cardinal(Degree, a) * cardinal(Degree, b);
          dx += coefficient * derivative(Degree, a) * cardinal(Degree, b);
          dy += coefficient * cardinal(Degree, a) * derivative(Degree, b);
        }
      auto result = Kernel::interpolate(table, point);
      max_error = std::max(
          {max_error,
           std::abs(expected - tmol::score::common::get<0>(result)),
           std::abs(expected - Kernel::interpolate_V(table, point)),
           std::abs(dx - tmol::score::common::get<1>(result)[0]),
           std::abs(dy - tmol::score::common::get<1>(result)[1])});
      point[0] += size;
      point[1] -= size + 1;
      max_error = std::max(
          max_error, std::abs(expected - Kernel::interpolate_V(table, point)));
    }
  }
  std::cout << Degree << " " << sizeof(Real) << " " << max_error << "\n";
  return max_error < (sizeof(Real) == 4 ? 3e-5L : 2e-12L);
}

int main() {
  bool valid = true;
#define CHECK_DEGREE(n) valid &= check<n, float>() & check<n, double>();
  CHECK_DEGREE(2)
  CHECK_DEGREE(3)
  CHECK_DEGREE(4)
  CHECK_DEGREE(5)
#undef CHECK_DEGREE
  return valid ? 0 : 1;
}
