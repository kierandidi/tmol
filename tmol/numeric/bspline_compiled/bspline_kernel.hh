// N-dimensional B-spline interpolation with periodic boundary conditions.
// From:
//    Thévenaz, Philippe, Thierry Blu, and Michael Unser.
//    "Interpolation revisited [medical images application]."
//    IEEE Transactions on medical imaging 19.7 (2000): 739-758.
//    http://bigwww.epfl.ch/publications/thevenaz0002.pdf

// Dimension- and degree-templated, header-only implementation.

#pragma once

#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace numeric {
namespace bspline {

template <
    std::size_t NDIM,
    std::size_t DEGREE,
    tmol::Device D,
    typename Real,
    typename Int>
struct ndspline_kernel {
  static auto TMOL_HOST_DEVICE_FUNC square(Real v) -> Real { return v * v; }
  static auto TMOL_HOST_DEVICE_FUNC cube(Real v) -> Real { return v * v * v; }

  static auto TMOL_HOST_DEVICE_FUNC
  _get_weights(tmol::portable::Vec<Real, NDIM> frac)
      -> tmol::portable::Mat<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts =
        tmol::portable::Mat<Real, NDIM, DEGREE + 1>::Constant(0.0);

    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts(dim, 1) = 3.0 / 4.0 - square(frac[dim]);
        wts(dim, 2) = (1.0 / 2.0) * (frac[dim] - wts(dim, 1) + 1.0);
        wts(dim, 0) = 1.0 - wts(dim, 1) - wts(dim, 2);
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts(dim, 3) = (1.0 / 6.0) * cube(frac[dim]);
        wts(dim, 0) = (1.0 / 6.0) + (1.0 / 2.0) * frac[dim] * (frac[dim] - 1.0)
                      - wts(dim, 3);
        wts(dim, 2) = frac[dim] + wts(dim, 0) - 2.0 * wts(dim, 3);
        wts(dim, 1) = 1.0 - wts(dim, 0) - wts(dim, 2) - wts(dim, 3);
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real t0 = frac[dim] * (t - 11.0 / 24.0);
        Real t1 = 19.0 / 96.0 + square(frac[dim]) * (1.0 / 4.0 - t);
        wts(dim, 0) = (1.0 / 24.0) * square(square(1.0 / 2.0 - frac[dim]));
        wts(dim, 1) = t1 + t0;
        wts(dim, 3) = t1 - t0;
        wts(dim, 4) = wts(dim, 0) + t0 + (1.0 / 2.0) * frac[dim];
        wts(dim, 2) =
            1.0 - wts(dim, 0) - wts(dim, 1) - wts(dim, 3) - wts(dim, 4);
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        // Cardinal quintic polynomials in reflected pairs; Horner form avoids
        // cancellation between large truncated powers near support boundaries.
        for (int side = 0; side < 2; ++side) {
          Real x = side ? 1 - frac[dim] : frac[dim];
          wts(dim, 5 * (1 - side)) = square(x) * cube(x) / 120;
          wts(dim, 1 + 3 * side) =
              (26 + x * (-50 + x * (20 + x * (20 + x * (-20 + 5 * x))))) / 120;
          wts(dim, 2 + side) =
              (66 + square(x) * (-60 + square(x) * (30 - 10 * x))) / 120;
        }
      }
    }

    return wts;
  }

  static auto TMOL_HOST_DEVICE_FUNC
  _get_dweights(tmol::portable::Vec<Real, NDIM> frac)
      -> tmol::portable::Mat<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> dwts =
        tmol::portable::Mat<Real, NDIM, DEGREE + 1>::Constant(0.0);

    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts(dim, 1) = -2.0 * frac[dim];
        dwts(dim, 2) = (1.0 / 2.0) * (1.0 - dwts(dim, 1));
        dwts(dim, 0) = -dwts(dim, 1) - dwts(dim, 2);
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts(dim, 3) = (1.0 / 2.0) * square(frac[dim]);
        dwts(dim, 0) = (frac[dim] - 0.5) - dwts(dim, 3);
        dwts(dim, 2) = 1.0 + dwts(dim, 0) - 2.0 * dwts(dim, 3);
        dwts(dim, 1) = -dwts(dim, 0) - dwts(dim, 2) - dwts(dim, 3);
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real dt = (1.0 / 3.0) * frac[dim];
        Real dt0 = (t - 11.0 / 24.0) + frac[dim] * dt;
        Real dt1 = 2.0 * frac[dim] * (1.0 / 4.0 - t) - square(frac[dim]) * dt;
        dwts(dim, 0) = -(1.0 / 6.0) * cube(1.0 / 2.0 - frac[dim]);
        dwts(dim, 1) = dt1 + dt0;
        dwts(dim, 3) = dt1 - dt0;
        dwts(dim, 4) = dwts(dim, 0) + dt0 + (1.0 / 2.0);
        dwts(dim, 2) =
            -dwts(dim, 0) - dwts(dim, 1) - dwts(dim, 3) - dwts(dim, 4);
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        for (int side = 0; side < 2; ++side) {
          Real x = side ? 1 - frac[dim] : frac[dim];
          Real sign = side ? -1 : 1;
          dwts(dim, 5 * (1 - side)) = sign * square(square(x)) / 24;
          dwts(dim, 1 + 3 * side) =
              sign * (-50 + x * (40 + x * (60 + x * (-80 + 25 * x)))) / 120;
          dwts(dim, 2 + side) =
              sign * x * (-120 + square(x) * (120 - 50 * x)) / 120;
        }
      }
    }
    return dwts;
  }

  static auto TMOL_HOST_DEVICE_FUNC interpolate_V(
      TensorAccessor<Real, NDIM, D> coeffs, tmol::portable::Vec<Real, NDIM> X)
      -> Real {
    typedef tmol::portable::Vec<Real, NDIM> RealN;
    typedef tmol::portable::Vec<Int, NDIM> IntN;

    Int nprods = 1;
    IntN idx;
    RealN frac;

    for (int dim = 0; dim < NDIM; ++dim) {
      if constexpr (DEGREE % 2 == 0)
        idx[dim] = (Int)std::floor(X[dim] + Real(0.5));
      else
        idx[dim] = (Int)std::floor(X[dim]);
      frac[dim] = X[dim] - idx[dim];
      idx[dim] -= (DEGREE / 2);
      nprods *= (DEGREE + 1);
    }

    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts = _get_weights(frac);
    Real interp = 0;
    for (int pt = 0; pt < nprods; ++pt) {
      Real weight = 1;
      Int stride = 1;
      Int pt_indexer = pt;
      Int idx_ij = 0;

      for (int dim = NDIM - 1; dim >= 0; --dim) {
        Int idx_box_i = pt_indexer % (DEGREE + 1);
        Int idx_i = (idx[dim] + idx_box_i) % coeffs.size(dim);
        if (idx_i < 0) idx_i += coeffs.size(dim);
        idx_ij += stride * idx_i;
        stride *= coeffs.size(dim);
        weight *= wts(dim, idx_box_i);
        pt_indexer /= (DEGREE + 1);
      }
      interp += weight * coeffs.data()[idx_ij];
    }
    return interp;
  }

  static auto TMOL_HOST_DEVICE_FUNC interpolate(
      TensorAccessor<Real, NDIM, D> coeffs, tmol::portable::Vec<Real, NDIM> X)
      -> tmol::score::common::tuple<Real, tmol::portable::Vec<Real, NDIM> > {
    typedef tmol::portable::Vec<Real, NDIM> RealN;
    typedef tmol::portable::Vec<Int, NDIM> IntN;

    Real interp = 0;
    RealN dinterp_dX = RealN::Constant(0.0);

    Int nprods = 1;
    Int pt_indexer, idx_box_i;
    int64_t idx_ij;
    IntN idx;
    RealN frac;

    Real weight = 1;
    RealN dweight;

    for (int dim = 0; dim < NDIM; ++dim) {
      if constexpr (DEGREE % 2 == 0)
        idx[dim] = (Int)std::floor(X[dim] + Real(0.5));
      else
        idx[dim] = (Int)std::floor(X[dim]);
      frac[dim] = X[dim] - idx[dim];
      idx[dim] -= (DEGREE / 2);
      nprods *= (DEGREE + 1);
    }

    // get weights in each dimension
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts = _get_weights(frac);
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> dwts = _get_dweights(frac);

    // Cache offsets on CPU and for small CUDA stencils. Larger CUDA stencils
    // keep only wrapped base indices, avoiding a per-thread offset array.
    constexpr bool cache_offsets = D == Device::CPU || NDIM <= 2;
    // A zero-row matrix is not a legal device variable, so the unused
    // instantiation keeps one dummy row rather than none.
    tmol::portable::Mat<int64_t, cache_offsets ? NDIM : 1, DEGREE + 1> offsets;
    if constexpr (cache_offsets) {
      for (int dim = 0; dim < NDIM; ++dim) {
        for (int tap = 0; tap <= DEGREE; ++tap) {
          Int wrapped = (idx[dim] + tap) % coeffs.size(dim);
          if (wrapped < 0) wrapped += coeffs.size(dim);
          offsets(dim, tap) = coeffs.stride(dim) * wrapped;
        }
      }
    } else {
      for (int dim = 0; dim < NDIM; ++dim) {
        idx[dim] %= coeffs.size(dim);
        if (idx[dim] < 0) idx[dim] += coeffs.size(dim);
      }
    }

    // do the dot product
    for (int pt = 0; pt < nprods; ++pt) {
      weight = 1;
      dweight = RealN::Constant(1.0);

      pt_indexer = pt;
      idx_ij = 0;

      for (int dim = NDIM - 1; dim >= 0; --dim) {
        idx_box_i = pt_indexer % (DEGREE + 1);
        if constexpr (cache_offsets) {
          idx_ij += offsets(dim, idx_box_i);
        } else {
          Int idx_i = idx[dim] + idx_box_i;
          while (idx_i >= coeffs.size(dim)) idx_i -= coeffs.size(dim);
          idx_ij += coeffs.stride(dim) * idx_i;
        }
        weight *= wts(dim, idx_box_i);

        for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
          dweight[d_dim] *=
              (d_dim == dim ? dwts(dim, idx_box_i) : wts(dim, idx_box_i));
        }

        pt_indexer /= (DEGREE + 1);
      }
      interp += weight * coeffs.data()[idx_ij];
      for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
        dinterp_dX[d_dim] += dweight[d_dim] * coeffs.data()[idx_ij];
      }
    }

    return {interp, dinterp_dX};
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
