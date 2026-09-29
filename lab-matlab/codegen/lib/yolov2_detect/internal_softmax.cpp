//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// internal_softmax.cpp
//
// Code generation for function 'internal_softmax'
//

// Include files
#include "internal_softmax.h"
#include "elementwiseOperationInPlace.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace deep {
namespace internal {
namespace coder {
namespace dlarray {
void iComputeSoftmaxForCpu(const float xdata[67600], float ydata[67600])
{
  double nonChannelSubscriptIndices[3];
  float dataExp[80];
  float f;
  float sumX;
  int b_k;
  int idx;
  int k;
  bool exitg1;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        sumX, dataExp, nonChannelSubscriptIndices, idx, b_k, f, k, exitg1)

  for (int nonChannelDimsProductIdx = 0; nonChannelDimsProductIdx < 845;
       nonChannelDimsProductIdx++) {
    sumX = xdata[nonChannelDimsProductIdx];
    if (!std::isnan(sumX)) {
      idx = 1;
    } else {
      idx = 0;
      k = 2;
      exitg1 = false;
      while (!exitg1 && (k < 81)) {
        if (!std::isnan(xdata[nonChannelDimsProductIdx + 845 * (k - 1)])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx != 0) {
      sumX = xdata[nonChannelDimsProductIdx + 845 * (idx - 1)];
      for (b_k = idx + 1; b_k < 81; b_k++) {
        f = xdata[nonChannelDimsProductIdx + 845 * (b_k - 1)];
        if (sumX < f) {
          sumX = f;
        }
      }
    }
    for (b_k = 0; b_k < 80; b_k++) {
      dataExp[b_k] = xdata[nonChannelDimsProductIdx + 845 * b_k] - sumX;
    }
    ::coder::internal::layer::lambdaForColumnMajorGeneric(dataExp);
    sumX = dataExp[0];
    for (b_k = 0; b_k < 79; b_k++) {
      sumX += dataExp[b_k + 1];
    }
    for (b_k = 0; b_k < 80; b_k++) {
      ydata[nonChannelDimsProductIdx + 845 * b_k] = dataExp[b_k] / sumX;
    }
  }
}

} // namespace dlarray
} // namespace coder
} // namespace internal
} // namespace deep
} // namespace coder

// End of code generation (internal_softmax.cpp)
