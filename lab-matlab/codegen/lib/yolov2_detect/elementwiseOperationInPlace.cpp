//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// elementwiseOperationInPlace.cpp
//
// Code generation for function 'elementwiseOperationInPlace'
//

// Include files
#include "elementwiseOperationInPlace.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace layer {
void lambdaForColumnMajorGeneric(dlarray &X)
{
#pragma omp parallel for num_threads(omp_get_max_threads())

  for (int iElem = 0; iElem < 2535; iElem++) {
    X.Data[iElem] = 1.0F / (std::exp(-X.Data[iElem]) + 1.0F);
  }
}

void lambdaForColumnMajorGeneric(b_dlarray &X)
{
#pragma omp parallel for num_threads(omp_get_max_threads())

  for (int iElem = 0; iElem < 1690; iElem++) {
    X.Data[iElem] = std::exp(X.Data[iElem]);
  }
}

void lambdaForColumnMajorGeneric(float X[80])
{
#pragma omp parallel for num_threads(omp_get_max_threads())

  for (int iElem = 0; iElem < 80; iElem++) {
    X[iElem] = std::exp(X[iElem]);
  }
}

} // namespace layer
} // namespace internal
} // namespace coder

// End of code generation (elementwiseOperationInPlace.cpp)
