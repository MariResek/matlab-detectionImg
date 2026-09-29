//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// poolingOperation.cpp
//
// Code generation for function 'poolingOperation'
//

// Include files
#include "poolingOperation.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace layer {
namespace poolingUtils {
void b_iPoolingForColumnMajorWithOpenMP(const float X[1384448], float Z[346112])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputHeightIdx_tmp;
  int inputWidthIdx;
  int inputWidthIdx_tmp;
  int varargout_7;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, varargout_7,       \
            inputHeightIdx_tmp, inputWidthIdx_tmp, filterWidthIdx,             \
            filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 346112; prodOutDimsIdx++) {
    inputWidthIdx =
        static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 10816U);
    varargout_7 = inputWidthIdx;
    inputHeightIdx = prodOutDimsIdx - inputWidthIdx * 10816;
    inputWidthIdx = inputHeightIdx / 104;
    inputHeightIdx -= inputWidthIdx * 104;
    inputHeightIdx_tmp = inputHeightIdx << 1;
    inputWidthIdx_tmp = inputWidthIdx << 1;
    opValue = -3.402823466E+38F;
    if ((inputHeightIdx_tmp + 1 > 0) && (inputWidthIdx_tmp + 1 > 0) &&
        (inputHeightIdx_tmp + 3 <= 208) && (inputWidthIdx_tmp + 3 <= 208)) {
      inputWidthIdx =
          (inputHeightIdx_tmp + 208 * inputWidthIdx_tmp) + 43264 * varargout_7;
      inputHeightIdx = (inputHeightIdx_tmp + 208 * (inputWidthIdx_tmp + 1)) +
                       43264 * varargout_7;
      opValue = std::fmax(
          std::fmax(std::fmax(std::fmax(-3.402823466E+38F, X[inputWidthIdx]),
                              X[inputWidthIdx + 1]),
                    X[inputHeightIdx]),
          X[inputHeightIdx + 1]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = (filterWidthIdx + inputWidthIdx_tmp) + 1;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = (filterHeightIdx + inputHeightIdx_tmp) + 1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 208) && (inputWidthIdx <= 208)) {
            inputPixel = X[((inputHeightIdx + 208 * (inputWidthIdx - 1)) +
                            43264 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

void c_iPoolingForColumnMajorWithOpenMP(const float X[692224], float Z[173056])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputHeightIdx_tmp;
  int inputWidthIdx;
  int inputWidthIdx_tmp;
  int varargout_7;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, varargout_7,       \
            inputHeightIdx_tmp, inputWidthIdx_tmp, filterWidthIdx,             \
            filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 173056; prodOutDimsIdx++) {
    inputWidthIdx =
        static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 2704U);
    varargout_7 = inputWidthIdx;
    inputHeightIdx = prodOutDimsIdx - inputWidthIdx * 2704;
    inputWidthIdx = inputHeightIdx / 52;
    inputHeightIdx -= inputWidthIdx * 52;
    inputHeightIdx_tmp = inputHeightIdx << 1;
    inputWidthIdx_tmp = inputWidthIdx << 1;
    opValue = -3.402823466E+38F;
    if ((inputHeightIdx_tmp + 1 > 0) && (inputWidthIdx_tmp + 1 > 0) &&
        (inputHeightIdx_tmp + 3 <= 104) && (inputWidthIdx_tmp + 3 <= 104)) {
      inputWidthIdx =
          (inputHeightIdx_tmp + 104 * inputWidthIdx_tmp) + 10816 * varargout_7;
      inputHeightIdx = (inputHeightIdx_tmp + 104 * (inputWidthIdx_tmp + 1)) +
                       10816 * varargout_7;
      opValue = std::fmax(
          std::fmax(std::fmax(std::fmax(-3.402823466E+38F, X[inputWidthIdx]),
                              X[inputWidthIdx + 1]),
                    X[inputHeightIdx]),
          X[inputHeightIdx + 1]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = (filterWidthIdx + inputWidthIdx_tmp) + 1;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = (filterHeightIdx + inputHeightIdx_tmp) + 1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 104) && (inputWidthIdx <= 104)) {
            inputPixel = X[((inputHeightIdx + 104 * (inputWidthIdx - 1)) +
                            10816 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

void d_iPoolingForColumnMajorWithOpenMP(const float X[346112], float Z[86528])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputHeightIdx_tmp;
  int inputWidthIdx;
  int inputWidthIdx_tmp;
  int varargout_7;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, varargout_7,       \
            inputHeightIdx_tmp, inputWidthIdx_tmp, filterWidthIdx,             \
            filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 86528; prodOutDimsIdx++) {
    inputWidthIdx =
        static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 676U);
    varargout_7 = inputWidthIdx;
    inputHeightIdx = prodOutDimsIdx - inputWidthIdx * 676;
    inputWidthIdx = inputHeightIdx / 26;
    inputHeightIdx -= inputWidthIdx * 26;
    inputHeightIdx_tmp = inputHeightIdx << 1;
    inputWidthIdx_tmp = inputWidthIdx << 1;
    opValue = -3.402823466E+38F;
    if ((inputHeightIdx_tmp + 1 > 0) && (inputWidthIdx_tmp + 1 > 0) &&
        (inputHeightIdx_tmp + 3 <= 52) && (inputWidthIdx_tmp + 3 <= 52)) {
      inputWidthIdx =
          (inputHeightIdx_tmp + 52 * inputWidthIdx_tmp) + 2704 * varargout_7;
      inputHeightIdx = (inputHeightIdx_tmp + 52 * (inputWidthIdx_tmp + 1)) +
                       2704 * varargout_7;
      opValue = std::fmax(
          std::fmax(std::fmax(std::fmax(-3.402823466E+38F, X[inputWidthIdx]),
                              X[inputWidthIdx + 1]),
                    X[inputHeightIdx]),
          X[inputHeightIdx + 1]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = (filterWidthIdx + inputWidthIdx_tmp) + 1;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = (filterHeightIdx + inputHeightIdx_tmp) + 1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 52) && (inputWidthIdx <= 52)) {
            inputPixel = X[((inputHeightIdx + 52 * (inputWidthIdx - 1)) +
                            2704 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

void e_iPoolingForColumnMajorWithOpenMP(const float X[173056], float Z[43264])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputHeightIdx_tmp;
  int inputWidthIdx;
  int inputWidthIdx_tmp;
  int varargout_7;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, varargout_7,       \
            inputHeightIdx_tmp, inputWidthIdx_tmp, filterWidthIdx,             \
            filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 43264; prodOutDimsIdx++) {
    inputWidthIdx =
        static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 169U);
    varargout_7 = inputWidthIdx;
    inputHeightIdx = prodOutDimsIdx - inputWidthIdx * 169;
    inputWidthIdx = inputHeightIdx / 13;
    inputHeightIdx -= inputWidthIdx * 13;
    inputHeightIdx_tmp = inputHeightIdx << 1;
    inputWidthIdx_tmp = inputWidthIdx << 1;
    opValue = -3.402823466E+38F;
    if ((inputHeightIdx_tmp + 1 > 0) && (inputWidthIdx_tmp + 1 > 0) &&
        (inputHeightIdx_tmp + 3 <= 26) && (inputWidthIdx_tmp + 3 <= 26)) {
      inputWidthIdx =
          (inputHeightIdx_tmp + 26 * inputWidthIdx_tmp) + 676 * varargout_7;
      inputHeightIdx = (inputHeightIdx_tmp + 26 * (inputWidthIdx_tmp + 1)) +
                       676 * varargout_7;
      opValue = std::fmax(
          std::fmax(std::fmax(std::fmax(-3.402823466E+38F, X[inputWidthIdx]),
                              X[inputWidthIdx + 1]),
                    X[inputHeightIdx]),
          X[inputHeightIdx + 1]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = (filterWidthIdx + inputWidthIdx_tmp) + 1;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = (filterHeightIdx + inputHeightIdx_tmp) + 1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 26) && (inputWidthIdx <= 26)) {
            inputPixel = X[((inputHeightIdx + 26 * (inputWidthIdx - 1)) +
                            676 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

void f_iPoolingForColumnMajorWithOpenMP(const float X[86528], float Z[86528])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputWidthIdx;
  int v1;
  int varargout_7;
  int vk;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, vk, varargout_7,   \
            v1, filterWidthIdx, filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 86528; prodOutDimsIdx++) {
    vk = static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 169U);
    varargout_7 = vk;
    v1 = prodOutDimsIdx - vk * 169;
    vk = v1 / 13 + 1;
    inputHeightIdx = (vk - 1) * 13;
    v1 = (v1 - inputHeightIdx) + 1;
    opValue = -3.402823466E+38F;
    if ((v1 > 0) && (vk > 0) && (v1 + 2 <= 13) && (vk + 2 <= 13)) {
      inputWidthIdx = (v1 + inputHeightIdx) + 169 * varargout_7;
      inputHeightIdx = (v1 + 13 * vk) + 169 * varargout_7;
      opValue = std::fmax(std::fmax(std::fmax(std::fmax(-3.402823466E+38F,
                                                        X[inputWidthIdx - 1]),
                                              X[inputWidthIdx]),
                                    X[inputHeightIdx - 1]),
                          X[inputHeightIdx]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = filterWidthIdx + vk;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = filterHeightIdx + v1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 13) && (inputWidthIdx <= 13)) {
            inputPixel = X[((inputHeightIdx + 13 * (inputWidthIdx - 1)) +
                            169 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

void iPoolingForColumnMajorWithOpenMP(const float X[2768896], float Z[692224])
{
  float inputPixel;
  float opValue;
  int filterHeightIdx;
  int filterWidthIdx;
  int inputHeightIdx;
  int inputHeightIdx_tmp;
  int inputWidthIdx;
  int inputWidthIdx_tmp;
  int varargout_7;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        inputPixel, inputHeightIdx, inputWidthIdx, opValue, varargout_7,       \
            inputHeightIdx_tmp, inputWidthIdx_tmp, filterWidthIdx,             \
            filterHeightIdx)

  for (int prodOutDimsIdx = 0; prodOutDimsIdx < 692224; prodOutDimsIdx++) {
    inputWidthIdx =
        static_cast<int>(static_cast<unsigned int>(prodOutDimsIdx) / 43264U);
    varargout_7 = inputWidthIdx;
    inputHeightIdx = prodOutDimsIdx - inputWidthIdx * 43264;
    inputWidthIdx = inputHeightIdx / 208;
    inputHeightIdx -= inputWidthIdx * 208;
    inputHeightIdx_tmp = inputHeightIdx << 1;
    inputWidthIdx_tmp = inputWidthIdx << 1;
    opValue = -3.402823466E+38F;
    if ((inputHeightIdx_tmp + 1 > 0) && (inputWidthIdx_tmp + 1 > 0) &&
        (inputHeightIdx_tmp + 3 <= 416) && (inputWidthIdx_tmp + 3 <= 416)) {
      inputWidthIdx =
          (inputHeightIdx_tmp + 416 * inputWidthIdx_tmp) + 173056 * varargout_7;
      inputHeightIdx = (inputHeightIdx_tmp + 416 * (inputWidthIdx_tmp + 1)) +
                       173056 * varargout_7;
      opValue = std::fmax(
          std::fmax(std::fmax(std::fmax(-3.402823466E+38F, X[inputWidthIdx]),
                              X[inputWidthIdx + 1]),
                    X[inputHeightIdx]),
          X[inputHeightIdx + 1]);
    } else {
      for (filterWidthIdx = 0; filterWidthIdx < 2; filterWidthIdx++) {
        inputWidthIdx = (filterWidthIdx + inputWidthIdx_tmp) + 1;
        for (filterHeightIdx = 0; filterHeightIdx < 2; filterHeightIdx++) {
          inputHeightIdx = (filterHeightIdx + inputHeightIdx_tmp) + 1;
          if ((inputHeightIdx > 0) && (inputWidthIdx > 0) &&
              (inputHeightIdx <= 416) && (inputWidthIdx <= 416)) {
            inputPixel = X[((inputHeightIdx + 416 * (inputWidthIdx - 1)) +
                            173056 * varargout_7) -
                           1];
          } else {
            inputPixel = -3.402823466E+38F;
          }
          opValue = std::fmax(opValue, inputPixel);
        }
      }
    }
    Z[prodOutDimsIdx] = opValue;
  }
}

} // namespace poolingUtils
} // namespace layer
} // namespace internal
} // namespace coder

// End of code generation (poolingOperation.cpp)
