//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// selectStrongestBboxCodegen.cpp
//
// Code generation for function 'selectStrongestBboxCodegen'
//

// Include files
#include "selectStrongestBboxCodegen.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace vision {
namespace internal {
namespace detector {
int selectStrongestBboxCodegen(const double inputBbox_data[],
                               const int inputBbox_size[2],
                               const double varargin_1_data[],
                               bool isKept_data[])
{
  double area_data[845];
  double x1_data[845];
  double x2_data[845];
  double y1_data[845];
  double y2_data[845];
  double label;
  double overlapRatio;
  double width;
  int b_i;
  int b_j;
  int currentBox;
  int isKept_size;
  int loop_ub;
  int numOfBbox;
  int ub_loop;
  loop_ub = inputBbox_size[0];
  isKept_size = inputBbox_size[0];
  for (int i{0}; i < loop_ub; i++) {
    double d;
    double d1;
    isKept_data[i] = true;
    label = inputBbox_data[i + inputBbox_size[0] * 2];
    d = inputBbox_data[i + inputBbox_size[0] * 3];
    area_data[i] = label * d;
    d1 = inputBbox_data[i];
    x1_data[i] = d1;
    x2_data[i] = d1 + label;
    label = inputBbox_data[i + inputBbox_size[0]];
    y1_data[i] = label;
    y2_data[i] = label + d;
  }
  numOfBbox = inputBbox_size[0] - 2;
  currentBox = -1;
  b_i = inputBbox_size[0];
  for (int i{0}; i < b_i; i++) {
    int c_i;
    c_i = i + 1;
    currentBox = i;
    label = varargin_1_data[i];
    if (std::isnan(label)) {
      isKept_data[i] = false;
    } else {
      if (isKept_data[i]) {
        ub_loop = 2;
      } else {
        ub_loop = 1;
      }
      if (ub_loop != 1) {
        int i1;
        i1 = i + 2;
        ub_loop = numOfBbox - i;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
        overlapRatio, width, b_j)

        for (int j = 0; j <= ub_loop; j++) {
          b_j = i1 + j;
          if (isKept_data[b_j - 1] && !(varargin_1_data[b_j - 1] != label)) {
            width = std::fmin(x2_data[c_i - 1], x2_data[b_j - 1]) -
                    std::fmax(x1_data[c_i - 1], x1_data[b_j - 1]);
            if (!(width <= 0.0)) {
              overlapRatio = std::fmin(y2_data[c_i - 1], y2_data[b_j - 1]) -
                             std::fmax(y1_data[c_i - 1], y1_data[b_j - 1]);
              if (!(overlapRatio <= 0.0)) {
                overlapRatio *= width;
                overlapRatio /=
                    (area_data[c_i - 1] + area_data[b_j - 1]) - overlapRatio;
                if (overlapRatio > 0.5) {
                  isKept_data[b_j - 1] = false;
                }
              }
            }
          }
        }
      }
    }
  }
  if (currentBox + 2 > isKept_size) {
    b_i = 0;
    loop_ub = 0;
  } else {
    b_i = currentBox + 1;
  }
  ub_loop = loop_ub - b_i;
  if (ub_loop - 1 >= 0) {
    std::memset(&isKept_data[b_i], 0,
                static_cast<unsigned int>(ub_loop) * sizeof(bool));
  }
  return isKept_size;
}

} // namespace detector
} // namespace internal
} // namespace vision
} // namespace coder

// End of code generation (selectStrongestBboxCodegen.cpp)
