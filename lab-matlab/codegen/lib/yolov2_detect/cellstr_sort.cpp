//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cellstr_sort.cpp
//
// Code generation for function 'cellstr_sort'
//

// Include files
#include "cellstr_sort.h"
#include "anonymous_function.h"
#include "introsort.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace matlab {
namespace internal {
namespace coder {
namespace datatypes {
int cellstr_sort(const cell_wrap_5 c_data[], int c_size,
                 cell_wrap_5 sorted_data[], int idx_data[], int &idx_size)
{
  anonymous_function b_this;
  int sorted_size;
  short y_data[845];
  y_data[0] = 1;
  sorted_size = 1;
  for (int k{2}; k <= c_size; k++) {
    sorted_size++;
    y_data[k - 1] = static_cast<short>(sorted_size);
  }
  idx_size = c_size;
  for (int k{0}; k < c_size; k++) {
    b_this.workspace.c.data[k] = c_data[k];
    idx_data[k] = y_data[k];
  }
  ::coder::internal::introsort(idx_data, c_size, b_this);
  sorted_size = c_size;
  for (int k{0}; k < c_size; k++) {
    int loop_ub;
    loop_ub = c_data[idx_data[k] - 1].f1.size[1];
    sorted_data[k].f1.size[0] = 1;
    sorted_data[k].f1.size[1] = loop_ub;
    for (int i{0}; i < loop_ub; i++) {
      sorted_data[k].f1.data[i] = c_data[idx_data[k] - 1].f1.data[i];
    }
  }
  return sorted_size;
}

bool cellstr_sort_anonFcn1(const cell_wrap_5 c_data[], int i, int j)
{
  int b_n_tmp;
  int k;
  int n;
  int n_tmp;
  bool varargout_1;
  n_tmp = c_data[i - 1].f1.size[1];
  b_n_tmp = c_data[j - 1].f1.size[1];
  n = static_cast<int>(
      std::fmin(static_cast<double>(n_tmp), static_cast<double>(b_n_tmp)));
  varargout_1 = (n_tmp < b_n_tmp);
  k = 0;
  int exitg1;
  do {
    exitg1 = 0;
    if (k <= n - 1) {
      if (c_data[i - 1].f1.data[k] != c_data[j - 1].f1.data[k]) {
        varargout_1 = (c_data[i - 1].f1.data[k] < c_data[j - 1].f1.data[k]);
        exitg1 = 1;
      } else {
        k++;
      }
    } else {
      if (n_tmp == b_n_tmp) {
        varargout_1 = (i < j);
      }
      exitg1 = 1;
    }
  } while (exitg1 == 0);
  return varargout_1;
}

} // namespace datatypes
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder

// End of code generation (cellstr_sort.cpp)
