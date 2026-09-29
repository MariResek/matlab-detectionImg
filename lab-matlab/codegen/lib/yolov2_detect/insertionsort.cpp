//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// insertionsort.cpp
//
// Code generation for function 'insertionsort'
//

// Include files
#include "insertionsort.h"
#include "anonymous_function.h"
#include "cellstr_sort.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
void insertionsort(int x_data[], int xstart, int xend,
                   const anonymous_function &cmp)
{
  for (int k{xstart + 1}; k <= xend; k++) {
    int idx;
    int xc;
    xc = x_data[k - 1];
    idx = k - 1;
    while ((idx >= xstart) &&
           matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
               cmp.workspace.c.data, xc, x_data[idx - 1])) {
      x_data[idx] = x_data[idx - 1];
      idx--;
    }
    x_data[idx] = xc;
  }
}

} // namespace internal
} // namespace coder

// End of code generation (insertionsort.cpp)
