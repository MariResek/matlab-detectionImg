//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// heapsort.cpp
//
// Code generation for function 'heapsort'
//

// Include files
#include "heapsort.h"
#include "anonymous_function.h"
#include "cellstr_sort.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <cstring>

// Function Declarations
namespace coder {
namespace internal {
static void heapify(int x_data[], int idx, int xstart, int xend,
                    const anonymous_function &cmp);

}
} // namespace coder

// Function Definitions
namespace coder {
namespace internal {
static void heapify(int x_data[], int idx, int xstart, int xend,
                    const anonymous_function &cmp)
{
  int extremum;
  int extremumIdx;
  int leftIdx;
  bool changed;
  bool exitg1;
  changed = true;
  extremumIdx = (idx + xstart) - 2;
  leftIdx = ((idx << 1) + xstart) - 1;
  exitg1 = false;
  while (!exitg1 && (leftIdx < xend)) {
    int cmpIdx;
    int xcmp;
    changed = false;
    extremum = x_data[extremumIdx];
    cmpIdx = leftIdx - 1;
    xcmp = x_data[leftIdx - 1];
    if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
            cmp.workspace.c.data, x_data[leftIdx - 1], x_data[leftIdx])) {
      cmpIdx = leftIdx;
      xcmp = x_data[leftIdx];
    }
    if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
            cmp.workspace.c.data, x_data[extremumIdx], xcmp)) {
      x_data[extremumIdx] = xcmp;
      x_data[cmpIdx] = extremum;
      extremumIdx = cmpIdx;
      leftIdx = ((((cmpIdx - xstart) + 2) << 1) + xstart) - 1;
      changed = true;
    } else {
      exitg1 = true;
    }
  }
  if (changed && (leftIdx <= xend)) {
    extremum = x_data[extremumIdx];
    if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
            cmp.workspace.c.data, x_data[extremumIdx], x_data[leftIdx - 1])) {
      x_data[extremumIdx] = x_data[leftIdx - 1];
      x_data[leftIdx - 1] = extremum;
    }
  }
}

void b_heapsort(int x_data[], int xstart, int xend,
                const anonymous_function &cmp)
{
  int n;
  n = xend - xstart;
  for (int idx{n + 1}; idx >= 1; idx--) {
    heapify(x_data, idx, xstart, xend, cmp);
  }
  for (int idx{0}; idx < n; idx++) {
    int t;
    int t_tmp;
    t_tmp = (xend - idx) - 1;
    t = x_data[t_tmp];
    x_data[t_tmp] = x_data[xstart - 1];
    x_data[xstart - 1] = t;
    heapify(x_data, 1, xstart, t_tmp, cmp);
  }
}

} // namespace internal
} // namespace coder

// End of code generation (heapsort.cpp)
