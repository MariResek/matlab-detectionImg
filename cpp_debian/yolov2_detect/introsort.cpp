//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// introsort.cpp
//
// Code generation for function 'introsort'
//

// Include files
#include "introsort.h"
#include "anonymous_function.h"
#include "cellstr_sort.h"
#include "heapsort.h"
#include "insertionsort.h"
#include "rt_nonfinite.h"
#include "stack1.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
void introsort(int x_data[], int xend, const anonymous_function &cmp)
{
  struct_T frame;
  if (xend > 1) {
    if (xend <= 32) {
      insertionsort(x_data, 1, xend, cmp);
    } else {
      stack st;
      int MAXDEPTH;
      int pmax;
      int pmin;
      int pow2p;
      int t;
      bool exitg1;
      pmax = 31;
      pmin = 0;
      exitg1 = false;
      while (!exitg1 && (pmax - pmin > 1)) {
        t = (pmin + pmax) >> 1;
        pow2p = 1 << t;
        if (pow2p == xend) {
          pmax = t;
          exitg1 = true;
        } else if (pow2p > xend) {
          pmax = t;
        } else {
          pmin = t;
        }
      }
      MAXDEPTH = (pmax - 1) << 1;
      frame.xstart = 1;
      frame.xend = xend;
      frame.depth = 0;
      st.d.data[0] = frame;
      st.n = 1;
      while (st.n > 0) {
        int frame_tmp_tmp;
        frame_tmp_tmp = st.n - 1;
        frame = st.d.data[st.n - 1];
        st.n--;
        pmax = frame.xend - frame.xstart;
        if (pmax + 1 <= 32) {
          insertionsort(x_data, frame.xstart, frame.xend, cmp);
        } else if (frame.depth == MAXDEPTH) {
          b_heapsort(x_data, frame.xstart, frame.xend, cmp);
        } else {
          pmax = (frame.xstart + pmax / 2) - 1;
          if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
                  cmp.workspace.c.data, x_data[pmax],
                  x_data[frame.xstart - 1])) {
            t = x_data[frame.xstart - 1];
            x_data[frame.xstart - 1] = x_data[pmax];
            x_data[pmax] = t;
          }
          if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
                  cmp.workspace.c.data, x_data[frame.xend - 1],
                  x_data[frame.xstart - 1])) {
            t = x_data[frame.xstart - 1];
            x_data[frame.xstart - 1] = x_data[frame.xend - 1];
            x_data[frame.xend - 1] = t;
          }
          if (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
                  cmp.workspace.c.data, x_data[frame.xend - 1], x_data[pmax])) {
            t = x_data[pmax];
            x_data[pmax] = x_data[frame.xend - 1];
            x_data[frame.xend - 1] = t;
          }
          pow2p = x_data[pmax];
          x_data[pmax] = x_data[frame.xend - 2];
          x_data[frame.xend - 2] = pow2p;
          pmax = frame.xstart - 1;
          pmin = frame.xend - 2;
          int exitg2;
          do {
            exitg2 = 0;
            pmax++;
            while (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
                cmp.workspace.c.data, x_data[pmax], pow2p)) {
              pmax++;
            }
            pmin--;
            while (matlab::internal::coder::datatypes::cellstr_sort_anonFcn1(
                cmp.workspace.c.data, pow2p, x_data[pmin])) {
              pmin--;
            }
            if (pmax + 1 >= pmin + 1) {
              exitg2 = 1;
            } else {
              t = x_data[pmax];
              x_data[pmax] = x_data[pmin];
              x_data[pmin] = t;
            }
          } while (exitg2 == 0);
          x_data[frame.xend - 2] = x_data[pmax];
          x_data[pmax] = pow2p;
          if (pmax + 2 < frame.xend) {
            st.d.data[frame_tmp_tmp].xstart = pmax + 2;
            st.d.data[frame_tmp_tmp].xend = frame.xend;
            st.d.data[frame_tmp_tmp].depth = frame.depth + 1;
            st.n = frame_tmp_tmp + 1;
          }
          if (frame.xstart < pmax + 1) {
            st.d.data[st.n].xstart = frame.xstart;
            st.d.data[st.n].xend = pmax + 1;
            st.d.data[st.n].depth = frame.depth + 1;
            st.n++;
          }
        }
      }
    }
  }
}

} // namespace internal
} // namespace coder

// End of code generation (introsort.cpp)
