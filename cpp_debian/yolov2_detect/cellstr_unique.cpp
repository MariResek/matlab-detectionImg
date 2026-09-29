//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cellstr_unique.cpp
//
// Code generation for function 'cellstr_unique'
//

// Include files
#include "cellstr_unique.h"
#include "cellstr_sort.h"
#include "findHelper.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace matlab {
namespace internal {
namespace coder {
namespace datatypes {
int cellstr_unique(const cell_wrap_5 a_data[], int a_size, cell_wrap_5 u_data[])
{
  cell_wrap_5 c_data[845];
  int idx_data[845];
  int tmp_data[845];
  int ia_data[80];
  int idx_size;
  int u_size;
  bool d_data[844];
  bool b_d_data[80];
  if (a_size == 0) {
    u_size = 0;
  } else {
    int c_size;
    int loop_ub;
    int minval;
    int ret;
    int tmp_size;
    c_size = cellstr_sort(a_data, a_size, c_data, idx_data, idx_size);
    loop_ub = c_size - 1;
    for (int i{0}; i <= c_size - 2; i++) {
      bool b_bool;
      b_bool = false;
      ret = c_data[i].f1.size[1];
      if ((ret == 0) && (c_data[i + 1].f1.size[1] == 0)) {
        b_bool = true;
      } else if (ret != 0) {
        minval = c_data[i + 1].f1.size[1];
        if ((minval != 0) && (ret == minval)) {
          ret = std::memcmp(&c_data[i].f1.data[0], &c_data[i + 1].f1.data[0],
                            static_cast<unsigned int>(minval));
          b_bool = (ret == 0);
        }
      }
      d_data[i] = !b_bool;
    }
    b_d_data[0] = true;
    if (loop_ub - 1 >= 0) {
      std::copy(&d_data[0], &d_data[loop_ub], &b_d_data[1]);
    }
    ret = b_d_data[0];
    for (int i{2}; i <= c_size; i++) {
      ret += b_d_data[i - 1];
    }
    u_size = static_cast<int>(
        std::fmin(static_cast<double>(ret), static_cast<double>(a_size)));
    tmp_size = ::coder::internal::findHelper(b_d_data, c_size, tmp_data);
    c_size = ::coder::internal::findHelper(b_d_data, c_size, tmp_data);
    for (int b_i{0}; b_i < c_size; b_i++) {
      if (b_i + 1 != static_cast<short>(tmp_size)) {
        double y_data[845];
        ret = tmp_data[b_i + 1];
        minval = tmp_data[b_i];
        if (static_cast<double>(ret) - 1.0 < minval) {
          loop_ub = 0;
        } else {
          loop_ub = ret - minval;
          ret = loop_ub - 1;
          for (int i{0}; i <= ret; i++) {
            y_data[i] = static_cast<double>(minval) + static_cast<double>(i);
          }
        }
        minval = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int i{2}; i <= loop_ub; i++) {
          ret = idx_data[static_cast<int>(y_data[i - 1]) - 1];
          if (minval > ret) {
            minval = ret;
          }
        }
        ia_data[b_i] = minval;
      } else {
        double y_data[845];
        ret = tmp_data[b_i];
        if (idx_size < ret) {
          loop_ub = 0;
        } else {
          ret = idx_size - ret;
          loop_ub = ret + 1;
          for (int i{0}; i <= ret; i++) {
            y_data[i] = tmp_data[b_i] + i;
          }
        }
        minval = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int i{2}; i <= loop_ub; i++) {
          ret = idx_data[static_cast<int>(y_data[i - 1]) - 1];
          if (minval > ret) {
            minval = ret;
          }
        }
        ia_data[b_i] = minval;
      }
      ret = a_data[minval - 1].f1.size[1];
      u_data[b_i].f1.size[0] = 1;
      u_data[b_i].f1.size[1] = ret;
      for (int i{0}; i < ret; i++) {
        u_data[b_i].f1.data[i] = a_data[ia_data[b_i] - 1].f1.data[i];
      }
    }
  }
  return u_size;
}

int cellstr_unique(const cell_wrap_5 a_data[], int a_size, cell_wrap_5 u_data[],
                   double ia_data[], int &ia_size)
{
  cell_wrap_5 c_data[845];
  int idx_data[845];
  int tmp_data[845];
  int idx_size;
  int u_size;
  bool d_data[844];
  bool b_d_data[80];
  if (a_size == 0) {
    u_size = 0;
    ia_size = 0;
  } else {
    int c_size;
    int loop_ub;
    int minval;
    int ret;
    int tmp_size;
    c_size = cellstr_sort(a_data, a_size, c_data, idx_data, idx_size);
    loop_ub = c_size - 1;
    for (int i{0}; i <= c_size - 2; i++) {
      bool b_bool;
      b_bool = false;
      ret = c_data[i].f1.size[1];
      if ((ret == 0) && (c_data[i + 1].f1.size[1] == 0)) {
        b_bool = true;
      } else if (ret != 0) {
        minval = c_data[i + 1].f1.size[1];
        if ((minval != 0) && (ret == minval)) {
          ret = std::memcmp(&c_data[i].f1.data[0], &c_data[i + 1].f1.data[0],
                            static_cast<unsigned int>(minval));
          b_bool = (ret == 0);
        }
      }
      d_data[i] = !b_bool;
    }
    b_d_data[0] = true;
    if (loop_ub - 1 >= 0) {
      std::copy(&d_data[0], &d_data[loop_ub], &b_d_data[1]);
    }
    ret = b_d_data[0];
    for (int i{2}; i <= c_size; i++) {
      ret += b_d_data[i - 1];
    }
    u_size = static_cast<int>(
        std::fmin(static_cast<double>(ret), static_cast<double>(a_size)));
    tmp_size = ::coder::internal::findHelper(b_d_data, c_size, tmp_data);
    c_size = ::coder::internal::findHelper(b_d_data, c_size, tmp_data);
    ia_size = c_size;
    for (int b_i{0}; b_i < c_size; b_i++) {
      if (b_i + 1 != static_cast<short>(tmp_size)) {
        double y_data[845];
        ret = tmp_data[b_i + 1];
        minval = tmp_data[b_i];
        if (static_cast<double>(ret) - 1.0 < minval) {
          loop_ub = 0;
        } else {
          loop_ub = ret - minval;
          ret = loop_ub - 1;
          for (int i{0}; i <= ret; i++) {
            y_data[i] = static_cast<double>(minval) + static_cast<double>(i);
          }
        }
        minval = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int i{2}; i <= loop_ub; i++) {
          ret = idx_data[static_cast<int>(y_data[i - 1]) - 1];
          if (minval > ret) {
            minval = ret;
          }
        }
        ia_data[b_i] = minval;
      } else {
        double y_data[845];
        ret = tmp_data[b_i];
        if (idx_size < ret) {
          loop_ub = 0;
        } else {
          ret = idx_size - ret;
          loop_ub = ret + 1;
          for (int i{0}; i <= ret; i++) {
            y_data[i] = tmp_data[b_i] + i;
          }
        }
        minval = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int i{2}; i <= loop_ub; i++) {
          ret = idx_data[static_cast<int>(y_data[i - 1]) - 1];
          if (minval > ret) {
            minval = ret;
          }
        }
        ia_data[b_i] = minval;
      }
      ret = a_data[minval - 1].f1.size[1];
      u_data[b_i].f1.size[0] = 1;
      u_data[b_i].f1.size[1] = ret;
      for (int i{0}; i < ret; i++) {
        u_data[b_i].f1.data[i] =
            a_data[static_cast<int>(ia_data[b_i]) - 1].f1.data[i];
      }
    }
  }
  return u_size;
}

} // namespace datatypes
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder

// End of code generation (cellstr_unique.cpp)
