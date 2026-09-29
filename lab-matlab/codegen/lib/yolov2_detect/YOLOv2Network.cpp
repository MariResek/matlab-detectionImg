//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// YOLOv2Network.cpp
//
// Code generation for function 'YOLOv2Network'
//

// Include files
#include "YOLOv2Network.h"
#include "callPredict.h"
#include "categorical.h"
#include "cellstr_sort.h"
#include "cellstr_unique.h"
#include "findHelper.h"
#include "imresize.h"
#include "nullAssignment.h"
#include "rt_nonfinite.h"
#include "selectStrongestBboxCodegen.h"
#include "sort.h"
#include "strtrim.h"
#include "yolov2_detect_internal_types.h"
#include "coder_bounded_array.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Variable Definitions
static const char cv[1120]{
    'p',    'b',    'c',    'm',    'a',    'b',    't',    't',    'b',
    't',    'f',    's',    'p',    'b',    'b',    'c',    'd',    'h',
    's',    'c',    'e',    'b',    'z',    'g',    'b',    'u',    'h',
    't',    's',    'f',    's',    's',    's',    'k',    'b',    'b',
    's',    's',    't',    'b',    'w',    'c',    'f',    'k',    's',
    'b',    'b',    'a',    's',    'o',    'b',    'c',    'h',    'p',
    'd',    'c',    'c',    's',    'p',    'b',    'd',    't',    't',
    'l',    'm',    'r',    'k',    'c',    'm',    'o',    't',    's',
    'r',    'b',    'c',    'v',    's',    't',    'h',    't',    'e',
    'i',    'a',    'o',    'e',    'u',    'r',    'r',    'o',    'r',
    'i',    't',    'a',    'e',    'i',    'a',    'o',    'o',    'h',
    'o',    'l',    'e',    'e',    'i',    'a',    'm',    'a',    'i',
    'u',    'r',    'k',    'n',    'p',    'i',    'a',    'a',    'k',
    'u',    'e',    'o',    'i',    'u',    'o',    'n',    'p',    'o',
    'a',    'p',    'a',    'r',    'r',    'a',    'o',    'i',    'o',
    'a',    'h',    'o',    'o',    'e',    'i',    'o',    'v',    'a',
    'o',    'e',    'e',    'e',    'i',    'v',    'o',    'i',    'e',
    'o',    'l',    'a',    'c',    'e',    'a',    'o',    'r',    'c',
    'r',    't',    'r',    's',    'a',    'u',    'a',    'a',    'r',
    'o',    'r',    'n',    'r',    't',    'g',    'r',    'e',    'w',
    'e',    'a',    'b',    'r',    'c',    'b',    'n',    'e',    'i',
    'i',    'i',    'o',    'o',    't',    's',    's',    'a',    'r',
    'n',    't',    'n',    'p',    'r',    'i',    'o',    'w',    'n',
    'p',    'n',    'a',    'o',    'r',    't',    'z',    'n',    'k',
    'a',    'f',    't',    'd',    'n',    'i',    'm',    'p',    'u',
    'm',    'y',    'l',    'c',    'e',    'a',    'n',    'f',    'o',
    'o',    's',    'i',    'd',    'i',    'o',    's',    'y',    '\x00',
    'o',    'o',    '\x00', 'i',    'c',    't',    'f',    'e',    'p',
    'k',    'c',    'd',    '\x00', '\x00', 's',    'e',    '\x00', 'p',
    'r',    'r',    'a',    'k',    'r',    'd',    '\x00', 't',    's',
    's',    'w',    'r',    'e',    'e',    'e',    't',    'f',    'n',
    't',    'e',    '\x00', 'k',    'f',    'o',    'l',    'a',    'l',
    'd',    'n',    'c',    'r',    ' ',    'z',    'u',    'e',    'i',
    'a',    't',    '\x00', 'i',    'l',    'o',    't',    's',    'o',
    'b',    'l',    'r',    'n',    's',    'k',    'r',    'k',    'c',
    'e',    's',    'd',    'r',    't',    'o',    'c',    '\x00', 'r',
    'p',    '\x00', 'n',    'k',    '\x00', 'f',    ' ',    ' ',    'i',
    'h',    '\x00', '\x00', '\x00', 'e',    'p',    '\x00', 'h',    '\x00',
    'a',    'f',    'p',    'e',    'b',    '\x00', 'c',    'b',    '\x00',
    'b',    't',    '\x00', 'b',    'b',    'e',    'b',    'i',    'l',
    ' ',    '\x00', '\x00', 'e',    'n',    '\x00', 'n',    'e',    'w',
    'g',    'c',    'o',    'd',    'a',    't',    '\x00', 'r',    '\x00',
    'e',    '\x00', 'n',    'e',    'n',    'o',    'e',    't',    'o',
    ' ',    'o',    '\x00', 't',    '\x00', 'i',    '\x00', 'k',    '\x00',
    's',    'y',    ' ',    'h',    'n',    'l',    '\x00', 'b',    'l',
    '\x00', '\x00', '\x00', '\x00', 'i',    'h',    's',    'n',    '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'a',    '\x00', '\x00',
    'f',    'a',    'l',    'a',    '\x00', 'a',    'e',    '\x00', 'o',
    's',    '\x00', 'a',    'a',    'b',    'o',    's',    'e',    'g',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'a',    '\x00', 'i',    'e',
    'o',    't',    'o',    '\x00', '\x00', '\x00', '\x00', '\x00', 'd',
    '\x00', 'g',    't',    'i',    'p',    '\x00', 'e',    'a',    'p',
    'w',    '\x00', 'e',    '\x00', 'g',    '\x00', '\x00', '\x00', 'o',
    ' ',    'd',    'b',    '\x00', 'e',    '\x00', 'i',    'a',    '\x00',
    '\x00', '\x00', '\x00', 'c',    'y',    'i',    'g',    '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'n',    '\x00', '\x00', 'e',
    'c',    'l',    'g',    '\x00', 's',    'e',    '\x00', 'a',    ' ',
    '\x00', 'l',    'l',    'o',    'a',    ' ',    '\x00', 'l',    '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'c',    '\x00', 'l',
    '\x00', 'g',    '\x00', '\x00', '\x00', '\x00', '\x00', 'p',    '\x00',
    't',    '\x00', 't',    '\x00', '\x00', '\x00', 'r',    'h',    'a',
    '\x00', 'r',    '\x00', 'e',    '\x00', '\x00', '\x00', 'r',    'b',
    'r',    'r',    '\x00', '\x00', '\x00', 'k',    'n',    '\x00', '\x00',
    '\x00', '\x00', ' ',    'd',    'g',    ' ',    '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', 't',    '\x00', '\x00', '\x00', 'k',
    'a',    '\x00', '\x00', 'e',    '\x00', '\x00', 'r',    'b',    '\x00',
    'l',    'l',    'a',    'r',    'r',    '\x00', 'a',    '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'h',    '\x00', 'i',    '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'l',    '\x00', 'a',
    '\x00', 'o',    '\x00', '\x00', '\x00', 'd',    'o',    'v',    '\x00',
    '\x00', '\x00', 'r',    '\x00', '\x00', '\x00', 's',    'e',    'i',
    'u',    '\x00', '\x00', '\x00', 'e',    'e',    '\x00', '\x00', '\x00',
    '\x00', 'l',    'r',    'n',    'm',    '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'd',    'a',    '\x00', ' ',
    ' ',    'r',    'd',    'a',    '\x00', 's',    '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'a',    '\x00', 'b',    '\x00',
    'r',    '\x00', '\x00', '\x00', '\x00', 'n',    'e',    '\x00', '\x00',
    '\x00', 'a',    '\x00', '\x00', '\x00', '\x00', 'a',    'e',    's',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    'i',    'a',    '\x00', 'e',    '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'l',    '\x00', 'b',    'g',
    'd',    '\x00', 'c',    '\x00', 's',    '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', 'n',    '\x00', 'l',    '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', 'e',    '\x00', '\x00', '\x00', '\x00',
    't',    '\x00', '\x00', '\x00', '\x00', 'r',    'r',    'h',    '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'g',
    'n',    '\x00', 't',    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', 'l',    '\x00', 'a',    'l',    '\x00',
    '\x00', 'k',    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', 't',    '\x00', 'e',    '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'o',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'h',    't',
    '\x00', 'e',    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 't',    'o',    '\x00', '\x00',
    'e',    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 'r',    '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', 't',    '\x00', '\x00',
    'r',    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', 'v',    '\x00', '\x00', 't',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', 'e',    '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x00',
    '\x00', '\x00', '\x00', '\x00'};

// Function Declarations
namespace coder {
namespace vision {
namespace internal {
namespace codegen {
static void YOLOv2Network_returnCategoricalLabels(double numBBoxes,
                                                  const float labels_data[],
                                                  categorical &labelNames);

static void YOLOv2Network_returnCategoricalLabels(categorical &labelNames);

} // namespace codegen
} // namespace internal
} // namespace vision
} // namespace coder

// Function Definitions
namespace coder {
namespace vision {
namespace internal {
namespace codegen {
static void YOLOv2Network_returnCategoricalLabels(double numBBoxes,
                                                  const float labels_data[],
                                                  categorical &labelNames)
{
  static cell_wrap_5 labelCells_data[845];
  bounded_array<cell_wrap_5, 80U, 1U> b_tmp_data;
  cell_wrap_5 inData_data[845];
  cell_wrap_5 uA_data[845];
  cell_wrap_5 tmp_data[80];
  cell_wrap_5 uB_data[80];
  cell_wrap_5 valueset_data[80];
  double dIdx_data[845];
  double icA_data[845];
  double y_data[845];
  double ib_data[80];
  double d;
  int a__2_data[845];
  int c_tmp_data[845];
  int idx_data[845];
  int b_i;
  int b_minSize;
  int j;
  int labelCells_size;
  int minSize;
  int minSize_tmp;
  int uA_size;
  int uB_size;
  char v_data[14];
  char c;
  bool b_d_data[845];
  bool d_data[844];
  bool is_less_than;
  minSize = static_cast<int>(numBBoxes);
  for (int i{0}; i < minSize; i++) {
    b_minSize = 0;
    b_i = -1;
    for (int k{0}; k < 14; k++) {
      c = cv[(static_cast<int>(labels_data[i]) + 80 * k) - 1];
      if (c != 0) {
        b_minSize++;
        b_i++;
        v_data[b_i] = c;
      }
    }
    labelCells_data[i].f1.size[1] = b_minSize;
    for (int k{0}; k < b_minSize; k++) {
      labelCells_data[i].f1.data[k] = v_data[k];
    }
  }
  j = 0;
  for (int i{0}; i < 80; i++) {
    b_minSize = 0;
    b_i = -1;
    for (int k{0}; k < 14; k++) {
      c = cv[i + 80 * k];
      if (c != 0) {
        b_minSize++;
        b_i++;
        v_data[b_i] = c;
      }
    }
    j++;
    valueset_data[j - 1].f1.size[1] = b_minSize;
    for (int k{0}; k < b_minSize; k++) {
      valueset_data[j - 1].f1.data[k] = v_data[k];
    }
  }
  for (int k{0}; k < minSize; k++) {
    strtrim(labelCells_data[k].f1.data, labelCells_data[k].f1.size,
            inData_data[k].f1.data, inData_data[k].f1.size);
  }
  for (int k{0}; k < j; k++) {
    strtrim(valueset_data[k].f1.data, valueset_data[k].f1.size,
            tmp_data[k].f1.data, tmp_data[k].f1.size);
  }
  matlab::internal::coder::datatypes::cellstr_unique(tmp_data, j,
                                                     b_tmp_data.data);
  if (static_cast<int>(numBBoxes) == 0) {
    uA_size = 0;
    labelCells_size = 0;
  } else {
    labelCells_size = matlab::internal::coder::datatypes::cellstr_sort(
        inData_data, static_cast<int>(numBBoxes), labelCells_data, idx_data,
        uB_size);
    minSize = labelCells_size - 1;
    for (int k{0}; k <= labelCells_size - 2; k++) {
      is_less_than = false;
      b_i = labelCells_data[k].f1.size[1];
      if ((b_i == 0) && (labelCells_data[k + 1].f1.size[1] == 0)) {
        is_less_than = true;
      } else if (b_i != 0) {
        b_minSize = labelCells_data[k + 1].f1.size[1];
        if ((b_minSize != 0) && (b_i == b_minSize)) {
          b_i = std::memcmp(&labelCells_data[k].f1.data[0],
                            &labelCells_data[k + 1].f1.data[0],
                            static_cast<unsigned int>(b_minSize));
          is_less_than = (b_i == 0);
        }
      }
      d_data[k] = !is_less_than;
    }
    b_d_data[0] = true;
    if (minSize - 1 >= 0) {
      std::copy(&d_data[0], &d_data[minSize], &b_d_data[1]);
    }
    b_i = b_d_data[0];
    for (int k{2}; k <= labelCells_size; k++) {
      b_i += b_d_data[k - 1];
    }
    uA_size = static_cast<int>(
        std::fmin(static_cast<double>(b_i),
                  static_cast<double>(static_cast<int>(numBBoxes))));
    minSize_tmp =
        ::coder::internal::findHelper(b_d_data, labelCells_size, c_tmp_data);
    for (int k{0}; k < minSize_tmp; k++) {
      dIdx_data[k] = c_tmp_data[k];
    }
    for (int i{0}; i < minSize_tmp; i++) {
      if (i + 1 != minSize_tmp) {
        d = dIdx_data[i + 1] - 1.0;
        b_i = static_cast<int>(dIdx_data[i]);
        if (d < b_i) {
          minSize = 0;
        } else {
          b_i = static_cast<int>(d) - b_i;
          minSize = b_i + 1;
          for (int k{0}; k <= b_i; k++) {
            y_data[k] = dIdx_data[i] + static_cast<double>(k);
          }
        }
        b_i = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int k{2}; k <= minSize; k++) {
          b_minSize = idx_data[static_cast<int>(y_data[k - 1]) - 1];
          if (b_i > b_minSize) {
            b_i = b_minSize;
          }
        }
        a__2_data[i] = b_i;
      } else {
        b_i = static_cast<int>(dIdx_data[i]);
        if (uB_size < b_i) {
          minSize = 0;
        } else {
          b_i = uB_size - b_i;
          minSize = b_i + 1;
          for (int k{0}; k <= b_i; k++) {
            y_data[k] = dIdx_data[i] + static_cast<double>(k);
          }
        }
        b_i = idx_data[static_cast<int>(y_data[0]) - 1];
        for (int k{2}; k <= minSize; k++) {
          b_minSize = idx_data[static_cast<int>(y_data[k - 1]) - 1];
          if (b_i > b_minSize) {
            b_i = b_minSize;
          }
        }
        a__2_data[i] = b_i;
      }
    }
    for (int k{0}; k < labelCells_size; k++) {
      icA_data[k] = b_d_data[k];
    }
    if (labelCells_size != 1) {
      for (int k{0}; k <= labelCells_size - 2; k++) {
        icA_data[k + 1] += icA_data[k];
      }
    }
    std::copy(&icA_data[0], &icA_data[labelCells_size], &y_data[0]);
    for (int k{0}; k < labelCells_size; k++) {
      icA_data[idx_data[k] - 1] = y_data[k];
    }
    for (int k{0}; k < minSize_tmp; k++) {
      b_i = inData_data[a__2_data[k] - 1].f1.size[1];
      uA_data[k].f1.size[1] = b_i;
      for (int i{0}; i < b_i; i++) {
        uA_data[k].f1.data[i] = inData_data[a__2_data[k] - 1].f1.data[i];
      }
    }
  }
  uB_size = matlab::internal::coder::datatypes::cellstr_unique(
      tmp_data, j, uB_data, ib_data, b_i);
  if (uA_size - 1 >= 0) {
    std::memset(&dIdx_data[0], 0,
                static_cast<unsigned int>(uA_size) * sizeof(double));
  }
  if (uB_size > 0) {
    int c_i;
    bool exitg1;
    j = 0;
    c_i = 0;
    exitg1 = false;
    while (!exitg1 && (c_i <= uA_size - 1)) {
      minSize_tmp = uA_data[c_i].f1.size[1];
      b_i = uB_data[j].f1.size[1];
      if (minSize_tmp <= b_i) {
        b_minSize = minSize_tmp;
      } else {
        b_minSize = b_i;
      }
      if (b_minSize == 0) {
        is_less_than = (minSize_tmp < b_i);
      } else {
        minSize = 0;
        while ((minSize + 1 <= b_minSize) &&
               (uA_data[c_i].f1.data[minSize] == uB_data[j].f1.data[minSize])) {
          minSize++;
        }
        if (minSize + 1 == b_minSize + 1) {
          is_less_than = (minSize_tmp < b_i);
        } else {
          is_less_than =
              (uA_data[c_i].f1.data[minSize] < uB_data[j].f1.data[minSize]);
        }
      }
      if (!is_less_than) {
        is_less_than = false;
        if ((minSize_tmp == 0) && (b_i == 0)) {
          is_less_than = true;
        } else if ((minSize_tmp != 0) && (b_i != 0) && (minSize_tmp == b_i)) {
          b_i = std::memcmp(&uA_data[c_i].f1.data[0], &uB_data[j].f1.data[0],
                            static_cast<unsigned int>(b_i));
          is_less_than = (b_i == 0);
        }
        if (is_less_than) {
          dIdx_data[c_i] = ib_data[j];
          j++;
        } else {
          bool exitg2;
          exitg2 = false;
          while (!exitg2 && (j + 1 <= uB_size)) {
            b_i = uB_data[j].f1.size[1];
            if (minSize_tmp <= b_i) {
              minSize = minSize_tmp;
            } else {
              minSize = b_i;
            }
            if (minSize == 0) {
              is_less_than = (minSize_tmp > b_i);
            } else {
              b_minSize = 0;
              while ((b_minSize + 1 <= minSize) &&
                     (uA_data[c_i].f1.data[b_minSize] ==
                      uB_data[j].f1.data[b_minSize])) {
                b_minSize++;
              }
              if (b_minSize + 1 == minSize + 1) {
                is_less_than = (minSize_tmp > b_i);
              } else {
                is_less_than = (uA_data[c_i].f1.data[b_minSize] >
                                uB_data[j].f1.data[b_minSize]);
              }
            }
            if (is_less_than) {
              j++;
            } else {
              exitg2 = true;
            }
          }
          if (j + 1 <= uB_size) {
            is_less_than = false;
            if ((minSize_tmp == 0) && (uB_data[j].f1.size[1] == 0)) {
              is_less_than = true;
            } else if (minSize_tmp != 0) {
              b_i = uB_data[j].f1.size[1];
              if ((b_i != 0) && (minSize_tmp == b_i)) {
                b_i = std::memcmp(&uA_data[c_i].f1.data[0],
                                  &uB_data[j].f1.data[0],
                                  static_cast<unsigned int>(b_i));
                is_less_than = (b_i == 0);
              }
            }
            if (is_less_than) {
              dIdx_data[c_i] = ib_data[j];
              j++;
            }
          }
        }
      }
      if (j + 1 > uB_size) {
        exitg1 = true;
      } else {
        c_i++;
      }
    }
  }
  labelNames.codes.size[0] = labelCells_size;
  for (int k{0}; k < labelCells_size; k++) {
    unsigned int u;
    d = dIdx_data[static_cast<int>(icA_data[k]) - 1];
    if (d >= 0.0) {
      u = static_cast<unsigned int>(d);
    } else {
      u = 0U;
    }
    labelNames.codes.data[k] = u;
  }
}

static void YOLOv2Network_returnCategoricalLabels(categorical &labelNames)
{
  bounded_array<cell_wrap_5, 80U, 1U> b_tmp_data;
  cell_wrap_5 tmp_data[80];
  cell_wrap_5 valueset_data[80];
  double ib_data[80];
  int b_i;
  int valueset_size_idx_1;
  valueset_size_idx_1 = 0;
  for (int i{0}; i < 80; i++) {
    int n;
    char v_data[14];
    n = 0;
    b_i = -1;
    for (int k{0}; k < 14; k++) {
      char c;
      c = cv[i + 80 * k];
      if (c != 0) {
        n++;
        b_i++;
        v_data[b_i] = c;
      }
    }
    valueset_size_idx_1++;
    valueset_data[valueset_size_idx_1 - 1].f1.size[1] = n;
    for (int k{0}; k < n; k++) {
      valueset_data[valueset_size_idx_1 - 1].f1.data[k] = v_data[k];
    }
  }
  for (int k{0}; k < valueset_size_idx_1; k++) {
    strtrim(valueset_data[k].f1.data, valueset_data[k].f1.size,
            tmp_data[k].f1.data, tmp_data[k].f1.size);
  }
  matlab::internal::coder::datatypes::cellstr_unique(
      tmp_data, valueset_size_idx_1, b_tmp_data.data);
  matlab::internal::coder::datatypes::cellstr_unique(
      tmp_data, valueset_size_idx_1, valueset_data, ib_data, b_i);
  labelNames.codes.size[0] = 0;
}

int YOLOv2Network_detect(const unsigned char b_I[2764800], double bboxes_data[],
                         int bboxes_size[2], float scores_data[],
                         categorical &varargout_1)
{
  static const double anchors[10]{0.6875, 2.0625, 5.46875, 3.53125, 9.15625,
                                  0.5625, 1.875,  3.34375, 7.875,   9.78125};
  static double bboxPred_data[3380];
  static float d_I[519168];
  static float tmpFeatureMap_Data[71825];
  static float boxOut[5070];
  static unsigned char c_I[519168];
  double x1_data[845];
  double x2_data[845];
  double c_ex;
  double d;
  float classPred_data[845];
  float y_data[845];
  int idx_size[2];
  int cx_tmp;
  int i;
  int ind;
  int scores_size;
  int trueCount;
  int y;
  short b_tmp_data[845];
  short tmp_data[845];
  unsigned char b_ex;
  unsigned char ex;
  bool b_selectedIndex_data[845];
  bool selectedIndex_data[845];
  imresize(b_I, c_I);
  ex = c_I[0];
  b_ex = c_I[0];
  for (int k{0}; k < 519167; k++) {
    unsigned char u;
    u = c_I[k + 1];
    if (ex > u) {
      ex = u;
    }
    if (b_ex < u) {
      b_ex = u;
    }
  }
  y = b_ex - ex;
  for (int k{0}; k < 519168; k++) {
    d_I[k] = static_cast<float>(c_I[k] - ex) / static_cast<float>(y);
  }
  ::coder::internal::ctarget::predict(d_I, tmpFeatureMap_Data);
  for (int anchorIdx{0}; anchorIdx < 5; anchorIdx++) {
    for (int colIdx{0}; colIdx < 13; colIdx++) {
      for (int rowIdx{0}; rowIdx < 13; rowIdx++) {
        double probPred[80];
        float b_xyBbox_tmp;
        float cx;
        float cy;
        float xyBbox_tmp;
        ind = (rowIdx * 13 + colIdx) * 5 + anchorIdx;
        cx_tmp = (rowIdx + 13 * colIdx) + 169 * anchorIdx;
        cx = (tmpFeatureMap_Data[cx_tmp + 845] + static_cast<float>(colIdx)) *
             32.0F;
        cy = (tmpFeatureMap_Data[cx_tmp + 1690] + static_cast<float>(rowIdx)) *
             32.0F;
        xyBbox_tmp = tmpFeatureMap_Data[cx_tmp + 2535] *
                     static_cast<float>(anchors[anchorIdx + 5]) * 32.0F / 2.0F;
        boxOut[ind] = cx - xyBbox_tmp;
        b_xyBbox_tmp = tmpFeatureMap_Data[cx_tmp + 3380] *
                       static_cast<float>(anchors[anchorIdx]) * 32.0F / 2.0F;
        boxOut[ind + 845] = cy - b_xyBbox_tmp;
        boxOut[ind + 1690] = cx + xyBbox_tmp;
        boxOut[ind + 2535] = cy + b_xyBbox_tmp;
        for (int k{0}; k < 80; k++) {
          probPred[k] = tmpFeatureMap_Data[cx_tmp + 845 * (k + 5)];
        }
        if (!std::isnan(probPred[0])) {
          i = 1;
        } else {
          bool exitg1;
          i = 0;
          y = 2;
          exitg1 = false;
          while (!exitg1 && (y < 81)) {
            if (!std::isnan(probPred[y - 1])) {
              i = y;
              exitg1 = true;
            } else {
              y++;
            }
          }
        }
        if (i == 0) {
          c_ex = probPred[0];
          y = 1;
        } else {
          c_ex = probPred[i - 1];
          y = i;
          for (int k{i + 1}; k < 81; k++) {
            d = probPred[k - 1];
            if (c_ex < d) {
              c_ex = d;
              y = k;
            }
          }
        }
        boxOut[ind + 3380] =
            tmpFeatureMap_Data[cx_tmp] * static_cast<float>(c_ex);
        boxOut[ind + 4225] = static_cast<float>(y);
      }
    }
  }
  trueCount = 0;
  y = 0;
  for (int k{0}; k < 845; k++) {
    if (boxOut[k + 3380] >= 0.4) {
      trueCount++;
      tmp_data[y] = static_cast<short>(k);
      y++;
    }
  }
  if (trueCount != 0) {
    double y1_data[845];
    double y2_data[845];
    int idx_data[845];
    int count;
    bboxes_size[0] = trueCount;
    for (int k{0}; k < 4; k++) {
      for (int anchorIdx{0}; anchorIdx < trueCount; anchorIdx++) {
        bboxes_data[anchorIdx + bboxes_size[0] * k] =
            boxOut[tmp_data[anchorIdx] + 845 * k];
      }
    }
    for (int k{0}; k < trueCount; k++) {
      double d1;
      double d2;
      c_ex = bboxes_data[k];
      x1_data[k] = c_ex;
      d = bboxes_data[k + bboxes_size[0]];
      y1_data[k] = d;
      d1 = bboxes_data[k + bboxes_size[0] * 2];
      x2_data[k] = d1;
      d2 = bboxes_data[k + bboxes_size[0] * 3];
      y2_data[k] = d2;
      if (c_ex < 1.0) {
        x1_data[k] = 1.0;
      }
      if (d < 1.0) {
        y1_data[k] = 1.0;
      }
      if (d1 > 416.0) {
        x2_data[k] = 416.0;
      }
      if (d2 > 416.0) {
        y2_data[k] = 416.0;
      }
    }
    for (int k{0}; k < trueCount; k++) {
      bboxes_data[k] = x1_data[k];
      bboxes_data[k + bboxes_size[0]] = y1_data[k];
      bboxes_data[k + bboxes_size[0] * 2] = x2_data[k];
      bboxes_data[k + bboxes_size[0] * 3] = y2_data[k];
    }
    for (int k{0}; k < trueCount; k++) {
      bboxPred_data[k] =
          ((bboxes_data[k] - 0.5) * 3.076923076923077 - 1.0384615384615385) +
          0.5;
      y = k + bboxes_size[0];
      bboxPred_data[y] =
          ((bboxes_data[y] - 0.5) * 1.7307692307692308 - 0.3653846153846154) +
          0.5;
      y = k + bboxes_size[0] * 2;
      bboxPred_data[y] =
          ((bboxes_data[y] + 0.5) * 3.076923076923077 - 1.0384615384615385) -
          0.5;
      y = k + bboxes_size[0] * 3;
      bboxPred_data[y] =
          ((bboxes_data[y] + 0.5) * 1.7307692307692308 - 0.3653846153846154) -
          0.5;
    }
    y = trueCount << 2;
    for (int k{0}; k < y; k++) {
      bboxPred_data[k] = std::floor(bboxPred_data[k]);
    }
    for (int k{0}; k < trueCount; k++) {
      x2_data[k] = (bboxPred_data[k + trueCount * 2] - bboxPred_data[k]) + 1.0;
    }
    for (int k{0}; k < trueCount; k++) {
      bboxPred_data[k + trueCount * 2] = x2_data[k];
    }
    for (int k{0}; k < trueCount; k++) {
      x2_data[k] =
          (bboxPred_data[k + trueCount * 3] - bboxPred_data[k + trueCount]) +
          1.0;
    }
    for (int k{0}; k < trueCount; k++) {
      bboxPred_data[k + trueCount * 3] = x2_data[k];
    }
    count = 1;
    scores_size = trueCount;
    for (int k{0}; k < trueCount; k++) {
      c_ex = bboxPred_data[k + trueCount * 3];
      if (c_ex >= 1.0) {
        d = bboxPred_data[k + trueCount * 2];
        if ((d >= 1.0) && (c_ex <= 720.0) && (d <= 1280.0)) {
          short b_i;
          count++;
          bboxes_data[count - 2] = bboxPred_data[k];
          bboxes_data[(count + bboxes_size[0]) - 2] =
              bboxPred_data[k + trueCount];
          bboxes_data[(count + bboxes_size[0] * 2) - 2] = d;
          bboxes_data[(count + bboxes_size[0] * 3) - 2] = c_ex;
          b_i = tmp_data[k];
          scores_data[count - 2] = boxOut[b_i + 3380];
          classPred_data[count - 2] = boxOut[b_i + 4225];
        }
      }
    }
    cx_tmp = bboxes_size[0] - count;
    for (int k{0}; k <= cx_tmp; k++) {
      idx_data[k] = count + k;
    }
    if (cx_tmp + 1 == 1) {
      ind = bboxes_size[0] - 1;
      y = static_cast<short>(idx_data[0]);
      for (int k{0}; k < 4; k++) {
        for (int anchorIdx{y}; anchorIdx <= ind; anchorIdx++) {
          i = anchorIdx + bboxes_size[0] * k;
          bboxes_data[i - 1] = bboxes_data[i];
        }
      }
    } else {
      std::memset(&selectedIndex_data[0], 0,
                  static_cast<unsigned int>(trueCount) * sizeof(bool));
      for (int k{0}; k <= cx_tmp; k++) {
        selectedIndex_data[static_cast<short>(idx_data[k]) - 1] = true;
      }
      y = 0;
      i = 0;
      for (int k{0}; k < trueCount; k++) {
        bool b;
        b = selectedIndex_data[k];
        y += b;
        if ((k + 1 > trueCount) || !b) {
          bboxes_data[i] = bboxes_data[k];
          bboxes_data[i + bboxes_size[0]] = bboxes_data[k + bboxes_size[0]];
          bboxes_data[i + bboxes_size[0] * 2] =
              bboxes_data[k + bboxes_size[0] * 2];
          bboxes_data[i + bboxes_size[0] * 3] =
              bboxes_data[k + bboxes_size[0] * 3];
          i++;
        }
      }
      ind = bboxes_size[0] - y;
    }
    if (ind < 1) {
      ind = 0;
    }
    for (int k{0}; k < 4; k++) {
      for (int anchorIdx{0}; anchorIdx < ind; anchorIdx++) {
        bboxes_data[anchorIdx + ind * k] =
            bboxes_data[anchorIdx + bboxes_size[0] * k];
      }
    }
    bboxes_size[0] = ind;
    bboxes_size[1] = 4;
    idx_size[1] = cx_tmp + 1;
    for (int k{0}; k <= cx_tmp; k++) {
      idx_data[k] = count + k;
    }
    ::coder::internal::nullAssignment(scores_data, scores_size, idx_data,
                                      idx_size);
    idx_size[1] = cx_tmp + 1;
    for (int k{0}; k <= cx_tmp; k++) {
      idx_data[k] = count + k;
    }
    ::coder::internal::nullAssignment(classPred_data, trueCount, idx_data,
                                      idx_size);
    if (ind == 0) {
      if (trueCount - 1 >= 0) {
        std::copy(&classPred_data[0], &classPred_data[trueCount], &y_data[0]);
      }
    } else {
      for (int k{0}; k < trueCount; k++) {
        x1_data[k] = classPred_data[k];
      }
      if (scores_size - 1 >= 0) {
        std::copy(&scores_data[0], &scores_data[scores_size], &y_data[0]);
      }
      i = ::coder::internal::sort(y_data, scores_size, idx_data);
      for (int k{0}; k < i; k++) {
        y = idx_data[k];
        y1_data[k] = y;
        x2_data[k] = x1_data[y - 1];
      }
      if (i - 1 >= 0) {
        std::copy(&x2_data[0], &x2_data[i], &x1_data[0]);
      }
      idx_size[0] = i;
      for (int k{0}; k < 4; k++) {
        for (int anchorIdx{0}; anchorIdx < i; anchorIdx++) {
          bboxPred_data[anchorIdx + i * k] = bboxes_data
              [(static_cast<int>(y1_data[anchorIdx]) + bboxes_size[0] * k) - 1];
        }
      }
      i = detector::selectStrongestBboxCodegen(bboxPred_data, idx_size, x1_data,
                                               selectedIndex_data);
      if (i - 1 >= 0) {
        std::copy(&selectedIndex_data[0], &selectedIndex_data[i],
                  &b_selectedIndex_data[0]);
      }
      for (int k{0}; k < i; k++) {
        selectedIndex_data[static_cast<int>(y1_data[k]) - 1] =
            b_selectedIndex_data[k];
      }
      scores_size = 0;
      y = 0;
      for (int k{0}; k < i; k++) {
        if (selectedIndex_data[k]) {
          scores_size++;
          b_tmp_data[y] = static_cast<short>(k);
          y++;
        }
      }
      for (int k{0}; k < 4; k++) {
        for (int anchorIdx{0}; anchorIdx < scores_size; anchorIdx++) {
          bboxPred_data[anchorIdx + scores_size * k] =
              bboxes_data[b_tmp_data[anchorIdx] + bboxes_size[0] * k];
        }
      }
      bboxes_size[0] = scores_size;
      bboxes_size[1] = 4;
      y = scores_size << 2;
      if (y - 1 >= 0) {
        std::copy(&bboxPred_data[0], &bboxPred_data[y], &bboxes_data[0]);
      }
      for (int k{0}; k < scores_size; k++) {
        y_data[k] = scores_data[b_tmp_data[k]];
      }
      for (int k{0}; k < scores_size; k++) {
        scores_data[k] = y_data[k];
        y_data[k] = classPred_data[b_tmp_data[k]];
      }
    }
    YOLOv2Network_returnCategoricalLabels(static_cast<double>(bboxes_size[0]),
                                          y_data, varargout_1);
  } else {
    bboxes_size[0] = 0;
    bboxes_size[1] = 4;
    scores_size = 0;
    YOLOv2Network_returnCategoricalLabels(varargout_1);
  }
  return scores_size;
}

} // namespace codegen
} // namespace internal
} // namespace vision
} // namespace coder

// End of code generation (YOLOv2Network.cpp)
