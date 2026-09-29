//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// yolov2_detect_internal_types.h
//
// Code generation for function 'yolov2_detect'
//

#ifndef YOLOV2_DETECT_INTERNAL_TYPES_H
#define YOLOV2_DETECT_INTERNAL_TYPES_H

// Include files
#include "rtwtypes.h"
#include "yolov2_detect_types.h"
#include "coder_bounded_array.h"

// Type Definitions
struct struct_T {
  int xstart;
  int xend;
  int depth;
};

namespace coder {
struct dlarray {
  float Data[2535];
};

struct b_dlarray {
  float Data[1690];
};

} // namespace coder
struct cell_wrap_5 {
  coder::bounded_array<char, 14U, 2U> f1;
};

struct b_struct_T {
  coder::bounded_array<cell_wrap_5, 845U, 1U> c;
};

#endif
// End of code generation (yolov2_detect_internal_types.h)
