//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// findHelper.cpp
//
// Code generation for function 'findHelper'
//

// Include files
#include "findHelper.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
int findHelper(const bool x_data[], int x_size, int i_data[])
{
  int i_size;
  int ii;
  bool exitg1;
  i_size = 0;
  ii = 0;
  exitg1 = false;
  while (!exitg1 && (ii <= x_size - 1)) {
    if (x_data[ii]) {
      i_size++;
      i_data[i_size - 1] = ii + 1;
      if (i_size >= x_size) {
        exitg1 = true;
      } else {
        ii++;
      }
    } else {
      ii++;
    }
  }
  if ((x_size == 1) && (i_size == 0)) {
    i_size = 0;
  } else if (i_size < 1) {
    i_size = 0;
  }
  return i_size;
}

} // namespace internal
} // namespace coder

// End of code generation (findHelper.cpp)
