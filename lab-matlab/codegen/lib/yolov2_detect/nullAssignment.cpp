//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// nullAssignment.cpp
//
// Code generation for function 'nullAssignment'
//

// Include files
#include "nullAssignment.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
void nullAssignment(float x_data[], int &x_size, const int idx_data[],
                    const int idx_size[2])
{
  int k0;
  int nxin;
  int nxout;
  bool b_data[845];
  nxin = x_size;
  if (nxin - 1 >= 0) {
    std::memset(&b_data[0], 0, static_cast<unsigned int>(nxin) * sizeof(bool));
  }
  nxout = idx_size[1];
  for (int k{0}; k < nxout; k++) {
    b_data[idx_data[k] - 1] = true;
  }
  nxout = 0;
  k0 = -1;
  for (int k{0}; k < nxin; k++) {
    bool b;
    b = b_data[k];
    nxout += b;
    if ((k + 1 > nxin) || !b) {
      k0++;
      x_data[k0] = x_data[k];
    }
  }
  nxout = x_size - nxout;
  if (nxout < 1) {
    x_size = 0;
  } else {
    x_size = nxout;
  }
}

} // namespace internal
} // namespace coder

// End of code generation (nullAssignment.cpp)
