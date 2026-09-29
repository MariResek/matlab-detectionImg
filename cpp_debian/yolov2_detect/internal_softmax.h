//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// internal_softmax.h
//
// Code generation for function 'internal_softmax'
//

#ifndef INTERNAL_SOFTMAX_H
#define INTERNAL_SOFTMAX_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace deep {
namespace internal {
namespace coder {
namespace dlarray {
void iComputeSoftmaxForCpu(const float xdata[67600], float ydata[67600]);

}
} // namespace coder
} // namespace internal
} // namespace deep
} // namespace coder

#endif
// End of code generation (internal_softmax.h)
