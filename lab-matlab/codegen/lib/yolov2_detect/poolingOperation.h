//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// poolingOperation.h
//
// Code generation for function 'poolingOperation'
//

#ifndef POOLINGOPERATION_H
#define POOLINGOPERATION_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace internal {
namespace layer {
namespace poolingUtils {
void b_iPoolingForColumnMajorWithOpenMP(const float X[1384448],
                                        float Z[346112]);

void c_iPoolingForColumnMajorWithOpenMP(const float X[692224], float Z[173056]);

void d_iPoolingForColumnMajorWithOpenMP(const float X[346112], float Z[86528]);

void e_iPoolingForColumnMajorWithOpenMP(const float X[173056], float Z[43264]);

void f_iPoolingForColumnMajorWithOpenMP(const float X[86528], float Z[86528]);

void iPoolingForColumnMajorWithOpenMP(const float X[2768896], float Z[692224]);

} // namespace poolingUtils
} // namespace layer
} // namespace internal
} // namespace coder

#endif
// End of code generation (poolingOperation.h)
