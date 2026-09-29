//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// callPredict.cpp
//
// Code generation for function 'callPredict'
//

// Include files
#include "callPredict.h"
#include "YOLOv2TransformLayer.h"
#include "conv2dDirectOptimizedColMajor.h"
#include "poolingOperation.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace ctarget {
void predict(const float inputsT_0_f1[519168], float outputs_0_f1[71825])
{
  static float c_X[2768896];
  static float X[1384448];
  static float Z[692224];
  static float b_Z[346112];
  static float d_Z[173056];
  static float b_X[86528];
  static float e_Z[86528];
  static float outT_f16_0_f1[71825];
  static float c_Z[43264];
  layer::conv2dDirectOptimizedColMajor(inputsT_0_f1, c_X);
  layer::poolingUtils::iPoolingForColumnMajorWithOpenMP(c_X, Z);
  layer::b_conv2dDirectOptimizedColMajor(Z, X);
  layer::poolingUtils::b_iPoolingForColumnMajorWithOpenMP(X, b_Z);
  layer::c_conv2dDirectOptimizedColMajor(b_Z, Z);
  layer::poolingUtils::c_iPoolingForColumnMajorWithOpenMP(Z, d_Z);
  layer::d_conv2dDirectOptimizedColMajor(d_Z, b_Z);
  layer::poolingUtils::d_iPoolingForColumnMajorWithOpenMP(b_Z, e_Z);
  layer::e_conv2dDirectOptimizedColMajor(e_Z, d_Z);
  layer::poolingUtils::e_iPoolingForColumnMajorWithOpenMP(d_Z, c_Z);
  layer::f_conv2dDirectOptimizedColMajor(c_Z, b_X);
  layer::poolingUtils::f_iPoolingForColumnMajorWithOpenMP(b_X, e_Z);
  layer::g_conv2dDirectOptimizedColMajor(e_Z, d_Z);
  layer::h_conv2dDirectOptimizedColMajor(d_Z, e_Z);
  layer::i_conv2dDirectOptimizedColMajor(e_Z, outT_f16_0_f1);
  layer::YOLOv2TransformLayer_predict(outT_f16_0_f1, outputs_0_f1);
}

} // namespace ctarget
} // namespace internal
} // namespace coder

// End of code generation (callPredict.cpp)
