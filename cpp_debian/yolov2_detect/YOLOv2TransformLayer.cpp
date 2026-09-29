//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// YOLOv2TransformLayer.cpp
//
// Code generation for function 'YOLOv2TransformLayer'
//

// Include files
#include "YOLOv2TransformLayer.h"
#include "elementwiseOperationInPlace.h"
#include "internal_softmax.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_internal_types.h"
#include <algorithm>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
namespace layer {
void YOLOv2TransformLayer_predict(const float X_Data[71825],
                                  float Z_Data[71825])
{
  static float b_X_Data[67600];
  b_dlarray expWH;
  dlarray iouPredAndSigmaXY;
  std::copy(&X_Data[0], &X_Data[2535], &iouPredAndSigmaXY.Data[0]);
  lambdaForColumnMajorGeneric(iouPredAndSigmaXY);
  std::copy(&X_Data[2535], &X_Data[4225], &expWH.Data[0]);
  lambdaForColumnMajorGeneric(expWH);
  deep::internal::coder::dlarray::iComputeSoftmaxForCpu(&X_Data[4225],
                                                        b_X_Data);
  std::copy(&iouPredAndSigmaXY.Data[0], &iouPredAndSigmaXY.Data[2535],
            &Z_Data[0]);
  std::copy(&expWH.Data[0], &expWH.Data[1690], &Z_Data[2535]);
  std::copy(&b_X_Data[0], &b_X_Data[67600], &Z_Data[4225]);
}

} // namespace layer
} // namespace internal
} // namespace coder

// End of code generation (YOLOv2TransformLayer.cpp)
