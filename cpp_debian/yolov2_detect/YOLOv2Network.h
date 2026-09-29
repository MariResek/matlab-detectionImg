//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// YOLOv2Network.h
//
// Code generation for function 'YOLOv2Network'
//

#ifndef YOLOV2NETWORK_H
#define YOLOV2NETWORK_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
namespace coder {
class categorical;

}

// Function Declarations
namespace coder {
namespace vision {
namespace internal {
namespace codegen {
int YOLOv2Network_detect(const unsigned char b_I[2764800], double bboxes_data[],
                         int bboxes_size[2], float scores_data[],
                         categorical &varargout_1);

}
} // namespace internal
} // namespace vision
} // namespace coder

#endif
// End of code generation (YOLOv2Network.h)
