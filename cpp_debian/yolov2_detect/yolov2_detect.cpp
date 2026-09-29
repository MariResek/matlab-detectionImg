//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// yolov2_detect.cpp
//
// Code generation for function 'yolov2_detect'
//

// Include files
#include "yolov2_detect.h"
#include "YOLOv2Network.h"
#include "categorical.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_data.h"
#include "yolov2_detect_initialize.h"
#include "coder_bounded_array.h"
#include <algorithm>
#include <cstring>

// Function Definitions
void yolov2_detect(const unsigned char in[2764800], double bboxes_data[],
                   int bboxes_size[2], float scores_data[], int scores_size[1],
                   unsigned int labels_data[], int labels_size[1])
{
  coder::categorical labelsCat;
  int loop_ub;
  if (!isInitialized_yolov2_detect) {
    yolov2_detect_initialize();
  }
  scores_size[0] = coder::vision::internal::codegen::YOLOv2Network_detect(
      in, bboxes_data, bboxes_size, scores_data, labelsCat);
  //  Convertir categorical a índices uint32 (compatible con C++)
  loop_ub = labelsCat.codes.size[0];
  labels_size[0] = labelsCat.codes.size[0];
  if (loop_ub - 1 >= 0) {
    std::copy(&labelsCat.codes.data[0], &labelsCat.codes.data[loop_ub],
              &labels_data[0]);
  }
}

// End of code generation (yolov2_detect.cpp)
