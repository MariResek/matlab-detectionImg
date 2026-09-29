//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// yolov2_detect.h
//
// Code generation for function 'yolov2_detect'
//

#ifndef YOLOV2_DETECT_H
#define YOLOV2_DETECT_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void yolov2_detect(const unsigned char in[2764800], double bboxes_data[],
                          int bboxes_size[2], float scores_data[],
                          int scores_size[1], unsigned int labels_data[],
                          int labels_size[1]);

#endif
// End of code generation (yolov2_detect.h)
