//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// yolov2_detect_initialize.cpp
//
// Code generation for function 'yolov2_detect_initialize'
//

// Include files
#include "yolov2_detect_initialize.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_data.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void yolov2_detect_initialize()
{
  omp_init_nest_lock(&yolov2_detect_nestLockGlobal);
  isInitialized_yolov2_detect = true;
}

// End of code generation (yolov2_detect_initialize.cpp)
