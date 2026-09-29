//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// yolov2_detect_terminate.cpp
//
// Code generation for function 'yolov2_detect_terminate'
//

// Include files
#include "yolov2_detect_terminate.h"
#include "rt_nonfinite.h"
#include "yolov2_detect_data.h"
#include "omp.h"
#include <cstring>

// Function Definitions
void yolov2_detect_terminate()
{
  omp_destroy_nest_lock(&yolov2_detect_nestLockGlobal);
  isInitialized_yolov2_detect = false;
}

// End of code generation (yolov2_detect_terminate.cpp)
