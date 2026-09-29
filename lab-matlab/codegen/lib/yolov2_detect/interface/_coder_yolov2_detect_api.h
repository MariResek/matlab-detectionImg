//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_yolov2_detect_api.h
//
// Code generation for function 'yolov2_detect'
//

#ifndef _CODER_YOLOV2_DETECT_API_H
#define _CODER_YOLOV2_DETECT_API_H

// Include files
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <algorithm>
#include <cstring>

// Variable Declarations
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

// Function Declarations
void yolov2_detect(const uint8_T in[2764800], real_T bboxes_data[],
                   int32_T bboxes_size[2], real32_T scores_data[],
                   int32_T scores_size[1], uint32_T labels_data[],
                   int32_T labels_size[1]);

void yolov2_detect_api(const mxArray *const prhs[2], int32_T nlhs,
                       const mxArray *plhs[3]);

void yolov2_detect_atexit();

void yolov2_detect_initialize();

void yolov2_detect_terminate();

void yolov2_detect_xil_shutdown();

void yolov2_detect_xil_terminate();

#endif
// End of code generation (_coder_yolov2_detect_api.h)
