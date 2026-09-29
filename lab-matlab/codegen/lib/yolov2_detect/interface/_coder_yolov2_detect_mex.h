//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_yolov2_detect_mex.h
//
// Code generation for function 'yolov2_detect'
//

#ifndef _CODER_YOLOV2_DETECT_MEX_H
#define _CODER_YOLOV2_DETECT_MEX_H

// Include files
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"

// Function Declarations
MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[],
                                     int32_T nrhs, const mxArray *prhs[]);

emlrtCTX mexFunctionCreateRootTLS();

void unsafe_yolov2_detect_mexFunction(int32_T nlhs, mxArray *plhs[3],
                                      int32_T nrhs, const mxArray *prhs[2]);

#endif
// End of code generation (_coder_yolov2_detect_mex.h)
