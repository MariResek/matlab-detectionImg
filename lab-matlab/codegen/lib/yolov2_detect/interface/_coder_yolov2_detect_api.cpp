//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_yolov2_detect_api.cpp
//
// Code generation for function 'yolov2_detect'
//

// Include files
#include "_coder_yolov2_detect_api.h"
#include "_coder_yolov2_detect_mex.h"

// Variable Definitions
emlrtCTX emlrtRootTLSGlobal{nullptr};

emlrtContext emlrtContextGlobal{
    true,                                                 // bFirstTime
    false,                                                // bInitialized
    131690U,                                              // fVersionInfo
    nullptr,                                              // fErrorFunction
    "yolov2_detect",                                      // fFunctionName
    nullptr,                                              // fRTCallStack
    false,                                                // bDebugMode
    {2045744189U, 2170104910U, 2743257031U, 4284093946U}, // fSigWrd
    nullptr                                               // fSigMem
};

// Function Declarations
static uint8_T (*b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[2764800];

static void emlrtExitTimeCleanupDtorFcn(const void *r);

static uint8_T (*emlrt_marshallIn(const emlrtStack &sp,
                                  const mxArray *b_nullptr,
                                  const char_T *identifier))[2764800];

static uint8_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                  const emlrtMsgIdentifier *parentId))[2764800];

static const mxArray *emlrt_marshallOut(real_T u_data[],
                                        const int32_T u_size[2]);

static const mxArray *emlrt_marshallOut(real32_T u_data[],
                                        const int32_T &u_size);

static const mxArray *emlrt_marshallOut(uint32_T u_data[],
                                        const int32_T &u_size);

// Function Definitions
static uint8_T (*b_emlrt_marshallIn(const emlrtStack &sp, const mxArray *src,
                                    const emlrtMsgIdentifier *msgId))[2764800]
{
  static const int32_T dims[3]{720, 1280, 3};
  int32_T iv[3];
  uint8_T(*ret)[2764800];
  boolean_T bv[3]{false, false, false};
  emlrtCheckVsBuiltInR2012b((emlrtConstCTX)&sp, msgId, src, "uint8", false, 3U,
                            (const void *)&dims[0], &bv[0], &iv[0]);
  ret = (uint8_T(*)[2764800])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static void emlrtExitTimeCleanupDtorFcn(const void *r)
{
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

static uint8_T (*emlrt_marshallIn(const emlrtStack &sp,
                                  const mxArray *b_nullptr,
                                  const char_T *identifier))[2764800]
{
  emlrtMsgIdentifier thisId;
  uint8_T(*y)[2764800];
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = emlrt_marshallIn(sp, emlrtAlias(b_nullptr), &thisId);
  emlrtDestroyArray(&b_nullptr);
  return y;
}

static uint8_T (*emlrt_marshallIn(const emlrtStack &sp, const mxArray *u,
                                  const emlrtMsgIdentifier *parentId))[2764800]
{
  uint8_T(*y)[2764800];
  y = b_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static const mxArray *emlrt_marshallOut(real_T u_data[],
                                        const int32_T u_size[2])
{
  static const int32_T iv[2]{0, 0};
  const mxArray *m;
  const mxArray *y;
  void *existingData;
  y = nullptr;
  m = emlrtCreateNumericArray(2, (const void *)&iv[0], mxDOUBLE_CLASS, mxREAL);
  existingData = emlrtMxGetData((mxArray *)m);
  if (existingData != (void *)&u_data[0]) {
    emlrtFreeMex(existingData);
  }
  emlrtMxSetData((mxArray *)m, &u_data[0]);
  emlrtSetDimensions((mxArray *)m, &u_size[0], 2);
  emlrtAssign(&y, m);
  return y;
}

static const mxArray *emlrt_marshallOut(real32_T u_data[],
                                        const int32_T &u_size)
{
  static const int32_T i{0};
  const mxArray *m;
  const mxArray *y;
  void *existingData;
  y = nullptr;
  m = emlrtCreateNumericArray(1, (const void *)&i, mxSINGLE_CLASS, mxREAL);
  existingData = emlrtMxGetData((mxArray *)m);
  if (existingData != (void *)&u_data[0]) {
    emlrtFreeMex(existingData);
  }
  emlrtMxSetData((mxArray *)m, &u_data[0]);
  emlrtSetDimensions((mxArray *)m, &u_size, 1);
  emlrtAssign(&y, m);
  return y;
}

static const mxArray *emlrt_marshallOut(uint32_T u_data[],
                                        const int32_T &u_size)
{
  static const int32_T i{0};
  const mxArray *m;
  const mxArray *y;
  void *existingData;
  y = nullptr;
  m = emlrtCreateNumericArray(1, (const void *)&i, mxUINT32_CLASS, mxREAL);
  existingData = emlrtMxGetData((mxArray *)m);
  if (existingData != (void *)&u_data[0]) {
    emlrtFreeMex(existingData);
  }
  emlrtMxSetData((mxArray *)m, &u_data[0]);
  emlrtSetDimensions((mxArray *)m, &u_size, 1);
  emlrtAssign(&y, m);
  return y;
}

void yolov2_detect_api(const mxArray *const prhs[2], int32_T nlhs,
                       const mxArray *plhs[3])
{
  static const uint32_T uv[4]{3309399318U, 1921326342U, 9999057U, 797956980U};
  static const char_T *s{"netMatFile"};
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  real_T(*bboxes_data)[3380];
  int32_T bboxes_size[2];
  int32_T labels_size;
  int32_T scores_size;
  real32_T(*scores_data)[845];
  uint32_T(*labels_data)[845];
  uint8_T(*in)[2764800];
  st.tls = emlrtRootTLSGlobal;
  bboxes_data = (real_T(*)[3380])mxMalloc(sizeof(real_T[3380]));
  scores_data = (real32_T(*)[845])mxMalloc(sizeof(real32_T[845]));
  labels_data = (uint32_T(*)[845])mxMalloc(sizeof(uint32_T[845]));
  // Check constant function inputs
  scores_size = 4;
  emlrtCheckArrayChecksumR2018b(&st, prhs[1], false, &scores_size,
                                (const char_T **)&s, &uv[0]);
  // Marshall function inputs
  in = emlrt_marshallIn(st, emlrtAlias(prhs[0]), "in");
  // Invoke the target function
  yolov2_detect(*in, *bboxes_data, bboxes_size, *scores_data, &scores_size,
                *labels_data, &labels_size);
  // Marshall function outputs
  plhs[0] = emlrt_marshallOut(*bboxes_data, bboxes_size);
  if (nlhs > 1) {
    plhs[1] = emlrt_marshallOut(*scores_data, scores_size);
  }
  if (nlhs > 2) {
    plhs[2] = emlrt_marshallOut(*labels_data, labels_size);
  }
}

void yolov2_detect_atexit()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtPushHeapReferenceStackR2021a(&st, false, nullptr,
                                    (void *)&emlrtExitTimeCleanupDtorFcn,
                                    nullptr, nullptr, nullptr);
  emlrtEnterRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  yolov2_detect_xil_terminate();
  yolov2_detect_xil_shutdown();
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void yolov2_detect_initialize()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

void yolov2_detect_terminate()
{
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

// End of code generation (_coder_yolov2_detect_api.cpp)
