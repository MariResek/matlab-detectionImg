//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_yolov2_detect_info.cpp
//
// Code generation for function 'yolov2_detect'
//

// Include files
#include "_coder_yolov2_detect_info.h"
#include "emlrt.h"
#include "tmwtypes.h"

// Function Declarations
static const mxArray *emlrtMexFcnResolvedFunctionsInfo();

// Function Definitions
static const mxArray *emlrtMexFcnResolvedFunctionsInfo()
{
  const mxArray *nameCaptureInfo;
  const char_T *data[7]{
      "789ced57cd6ed340105e5b499aa2b684038823278e55848444101cd2b8281109ad882b0a"
      "35aa2367690cf18f6c2734b7be013c02c7be03178e8813472e95405c"
      "780c9cc4e338238fec86d6858a9156bb9f6776bec9eee65b2d131a2d8131b6c6a6565a99"
      "f6ab80835e64f386fd028a13e6c3599ee5e6e681ff5dd06b96e9f143",
      "6f0acc8ec1c399ba690f3c796473d7070e77adfe9077279e577a9fcbbac1db51f0648c8c"
      "47115708c6aef1b8d6e3da9bf6c0604ecf9d55d88f82703d54e2f7e6"
      "12d6031b5e0f1c077cf6827c02fa4ef1817f64f5ade19dfd2ef7b8e645f8bf11f9afa17c"
      "38bf1054d8aa6db5a7b830c1aee7e8e6c12cff21911f3023fc05c48b",
      "f9c17f73f6e9440ce2a20deaf840f0a45de707441d25e4dfdb7c59bbafecb8dc7115c33f"
      "bd5cd932b9e4e843ae48963630b8e959aed2aacacdea8632b729eb06"
      "63c9e7f06aca7aa9755b65c549df6d9e0859f2fdb8fdf363967c6017c5479dfbb4e7ed3a"
      "c15742fedd664bbadba8bcad6d4acf0f1cede96bad5b9123756c27f0",
      "24d5c1089c557e95987f5e7a9c35df65d55f189f95fe3e24ea2821ffdece29f5579a88af"
      "e5ac1b9de8bda812f59c955ea819ebef154ffc92251fd8bfaabf3708"
      "be12f2571bd56d5baac865adfdd875eb354f975f34372e8ffe7e2ac6cf5f4e995f2dc6e7"
      "17919fa1774979293e6fda7709755ee05db28cfc80414fc142fd2ac4",
      "f39d5ea7e7f38cdf27cf9c8e6d73677f3fc25726f892f67509c551fb5a17e2e727edebaf"
      "84fc228acb05dfeef96dd76f477efbecb7af9136be033be628bcbb28"
      "c37b0ffcd030c6b5e17949e338be68dcf7085e437ec0c741bfa8eee1738ae3ebc88fdfe3"
      "80f3410ffa37603a3399e7ef8bc67aacc31c54f7d11fd69d4318c7df",
      "4aa81b2cd439e2bc26fd1f705e2a3ff50e4fab378cc0a0377f3bbf4ae45f49e04f5adf3c"
      "713fffdfcff3e52f2ea8ef90ef18613011f9f1bdfd9ec87b51f736d8"
      "6f42ecb321",
      ""};
  nameCaptureInfo = nullptr;
  emlrtNameCaptureMxArrayR2016a(&data[0], 5368U, &nameCaptureInfo);
  return nameCaptureInfo;
}

mxArray *emlrtMexFcnProperties()
{
  mxArray *xEntryPoints;
  mxArray *xInputs;
  mxArray *xResult;
  const char_T *epFieldName[7]{
      "QualifiedName",    "NumberOfInputs", "NumberOfOutputs", "ConstantInputs",
      "ResolvedFilePath", "TimeStamp",      "Visible"};
  const char_T *propFieldName[7]{
      "Version",      "ResolvedFunctions", "Checksum", "EntryPoints",
      "CoverageInfo", "IsPolymorphic",     "AuxData"};
  uint8_T v[216]{
      0U,   1U,   73U,  77U,  0U,   0U,   0U,   0U,   14U,  0U,   0U,   0U,
      200U, 0U,   0U,   0U,   6U,   0U,   0U,   0U,   8U,   0U,   0U,   0U,
      2U,   0U,   0U,   0U,   0U,   0U,   0U,   0U,   5U,   0U,   0U,   0U,
      8U,   0U,   0U,   0U,   1U,   0U,   0U,   0U,   1U,   0U,   0U,   0U,
      1U,   0U,   0U,   0U,   0U,   0U,   0U,   0U,   5U,   0U,   4U,   0U,
      17U,  0U,   0U,   0U,   1U,   0U,   0U,   0U,   17U,  0U,   0U,   0U,
      67U,  108U, 97U,  115U, 115U, 69U,  110U, 116U, 114U, 121U, 80U,  111U,
      105U, 110U, 116U, 115U, 0U,   0U,   0U,   0U,   0U,   0U,   0U,   0U,
      14U,  0U,   0U,   0U,   112U, 0U,   0U,   0U,   6U,   0U,   0U,   0U,
      8U,   0U,   0U,   0U,   2U,   0U,   0U,   0U,   0U,   0U,   0U,   0U,
      5U,   0U,   0U,   0U,   8U,   0U,   0U,   0U,   1U,   0U,   0U,   0U,
      0U,   0U,   0U,   0U,   1U,   0U,   0U,   0U,   0U,   0U,   0U,   0U,
      5U,   0U,   4U,   0U,   14U,  0U,   0U,   0U,   1U,   0U,   0U,   0U,
      56U,  0U,   0U,   0U,   81U,  117U, 97U,  108U, 105U, 102U, 105U, 101U,
      100U, 78U,  97U,  109U, 101U, 0U,   77U,  101U, 116U, 104U, 111U, 100U,
      115U, 0U,   0U,   0U,   0U,   0U,   0U,   0U,   80U,  114U, 111U, 112U,
      101U, 114U, 116U, 105U, 101U, 115U, 0U,   0U,   0U,   0U,   72U,  97U,
      110U, 100U, 108U, 101U, 0U,   0U,   0U,   0U,   0U,   0U,   0U,   0U};
  xEntryPoints =
      emlrtCreateStructMatrix(1, 1, 7, (const char_T **)&epFieldName[0]);
  xInputs = emlrtCreateLogicalMatrix(1, 2);
  emlrtSetField(xEntryPoints, 0, "QualifiedName",
                emlrtMxCreateString("yolov2_detect"));
  emlrtSetField(xEntryPoints, 0, "NumberOfInputs",
                emlrtMxCreateDoubleScalar(1.0));
  emlrtSetField(xEntryPoints, 0, "NumberOfOutputs",
                emlrtMxCreateDoubleScalar(3.0));
  emlrtSetField(xEntryPoints, 0, "ConstantInputs", xInputs);
  emlrtSetField(
      xEntryPoints, 0, "ResolvedFilePath",
      emlrtMxCreateString(
          "C:\\Users\\mrese\\OneDrive\\Documentos\\MATLAB\\yolov2_detect.m"));
  emlrtSetField(xEntryPoints, 0, "TimeStamp",
                emlrtMxCreateDoubleScalar(740249.7354398149));
  emlrtSetField(xEntryPoints, 0, "Visible", emlrtMxCreateLogicalScalar(true));
  xResult =
      emlrtCreateStructMatrix(1, 1, 7, (const char_T **)&propFieldName[0]);
  emlrtSetField(xResult, 0, "Version",
                emlrtMxCreateString("26.1.0.3346908 (R2026a) Update 5"));
  emlrtSetField(xResult, 0, "ResolvedFunctions",
                (mxArray *)emlrtMexFcnResolvedFunctionsInfo());
  emlrtSetField(xResult, 0, "Checksum",
                emlrtMxCreateString("AWO8XVWbPeLLcT3NprrwDC"));
  emlrtSetField(xResult, 0, "EntryPoints", xEntryPoints);
  emlrtSetField(xResult, 0, "AuxData",
                emlrtMxCreateRowVectorUINT8((const uint8_T *)&v, 216U));
  return xResult;
}

// End of code generation (_coder_yolov2_detect_info.cpp)
