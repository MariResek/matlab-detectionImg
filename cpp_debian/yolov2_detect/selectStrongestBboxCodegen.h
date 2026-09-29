//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// selectStrongestBboxCodegen.h
//
// Code generation for function 'selectStrongestBboxCodegen'
//

#ifndef SELECTSTRONGESTBBOXCODEGEN_H
#define SELECTSTRONGESTBBOXCODEGEN_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace vision {
namespace internal {
namespace detector {
int selectStrongestBboxCodegen(const double inputBbox_data[],
                               const int inputBbox_size[2],
                               const double varargin_1_data[],
                               bool isKept_data[]);

}
} // namespace internal
} // namespace vision
} // namespace coder

#endif
// End of code generation (selectStrongestBboxCodegen.h)
