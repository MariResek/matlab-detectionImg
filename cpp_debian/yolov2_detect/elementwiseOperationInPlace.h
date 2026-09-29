//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// elementwiseOperationInPlace.h
//
// Code generation for function 'elementwiseOperationInPlace'
//

#ifndef ELEMENTWISEOPERATIONINPLACE_H
#define ELEMENTWISEOPERATIONINPLACE_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
namespace coder {
struct dlarray;

struct b_dlarray;

} // namespace coder

// Function Declarations
namespace coder {
namespace internal {
namespace layer {
void lambdaForColumnMajorGeneric(dlarray &X);

void lambdaForColumnMajorGeneric(b_dlarray &X);

void lambdaForColumnMajorGeneric(float X[80]);

} // namespace layer
} // namespace internal
} // namespace coder

#endif
// End of code generation (elementwiseOperationInPlace.h)
