//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// categorical.h
//
// Code generation for function 'categorical'
//

#ifndef CATEGORICAL_H
#define CATEGORICAL_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "coder_bounded_array.h"
#include <cstddef>
#include <cstdlib>

// Type Definitions
namespace coder {
class categorical {
public:
  bounded_array<unsigned int, 845U, 1U> codes;
};

class b_categorical {
protected:
  array<unsigned int, 1U> codes;
};

} // namespace coder

#endif
// End of code generation (categorical.h)
