//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cellstr_unique.h
//
// Code generation for function 'cellstr_unique'
//

#ifndef CELLSTR_UNIQUE_H
#define CELLSTR_UNIQUE_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
struct cell_wrap_5;

// Function Declarations
namespace coder {
namespace matlab {
namespace internal {
namespace coder {
namespace datatypes {
int cellstr_unique(const cell_wrap_5 a_data[], int a_size,
                   cell_wrap_5 u_data[]);

int cellstr_unique(const cell_wrap_5 a_data[], int a_size, cell_wrap_5 u_data[],
                   double ia_data[], int &ia_size);

} // namespace datatypes
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder

#endif
// End of code generation (cellstr_unique.h)
