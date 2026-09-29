//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cellstr_sort.h
//
// Code generation for function 'cellstr_sort'
//

#ifndef CELLSTR_SORT_H
#define CELLSTR_SORT_H

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
int cellstr_sort(const cell_wrap_5 c_data[], int c_size,
                 cell_wrap_5 sorted_data[], int idx_data[], int &idx_size);

bool cellstr_sort_anonFcn1(const cell_wrap_5 c_data[], int i, int j);

} // namespace datatypes
} // namespace coder
} // namespace internal
} // namespace matlab
} // namespace coder

#endif
// End of code generation (cellstr_sort.h)
