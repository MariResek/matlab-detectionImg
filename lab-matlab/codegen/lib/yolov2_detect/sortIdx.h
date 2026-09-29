//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sortIdx.h
//
// Code generation for function 'sortIdx'
//

#ifndef SORTIDX_H
#define SORTIDX_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
namespace internal {
void merge_block(int idx_data[], float x_data[], int offset, int n,
                 int preSortLevel, int iwork_data[], float xwork_data[]);

}
} // namespace coder

#endif
// End of code generation (sortIdx.h)
