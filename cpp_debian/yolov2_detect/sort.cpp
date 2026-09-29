//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sort.cpp
//
// Code generation for function 'sort'
//

// Include files
#include "sort.h"
#include "rt_nonfinite.h"
#include "sortIdx.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
int sort(float x_data[], const int &x_size, int idx_data[])
{
  float vwork_data[845];
  float xwork_data[845];
  int iidx_data[845];
  int iwork_data[845];
  int dim;
  int idx_size;
  int vstride;
  int vwork_size;
  dim = 2;
  if (x_size != 1) {
    dim = 1;
  }
  if (dim <= 1) {
    vwork_size = x_size;
  } else {
    vwork_size = 1;
  }
  idx_size = x_size;
  vstride = 1;
  dim -= 2;
  for (int k{0}; k <= dim; k++) {
    vstride *= x_size;
  }
  for (int j{0}; j < vstride; j++) {
    for (int k{0}; k < vwork_size; k++) {
      vwork_data[k] = x_data[j + k * vstride];
      iidx_data[k] = 0;
    }
    if (vwork_size != 0) {
      float x4[4];
      int bLen;
      int bLen2;
      int i1;
      int i2;
      int i4;
      int ib;
      int nNaNs;
      int wOffset;
      short idx4[4];
      x4[0] = 0.0F;
      idx4[0] = 0;
      x4[1] = 0.0F;
      idx4[1] = 0;
      x4[2] = 0.0F;
      idx4[2] = 0;
      x4[3] = 0.0F;
      idx4[3] = 0;
      nNaNs = 0;
      ib = 0;
      for (int k{0}; k < vwork_size; k++) {
        iwork_data[k] = 0;
        if (std::isnan(vwork_data[k])) {
          dim = (vwork_size - nNaNs) - 1;
          iidx_data[dim] = k + 1;
          xwork_data[dim] = vwork_data[k];
          nNaNs++;
        } else {
          ib++;
          idx4[ib - 1] = static_cast<short>(k + 1);
          x4[ib - 1] = vwork_data[k];
          if (ib == 4) {
            float f;
            float f1;
            dim = k - nNaNs;
            if (x4[0] >= x4[1]) {
              i1 = 1;
              i2 = 2;
            } else {
              i1 = 2;
              i2 = 1;
            }
            if (x4[2] >= x4[3]) {
              ib = 3;
              i4 = 4;
            } else {
              ib = 4;
              i4 = 3;
            }
            f = x4[i1 - 1];
            f1 = x4[ib - 1];
            if (f >= f1) {
              if (x4[i2 - 1] >= f1) {
                bLen = i1;
                bLen2 = i2;
                i1 = ib;
                i2 = i4;
              } else if (x4[i2 - 1] >= x4[i4 - 1]) {
                bLen = i1;
                bLen2 = ib;
                i1 = i2;
                i2 = i4;
              } else {
                bLen = i1;
                bLen2 = ib;
                i1 = i4;
              }
            } else if (f >= x4[i4 - 1]) {
              if (x4[i2 - 1] >= x4[i4 - 1]) {
                bLen = ib;
                bLen2 = i1;
                i1 = i2;
                i2 = i4;
              } else {
                bLen = ib;
                bLen2 = i1;
                i1 = i4;
              }
            } else {
              bLen = ib;
              bLen2 = i4;
            }
            iidx_data[dim - 3] = idx4[bLen - 1];
            iidx_data[dim - 2] = idx4[bLen2 - 1];
            iidx_data[dim - 1] = idx4[i1 - 1];
            iidx_data[dim] = idx4[i2 - 1];
            vwork_data[dim - 3] = x4[bLen - 1];
            vwork_data[dim - 2] = x4[bLen2 - 1];
            vwork_data[dim - 1] = x4[i1 - 1];
            vwork_data[dim] = x4[i2 - 1];
            ib = 0;
          }
        }
      }
      wOffset = vwork_size - nNaNs;
      if (ib > 0) {
        signed char perm[4];
        perm[1] = 0;
        perm[2] = 0;
        perm[3] = 0;
        if (ib == 1) {
          perm[0] = 1;
        } else if (ib == 2) {
          if (x4[0] >= x4[1]) {
            perm[0] = 1;
            perm[1] = 2;
          } else {
            perm[0] = 2;
            perm[1] = 1;
          }
        } else if (x4[0] >= x4[1]) {
          if (x4[1] >= x4[2]) {
            perm[0] = 1;
            perm[1] = 2;
            perm[2] = 3;
          } else if (x4[0] >= x4[2]) {
            perm[0] = 1;
            perm[1] = 3;
            perm[2] = 2;
          } else {
            perm[0] = 3;
            perm[1] = 1;
            perm[2] = 2;
          }
        } else if (x4[0] >= x4[2]) {
          perm[0] = 2;
          perm[1] = 1;
          perm[2] = 3;
        } else if (x4[1] >= x4[2]) {
          perm[0] = 2;
          perm[1] = 3;
          perm[2] = 1;
        } else {
          perm[0] = 3;
          perm[1] = 2;
          perm[2] = 1;
        }
        dim = static_cast<unsigned char>(ib);
        for (int k{0}; k < dim; k++) {
          i1 = (wOffset - ib) + k;
          i2 = perm[k];
          iidx_data[i1] = idx4[i2 - 1];
          vwork_data[i1] = x4[i2 - 1];
        }
      }
      dim = nNaNs >> 1;
      for (int k{0}; k < dim; k++) {
        i1 = wOffset + k;
        i2 = iidx_data[i1];
        ib = (vwork_size - k) - 1;
        iidx_data[i1] = iidx_data[ib];
        iidx_data[ib] = i2;
        vwork_data[i1] = xwork_data[ib];
        vwork_data[ib] = xwork_data[i1];
      }
      if ((static_cast<unsigned int>(nNaNs) & 1U) != 0U) {
        dim += wOffset;
        vwork_data[dim] = xwork_data[dim];
      }
      dim = 2;
      if (wOffset > 1) {
        if (vwork_size >= 256) {
          int nBlocks;
          nBlocks = wOffset >> 8;
          if (nBlocks > 0) {
            for (int b{0}; b < nBlocks; b++) {
              float xwork[256];
              short iwork[256];
              i4 = (b << 8) - 1;
              for (int b_b{0}; b_b < 6; b_b++) {
                int i;
                bLen = 1 << (b_b + 2);
                bLen2 = bLen << 1;
                i = 256 >> (b_b + 3);
                for (int b_k{0}; b_k < i; b_k++) {
                  i1 = (i4 + b_k * bLen2) + 1;
                  for (int k{0}; k < bLen2; k++) {
                    dim = i1 + k;
                    iwork[k] = static_cast<short>(iidx_data[dim]);
                    xwork[k] = vwork_data[dim];
                  }
                  i2 = 0;
                  ib = bLen;
                  dim = i1 - 1;
                  int exitg1;
                  do {
                    exitg1 = 0;
                    dim++;
                    if (xwork[i2] >= xwork[ib]) {
                      iidx_data[dim] = iwork[i2];
                      vwork_data[dim] = xwork[i2];
                      if (i2 + 1 < bLen) {
                        i2++;
                      } else {
                        exitg1 = 1;
                      }
                    } else {
                      iidx_data[dim] = iwork[ib];
                      vwork_data[dim] = xwork[ib];
                      if (ib + 1 < bLen2) {
                        ib++;
                      } else {
                        dim -= i2;
                        for (int k{i2 + 1}; k <= bLen; k++) {
                          ib = dim + k;
                          iidx_data[ib] = iwork[k - 1];
                          vwork_data[ib] = xwork[k - 1];
                        }
                        exitg1 = 1;
                      }
                    }
                  } while (exitg1 == 0);
                }
              }
            }
            dim = nBlocks << 8;
            i1 = wOffset - dim;
            if (i1 > 0) {
              merge_block(iidx_data, vwork_data, dim, i1, 2, iwork_data,
                          xwork_data);
            }
            dim = 8;
          }
        }
        merge_block(iidx_data, vwork_data, 0, wOffset, dim, iwork_data,
                    xwork_data);
      }
      if ((nNaNs > 0) && (wOffset > 0)) {
        for (int k{0}; k < nNaNs; k++) {
          dim = wOffset + k;
          xwork_data[k] = vwork_data[dim];
          iwork_data[k] = iidx_data[dim];
        }
        for (int k{wOffset}; k >= 1; k--) {
          dim = (nNaNs + k) - 1;
          vwork_data[dim] = vwork_data[k - 1];
          iidx_data[dim] = iidx_data[k - 1];
        }
        std::copy(&xwork_data[0], &xwork_data[nNaNs], &vwork_data[0]);
        std::copy(&iwork_data[0], &iwork_data[nNaNs], &iidx_data[0]);
      }
    }
    for (int k{0}; k < vwork_size; k++) {
      dim = j + k * vstride;
      x_data[dim] = vwork_data[k];
      idx_data[dim] = iidx_data[k];
    }
  }
  return idx_size;
}

} // namespace internal
} // namespace coder

// End of code generation (sort.cpp)
