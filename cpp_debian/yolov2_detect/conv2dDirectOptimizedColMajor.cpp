//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// conv2dDirectOptimizedColMajor.cpp
//
// Code generation for function 'conv2dDirectOptimizedColMajor'
//

// Include files
#include "conv2dDirectOptimizedColMajor.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function Declarations
static void b_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void b_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void c_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void c_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static char *computeFilePathUsingEnvVariable(const char *unresolvedFilePath);
static void convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool outputChannelTailCase,
  bool canInitializeOutputRegistersWithBiasValues, const float &biasBufferPtr);
static void d_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void d_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void e_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void e_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void f_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void f_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void g_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void g_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static char *getCustomUserDataPathEnvVar(const char *unresolvedFilePath);
static int getPositionOfLastFileSeparator(const char *filePath);
static char *getRelativePathToParentFolder(const char *filePath);
static char *getResolvedFilePath(const char *unresolvedFilePath);
static void h_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void h_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void i_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor);
static void i_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr);
static void readDnnConstants_real32_T(float &inputBufferPtr, const char
  *unresolvedFilePath, int numElementsToRead);
static char *resolveBinaryFilePath(const char *unresolvedFilePath);
static char *sanitizeFilePathForHSP(const char *unSanitizedFilePath);
static void stringConcat(char *destinationString, const char *sourceString,
  size_t destBufferSize);

// Function Definitions
static void b_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[705600];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 705600));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       692224; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 43264U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 43264;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 208U);
    fusedInputWidthAndHeightIdx %= 208;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 210) +
      inputChannelMiniBlockIdx * 44100) + 211] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 208) +
      inputChannelMiniBlockIdx * 43264];
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

  for (int fusedOutputWidthAndChannelBlockIdx = 0;
       fusedOutputWidthAndChannelBlockIdx < 416;
       fusedOutputWidthAndChannelBlockIdx++) {
    outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedOutputWidthAndChannelBlockIdx) / 208U);
    outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 208;
    outputBufferIdx = outputBufferIdx_tmp * 208 + outputChannelBlockIdx * 692224;
    outputBufferIdx_tmp *= 210;
    weightsIdx = outputChannelBlockIdx * 2304;
    for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
         inputChannelMiniblockIdx++) {
      c_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                          inputChannelMiniblockIdx * 44100], (&outputTensor)
                          [outputBufferIdx], (&weightsTensor)[weightsIdx +
                          inputChannelMiniblockIdx * 9],
                          inputChannelMiniblockIdx == 0,
                          inputChannelMiniblockIdx == 15, (&biasTensor)
                          [outputChannelBlockIdx << 4]);
    }
  }
}

static void b_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 416;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 173056];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 346112];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 519168];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 692224];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 865280];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1038336];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1211392];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1384448];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1557504];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1730560];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1903616];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2076672];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2249728];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2422784];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2595840];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 54];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 81];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 108];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 135];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 162];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 189];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 216];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 243];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 270];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 297];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 324];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 351];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 378];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 405];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 28];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 82];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 109];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 136];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 163];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 190];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 217];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 244];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 271];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 298];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 325];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 352];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 379];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 406];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 29];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 56];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 83];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 110];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 137];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 164];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 191];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 218];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 245];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 272];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 299];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 326];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 353];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 380];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 407];
      idxToStrideInputBufferAlongWidth += 418;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 173056] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 346112] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 519168] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 692224] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 865280] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1038336] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1211392] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1384448] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1557504] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1730560] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1903616] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2076672] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2249728] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2422784] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2595840] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 173056] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 346112] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 519168] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 692224] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 865280] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1038336] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1211392] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1384448] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1557504] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1730560] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1903616] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2076672] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2249728] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2422784] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2595840] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void c_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[359552];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 359552));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       346112; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 10816U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 10816;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 104U);
    fusedInputWidthAndHeightIdx %= 104;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 106) +
      inputChannelMiniBlockIdx * 11236) + 107] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 104) +
      inputChannelMiniBlockIdx * 10816];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 2;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 179776;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 416;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 104U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 104;
      outputBufferIdx = outputBufferIdx_tmp * 104 + outputChannelBlockIdx *
        173056;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 106 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 4608 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        d_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 11236], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 1) &&
                            (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void c_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 208;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 43264];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 86528];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 129792];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 173056];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 216320];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 259584];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 302848];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 346112];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 389376];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 432640];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 475904];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 519168];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 562432];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 605696];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 648960];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 144];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 288];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 432];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 576];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 720];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 864];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1008];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1152];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1296];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1440];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1584];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1728];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1872];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2016];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2160];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 145];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 289];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 433];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 577];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 721];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 865];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1009];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1153];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1297];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1441];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1585];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1729];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1873];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2017];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2161];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 146];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 290];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 434];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 578];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 722];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 866];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1010];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1154];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1298];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1442];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1586];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1730];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1874];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2018];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2162];
      idxToStrideInputBufferAlongWidth += 210;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 43264] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 86528] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 129792] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 173056] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 216320] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 259584] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 302848] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 346112] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 389376] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 432640] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 475904] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 519168] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 562432] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 605696] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 648960] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 43264] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 86528] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 129792] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 173056] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 216320] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 259584] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 302848] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 346112] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 389376] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 432640] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 475904] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 519168] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 562432] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 605696] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 648960] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static char *computeFilePathUsingEnvVariable(const char *unresolvedFilePath)
{
  char *resolvedFilePath;
  char *stringDuplicate;

#if defined(MW_RUNTIME_DL_DATA_PATH)

  extern char* mwGetRuntimeDLDataPath(const char*);
  resolvedFilePath = mwGetRuntimeDLDataPath(const_cast<char *>
    (unresolvedFilePath));

#elif defined(MW_DL_DATA_PATH)

  resolvedFilePath = resolveBinaryFilePath(unresolvedFilePath);

#else

  char *coderDataPath;
  coderDataPath = getenv("CODER_DATA_PATH");
  if (coderDataPath != NULL) {
    resolvedFilePath = resolveBinaryFilePath(unresolvedFilePath);
  } else {
    size_t filePathLen;
    size_t sizeOfChar;
    filePathLen = strlen(unresolvedFilePath) + 1;
    sizeOfChar = 1;
    stringDuplicate = static_cast<char *>(calloc(filePathLen, sizeOfChar));
    stringConcat(stringDuplicate, unresolvedFilePath, filePathLen);
    resolvedFilePath = stringDuplicate;
  }

#endif

  return resolvedFilePath;
}

static void convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[2795584];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 2795584));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       519168; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 173056U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 173056;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 416U);
    fusedInputWidthAndHeightIdx %= 416;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 418) +
      inputChannelMiniBlockIdx * 174724) + 419] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 416) +
      inputChannelMiniBlockIdx * 173056];
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputBufferIdx_tmp,outputBufferIdx)

  for (int fusedOutputWidthAndChannelBlockIdx = 0;
       fusedOutputWidthAndChannelBlockIdx < 416;
       fusedOutputWidthAndChannelBlockIdx++) {
    outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 416;
    outputBufferIdx = outputBufferIdx_tmp * 416;
    outputBufferIdx_tmp *= 418;
    b_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp],
                        (&outputTensor)[outputBufferIdx], (&weightsTensor)[0],
                        true, false, (&biasTensor)[0]);
    b_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp + 174724],
                        (&outputTensor)[outputBufferIdx], (&weightsTensor)[9],
                        false, false, (&biasTensor)[0]);
    b_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp + 349448],
                        (&outputTensor)[outputBufferIdx], (&weightsTensor)[18],
                        false, true, (&biasTensor)[0]);
  }
}

static void convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool outputChannelTailCase,
  bool canInitializeOutputRegistersWithBiasValues, const float &biasBufferPtr)
{
  if (!outputChannelTailCase) {
    for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 13;
         outputHeightBlockIdx++) {
      float b_outputRegister;
      float c_outputRegister;
      float d_outputRegister;
      float e_outputRegister;
      float f_outputRegister;
      float g_outputRegister;
      float h_outputRegister;
      float i_outputRegister;
      float inputRegister0_0;
      float j_outputRegister;
      float k_outputRegister;
      float l_outputRegister;
      float m_outputRegister;
      float n_outputRegister;
      float o_outputRegister;
      float outputRegister;
      float p_outputRegister;
      if (canInitializeOutputRegistersWithBiasValues) {
        outputRegister = (&biasBufferPtr)[0];
        b_outputRegister = (&biasBufferPtr)[1];
        c_outputRegister = (&biasBufferPtr)[2];
        d_outputRegister = (&biasBufferPtr)[3];
        e_outputRegister = (&biasBufferPtr)[4];
        f_outputRegister = (&biasBufferPtr)[5];
        g_outputRegister = (&biasBufferPtr)[6];
        h_outputRegister = (&biasBufferPtr)[7];
        i_outputRegister = (&biasBufferPtr)[8];
        j_outputRegister = (&biasBufferPtr)[9];
        k_outputRegister = (&biasBufferPtr)[10];
        l_outputRegister = (&biasBufferPtr)[11];
        m_outputRegister = (&biasBufferPtr)[12];
        n_outputRegister = (&biasBufferPtr)[13];
        o_outputRegister = (&biasBufferPtr)[14];
        p_outputRegister = (&biasBufferPtr)[15];
      } else {
        outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
        b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 169];
        c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 338];
        d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 507];
        e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
        f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 845];
        g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1014];
        h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1183];
        i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
        j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1521];
        k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1690];
        l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1859];
        m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2028];
        n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2197];
        o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2366];
        p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2535];
      }

      inputRegister0_0 = (&inputBufferPtr)[outputHeightBlockIdx];
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[0];
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = b_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[512];
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = c_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[1024];
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = d_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[1536];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[2048];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[2560];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[3072];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[3584];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[4096];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[4608];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[5120];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[5632];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[6144];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[6656];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[7168];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[7680];
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = p_outputRegister;
    }
  } else {
    for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 13;
         outputHeightBlockIdx++) {
      float b_outputRegister;
      float c_outputRegister;
      float d_outputRegister;
      float e_outputRegister;
      float f_outputRegister;
      float g_outputRegister;
      float h_outputRegister;
      float i_outputRegister;
      float inputRegister0_0;
      float outputRegister;
      if (canInitializeOutputRegistersWithBiasValues) {
        outputRegister = (&biasBufferPtr)[0];
        b_outputRegister = (&biasBufferPtr)[1];
        c_outputRegister = (&biasBufferPtr)[2];
        d_outputRegister = (&biasBufferPtr)[3];
        e_outputRegister = (&biasBufferPtr)[4];
        f_outputRegister = (&biasBufferPtr)[5];
        g_outputRegister = (&biasBufferPtr)[6];
        h_outputRegister = (&biasBufferPtr)[7];
        i_outputRegister = (&biasBufferPtr)[8];
      } else {
        outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
        b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 169];
        c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 338];
        d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 507];
        e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
        f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 845];
        g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1014];
        h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1183];
        i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
      }

      inputRegister0_0 = (&inputBufferPtr)[outputHeightBlockIdx];
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[0];
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = b_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[512];
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = c_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[1024];
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = d_outputRegister +
        inputRegister0_0 * (&weightsBufferPtr)[1536];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[2048];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[2560];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[3072];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[3584];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[4096];
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = i_outputRegister;
    }
  }
}

static void d_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[186624];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 186624));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       173056; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 2704U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 2704;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 52U);
    fusedInputWidthAndHeightIdx %= 52;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 54) +
      inputChannelMiniBlockIdx * 2916) + 55] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 52) +
      inputChannelMiniBlockIdx * 2704];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 4;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 46656;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 416;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 52U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 52;
      outputBufferIdx = outputBufferIdx_tmp * 52 + outputChannelBlockIdx * 43264;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 54 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 9216 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        e_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 2916], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 3) &&
                            (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void d_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 104;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 10816];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 21632];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 32448];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 43264];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 54080];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 64896];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 75712];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 86528];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 97344];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 108160];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 118976];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 129792];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 140608];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 151424];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 162240];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 288];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 576];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 864];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1152];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1440];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1728];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2016];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2304];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2592];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2880];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3168];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3456];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3744];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4032];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4320];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 289];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 577];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 865];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1153];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1441];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1729];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2017];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2305];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2593];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2881];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3169];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3457];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3745];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4033];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4321];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 290];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 578];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 866];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1154];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1442];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1730];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2018];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2306];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2594];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2882];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3170];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3458];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3746];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4034];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4322];
      idxToStrideInputBufferAlongWidth += 106;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 10816] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 21632] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 32448] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 43264] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 54080] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 64896] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 75712] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 86528] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 97344] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 108160] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 118976] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 129792] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 140608] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 151424] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 162240] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 10816] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 21632] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 32448] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 43264] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 54080] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 64896] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 75712] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 86528] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 97344] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 108160] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 118976] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 129792] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 140608] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 151424] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 162240] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void e_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[100352];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 100352));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       86528; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 676U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 676;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 26U);
    fusedInputWidthAndHeightIdx %= 26;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 28) +
      inputChannelMiniBlockIdx * 784) + 29] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 26) +
      inputChannelMiniBlockIdx * 676];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 8;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 12544;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 416;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 26U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 26;
      outputBufferIdx = outputBufferIdx_tmp * 26 + outputChannelBlockIdx * 10816;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 28 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 18432 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        f_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 784], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 7) &&
                            (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void e_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 52;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2704];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 5408];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 8112];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 10816];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 13520];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 16224];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 18928];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 21632];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 24336];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 27040];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 29744];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 32448];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 35152];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 37856];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 40560];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 576];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1152];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1728];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2304];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2880];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3456];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4032];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4608];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5184];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5760];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6336];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6912];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 7488];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8064];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8640];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 577];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1153];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1729];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2305];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2881];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3457];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4033];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4609];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5185];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5761];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6337];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6913];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 7489];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8065];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8641];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 578];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1154];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1730];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2306];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2882];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3458];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4034];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4610];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5186];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5762];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6338];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6914];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 7490];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8066];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8642];
      idxToStrideInputBufferAlongWidth += 54;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2704] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 5408] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 8112] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 10816] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 13520] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 16224] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 18928] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 21632] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 24336] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 27040] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 29744] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 32448] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 35152] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 37856] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 40560] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2704] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 5408] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 8112] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 10816] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 13520] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 16224] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 18928] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 21632] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 24336] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 27040] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 29744] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 32448] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 35152] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 37856] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 40560] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void f_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[57600];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 57600));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       43264; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 169U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 169;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 13U);
    fusedInputWidthAndHeightIdx %= 13;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 15) +
      inputChannelMiniBlockIdx * 225) + 16] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 13) +
      inputChannelMiniBlockIdx * 169];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 16;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 3600;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 416;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 13U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 13;
      outputBufferIdx = outputBufferIdx_tmp * 13 + outputChannelBlockIdx * 2704;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 15 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 36864 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        g_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 225], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 15)
                            && (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void f_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 26;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2028];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2704];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 3380];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 4056];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 4732];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 5408];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 6084];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 6760];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 7436];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 8112];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 8788];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 9464];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 10140];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1152];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2304];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3456];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4608];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5760];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6912];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8064];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9216];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 10368];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11520];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 12672];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13824];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 14976];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16128];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 17280];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1153];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2305];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3457];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4609];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5761];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6913];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8065];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9217];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 10369];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11521];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 12673];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13825];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 14977];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16129];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 17281];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1154];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2306];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 3458];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4610];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 5762];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6914];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 8066];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9218];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 10370];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11522];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 12674];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13826];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 14978];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16130];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 17282];
      idxToStrideInputBufferAlongWidth += 28;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2704] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 3380] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 4056] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 4732] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 5408] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 6084] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 6760] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 7436] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 8112] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 8788] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 9464] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 10140] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2704] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 3380] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 4056] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 4732] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 5408] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 6084] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 6760] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 7436] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 8112] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 8788] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 9464] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 10140] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void g_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[115200];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 115200));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       86528; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 169U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 169;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 13U);
    fusedInputWidthAndHeightIdx %= 13;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 15) +
      inputChannelMiniBlockIdx * 225) + 16] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 13) +
      inputChannelMiniBlockIdx * 169];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 32;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 3600;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 832;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 13U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 13;
      outputBufferIdx = outputBufferIdx_tmp * 13 + outputChannelBlockIdx * 2704;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 15 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 73728 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        h_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 225], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 31)
                            && (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void g_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 13;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 169];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 338];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 507];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 845];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1014];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1183];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1521];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1690];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1859];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2028];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2197];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2366];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2535];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2304];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4608];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6912];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9216];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11520];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13824];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16128];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18432];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 20736];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23040];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 25344];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27648];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 29952];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32256];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 34560];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2305];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4609];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6913];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9217];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11521];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13825];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16129];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18433];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 20737];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23041];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 25345];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27649];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 29953];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32257];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 34561];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2306];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4610];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 6914];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9218];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 11522];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13826];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 16130];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18434];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 20738];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23042];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 25346];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27650];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 29954];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32258];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 34562];
      idxToStrideInputBufferAlongWidth += 15;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static char *getCustomUserDataPathEnvVar(const char *unresolvedFilePath)
{
  const char *fileName;
  char *coderDataPath;
  char *resolvedFilePath;
  coderDataPath = getenv("CODER_DATA_PATH");
  if (coderDataPath != NULL) {
    int posOfLastPathSeparator;
    size_t filePathLength;
    size_t sizeOfChar;
    posOfLastPathSeparator = getPositionOfLastFileSeparator(unresolvedFilePath);
    fileName = &unresolvedFilePath[posOfLastPathSeparator];
    filePathLength = (strlen(coderDataPath) + strlen(fileName)) + 1;
    sizeOfChar = 1;
    resolvedFilePath = static_cast<char *>(calloc(filePathLength, sizeOfChar));
    stringConcat(resolvedFilePath, coderDataPath, filePathLength);
    stringConcat(resolvedFilePath, fileName, filePathLength);
  } else {
    resolvedFilePath = NULL;
  }

  return resolvedFilePath;
}

static int getPositionOfLastFileSeparator(const char *filePath)
{
  int lastPathSeparatorUnix;
  int posOfLastPathSeparator;
  const char *ptrToLastPathSeparator;
  lastPathSeparatorUnix = '/';
  ptrToLastPathSeparator = strrchr(filePath, lastPathSeparatorUnix);
  if (ptrToLastPathSeparator != NULL) {
    posOfLastPathSeparator = (int)(ptrToLastPathSeparator - filePath);
  } else {
    int lastPathSeparatorWindows;
    lastPathSeparatorWindows = '\\';
    ptrToLastPathSeparator = strrchr(filePath, lastPathSeparatorWindows);
    if (ptrToLastPathSeparator != NULL) {
      posOfLastPathSeparator = (int)(ptrToLastPathSeparator - filePath);
    } else {
      posOfLastPathSeparator = -1;
    }
  }

  return posOfLastPathSeparator;
}

static char *getRelativePathToParentFolder(const char *filePath)
{
  int posOfLastPathSeparator;
  const char *fileName;
  const char *parentDir;
  char *resolvedFilePath;
  size_t filePathLength;
  size_t sizeOfChar;
  parentDir = "..";
  posOfLastPathSeparator = getPositionOfLastFileSeparator(filePath);
  fileName = &filePath[posOfLastPathSeparator];
  filePathLength = (strlen(parentDir) + strlen(fileName)) + 1;
  sizeOfChar = 1;
  resolvedFilePath = static_cast<char *>(calloc(filePathLength, sizeOfChar));
  stringConcat(resolvedFilePath, parentDir, filePathLength);
  stringConcat(resolvedFilePath, fileName, filePathLength);
  return resolvedFilePath;
}

static char *getResolvedFilePath(const char *unresolvedFilePath)
{
  const char *fileOpenMode;
  char *computedPathUsingEnvVars;
  char *pathUsingEnvVarAndSanitizedPath;
  char *relativePathToParent;
  char *resolvedFilePath;
  char *sanitizedFilePath;
  char *stringDuplicate;
  FILE* filePtr;
  resolvedFilePath = NULL;
  fileOpenMode = "rb";
  filePtr = fopen(unresolvedFilePath, fileOpenMode);
  if (filePtr) {
    size_t filePathLen;
    size_t sizeOfChar;
    filePathLen = strlen(unresolvedFilePath) + 1;
    sizeOfChar = 1;
    stringDuplicate = static_cast<char *>(calloc(filePathLen, sizeOfChar));
    stringConcat(stringDuplicate, unresolvedFilePath, filePathLen);
    resolvedFilePath = stringDuplicate;
    fclose(filePtr);
  } else {
    computedPathUsingEnvVars = computeFilePathUsingEnvVariable
      (unresolvedFilePath);
    filePtr = fopen(computedPathUsingEnvVars, fileOpenMode);
    if (filePtr) {
      resolvedFilePath = computedPathUsingEnvVars;
      fclose(filePtr);
    } else {
      std::free(computedPathUsingEnvVars);
      sanitizedFilePath = sanitizeFilePathForHSP(unresolvedFilePath);
      filePtr = fopen(sanitizedFilePath, fileOpenMode);
      if (filePtr) {
        resolvedFilePath = sanitizedFilePath;
        fclose(filePtr);
      } else {
        relativePathToParent = getRelativePathToParentFolder(unresolvedFilePath);
        filePtr = fopen(relativePathToParent, fileOpenMode);
        if (filePtr) {
          resolvedFilePath = relativePathToParent;
          fclose(filePtr);
        } else {
          std::free(relativePathToParent);
          pathUsingEnvVarAndSanitizedPath = computeFilePathUsingEnvVariable
            (sanitizedFilePath);
          filePtr = fopen(pathUsingEnvVarAndSanitizedPath, fileOpenMode);
          if (filePtr) {
            resolvedFilePath = pathUsingEnvVarAndSanitizedPath;
            fclose(filePtr);
          } else {
            std::free(pathUsingEnvVarAndSanitizedPath);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }

  return resolvedFilePath;
}

static void h_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  static float inputScratchpadBuffer[230400];
  static bool bufferInitialized;
  int fusedInputWidthAndHeightIdx;
  int inputChannelMiniBlockIdx;
  int inputChannelMiniblockIdx;
  int inputWidthIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  if (!bufferInitialized) {
    std::memset(&inputScratchpadBuffer[0], 0, static_cast<unsigned int>(
      static_cast<int>(sizeof(float)) * 230400));
    bufferInitialized = true;
  }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(inputChannelMiniBlockIdx,fusedInputWidthAndHeightIdx,inputWidthIdx)

  for (int fusedInputChannelMiniBlockIdx = 0; fusedInputChannelMiniBlockIdx <
       173056; fusedInputChannelMiniBlockIdx++) {
    inputChannelMiniBlockIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputChannelMiniBlockIdx) / 169U);
    fusedInputWidthAndHeightIdx = fusedInputChannelMiniBlockIdx % 169;
    inputWidthIdx = static_cast<int>(static_cast<unsigned int>
      (fusedInputWidthAndHeightIdx) / 13U);
    fusedInputWidthAndHeightIdx %= 13;
    inputScratchpadBuffer[((fusedInputWidthAndHeightIdx + inputWidthIdx * 15) +
      inputChannelMiniBlockIdx * 225) + 16] = (&inputTensor)
      [(fusedInputWidthAndHeightIdx + inputWidthIdx * 13) +
      inputChannelMiniBlockIdx * 169];
  }

  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 64;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx * 144;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 3600;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 416;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 13U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 13;
      outputBufferIdx = outputBufferIdx_tmp * 13 + outputChannelBlockIdx * 2704;
      outputBufferIdx_tmp = outputBufferIdx_tmp * 15 +
        inputScratchpadBufferBaseIdx;
      weightsIdx = outputChannelBlockIdx * 147456 + inputChannelBlockIdxOffset;
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        i_convolutionKernel(inputScratchpadBuffer[outputBufferIdx_tmp +
                            inputChannelMiniblockIdx * 225], (&outputTensor)
                            [outputBufferIdx], (&weightsTensor)[weightsIdx +
                            inputChannelMiniblockIdx * 9], (inputChannelBlockIdx
          == 0) && (inputChannelMiniblockIdx == 0), (inputChannelBlockIdx == 63)
                            && (inputChannelMiniblockIdx == 15), (&biasTensor)
                            [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void h_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 13;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 169];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 338];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 507];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 845];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1014];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1183];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1521];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1690];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1859];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2028];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2197];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2366];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2535];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4608];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9216];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13824];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18432];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23040];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27648];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32256];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36864];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 41472];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46080];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 50688];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55296];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 59904];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64512];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 69120];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4609];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9217];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13825];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18433];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23041];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27649];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32257];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36865];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 41473];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46081];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 50689];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55297];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 59905];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64513];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 69121];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 4610];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9218];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 13826];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18434];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 23042];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27650];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 32258];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36866];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 41474];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46082];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 50690];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55298];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 59906];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64514];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 69122];
      idxToStrideInputBufferAlongWidth += 15;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void i_convolution(const float &inputTensor, float &outputTensor, const
  float &weightsTensor, const float &biasTensor)
{
  int inputChannelMiniblockIdx;
  int outputBufferIdx;
  int outputBufferIdx_tmp;
  int outputChannelBlockIdx;
  int weightsIdx;
  bool outputChannelTailCase;
  for (int inputChannelBlockIdx{0}; inputChannelBlockIdx < 32;
       inputChannelBlockIdx++) {
    int inputChannelBlockIdxOffset;
    int inputScratchpadBufferBaseIdx;
    inputChannelBlockIdxOffset = inputChannelBlockIdx << 4;
    inputScratchpadBufferBaseIdx = inputChannelBlockIdx * 2704;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(outputChannelBlockIdx,outputBufferIdx_tmp,outputBufferIdx,weightsIdx,outputChannelTailCase,inputChannelMiniblockIdx)

    for (int fusedOutputWidthAndChannelBlockIdx = 0;
         fusedOutputWidthAndChannelBlockIdx < 351;
         fusedOutputWidthAndChannelBlockIdx++) {
      outputChannelBlockIdx = static_cast<int>(static_cast<unsigned int>
        (fusedOutputWidthAndChannelBlockIdx) / 13U);
      outputBufferIdx_tmp = fusedOutputWidthAndChannelBlockIdx % 13 * 13;
      outputBufferIdx = outputBufferIdx_tmp + outputChannelBlockIdx * 2704;
      weightsIdx = (outputChannelBlockIdx << 13) + inputChannelBlockIdxOffset;
      outputBufferIdx_tmp += inputScratchpadBufferBaseIdx;
      outputChannelTailCase = (outputChannelBlockIdx == 26);
      for (inputChannelMiniblockIdx = 0; inputChannelMiniblockIdx < 16;
           inputChannelMiniblockIdx++) {
        convolutionKernel((&inputTensor)[outputBufferIdx_tmp +
                          inputChannelMiniblockIdx * 169], (&outputTensor)
                          [outputBufferIdx], (&weightsTensor)[weightsIdx +
                          inputChannelMiniblockIdx], outputChannelTailCase,
                          (inputChannelBlockIdx == 0) &&
                          (inputChannelMiniblockIdx == 0), (&biasTensor)
                          [outputChannelBlockIdx << 4]);
      }
    }
  }
}

static void i_convolutionKernel(const float &inputBufferPtr, float
  &outputBufferPtr, const float &weightsBufferPtr, bool
  canInitializeOutputRegistersWithBiasValues, bool canApplyActivationOperation,
  const float &biasBufferPtr)
{
  int idxToStrideInputBufferAlongHeight;
  idxToStrideInputBufferAlongHeight = 0;
  for (int outputHeightBlockIdx{0}; outputHeightBlockIdx < 13;
       outputHeightBlockIdx++) {
    float b_outputRegister;
    float c_outputRegister;
    float d_outputRegister;
    float e_outputRegister;
    float f_outputRegister;
    float g_outputRegister;
    float h_outputRegister;
    float i_outputRegister;
    float j_outputRegister;
    float k_outputRegister;
    float l_outputRegister;
    float m_outputRegister;
    float n_outputRegister;
    float o_outputRegister;
    float outputRegister;
    float p_outputRegister;
    int idxToStrideInputBufferAlongWidth;
    if (canInitializeOutputRegistersWithBiasValues) {
      outputRegister = (&biasBufferPtr)[0];
      b_outputRegister = (&biasBufferPtr)[1];
      c_outputRegister = (&biasBufferPtr)[2];
      d_outputRegister = (&biasBufferPtr)[3];
      e_outputRegister = (&biasBufferPtr)[4];
      f_outputRegister = (&biasBufferPtr)[5];
      g_outputRegister = (&biasBufferPtr)[6];
      h_outputRegister = (&biasBufferPtr)[7];
      i_outputRegister = (&biasBufferPtr)[8];
      j_outputRegister = (&biasBufferPtr)[9];
      k_outputRegister = (&biasBufferPtr)[10];
      l_outputRegister = (&biasBufferPtr)[11];
      m_outputRegister = (&biasBufferPtr)[12];
      n_outputRegister = (&biasBufferPtr)[13];
      o_outputRegister = (&biasBufferPtr)[14];
      p_outputRegister = (&biasBufferPtr)[15];
    } else {
      outputRegister = (&outputBufferPtr)[outputHeightBlockIdx];
      b_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 169];
      c_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 338];
      d_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 507];
      e_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 676];
      f_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 845];
      g_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1014];
      h_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1183];
      i_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1352];
      j_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1521];
      k_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1690];
      l_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 1859];
      m_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2028];
      n_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2197];
      o_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2366];
      p_outputRegister = (&outputBufferPtr)[outputHeightBlockIdx + 2535];
    }

    idxToStrideInputBufferAlongWidth = idxToStrideInputBufferAlongHeight;
    for (int kernelWidthIdx{0}; kernelWidthIdx < 3; kernelWidthIdx++) {
      float inputRegister0_0;
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9216];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18432];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27648];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36864];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46080];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55296];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64512];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 73728];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 82944];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 92160];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 101376];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 110592];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 119808];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 129024];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 138240];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 1];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 1];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9217];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18433];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27649];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36865];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46081];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55297];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64513];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 73729];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 82945];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 92161];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 101377];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 110593];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 119809];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 129025];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 138241];
      inputRegister0_0 = (&inputBufferPtr)[idxToStrideInputBufferAlongWidth + 2];
      outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 2];
      b_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 9218];
      c_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 18434];
      d_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 27650];
      e_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 36866];
      f_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 46082];
      g_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 55298];
      h_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 64514];
      i_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 73730];
      j_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 82946];
      k_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 92162];
      l_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 101378];
      m_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 110594];
      n_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 119810];
      o_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 129026];
      p_outputRegister += inputRegister0_0 * (&weightsBufferPtr)[kernelWidthIdx *
        3 + 138242];
      idxToStrideInputBufferAlongWidth += 15;
    }

    if (canApplyActivationOperation) {
      (&outputBufferPtr)[outputHeightBlockIdx] = std::fmax(outputRegister, 0.0F)
        + 0.1F * std::fmin(outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = std::fmax
        (b_outputRegister, 0.0F) + 0.1F * std::fmin(b_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = std::fmax
        (c_outputRegister, 0.0F) + 0.1F * std::fmin(c_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = std::fmax
        (d_outputRegister, 0.0F) + 0.1F * std::fmin(d_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = std::fmax
        (e_outputRegister, 0.0F) + 0.1F * std::fmin(e_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = std::fmax
        (f_outputRegister, 0.0F) + 0.1F * std::fmin(f_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = std::fmax
        (g_outputRegister, 0.0F) + 0.1F * std::fmin(g_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = std::fmax
        (h_outputRegister, 0.0F) + 0.1F * std::fmin(h_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = std::fmax
        (i_outputRegister, 0.0F) + 0.1F * std::fmin(i_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = std::fmax
        (j_outputRegister, 0.0F) + 0.1F * std::fmin(j_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = std::fmax
        (k_outputRegister, 0.0F) + 0.1F * std::fmin(k_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = std::fmax
        (l_outputRegister, 0.0F) + 0.1F * std::fmin(l_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = std::fmax
        (m_outputRegister, 0.0F) + 0.1F * std::fmin(m_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = std::fmax
        (n_outputRegister, 0.0F) + 0.1F * std::fmin(n_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = std::fmax
        (o_outputRegister, 0.0F) + 0.1F * std::fmin(o_outputRegister, 0.0F);
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = std::fmax
        (p_outputRegister, 0.0F) + 0.1F * std::fmin(p_outputRegister, 0.0F);
    } else {
      (&outputBufferPtr)[outputHeightBlockIdx] = outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 169] = b_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 338] = c_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 507] = d_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 676] = e_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 845] = f_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1014] = g_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1183] = h_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1352] = i_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1521] = j_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1690] = k_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 1859] = l_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2028] = m_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2197] = n_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2366] = o_outputRegister;
      (&outputBufferPtr)[outputHeightBlockIdx + 2535] = p_outputRegister;
    }

    idxToStrideInputBufferAlongHeight++;
  }
}

static void readDnnConstants_real32_T(float &inputBufferPtr, const char
  *unresolvedFilePath, int numElementsToRead)
{
  int elementSizeInBytes;
  const char *fileOpenMode;
  char *resolvedFilePath;
  FILE* filePtr;
  void *dataBufferPtr;
  resolvedFilePath = getResolvedFilePath(unresolvedFilePath);
  fileOpenMode = "rb";
  filePtr = fopen(resolvedFilePath, fileOpenMode);
  dataBufferPtr = &(&inputBufferPtr)[0];
  elementSizeInBytes = 4;
  fread(dataBufferPtr, elementSizeInBytes, numElementsToRead, filePtr);
  fclose(filePtr);
  std::free(resolvedFilePath);
}

static char *resolveBinaryFilePath(const char *unresolvedFilePath)
{
  const char *filePathAfterSlicingRelativePathOp;
  const char *lastDirName;
  const char *lastPathSeparator;
  const char *leadingPathSeparatorUnixAndWindows;
  const char *mwDLDataPath;
  char *codegenDir;
  char *coderDataPath;
  char *resolvedFilePath;
  char *updatedStartDir;
  size_t sizeOfChar;

#define XSTR(x)                        #x
#define STR(x)                         XSTR(x)

  coderDataPath = getenv("CODER_DATA_PATH");
  sizeOfChar = 1;
  if (coderDataPath != NULL) {
    resolvedFilePath = getCustomUserDataPathEnvVar(unresolvedFilePath);
  } else {
    int lastPathSeparatorWindows;
    size_t codegenDirLength;
    size_t posOfCodegenDir;
    size_t posOfLeadingPathSeparator;
    mwDLDataPath = STR(MW_DL_DATA_PATH);
    filePathAfterSlicingRelativePathOp = &unresolvedFilePath[2];
    leadingPathSeparatorUnixAndWindows = "/\\";
    posOfLeadingPathSeparator = strcspn(filePathAfterSlicingRelativePathOp,
      leadingPathSeparatorUnixAndWindows);
    codegenDirLength = posOfLeadingPathSeparator + 1;
    codegenDir = static_cast<char *>(calloc(codegenDirLength, sizeOfChar));
    strncpy(codegenDir, filePathAfterSlicingRelativePathOp,
            posOfLeadingPathSeparator);
    lastPathSeparatorWindows = '\\';
    lastPathSeparator = strrchr(mwDLDataPath, lastPathSeparatorWindows);
    if (lastPathSeparator == NULL) {
      int lastPathSeparatorUnix;
      lastPathSeparatorUnix = '/';
      lastPathSeparator = strrchr(mwDLDataPath, lastPathSeparatorUnix);
    }

    if (lastPathSeparator == NULL) {
      lastDirName = mwDLDataPath;
    } else {
      lastDirName = lastPathSeparator + 1;
    }

    if (strcmp(lastDirName, codegenDir) == 0) {
      posOfCodegenDir = lastDirName - mwDLDataPath;
    } else {
      posOfCodegenDir = strlen(mwDLDataPath);
    }

    if (posOfCodegenDir == strlen(mwDLDataPath)) {
      size_t filePathLen_a;
      filePathAfterSlicingRelativePathOp = &unresolvedFilePath[1];
      filePathLen_a = (strlen(mwDLDataPath) + strlen
                       (filePathAfterSlicingRelativePathOp)) + 1;
      resolvedFilePath = static_cast<char *>(calloc(filePathLen_a, sizeOfChar));
      stringConcat(resolvedFilePath, mwDLDataPath, filePathLen_a);
      stringConcat(resolvedFilePath, filePathAfterSlicingRelativePathOp,
                   filePathLen_a);
    } else {
      size_t filePathLen_b;
      filePathLen_b = posOfCodegenDir + 1;
      updatedStartDir = static_cast<char *>(calloc(filePathLen_b, sizeOfChar));
      strncpy(updatedStartDir, mwDLDataPath, posOfCodegenDir);
      filePathLen_b = (strlen(updatedStartDir) + strlen
                       (filePathAfterSlicingRelativePathOp)) + 1;
      resolvedFilePath = static_cast<char *>(calloc(filePathLen_b, sizeOfChar));
      stringConcat(resolvedFilePath, updatedStartDir, filePathLen_b);
      stringConcat(resolvedFilePath, filePathAfterSlicingRelativePathOp,
                   filePathLen_b);
      std::free(updatedStartDir);
    }

    std::free(codegenDir);
  }

#undef XSTR
#undef STR

  return resolvedFilePath;
}

static char *sanitizeFilePathForHSP(const char *unSanitizedFilePath)
{
  char *sanitizedFilePath;
  char *stringDuplicate;
  size_t filePathLen;
  size_t sizeOfChar;
  filePathLen = strlen(unSanitizedFilePath) + 1;
  sizeOfChar = 1;
  stringDuplicate = static_cast<char *>(calloc(filePathLen, sizeOfChar));
  stringConcat(stringDuplicate, unSanitizedFilePath, filePathLen);
  sanitizedFilePath = stringDuplicate;
  for (size_t charIdx = 0; charIdx < strlen(unSanitizedFilePath); charIdx++) {
    char charToCheckFor;
    charToCheckFor = unSanitizedFilePath[charIdx];
    if (isspace(charToCheckFor)) {
      sanitizedFilePath[charIdx] = '_';
    }
  }

  return sanitizedFilePath;
}

static void stringConcat(char *destinationString, const char *sourceString,
  size_t destBufferSize)
{
  size_t dstStringLen;
  size_t srcBuffIdx;
  dstStringLen = strlen(destinationString);
  srcBuffIdx = 0;
  while ((sourceString[srcBuffIdx] != '\x00') && (dstStringLen < destBufferSize
          - 1)) {
    destinationString[dstStringLen] = sourceString[srcBuffIdx];
    dstStringLen++;
    srcBuffIdx++;
  }

  destinationString[dstStringLen] = '\x00';
}

namespace coder
{
  namespace internal
  {
    namespace layer
    {
      void b_conv2dDirectOptimizedColMajor(const float X[692224], float Z
        [1384448])
      {
        static const float reformattedAndTruncatedWeights[4608]{ -0.047151007F,
          0.037254326F, -0.008188381F, -0.037747696F, 0.06412642F,
          -0.0023711966F, -0.060386427F, 0.027673332F, 0.008664559F,
          0.050614737F, -0.019523893F, -0.010648383F, 0.016675036F, -0.0749529F,
          -0.047626555F, -0.00033940302F, -0.07826747F, -0.07087442F,
          0.008882499F, -0.03407472F, -0.2537001F, 0.13335371F, 0.10346272F,
          -0.10499348F, 0.09106082F, 0.07864939F, -0.10355763F, 0.013179905F,
          -0.0023538484F, -0.30751586F, 0.15677337F, 0.19255875F, -0.10641023F,
          0.094537705F, 0.14727247F, -0.114237875F, 0.06631846F, 0.03403607F,
          0.04877578F, 0.06531173F, 0.059650585F, 0.048416235F, 0.05928718F,
          0.066571765F, 0.027280819F, -0.11544261F, 0.12830687F, -0.01456107F,
          0.010712754F, 0.22673571F, 0.09317571F, -0.13094133F, 0.14012054F,
          0.11099349F, 0.0639756F, 0.023493571F, 0.03854859F, 0.0199876F,
          0.021466335F, 0.035215877F, 0.018892437F, 0.03640516F, 0.016918242F,
          0.02427804F, 0.0036013487F, -0.0032400496F, -0.037390307F,
          -0.081299156F, -0.09662228F, 0.007693068F, -0.0038063214F,
          -0.025956796F, 0.024893781F, -0.029242864F, -0.04092945F, 0.01877548F,
          0.008472511F, 0.04083381F, 0.0126164295F, -0.010631932F, -0.013171623F,
          -0.079424925F, -0.15322012F, -0.1637012F, 0.047744915F, 0.057894148F,
          0.047691222F, -0.0025781142F, 0.024101589F, -0.0124404635F,
          -0.031507462F, -0.01496639F, 0.008973239F, -0.048195194F, 0.002333862F,
          -0.05081288F, -0.12928616F, -0.028556129F, -0.050024357F, -0.09004691F,
          -0.13558003F, -0.14388663F, 0.050770022F, 0.030092731F, 0.031422365F,
          -0.020904878F, -0.012609844F, -0.0056893434F, -0.06881888F,
          0.008785737F, 0.17470449F, -0.050828133F, -0.012260769F, 0.09090137F,
          -0.16685234F, -0.14457655F, -0.035909742F, 0.014208187F, -0.012505192F,
          0.005319819F, -0.0034218163F, -0.008776195F, -0.015251968F,
          0.0013146844F, -0.005967266F, -0.04438257F, -0.058207F, -0.0200332F,
          2.0865522E-5F, -0.08982952F, -0.014229501F, -0.029114768F,
          -0.13000979F, -0.089859694F, -0.07102617F, -0.04837942F, 0.07129524F,
          0.2131528F, -0.048971083F, 0.05108203F, 0.14680663F, -0.0880993F,
          -0.0037085605F, 0.090188764F, 0.00052206294F, -0.009609086F,
          0.005671955F, -0.0034594527F, -0.009329459F, 0.016453773F,
          0.0015421727F, 0.013404192F, 0.023292065F, -0.009334429F, 0.004198166F,
          -0.0007816921F, 0.008850108F, 0.010589693F, -0.001469091F,
          0.004458691F, 0.0029668445F, -0.009283106F, 0.06329194F, 0.09776375F,
          0.037264943F, 0.11318254F, 0.17221339F, 0.08680172F, 0.06412459F,
          0.12709942F, 0.057280354F, -0.0023058904F, -0.012091701F, 0.002831309F,
          -0.047327947F, -0.059482284F, -0.01938231F, -0.019868204F,
          -0.021512851F, 0.014469982F, -0.0075458423F, 0.009482512F,
          -0.0016481541F, -0.005319817F, -0.0010310581F, 0.008709907F,
          -0.010552006F, 0.0049275397F, -0.0011370028F, -0.0020619743F,
          -0.012829219F, -0.005997002F, 0.016343402F, 0.017690908F,
          0.0069028777F, 0.0020074332F, 0.028208885F, 0.0024403983F,
          -0.010380172F, -0.009307545F, -0.015513004F, -0.0013553337F,
          -0.012337464F, -0.014900533F, -0.0020299105F, -0.011205314F,
          -0.012511275F, 0.00016878566F, -0.009924987F, -0.006791872F,
          -0.0030811692F, -0.037859574F, -0.017857581F, 0.011640136F,
          -0.0008641505F, 0.010591009F, 0.0062874863F, 0.009221619F,
          0.0018826197F, 0.001515949F, 0.0055579823F, 0.0008444957F,
          0.0038476577F, 0.00689235F, 0.0025971124F, -0.0023797452F, 0.00197107F,
          -0.008693209F, 0.0019497054F, -0.003248848F, 0.0006189765F,
          0.002051274F, -0.0049673473F, 0.004456148F, 0.0058575254F,
          0.018132068F, -0.00046374885F, 0.009457708F, 0.010887654F,
          0.009132256F, -6.6833985E-5F, 0.0071447985F, 0.0012180075F,
          0.0021687548F, 0.0016163224F, 0.00025952733F, 0.002891033F,
          0.0016657767F, 0.0035846042F, 0.004000895F, -0.0019590852F,
          0.00050236617F, 0.012788241F, -0.017218305F, 0.019313365F,
          -0.021524489F, -0.06483277F, -0.007824995F, 0.004694418F, -0.03671799F,
          0.008417555F, 0.0031499898F, 0.00079468807F, 0.000847119F,
          0.0067588473F, -0.0029033686F, 0.0007250258F, 0.00029342048F,
          0.0036125456F, -0.0067158244F, 0.0019725088F, -0.0016967987F,
          -0.0040759863F, 0.0074825324F, -0.0058599054F, -0.0013237849F,
          0.008080761F, -0.0014573386F, 0.0009916602F, -0.043835435F,
          -0.09113189F, -0.0012929122F, -0.12693228F, -0.21067096F, -0.06438956F,
          -0.054516766F, -0.11356205F, -0.0020877798F, -0.11102428F, -0.2686684F,
          -0.03281854F, -0.20241672F, 0.056151826F, 0.22500071F, 0.052080274F,
          0.27888387F, 0.05344878F, -0.17463462F, -0.24243625F, -0.0640215F,
          -0.20610152F, -0.036957774F, 0.15648545F, -0.07069719F, 0.16457526F,
          0.12639639F, 0.067643724F, 0.021027181F, 0.029151749F, 0.033491667F,
          -0.037331473F, -0.030087998F, 0.05456772F, -0.059496816F, -0.09567415F,
          0.011346964F, -0.04264423F, 0.034162786F, -0.033756185F, -0.043573122F,
          0.030823268F, 0.05752352F, 0.014466415F, -0.019332789F, -0.0005908305F,
          -0.05400536F, -0.09479621F, -0.09111005F, -0.025425034F, 0.051252063F,
          -0.008505139F, 0.057551675F, 0.0040177843F, -0.0656039F, -0.039958034F,
          0.053736437F, -0.092424825F, 0.04839896F, 0.06798181F, 0.017537033F,
          0.086695F, -0.05057138F, 0.16673556F, -0.0063594696F, -0.077816986F,
          -0.06708075F, -0.02537045F, -0.046792343F, 0.09453475F, -0.1105983F,
          -0.23801501F, -0.035262495F, -0.191806F, -0.14790536F, -0.16177492F,
          -0.2864675F, 0.24996921F, -0.12359941F, 0.12418642F, 0.5014622F,
          -0.034149237F, -0.02710954F, 0.00046952788F, -0.14376977F,
          -0.12100192F, 0.085100845F, -0.08396839F, 0.058707975F, 0.089172006F,
          0.02047669F, -0.014351091F, -0.05518755F, -0.10946699F, 0.12466926F,
          -0.08909173F, -0.014545046F, -0.106247835F, -0.045914203F,
          -0.10957102F, -0.21814027F, 0.061167307F, -0.17929055F, -0.12548141F,
          0.07637111F, -0.0089727715F, 0.3683433F, 0.1543938F, -0.051150013F,
          -0.17550677F, -0.002050261F, -0.31707686F, -0.24634987F, 0.31661245F,
          -0.009207134F, 0.07666169F, 0.08764591F, 0.06444406F, 0.022164537F,
          0.019425485F, 0.030919021F, -0.0057447343F, -0.03426369F, 0.011466857F,
          0.008864401F, -0.10679583F, -0.038897708F, -0.20929171F, -0.082714945F,
          -0.056375712F, -0.1189196F, 0.084810525F, 0.041542932F, 0.111719824F,
          0.07164297F, 0.04186483F, -0.031110002F, 0.04116558F, 0.025622817F,
          0.14811252F, -0.14340056F, -0.06269428F, -0.1321994F, -0.0061463434F,
          0.0022526872F, -0.068355836F, 0.010038031F, -0.10275515F,
          -0.060583584F, 0.074005894F, -0.032305174F, 0.13939719F, 0.054205175F,
          0.115292534F, 0.017549109F, -0.13060157F, 0.03449483F, 0.052649F,
          -0.07758942F, -0.039858386F, 0.03743534F, 0.11483316F, -0.051696178F,
          0.17434616F, 0.15093993F, -0.076813176F, -0.15402861F, 0.19391938F,
          -0.018115627F, -0.08659035F, -0.09286048F, -0.045474675F,
          -0.00012917802F, 0.05046019F, -0.014137006F, -0.0128088035F,
          0.027189003F, 0.0063003995F, 0.009911138F, 0.021059632F, -0.032598633F,
          0.0368661F, 0.016798113F, -0.009762654F, 0.009282893F, -0.0020052542F,
          -0.017569413F, -0.025497992F, -0.012463757F, -0.10325989F, -0.2938013F,
          -0.21745856F, 0.13816755F, -0.04859371F, -0.1617277F, 0.18301712F,
          0.2612241F, 0.12569715F, -0.029942514F, 0.077830896F, -0.008725628F,
          -0.059310492F, -0.03818951F, 0.05892164F, 0.03278003F, -0.05694947F,
          -0.007965902F, -0.03366977F, 0.013088271F, -0.10218309F, -0.099743396F,
          -0.23212537F, -0.13120815F, -0.0040770285F, -0.07594638F, -0.1364678F,
          0.102374874F, 0.12811911F, 0.010917259F, -0.07985278F, -0.002584512F,
          0.10792925F, -0.12288227F, -0.11077446F, 0.0042170947F,
          -0.00089650357F, 0.021428915F, -0.07312727F, -0.12445673F, 0.11683338F,
          0.053395152F, -0.034633946F, -0.074726865F, 0.06307976F, 0.048031062F,
          -0.2461343F, -0.05129735F, 0.0041613337F, 0.039284762F, -0.1492539F,
          0.046268515F, -0.017203676F, -0.06679646F, -0.037179593F, -0.16714545F,
          0.08182632F, 0.046443917F, 0.18288921F, -0.20104083F, 0.06372876F,
          0.050984398F, 0.07485815F, -0.009544355F, 0.08501389F, -0.02118261F,
          -0.23029956F, -0.123042054F, 0.29396695F, -0.044023965F, -0.36126682F,
          0.025587127F, -0.05909061F, -0.0055971565F, 0.099218346F,
          -0.029250303F, -0.09971928F, 0.08278892F, 0.073437765F, -0.040151384F,
          -0.02684569F, -0.13662642F, 0.17243755F, -0.18598485F, -0.100398555F,
          -0.16756858F, -0.049303968F, 0.015560408F, -0.10358868F, 0.042594682F,
          0.080032386F, 0.5102177F, 0.16508335F, -0.26364756F, -0.15841307F,
          0.2662129F, -0.05301057F, -0.3535218F, -0.14136575F, -0.03341626F,
          0.060819868F, 0.05049181F, -0.047429968F, -0.026121922F, 0.080158286F,
          -0.0068939454F, -0.0607804F, -0.024005895F, 0.096245535F, -0.07057235F,
          -0.043337908F, -0.14090095F, -0.06118652F, -0.009419646F, -0.06727158F,
          -0.0017159746F, 0.0428583F, 0.19271716F, 0.07409578F, -0.05323967F,
          0.02692058F, -0.14758168F, -0.11411829F, -0.06913967F, -0.1582117F,
          -0.08151323F, -0.13647188F, -0.0065842425F, 0.037420195F, -0.0497636F,
          0.024537794F, 0.057815816F, 0.015923338F, 0.0272408F, 0.051547524F,
          -0.044789832F, 0.027944967F, -0.022416335F, 0.0035097946F,
          0.018205564F, -0.04126453F, 0.019246535F, 0.009788356F, -0.023282325F,
          -0.37981957F, -0.22500138F, 0.040530376F, -0.28924063F, -0.015979944F,
          0.17915364F, -0.043949634F, 0.0641978F, 0.1988198F, 0.14881249F,
          0.018088844F, -0.072661825F, 0.11057641F, -0.04739712F, -0.12399129F,
          0.045566086F, -0.05902432F, -0.045818806F, -0.03889002F, -0.062056918F,
          -0.020282293F, -0.14702517F, -0.078756206F, -0.01014128F,
          -0.077042766F, -0.0028127353F, 0.050927464F, 0.35148934F, 0.115688525F,
          -0.042037833F, 0.2560879F, -0.22117516F, -0.17010477F, 0.14788456F,
          -0.18817487F, -0.12680903F, 0.15403216F, -0.063440345F, -0.06315062F,
          0.04501532F, -0.22024442F, -0.07994521F, 0.0034392055F, -0.085872225F,
          0.048191924F, 0.12592927F, -0.026659966F, -0.013744838F, 0.19058791F,
          -0.23759693F, -0.25931635F, -0.0009459488F, -0.1443015F, -0.054642264F,
          0.025237422F, 0.07204422F, -0.024610547F, 0.005780465F, 0.071695805F,
          -0.022685407F, 0.04803948F, -0.01632099F, -0.038178865F, 0.02744396F,
          0.0100623155F, -0.05794507F, -0.016777374F, -0.14527468F, -0.2065375F,
          -0.059951503F, -0.07209959F, 0.003046336F, -0.13186486F, 0.012537714F,
          0.058229353F, -0.11732674F, -0.008971643F, 0.032950636F, -0.008619532F,
          0.054058094F, 0.07980408F, 0.25282016F, -0.06554479F, -0.10028651F,
          0.054005165F, -0.13880897F, -0.078222364F, -0.003444497F, -0.15330839F,
          -0.102202535F, 0.35887682F, 0.285303F, -0.06010109F, 0.2666698F,
          -0.06417798F, -0.121764414F, 0.14393872F, -0.13322505F, -0.18144056F,
          0.026484195F, 0.04129254F, 0.008767826F, -0.011639518F, -0.054680042F,
          -0.024864353F, 0.0059034387F, 0.0016887819F, 0.027246485F, 0.03697351F,
          0.027547503F, 0.0054580946F, 0.03133718F, 0.0474944F, 0.024736613F,
          0.018548412F, -0.0012044519F, -0.0031156181F, 0.009998204F,
          0.015424582F, 0.006717911F, 0.005675713F, 0.032782268F, 0.027583677F,
          8.2663886E-5F, 0.012319503F, -0.0012646579F, 0.062197328F,
          -0.0055574067F, -0.054225616F, 0.05222002F, 0.06642651F, -0.026644856F,
          0.0099828F, 0.06311496F, 0.049022354F, 0.0650233F, 0.056760106F,
          -0.13100292F, 0.20250769F, 0.28814697F, -0.060300406F, -0.024724944F,
          0.121397376F, -0.021683773F, -0.020371467F, -0.004550052F,
          -0.0047938805F, -0.008457311F, -0.015532783F, -0.023603786F,
          -0.011828037F, 0.0025962365F, 0.007334077F, -0.052311733F,
          -0.06618634F, 0.006595661F, -0.017160792F, 0.08744387F, -0.029855391F,
          -0.020851787F, 0.034267686F, 0.0061819693F, -0.020551654F,
          -0.009668916F, -0.0035803786F, -0.011751215F, -0.018204479F,
          -0.0063928254F, -0.011422835F, -0.0051328437F, -0.0076128477F,
          -0.0025158324F, -0.013555655F, -0.0023959547F, -0.007723678F,
          -0.041094843F, -0.022215627F, 0.0015585707F, -0.005881979F,
          0.0066581494F, 0.0013060705F, 0.0053942157F, -0.008672507F,
          0.0043827714F, 0.012756932F, -0.0048144097F, 0.010644806F,
          -0.008653986F, 0.0017040323F, 6.1939376E-5F, -0.0009987836F,
          -0.004236172F, 0.007215345F, 0.0048555266F, 0.0013224775F,
          0.0038724246F, 0.002410789F, -0.00110179F, 0.001793252F, 0.007223861F,
          -0.0067879497F, 0.010764749F, 0.0051682508F, -0.00011714407F,
          -0.007964054F, -0.0053757485F, -0.010360884F, -0.003604351F,
          -0.006004474F, 0.002448363F, 0.005436491F, -0.0014946237F,
          0.0049257823F, 0.00096386997F, -0.0022311022F, -0.0024975017F,
          -0.089133374F, -0.10318036F, 0.1274643F, -0.22571869F, -0.3992521F,
          0.02779923F, 0.015826903F, -0.11572996F, 0.019293688F, 0.015299163F,
          0.01744499F, 0.0016052072F, 0.006614561F, 0.014577155F, 0.0035861041F,
          -0.005768753F, 0.0069422843F, -0.0073932265F, 0.005732505F,
          0.006106621F, 0.00043753267F, 0.02141913F, -0.0023031333F,
          0.011276479F, -0.0015374407F, -0.0022988804F, -0.0007379101F,
          0.0018335159F, 0.06908183F, 0.025363225F, 0.09001782F, 0.09025805F,
          0.074450016F, 0.048402052F, 0.007310738F, -0.019287694F, 0.055957433F,
          0.022492757F, -0.0025011105F, 0.03638173F, 0.03551409F, 0.011217141F,
          0.027739342F, 0.029935664F, 0.022245344F, 0.057633765F, 0.0014208147F,
          -0.0030647563F, 0.04844102F, -0.0048455335F, -4.103719E-5F,
          0.051037394F, -0.0029698065F, 0.011723949F, -0.018828025F,
          -0.01399748F, -0.0034811907F, 0.018449608F, 0.01140106F, 0.013530026F,
          0.02421919F, 0.015405867F, 0.019390693F, -0.03127407F, -0.031254184F,
          -0.0014801016F, 0.0046924227F, -0.010039223F, 0.017316604F,
          0.026484622F, 0.003536269F, 0.035961386F, -0.02186243F, 0.09563837F,
          0.005806773F, -0.0064591644F, 0.007329937F, -0.029433185F,
          0.048715338F, -0.025384417F, 0.003149605F, 0.032799847F, -0.011034086F,
          -0.015223589F, -0.0010841363F, -0.00047241338F, -0.00060097495F,
          -0.024469346F, -0.00607721F, 0.01756367F, -0.008941429F, -0.016788494F,
          -0.004241933F, -0.075268716F, -0.03087267F, -0.031695485F,
          -0.08524074F, -0.025605913F, -0.077596724F, 0.046351008F, 0.022622274F,
          0.03268713F, -0.0041136434F, 0.0041367663F, 0.010378474F,
          0.0013373981F, 0.010257343F, 0.0027079005F, -0.059498176F,
          -0.046105485F, -0.023889614F, -0.07957666F, -0.051444862F,
          -0.052290134F, -0.08330931F, -0.07593009F, -0.04531947F, -0.10734919F,
          -0.113343544F, -0.07473722F, -0.121478036F, -0.12385617F, -0.07610631F,
          -0.09702691F, -0.13909529F, -0.042610638F, 0.020950975F,
          -0.0018752523F, 0.026946124F, -0.0005757629F, 0.009838309F,
          0.03536449F, 0.060943745F, -0.009947491F, 0.0020539379F, -0.0959476F,
          -0.12939373F, -0.095989585F, -0.14718358F, -0.10697332F, -0.09454639F,
          -0.14139348F, -0.1320944F, -0.10025372F, 0.029236613F, 0.0242583F,
          -0.019625332F, 0.0120183695F, 0.022288032F, -0.021873862F,
          -0.010287114F, 0.0034141324F, -0.03847167F, -0.03205238F,
          0.0052330685F, -0.048009377F, -0.03951315F, -0.028214246F,
          -0.018350868F, 0.017183993F, -0.06032485F, 0.07789725F, -0.025352504F,
          -0.01780295F, 0.00430355F, 0.009407635F, -0.00029323576F,
          -0.0009848204F, 0.02188581F, -0.041533116F, 0.06825874F, 0.023002949F,
          0.037319798F, -0.00711395F, 0.007485943F, 0.0311616F, -0.013484491F,
          -0.016170835F, 0.00019555187F, -0.041331165F, -0.020268137F,
          -0.008429749F, -0.009936862F, -0.0109674055F, -0.019916054F,
          0.009259171F, 0.008943173F, -0.014193252F, -0.0008103966F, 0.01621782F,
          0.035646956F, 0.020243062F, 0.009095439F, 0.0068552042F, 0.021457009F,
          0.026158605F, 0.019702205F, -0.015271674F, -0.038234796F,
          -0.043316536F, 0.016208014F, -0.04865899F, -0.06817934F, -0.01803448F,
          0.009670522F, -0.04685241F, -0.017333183F, -0.15691726F, -0.16250598F,
          -0.102546915F, -0.1880159F, -0.20191209F, -0.1585508F, -0.12850384F,
          -0.16479416F, -0.11390152F, -0.02740577F, 0.000581176F, -0.009031084F,
          -0.03567162F, 0.001775939F, -0.027324336F, -0.03751436F, -0.012237232F,
          -0.041144025F, -0.04883535F, -0.048549533F, -0.0371516F, -0.05228537F,
          -0.04175361F, -0.052505538F, -0.054383606F, -0.07027203F,
          -0.059497934F, 0.0036022773F, -0.019539632F, -0.018539613F, 0.0040047F,
          -0.022425616F, -0.008579198F, -0.018000819F, -0.0050711897F,
          -0.014133981F, -0.0059731095F, 0.0030822798F, 0.04156254F,
          0.0037153754F, -0.023935856F, 0.024710571F, 0.017279997F, 0.021186087F,
          0.029803028F, -0.0068397517F, -0.0067430604F, -0.00040086257F,
          0.008116094F, 0.010599935F, -0.005291401F, 0.03983526F, -0.0048596174F,
          0.00085997634F, -0.0024622083F, -0.005310979F, -0.005625381F,
          -0.0051123644F, 0.018724926F, 0.0017088755F, -0.0052346173F,
          0.0029222153F, 0.014467624F, -0.001026017F, 0.010673983F, 0.014359796F,
          0.009316524F, 0.015499696F, 0.008087278F, -0.00019533232F,
          0.0124241775F, -0.024724329F, -0.014037219F, -0.0062958277F,
          -0.0058841943F, 0.015057539F, 0.005277445F, 0.012311165F,
          0.0007251056F, 0.012464273F, -0.0032011832F, -0.032015372F,
          -0.035700694F, -0.042017125F, -0.0045405827F, -0.0031265167F,
          -0.009068178F, -0.047336936F, -0.0050541516F, -0.0014279484F,
          0.031833094F, 0.04161642F, 0.028132834F, -0.0015434253F, 0.034095664F,
          0.013057213F, 0.018125597F, -0.0014084819F, 0.024598306F, 0.02095351F,
          -0.014338573F, -0.004811816F, 0.039345503F, -0.016938768F,
          0.013078285F, 0.015460864F, 0.013304274F, 0.024961336F, -0.002041185F,
          -0.008436725F, -0.036893982F, -0.0020926048F, -0.01432593F,
          -0.0394321F, -0.058509618F, -0.048062995F, -0.05039332F, 0.080367714F,
          0.04283854F, -0.08382387F, 0.07363213F, 0.0491912F, -0.086394936F,
          -0.006591895F, 0.05835495F, -0.0804876F, 0.11867869F, -0.071717784F,
          -0.057906903F, 0.14526117F, -0.07719192F, -0.041886143F, 0.06873569F,
          -0.026048746F, -0.029910618F, 0.0032155185F, -0.02512106F,
          -0.025679644F, 0.034973167F, -0.021698132F, -0.041671857F,
          0.032611925F, 0.016620371F, -0.028516507F, -0.008714997F,
          -0.024710502F, 0.006596847F, 0.0024511823F, -0.026551537F,
          -0.001220381F, 0.01960028F, -0.01588571F, -0.011504111F, 0.031825524F,
          -0.046812247F, -0.029671118F, -0.038173746F, -0.10089893F,
          0.039048996F, 0.07437082F, 0.0050547617F, -0.020823654F, 0.05454734F,
          -0.07808293F, -0.01560846F, 0.05351772F, -0.055899933F, -0.02174614F,
          0.03394413F, -0.0035792475F, 0.015348522F, -0.19665584F, 0.0064746467F,
          0.11713587F, -0.16121627F, -0.04861075F, 0.06933236F, -0.16847597F,
          -0.00031869262F, 0.028558858F, 0.0041817F, 0.07121895F, -0.05345379F,
          -0.022798024F, 0.046300627F, -0.053780068F, -0.038331453F, 0.09275103F,
          -0.02246678F, -0.030439578F, -0.07048243F, -0.00072736567F,
          0.015331392F, -0.026264051F, 0.0065542944F, -0.023243273F,
          0.024957053F, 0.012270887F, 0.017319445F, -0.065088235F, 0.02910666F,
          0.0041342773F, -0.07186177F, -0.018623607F, -0.02774487F, 0.06256323F,
          0.011670898F, 0.30886826F, -0.026313787F, -0.18756834F, 0.3142073F,
          -0.066460475F, -0.21736744F, 0.18713015F, 0.053683203F, -0.175359F,
          -0.04173687F, -0.013586676F, 0.02286332F, -0.066353686F, -0.08083866F,
          -0.009915895F, 0.08167474F, -0.0699849F, 0.030350724F, 0.020541644F,
          -0.039365496F, -0.024426937F, 0.014108863F, -0.024396224F,
          -0.003631425F, -0.00979479F, -0.01657799F, -0.012769082F, 0.042181473F,
          -0.0042017647F, -0.18725856F, 0.079418585F, 0.028138231F, -0.13781105F,
          -0.020826468F, 0.032952566F, -0.15175043F, -0.030733716F,
          -0.034099974F, -0.014471701F, -0.06320509F, -0.005153855F,
          -0.0059938068F, -0.021375181F, -0.04982264F, -0.04699586F, 0.04835365F,
          -0.032725383F, -0.0326107F, 0.034109175F, -0.011917575F,
          -0.0153063275F, 0.013749287F, 0.001531326F, -0.009954151F,
          -0.031928394F, -0.031085745F, 0.033440396F, -0.040565245F,
          -0.036823016F, 0.013249587F, 0.0091601135F, 0.012668113F, 0.023701664F,
          0.022853982F, 0.0026264365F, 0.0013297828F, -0.009452587F,
          -0.027973443F, -0.008643515F, -0.012928492F, -0.017351953F,
          -0.017052855F, -0.05669245F, -0.08058761F, -0.019015307F, -0.07246344F,
          -0.11772972F, -0.036205582F, 0.016950503F, -0.027230058F, -0.01569956F,
          -0.08719645F, -0.056960166F, 0.012136519F, -0.07991097F, -0.058849484F,
          0.01981306F, 0.0016781723F, -0.0060268147F, 0.0062520076F,
          -0.021936422F, -0.004304964F, -0.0058096056F, -0.010491132F,
          0.003932805F, -0.0077008167F, -0.001923185F, 0.0014705047F,
          -0.000153507F, 0.13530293F, 0.07888766F, -0.059817564F, 0.07305918F,
          0.0024684453F, 0.037446026F, -0.04006072F, -0.048345562F, -0.05404539F,
          -0.010325837F, -0.0062892665F, -0.009859692F, -0.007648951F,
          -0.016370704F, -0.018856969F, -0.0026893623F, -0.012504785F,
          -0.017842563F, 0.016923843F, -0.0122387735F, 0.015055437F,
          -0.022203889F, -0.08185746F, -0.015860325F, -0.006647093F,
          -0.032905612F, -0.0065665436F, 0.003028127F, -0.0143470215F,
          0.0062934407F, 0.011651712F, -0.0059906025F, -0.00040092383F,
          0.014546834F, 0.0067369994F, -0.0022304812F, 0.0046045333F,
          -0.0007224435F, 0.009355001F, -0.0022641649F, 0.003790417F,
          0.0019878778F, -0.0027016213F, 0.014923909F, 0.009555516F,
          0.0047465097F, 0.00449337F, 0.004024992F, 0.014463625F, 0.008240124F,
          0.0039075473F, 0.001910394F, 0.013135631F, 0.008144176F,
          -0.0035514024F, 0.005026607F, -0.0013905208F, 0.0032498017F,
          -0.0004903329F, 0.002459611F, -0.00044426115F, -0.00027738189F,
          2.4933335E-5F, 0.09398304F, 0.10345492F, -0.022243619F, 0.11855024F,
          0.115379415F, -0.0048724636F, 0.011100058F, 0.041556247F, 0.014099937F,
          0.009662201F, 0.0077649606F, -0.0005716612F, 0.0067477063F,
          0.009132145F, 0.008803811F, -0.0026261897F, -0.006300141F,
          -0.0035028737F, 0.008344475F, 0.0020041966F, 0.01161764F, 0.007609813F,
          0.0018400743F, 0.0008226876F, 0.004344456F, 0.0003012302F,
          0.004100632F, 0.048058726F, 0.05495414F, -0.03483639F, 0.07850644F,
          0.08751639F, -0.0070770024F, -0.022181628F, 0.024411261F,
          -0.023846077F, -0.032618295F, 0.21319842F, -0.121955656F, 0.059263792F,
          0.03167306F, -0.004258065F, 0.0075315908F, -0.099654526F, 0.00958194F,
          0.087246425F, -0.09780779F, 0.033068653F, 0.038258087F, -0.023049826F,
          0.019801801F, -0.107112534F, 0.016518645F, -0.044442806F,
          -0.017456815F, -0.015722143F, -0.003947882F, -0.010334955F,
          -0.0050924593F, 0.034186363F, -0.0014711372F, 0.0313641F, 0.048274696F,
          -0.04720513F, -0.015919736F, -0.00512362F, 0.0031611298F, 0.009858481F,
          0.03590084F, 0.018444557F, 0.021007994F, 0.03628099F, 0.10106805F,
          -0.032023117F, -0.12536235F, 0.059502203F, -0.18617225F, -0.1460002F,
          -0.103608534F, -0.10267112F, 0.09565174F, 0.022505077F, -0.02224127F,
          -0.00011777529F, -0.0022565264F, 0.019238524F, -0.0420782F,
          -0.012177875F, -0.015131341F, -0.00057286874F, -0.07320099F,
          -0.14685299F, 0.012336611F, -0.0680648F, -0.09683807F, 0.055407118F,
          -0.050376114F, 0.03584723F, -0.09011043F, -0.077813976F, 0.10903028F,
          0.008662844F, 0.030644171F, 0.07001498F, -0.11852088F, 0.119647056F,
          -0.037405226F, -0.08124983F, -0.023504917F, -0.05254673F,
          -0.015771244F, -0.025081784F, -0.036472972F, -0.102285996F,
          -0.026710523F, -0.0935123F, 0.039451994F, -0.060813412F, -0.07450925F,
          0.053417392F, -0.26102707F, 0.26381338F, -0.16861549F, -0.098315224F,
          0.17451899F, -0.082684904F, 0.0033147046F, 0.24623953F, -0.17942208F,
          0.13904679F, -0.49087864F, 0.13440612F, 0.03793373F, -0.110420905F,
          -0.0012243575F, 0.0724563F, -0.049235977F, -0.17683418F, 0.20145388F,
          -0.4935619F, 0.009183841F, 0.066572525F, -0.28343442F, 0.104227774F,
          0.01612543F, 0.004597728F, 0.002311393F, -0.009400405F, -0.02778447F,
          0.03131593F, -0.0044225175F, -0.006806043F, 0.055703066F, -0.12462737F,
          0.16918455F, -0.1343695F, 0.016127229F, -0.051243808F, -0.03292508F,
          0.08161154F, -0.120839946F, 0.0926104F, 0.022113504F, -0.11903344F,
          0.28246248F, -0.29492983F, 0.29764715F, -0.05770291F, 0.09722208F,
          0.08127885F, -0.11405364F, 0.037322436F, 0.023794588F, 0.014292753F,
          0.04038284F, -0.020826107F, -0.01656432F, 0.0035894483F, -0.026011843F,
          0.0070965043F, -0.028555157F, -0.017206822F, -0.015711818F,
          -0.023382632F, 0.002392333F, -0.0011490117F, -0.017023552F,
          -0.0014910139F, -0.012607921F, 0.018610548F, 0.015063915F,
          0.011939727F, 0.011382718F, -0.008976978F, 0.00887903F, 0.013349654F,
          0.0020876552F, 0.0037073698F, -0.027743936F, -0.0128748305F,
          -0.009589084F, -0.029350393F, -0.0063516167F, -0.0071493303F,
          -0.027165722F, -0.009142546F, -0.0071688103F, -0.013648706F,
          0.011633456F, 0.0016173673F, -0.014108515F, 0.018282745F, 0.007154858F,
          -0.0023670045F, 0.024192534F, 0.007369374F, -0.0041723945F,
          0.04235227F, 0.029889936F, -0.00080371834F, 0.007039209F, 0.024089865F,
          -0.030959744F, 0.016206607F, 0.024621217F, -0.017303942F,
          -0.0023421722F, -0.005035701F, -0.03130078F, 0.0330163F, 0.023698514F,
          -0.029413184F, 0.028503645F, -0.022560293F, -0.025639124F,
          -0.01827605F, -0.000799345F, 0.01582587F, 0.017128099F, 0.015097733F,
          -0.0039302995F, 0.0013725474F, 0.012905543F, -0.07149094F,
          -0.08450696F, -0.07468566F, -0.05495737F, -0.08326292F, -0.059027568F,
          -0.04989058F, -0.050497673F, -0.053339038F, -0.013521169F,
          0.0033673062F, -0.015543868F, -0.03328431F, -0.010473895F,
          0.0050815395F, 0.0041640857F, -0.00133439F, -0.015490806F,
          0.016964553F, 0.020650977F, 0.00949094F, -0.008303392F, -0.030632392F,
          -0.029969795F, -0.012959161F, -0.0062812977F, -0.006843743F,
          -0.021187892F, 0.035135932F, 0.0010251143F, 0.007648029F, 0.0388525F,
          0.009316919F, 0.010276458F, 0.038112022F, -0.001726916F, -0.021129297F,
          0.011248163F, 0.017898096F, -0.00169561F, 0.02963836F, 0.032354094F,
          0.021216126F, 0.025496434F, 0.0076473625F, -0.0041765086F,
          -0.020239156F, -0.013426176F, -0.005727686F, -0.017298788F,
          -0.006779583F, 0.0116230035F, 0.005403794F, -0.0018257848F,
          -0.0004722468F, -0.04228843F, -0.02656195F, 0.006245635F,
          -0.038499374F, -0.014416662F, -0.02857241F, -0.007811234F,
          -0.035610035F, 0.063067734F, -0.009071839F, -0.034820806F, 0.06302955F,
          -0.042717755F, -0.041354254F, 0.063092194F, -0.01951442F,
          -0.023455223F, -0.011762535F, -0.018744439F, -0.009299846F,
          -0.011286785F, -0.015076519F, -0.0013009419F, 0.016165392F,
          0.010878559F, 0.00017679468F, 0.017980007F, 0.01871828F, 0.011274649F,
          0.047577213F, 0.034268215F, 0.001873424F, 0.017271727F, 0.021324102F,
          0.030106308F, -0.027215524F, -0.013800364F, -0.008418431F,
          -0.01451671F, -0.008903531F, -0.007703172F, -0.0139707215F,
          -0.019602625F, -0.00508689F, 0.0011779639F, 0.0021533775F,
          -0.0010690278F, 0.0063268095F, 0.0056202584F, 0.0028497595F,
          -0.004792477F, 0.003483981F, 0.0052054143F, -0.0014283505F,
          -0.003022107F, -0.0075026085F, -0.0011636114F, 0.0045774737F,
          0.0038813462F, -0.008501279F, 0.0042110314F, 0.009633212F,
          0.028759375F, 0.00292713F, -0.014217972F, 0.018057805F, 0.016914725F,
          -0.015459616F, 0.006445902F, 0.0034830053F, -0.016090242F,
          -0.027204748F, 0.0015212771F, -0.0016218467F, -0.020539768F,
          0.02079014F, 0.005260962F, -0.010045251F, 0.02896637F, 0.0023323505F,
          -0.016706184F, -0.013421807F, -0.0038471315F, -0.021435102F,
          0.0033020028F, -0.003265932F, -0.013130055F, 0.008250855F, 0.00865639F,
          -0.011818259F, -0.008064185F, -0.007930086F, -0.021130038F,
          0.010980732F, -0.008052501F, -0.026348207F, 0.0107268505F,
          -0.0016850968F, -0.010065229F, -0.0024028285F, 0.0017342783F,
          0.0004940897F, -6.0431807E-6F, -0.013503264F, -0.0054647634F,
          0.011171819F, -0.004716086F, 0.005039133F, 0.012175977F, 0.015538443F,
          0.011023222F, 0.02479724F, 0.01598015F, 0.03141171F, 0.049681183F,
          -0.0032156692F, 0.010673955F, 0.014472699F, 0.0026709998F, 0.06558004F,
          -0.003030118F, -0.014808253F, 0.017652282F, 0.011336061F, 0.006869095F,
          0.023938084F, 0.033672135F, 0.00015666503F, 0.03549774F, 0.04111862F,
          0.024639035F, -0.0076584374F, -0.0018908262F, 0.016642712F,
          -0.010845895F, -0.006435998F, 0.0178971F, -0.006347438F, -0.01393525F,
          0.019427845F, -0.005353173F, -0.009596995F, 0.009178484F,
          -0.0031531404F, 0.0015661084F, -0.0070350845F, 0.02495317F,
          0.026521122F, 0.010526388F, -0.012134508F, -0.007726597F,
          -0.009859832F, -0.009663507F, 0.009911886F, 0.00324354F, -0.017880995F,
          0.048832867F, 0.039805997F, -0.01922677F, 0.021111753F, 0.0065755174F,
          -0.012422861F, -0.0041353684F, 0.016579952F, -0.011297978F,
          -0.007697392F, 0.015633376F, -0.01169093F, -0.0056052934F,
          0.009285716F, -0.019974116F, 0.064035736F, 0.061607085F, 0.039093673F,
          -0.07620684F, 0.004942442F, -0.017059779F, -0.027270135F, -0.03719261F,
          0.05964763F, 0.019594708F, -0.03378131F, -0.019961663F, 0.11594556F,
          0.09188855F, -0.03343315F, -0.09193026F, 0.03798715F, 0.0049953475F,
          -0.0016372788F, -0.016892286F, 0.011357688F, -0.01074302F,
          0.022997608F, -0.010048402F, 0.010828143F, -0.0034324236F,
          0.011831428F, -0.007788835F, -0.043896604F, -0.008126439F,
          0.004974302F, 0.069980405F, -0.06531391F, -0.04470897F, 0.012762344F,
          0.12322291F, 0.109799F, -0.028969824F, -0.26528397F, -0.16200836F,
          0.11638314F, 0.038310148F, -0.27632797F, -0.22004314F, 0.008036161F,
          0.05090753F, -0.010268348F, -0.02143913F, -0.05107632F, 0.020687245F,
          0.023422057F, 0.023845606F, -0.0057111243F, -0.08674473F, -0.32181132F,
          -0.12444587F, -0.0036837484F, 0.02413336F, 0.00716377F, 0.043280635F,
          -0.09677868F, -0.031541053F, -0.034863934F, 0.09755535F, 0.14290899F,
          0.07864198F, -0.09232896F, -0.051574755F, -0.04261325F, -0.023557674F,
          -0.008838172F, 0.008178305F, 0.17210236F, -0.10080691F, -0.084606275F,
          -0.16490395F, 0.09164198F, -0.04960434F, 0.09982399F, 0.007117661F,
          0.09619789F, 0.000138079F, -0.14363743F, -0.40259433F, 0.10574899F,
          0.20618634F, 0.062989645F, -0.015181263F, -0.23092987F, 0.21401815F,
          0.24827272F, -0.13670336F, -0.33758545F, -0.15033178F, 0.28275537F,
          0.06571362F, -0.109614596F, -0.19084208F, -0.17378753F, -0.06043665F,
          0.09743428F, 0.07946793F, 0.22141954F, -0.37431702F, -0.09010799F,
          -0.039566126F, 0.09791133F, -0.008545183F, -0.0037510162F,
          -0.06237004F, 0.014868743F, -0.0051367558F, -0.019639151F, 0.00935801F,
          0.01615809F, -0.004104201F, -0.08096145F, -0.051581684F, 0.0034638115F,
          -0.037918586F, -0.18091118F, -0.09072358F, -0.011534754F, 0.039119013F,
          -0.061383404F, -0.22769202F, -0.1953897F, 0.04037768F, 0.3280603F,
          0.079303384F, -0.39281806F, -0.11299714F, 0.11904624F, 0.29779926F,
          0.023326159F, 0.058950145F, -0.06080545F, -0.030359339F, -0.016097274F,
          0.087198645F, -0.0036446496F, -0.020129437F, -0.020233555F,
          0.13350846F, 0.110978104F, 0.093242526F, -0.020954272F, -0.03221049F,
          -0.013995341F, -0.09638894F, -0.10166367F, -0.014429264F,
          -0.053769138F, -0.080665916F, -0.057331F, 0.010330628F, 0.05280041F,
          0.059986006F, 0.025414163F, 0.044573095F, 0.072657734F, 0.0027527455F,
          0.012657175F, 0.019021623F, 0.0053867637F, -0.011623472F,
          -0.012241797F, 0.043976713F, 0.017234562F, -0.0016312212F,
          -0.008436863F, -0.033518355F, -0.042882495F, -0.0046966523F,
          -0.034534767F, -0.031128F, 0.054668248F, 0.052212454F, 0.043834105F,
          -0.035504945F, -0.06768813F, -0.07246813F, -0.022530247F,
          -0.050944228F, -0.053582758F, -0.07317834F, -0.06595103F,
          -0.0022731677F, -0.004265422F, 0.03425902F, 0.02720867F,
          -0.0062991725F, -0.040087238F, -0.011692848F, 0.03202128F,
          -0.008986052F, 0.027362412F, -0.059362467F, -0.050033677F,
          -0.06148371F, -0.04265149F, -0.06508363F, -0.06881972F, 0.05997772F,
          0.006509872F, -0.047963474F, 0.11517216F, 0.13417706F, 0.15729323F,
          -0.020234155F, 0.033522274F, 0.03678962F, -0.16074057F, -0.1876184F,
          -0.16642648F, -0.061260115F, -0.07871579F, 0.011477683F, -0.083901644F,
          -0.059132855F, -0.0031498277F, -0.0021397523F, -0.03886145F,
          0.0073489877F, 0.0645491F, -0.021385731F, 0.030537179F, 0.017646922F,
          -0.035835873F, 0.027932175F, 0.054833002F, 0.023058053F, 0.064162746F,
          0.08695516F, 0.05017907F, 0.07622384F, 0.00454715F, -0.02650802F,
          0.037889417F, -0.011112257F, 0.016244633F, 0.035343897F, -0.21670403F,
          -0.39191738F, -0.11962933F, -0.3026028F, -0.54865676F, -0.23395483F,
          -0.03515066F, -0.14957006F, -0.03820414F, -0.052966274F, -0.06845406F,
          -0.075739354F, 0.007754282F, 0.031619564F, 0.0065293247F, 0.03614204F,
          0.055590264F, 0.04299753F, 0.02656549F, 0.03448898F, 0.05069736F,
          0.00037681215F, -0.0104776835F, -0.006410982F, 0.05579755F,
          0.07829494F, 0.06238703F, 0.02707787F, 0.06493295F, 0.080395214F,
          0.025083177F, 0.0025096443F, 0.025946412F, 0.08671145F, 0.057173673F,
          0.022276936F, 0.016870715F, -0.0008518829F, 0.018192954F,
          -0.022092488F, -0.0025923334F, 0.031763714F, -0.048153006F,
          -0.0017226011F, 0.05921916F, -0.17332877F, -0.0967055F, -0.044585615F,
          0.0018240506F, -0.087487996F, -0.043084163F, 0.11103524F, 0.01366952F,
          0.030723993F, -0.034803823F, -0.19670062F, -0.11734683F, 0.1671446F,
          -0.12935506F, -0.14938079F, 0.25067624F, 0.14796488F, -0.05067007F,
          -0.0013262044F, 0.059131853F, 0.052869193F, -0.07938025F,
          0.0092363255F, 0.01651064F, -0.16185012F, -0.058147326F,
          -0.0007605687F, 0.018129151F, 0.07132094F, 0.026870184F, -0.056130003F,
          0.00778318F, -0.008543942F, -0.10453585F, 0.008633637F, 0.049744435F,
          0.054064654F, 0.051201962F, 0.063408725F, -0.15319327F, -0.06900197F,
          0.09820232F, -0.0636201F, -0.13958663F, -0.097047135F, -0.025456902F,
          -0.10031801F, -0.011765592F, 0.11419961F, -0.025129782F, -0.124008834F,
          0.06165689F, 0.088282466F, 0.04793646F, -0.12738253F, -0.007620016F,
          0.07371414F, -0.08822012F, -0.08941732F, -0.028366446F, -0.07216302F,
          -0.089251265F, 0.03536773F, -0.007776994F, -0.22156388F, -0.25317535F,
          0.25805837F, -0.08946786F, -0.2577995F, 0.21081533F, 0.31277373F,
          0.08606068F, -0.12600727F, -0.08172635F, -0.020912468F, 0.14869274F,
          -0.17900082F, -0.1616055F, 0.19640289F, 0.029537626F, -0.05669293F,
          -0.0816064F, -0.13056564F, -0.091726184F, 0.024995992F, -0.07742232F,
          -0.20332469F, 0.036798943F, 0.0387296F, -0.06943645F, 0.018145746F,
          -0.09956845F, -0.06095229F, 0.029021664F, -0.012117732F, -0.034289353F,
          0.089284115F, 0.06989827F, -0.0793603F, -0.033546135F, -0.13625973F,
          -0.07204466F, 0.1731173F, -0.1488556F, -0.17974085F, 0.06307659F,
          0.050634176F, 0.025022553F, 0.02687791F, -0.038346156F, 0.012694043F,
          0.05330698F, 0.014572342F, 0.026193641F, -0.12664036F, -0.017976675F,
          -0.039552353F, 0.017711682F, -0.13493004F, -0.00036898F, 0.08044104F,
          -0.03208678F, -0.04195793F, 0.11501347F, 0.19273691F, 0.043372188F,
          0.032393605F, -0.099285044F, -0.065652005F, 0.102986045F, -0.04600126F,
          -0.05020089F, 0.05676654F, 0.22562166F, 0.047719494F, -0.037219707F,
          -0.11084042F, -0.03999013F, 0.105768755F, -0.068931445F, -0.077554345F,
          0.07253278F, 0.111604385F, -0.029262649F, 0.046310455F, -0.047217384F,
          -0.09842632F, 0.10696793F, -0.086078025F, -0.049210746F, 0.10069885F,
          -0.028341597F, -0.01450843F, -0.0464686F, -0.04289295F, 0.11278323F,
          -0.00080440345F, -0.035505105F, 0.12826075F, -0.037599873F,
          -0.044442713F, 0.12850358F, 0.008224164F, -0.049606986F, 0.026245948F,
          -0.0069290046F, -0.06855559F, 0.049220286F, 0.002466882F,
          -0.021956401F, 0.06588144F, -0.052650638F, -0.038465865F, 0.027076717F,
          -0.016740212F, -0.018844001F, 0.058597535F, -0.036160093F,
          0.016470345F, 0.08784498F, 0.14694002F, -0.25735068F, 0.08798291F,
          0.17911842F, -0.2998644F, 0.053522926F, 0.12916876F, -0.28895777F,
          0.055552732F, 0.0057646064F, -0.004279286F, 0.015126713F,
          0.00080391846F, 0.007615096F, -0.011545356F, 0.00233309F, 0.010475975F,
          0.02364384F, -0.16084313F, 0.07185676F, -0.15373647F, -0.20389771F,
          -0.043509368F, -0.18794407F, -0.12540853F, 0.05562353F, -0.041394632F,
          0.08400293F, 0.13818614F, -0.16936974F, 0.07059501F, 0.2045655F,
          -0.17672923F, 0.108565554F, 0.09257239F, -0.18460833F, 0.029913332F,
          -0.067385145F, 0.012291594F, 0.084480375F, -0.10201215F, -0.019197661F,
          0.04153189F, -0.054294538F, 0.011699732F, 0.047163725F, -0.026044967F,
          0.01901084F, -0.11598268F, -0.18777451F, -0.13117132F, 0.057664F,
          -0.015146849F, 0.027167076F, 0.11477807F, -0.40812138F, 0.048240688F,
          0.24107227F, -0.43476316F, 0.13979283F, 0.0868958F, -0.36082026F,
          0.08595578F, 0.040406104F, 0.03181833F, 0.02454418F, -0.12323836F,
          -0.14273691F, -0.12343689F, 0.080656F, -0.055772867F, 0.0096729435F,
          -0.03888754F, -0.01567328F, 0.080979325F, -0.035921704F, -0.059916727F,
          0.070628755F, -0.034266803F, -0.05227037F, 0.051615484F, -0.062142897F,
          -0.060443424F, 0.069074884F, -0.070909865F, -0.040258665F,
          0.096597105F, -0.050658695F, -0.059951495F, 0.004944998F, -0.19432682F,
          0.22871684F, -0.14602323F, -0.110420406F, 0.41057074F, -0.08937306F,
          -0.16218898F, 0.264153F, -0.097931325F, -0.009390712F, -0.035837717F,
          0.027098238F, 0.035899162F, -0.031817447F, 0.017932042F, -0.0174284F,
          -0.033295553F, 0.021795215F, -0.08366108F, -0.057824716F,
          -0.0059453337F, -0.11296346F, -0.09200106F, -0.032063585F,
          0.0038697715F, 0.008195085F, 0.04980391F, -0.00035804592F,
          -0.024586715F, -0.07283472F, -0.018548986F, -0.07519763F, -0.11430426F,
          0.06982963F, 0.03232149F, 0.022237718F, 0.017633727F, 0.02131809F,
          0.012926276F, -0.006845355F, -0.013476266F, -0.017401747F,
          0.023543103F, 0.016383434F, 0.003969269F, -0.012006987F, -0.011109904F,
          -0.0075724605F, 0.011092688F, 0.0064562913F, 0.030624568F,
          -0.007950218F, 0.0112124F, 0.04764256F, 0.06792276F, 0.09112062F,
          0.071994685F, 0.01696421F, -0.028771488F, -0.051913183F, 0.03306412F,
          0.0372757F, 0.027530516F, -0.005351224F, -0.04403329F, -0.077248335F,
          0.042089924F, 0.056934927F, 0.043381877F, -0.0004874751F, 0.024897698F,
          0.055898283F, 0.025299773F, -0.056374706F, -0.063300915F, -0.01029877F,
          -0.014219565F, -0.07586209F, -0.0017324303F, -0.017211797F,
          -0.035091873F, -0.0130025325F, -0.064984806F, -0.030323345F,
          -0.019362161F, -0.0910217F, -0.052774064F, 0.0941746F, 0.084363066F,
          0.10476076F, 0.03474447F, -0.0027401492F, 0.077751204F, 0.030976534F,
          0.01185914F, 0.056407664F, 0.047006886F, 0.019005332F, 0.034027413F,
          0.07723485F, 0.02296662F, 0.08330945F, 0.07360709F, 0.009851011F,
          0.033193503F, 0.040051732F, 0.021990644F, 0.052020583F, -0.04722004F,
          -0.090484075F, -0.09022623F, -0.118794255F, -0.19675916F, -0.23602408F,
          0.0677168F, 0.060776934F, 0.08884234F, 0.08993072F, 0.017898817F,
          0.06867666F, 0.055147428F, 0.03695708F, 0.042106107F, 0.04342117F,
          0.009938404F, 0.08626922F, 0.0488819F, 0.037580363F, -0.008765083F,
          -0.012197093F, -0.017719505F, -0.070588596F, 0.010795852F,
          0.024738837F, -0.020158404F, -0.29153857F, -0.39440638F, -0.38835803F,
          -0.19895864F, -0.24269791F, -0.26356298F, 0.037891407F, 0.084710784F,
          0.11115397F, -0.07657175F, -0.08922304F, -0.08894892F, -0.16351439F,
          -0.19831581F, -0.20997189F, 0.02101062F, 0.008568466F, 0.0511976F,
          0.006761414F, 0.008674794F, -0.013316056F, -0.0009578456F,
          -0.005540616F, -0.013621282F, -0.006612544F, 0.01825203F, 0.022891128F,
          0.08082481F, 0.055996947F, 0.046459086F, 0.028514128F, -0.097583294F,
          0.060573723F, -0.010369698F, -0.058251012F, 0.0011872898F,
          -0.115573645F, 0.08119585F, 0.047957797F, -0.05820594F, 0.09303381F,
          0.030340234F, -0.033391744F, 0.024275212F, 0.010047592F, 0.009791968F,
          -0.06823114F, -0.054164726F, 0.026733534F, -0.042560384F,
          -0.028525785F, 0.023508772F, -0.012634553F, -0.016203774F,
          -0.055181634F, -0.07907044F, -0.014070124F, 0.007093866F,
          -0.014005016F, 0.016439183F, -0.0034763694F, -0.0102772415F,
          0.017545886F, 0.028537486F, -0.283416F, -0.05151562F, 0.01364183F,
          -0.25473937F, 0.0014527781F, -0.008354326F, -7.9068195E-5F,
          0.014309889F, -0.061758984F, 0.15647855F, -0.0018252567F,
          -0.038670607F, 0.018810406F, 0.020461008F, 0.007416601F, -0.013572474F,
          -0.011276791F, 0.0030637227F, 0.024150789F, -0.032003794F,
          -0.045634937F, -0.014072438F, -0.08643778F, 0.015490612F, -0.06644823F,
          -0.038457196F, -0.048216876F, 0.019879974F, 0.054987565F, -0.04961185F,
          -0.024003271F, 0.011322923F, -0.04915039F, -0.053814452F, -0.06614789F,
          -0.06528912F, 0.028382903F, -0.005507826F, -0.08697642F, 0.076731645F,
          -0.046523266F, 0.0026607225F, -0.022150056F, -0.028367355F,
          -0.040047903F, 0.054646015F, 0.04535967F, -0.0769991F, 0.045622975F,
          0.022692999F, -0.027336093F, -0.050703418F, -0.046844307F,
          -0.07754508F, -0.0015348343F, 0.05901841F, -0.021963801F, 0.032271374F,
          0.12675016F, -0.060138404F, -0.026861843F, -0.04755931F, -0.13764681F,
          -0.06363793F, -0.04272185F, -0.116353504F, 0.11755929F, 0.12512729F,
          -0.042455673F, -0.0013641235F, -0.036630016F, -0.047434725F,
          0.0060127657F, -0.013080812F, -0.045902293F, -0.015610328F,
          0.023459593F, -0.021590723F, -0.014417333F, -0.032786116F,
          -0.008108132F, 0.1335297F, 0.09096735F, -0.009287196F, 0.038576923F,
          0.016013702F, -0.025795627F, -0.033063523F, -0.038384046F,
          -0.043431967F, 0.2984419F, 0.028420702F, -0.03038744F, 0.3130083F,
          0.005859783F, -0.019369122F, -0.021020485F, -0.052903548F,
          -0.051898498F, 0.066710815F, 0.039796513F, -0.037561808F, 0.016098361F,
          0.008624295F, -0.022812847F, 0.003757991F, -0.031127375F, 0.08821492F,
          -0.059687406F, -0.102195054F, 0.059904326F, -0.07377704F, -0.10780489F,
          0.059621535F, -0.05157218F, -0.0262774F, -0.056642335F, -0.01637486F,
          0.040877275F, -0.084477335F, -0.005329113F, 0.05588783F, -0.054042168F,
          0.017105076F, 0.08382498F, 0.012672973F, 0.008574107F, 0.005085715F,
          0.0050144433F, -0.0061564767F, 0.00318717F, 0.018866548F,
          -0.0014819312F, 0.011095812F, -0.028086009F, -0.0065450193F,
          0.0504992F, -0.029643929F, -0.027748939F, 0.03827837F, -0.024699666F,
          -0.0091086235F, 0.0687551F, -0.030514237F, 0.015674122F, -0.049453404F,
          -0.07320683F, -0.045870114F, -0.15222165F, -0.07292774F, -0.05364738F,
          -0.06346497F, -0.027956964F, -0.013600165F, 0.01390694F,
          -0.0019009743F, 0.0040354636F, 0.0115357265F, 0.014760646F,
          0.017320763F, 0.046339147F, 0.052110054F, -0.058932792F, 0.009229024F,
          -0.026774686F, -0.0633948F, 0.060712937F, 0.009359698F, -0.005886876F,
          0.029092439F, 0.14963041F, 0.020236285F, -0.14349353F, 0.10044758F,
          0.02900304F, -0.13659725F, 0.111762606F, 0.01311121F, -0.14019272F,
          -0.0022952463F, 0.024351625F, 0.026569067F, 0.00088588597F,
          0.004367003F, 0.02202431F, 0.02321197F, 0.0102897845F, 0.0048476094F,
          0.0047727125F, -0.017068053F, 0.025011258F, -0.0014642936F,
          -0.00998306F, 0.033915848F, 0.052832335F, 0.024727153F, 0.01508236F,
          -0.08986797F, -0.26771098F, -0.0025085092F, -0.37270823F, -0.48844364F,
          -0.10839035F, -0.18808259F, -0.27622038F, -0.026899664F, 0.036141504F,
          0.017913248F, 0.015027622F, 0.02148313F, -0.017430922F, 0.015167175F,
          0.007064066F, 0.00018037004F, 0.025653146F, -0.060121007F,
          0.008214891F, 0.050591215F, -0.048599448F, 0.017681072F, 0.034762945F,
          -0.049784794F, 0.0091134785F, 0.02532317F, -0.0070876777F,
          -0.11785884F, -0.02938606F, -0.12068885F, -0.111910805F, -0.02623418F,
          0.05136569F, -0.052213624F, 0.04425863F, 0.06475081F, 0.0049035447F,
          0.11774881F, 0.04286647F, -0.011731723F, 0.10941885F, 0.090046175F,
          0.044335242F, 0.117257595F, -0.0006315625F, -0.02247403F, 0.018886954F,
          0.018314658F, -0.023033036F, -0.003730882F, 0.016781352F,
          -0.004006526F, 0.028709067F, 0.066726185F, 0.09841207F, 0.0053002797F,
          0.09338749F, 0.110810354F, -0.027738886F, -0.00785908F, -0.030354718F,
          -0.09894069F, -0.033980213F, -0.05006619F, -0.013180662F,
          -0.043404177F, -0.06605706F, 0.0013145097F, -0.021182396F,
          -0.014360133F, 0.0075178198F, 0.0051460797F, -0.021359537F,
          0.0023210796F, -0.025204144F, -0.052511588F, -0.0055743787F,
          -0.015621271F, -0.010404012F, 0.03878567F, -0.020882089F,
          -0.047039628F, 0.013371924F, -0.04438731F, -0.05969749F, 0.034205385F,
          -0.016751578F, 0.011534069F, 0.09310964F, 0.037593093F, 0.008848526F,
          0.018890267F, 0.011544507F, -0.027183352F, 0.01487438F, 0.018640194F,
          -0.0057286113F, -0.017368F, 0.024774404F, 0.020184383F, -0.013177099F,
          0.034401093F, 0.0076134005F, -0.03024134F, -0.006544855F,
          -0.041321248F, -0.03174209F, -0.01469889F, -0.0104861315F, 0.00615873F,
          -0.012761223F, -0.024443327F, 0.0055065206F, 0.009198321F,
          -0.0015182776F, -0.0026982664F, 0.012895579F, 0.057201866F,
          0.045454722F, 0.048175134F, 0.09458669F, 0.06271557F, 0.025135353F,
          0.04878042F, -0.019263366F, -0.019310476F, -0.017194945F, -0.00749586F,
          -0.0017729872F, 0.021418475F, -0.00354287F, 0.008284798F, 0.008544624F,
          -0.033717476F, -0.026136756F, -0.0076717073F, 0.0029511817F,
          -0.017778138F, 0.015619676F, 0.037935082F, 0.0048572985F, 0.0367991F,
          0.0154406205F, 0.011196605F, 0.01580577F, -0.010174888F, 0.007363011F,
          0.011964535F, -0.015168811F, -0.012373523F, 0.004757128F,
          -0.015592993F, 0.0035427182F, -0.0040995204F, -0.008907896F,
          0.000111661575F, -0.004063107F, -0.027528362F, -0.009860339F,
          -0.029715989F, -0.028017098F, 0.028949955F, 0.015002268F, 0.014134141F,
          0.0213845F, -0.003942719F, -0.015989425F, 0.013123062F, -0.024299735F,
          -0.039435763F, -0.04362701F, -0.018941265F, -0.010559553F, -0.0218774F,
          0.006951733F, 0.003640103F, -0.020337613F, -0.00089465844F,
          -0.0191265F, -0.04508505F, -0.012944833F, 0.013113146F, -0.024144264F,
          0.034520403F, 0.051667083F, -0.0054478273F, 0.06644056F, 0.04203418F,
          -0.0017275742F, 0.0024652658F, -0.01355532F, 0.008488353F,
          0.0035072125F, -0.016936889F, -0.008816698F, -0.021747494F,
          -0.035560437F, -0.016900543F, 0.1468278F, 0.029886635F, -0.07967809F,
          0.09543435F, 0.034377854F, -0.10060358F, 0.014785506F, 0.031097686F,
          0.05166634F, 0.015963761F, 0.045023344F, 0.030492056F, -0.035120677F,
          -0.009653711F, 0.044797238F, -0.018015144F, -0.05919793F, 0.012066795F,
          0.0070725395F, -0.014486371F, 0.029053787F, -0.007955516F,
          -0.025184985F, 0.026643043F, -0.008869237F, -0.00633054F, 0.036846716F,
          0.0028533353F, -0.043184895F, 0.03549784F, -0.017920174F,
          -0.008011814F, 0.045896325F, -0.03389223F, -0.00916278F, 0.16325516F,
          -0.01930087F, -0.10609147F, 0.13534558F, -0.08578321F, 0.027830254F,
          0.1255966F, 0.052772857F, 0.016301736F, -0.006296076F, -0.060414165F,
          0.06415549F, -0.009101353F, -0.097797856F, 0.04308975F, 0.021878615F,
          -0.020229248F, 0.01440089F, -0.06543584F, -0.016841458F, 0.001659283F,
          -0.081405215F, -0.07185602F, -0.21046509F, 0.023398953F, 0.0812226F,
          -0.036576245F, -0.15342748F, 0.07502663F, 0.1769185F, -0.22869347F,
          -0.0008787594F, 0.15573911F, -0.19721828F, -0.07248402F, 0.12547536F,
          -0.02649014F, -0.0055246446F, 0.044479597F, -0.041169144F,
          -0.015256753F, 0.055603452F, 0.013615993F, -0.037037604F, 0.016267523F,
          0.012854634F, 0.003585325F, 0.030572534F, 0.0016406972F, -0.028520504F,
          0.022593426F, 0.019105406F, 0.037536185F, 0.018980578F, 0.08655212F,
          0.028110366F, 0.07525791F, 0.08368304F, -0.062384535F, 0.095035635F,
          0.04994898F, -0.051633954F, 0.056841493F, -0.031495713F, -0.013593406F,
          0.06073627F, -0.037030846F, -0.17877883F, 0.05881872F, 0.0711175F,
          -0.079289645F, 0.025175883F, 0.04317977F, -0.011340711F, -0.059347242F,
          0.058671683F, 0.011053804F, -0.061164346F, 0.06289397F, 0.026243156F,
          -0.033117454F, -0.0683204F, -0.11188583F, 0.08444875F, -0.03916565F,
          -0.18182328F, 0.060877692F, -0.006495007F, -0.059921116F,
          -0.0061317626F, -0.14704344F, -0.2694435F, 0.09263903F, -0.21095219F,
          -0.5597507F, -0.034824796F, -0.04642913F, -0.23723593F, -0.025649423F,
          -0.006810014F, 0.0032924893F, 0.017351886F, -0.015703434F,
          -0.010033485F, 0.03250498F, 0.009877122F, -0.016196242F, 0.009415422F,
          0.050616965F, 0.014638624F, 0.029080119F, 0.022410871F, 0.004571119F,
          0.027241437F, 0.038459424F, 0.04433695F, 0.066000395F, -0.008824536F,
          -0.00662779F, -0.007912261F, 0.014548542F, -0.008821652F,
          -0.0075379554F, -0.01637615F, -0.023212953F, 0.013172245F,
          0.049855486F, 0.014515168F, -0.040047795F, 0.059985F, 0.01992323F,
          -0.064854994F, 0.048569065F, -0.011634761F, -0.09029848F, -0.18966353F,
          -0.23253526F, -0.10277271F, -0.2231737F, -0.28519398F, -0.13616495F,
          -0.14127854F, -0.21286461F, -0.0803563F, -0.029594934F, -0.0046361517F,
          0.012208428F, 0.00023103441F, 0.026124602F, 0.013123011F,
          -0.023699392F, 0.023504904F, 0.02284259F, 0.021929404F, 0.015314164F,
          -0.019610075F, 0.024082279F, 0.0458985F, -0.06739151F, 0.012885204F,
          0.035963837F, -0.10032428F, 0.006078496F, -0.0022705388F,
          0.0011436787F, 0.022256635F, -0.00410035F, 0.026456987F,
          -0.0101246275F, 0.0051225717F, 0.0020995985F, 0.040423572F,
          -0.013104968F, 0.0015577133F, 8.176127E-6F, -0.05721705F,
          -0.037084542F, 0.014266105F, -0.041029643F, 0.007730129F,
          0.0006848049F, 0.01069267F, 0.0017935311F, 0.005414476F, 0.0038387706F,
          -0.014432002F, 0.027071072F, -0.0018528758F, -0.0064538163F,
          0.019341473F, -0.012056119F, 0.004308555F, 0.0055701477F,
          -0.002834965F, 0.0362798F, 0.00027918356F, 0.022073202F, -0.011065675F,
          -0.0048883995F, -0.015412585F, -0.003852075F, 0.009520393F,
          -0.014474036F, -0.040030424F, 0.015089703F, 0.0055565336F,
          -0.013142067F, -0.005822106F, 0.005636409F, 0.004464163F, 0.031679284F,
          -0.008667515F, -0.014541782F, 0.0015032444F, -0.010800909F,
          -0.023382552F, -0.011294961F, 0.013905923F, 0.03706969F, -0.025652884F,
          0.028540505F, 0.05519611F, -0.023483243F, 0.052920643F, 0.069712274F,
          0.0019388335F, -0.0029129768F, -0.0019073331F, 0.003595813F,
          -0.011351322F, -0.015540627F, -0.010971932F, 0.009133454F,
          -0.009665643F, 0.011662792F, -0.01391297F, 0.0077039846F, 0.019233257F,
          -0.026635036F, -0.0007151738F, 0.012503032F, -0.009910574F,
          -0.00919504F, -0.23295447F, -0.19555032F, 0.002368534F, -0.25131476F,
          -0.21245971F, 0.027235584F, -0.16020547F, -0.07796384F, 0.11274945F,
          0.05885135F, 0.03859252F, 0.0805183F, -0.0052695507F, -0.062476195F,
          0.0066008153F, -0.044805218F, -0.10319029F, -0.027497273F,
          0.072542764F, 0.057043213F, 0.024853837F, -0.05431729F, -0.075174555F,
          -0.07004563F, -0.042602308F, -0.023525989F, -0.04046158F,
          -0.016972639F, -0.027354648F, -0.017578755F, 0.0011154041F,
          -0.016289236F, -0.013084671F, 0.037061088F, 0.003819685F,
          0.0039908634F, -0.013330995F, -0.014560854F, 0.025050653F,
          -0.011269651F, -0.01893353F, -0.0012300422F, 0.0071166134F,
          -0.0076599456F, 0.01266459F, -0.02900343F, -0.03792275F, -0.01985059F,
          -0.041847613F, -0.014666623F, 0.027952833F, 0.053448893F, 0.063902445F,
          -0.0067766327F, 0.094649166F, 0.09701565F, 0.10154284F, -0.08278037F,
          -0.062272932F, -0.023057334F, -0.0577513F, -0.08491313F, -0.044830568F,
          0.009539572F, 0.027567323F, -0.015308771F, -0.033355176F, -0.05918083F,
          -0.0686745F, 0.047969554F, 0.06079196F, 0.052572776F, 0.111137085F,
          0.10753743F, 0.14120074F, 0.052712485F, 0.013877304F, 0.03331005F,
          -0.1105594F, -0.12158598F, -0.10559312F, 0.10591318F, 0.097691864F,
          0.05951415F, 0.026184808F, -0.003980625F, -0.03299189F, -0.116264306F,
          -0.10038252F, -0.1417354F, 0.045773838F, 0.07658171F, 0.059322305F,
          0.0070679532F, 0.037068505F, -0.036380906F, -0.097881414F,
          0.017233243F, -0.05822994F, 0.003199803F, -0.06746653F, 0.011095232F,
          -0.043982424F, -0.16528845F, -0.024072254F, -0.02401675F,
          -0.089909054F, 0.026428359F, 0.12029532F, 0.13262446F, 0.12708364F,
          -0.011437079F, -0.03177441F, -0.054558743F, -0.19015607F,
          -0.097126566F, -0.16477759F, -0.003673411F, 0.0068161064F,
          -0.0117065385F, -0.0144271385F, -0.006821402F, -0.029056735F,
          -0.010520757F, 0.02008394F, -0.0027167252F, 0.013741979F, -0.1771189F,
          0.0009919688F, 0.0374869F, -0.1695466F, 0.021418618F, 0.081639536F,
          -0.0801222F, 0.07052912F, 0.016364604F, -0.09122003F, -0.014763024F,
          -0.019896215F, -0.121380344F, -0.024969393F, 0.006572616F,
          -0.07258837F, 0.021491287F, 0.036085837F, 0.042564627F, 0.029705444F,
          -0.007967446F, -0.0071723354F, -0.018960299F, -0.030183714F,
          -0.002657584F, -0.013977979F, -0.08791526F, 0.00040166767F,
          0.06792502F, 0.07663609F, 0.1291406F, -0.080058716F, -0.008424154F,
          -0.043491F, 0.024005257F, -0.023407193F, 0.011087075F, -0.043978117F,
          9.749009E-5F, -0.05972071F, 0.022244452F, 0.05041925F, 0.103154905F,
          -0.030977052F, 0.004697827F, -0.007861606F, 0.026114427F,
          -0.017334402F, -0.009646014F, 0.01755222F, -0.020264795F,
          0.0006730932F, 0.020924961F, -0.014380704F, -0.029798582F,
          0.013259637F, -0.014453405F, -0.006462522F, -0.0029481992F,
          0.013880012F, -0.0009797576F, 0.012943818F, -0.01648479F, 0.15938054F,
          0.15294784F, 0.14807165F, -0.1889246F, -0.22340924F, -0.21909352F,
          -0.23937245F, -0.07213301F, -0.068083026F, -0.06511391F, 0.06779751F,
          0.0047821673F, 0.086804524F, 0.002345506F, 0.05195545F, 0.0045752805F,
          -0.069955684F, -0.094415694F, -0.062283453F, 0.014411998F,
          -0.14902577F, -0.23493747F, -0.029317634F, 0.088727534F, 0.11535422F,
          -0.050284933F, -0.059102472F, -0.13862297F, -0.015653301F, -0.0740969F,
          0.06581321F, 0.1349342F, 0.14157467F, 0.04096801F, -0.07202532F,
          0.004558159F, -0.23221412F, 0.055055555F, -0.024058465F, 0.09231225F,
          -0.040289883F, 0.017180994F, -0.15511277F, 0.051094323F, 0.057103623F,
          -0.01601974F, -0.18569396F, -0.40853032F, 0.18516707F, 0.12875058F,
          0.18241982F, 0.10842949F, -0.23621884F, 0.030314615F, 0.065173045F,
          0.17537677F, 0.22545409F, -0.025355631F, -0.36866382F, -0.22392632F,
          -0.1828808F, 0.07004629F, -0.047186747F, -0.050581533F, 0.13308015F,
          0.08705875F, 0.17076987F, -0.4768016F, -0.22841991F, 0.03193005F,
          0.085966915F, 0.033317313F, -0.021696396F, -0.007801946F, -0.01295134F,
          -0.015546398F, -0.023579068F, -0.04271967F, -0.012436145F,
          0.058135524F, -0.05947782F, -0.19518629F, -0.03128639F, -0.096983835F,
          0.09475137F, -0.13511924F, 0.013926932F, -0.15305355F, -0.053753447F,
          0.06134827F, -0.20978883F, -0.3164741F, -0.39273173F, -0.066814095F,
          0.41916415F, 0.31268945F, 0.2518123F, -0.10408946F, -0.008076739F,
          -0.010141032F, 0.033131868F, 0.00969652F, 0.0033572195F, -0.03424059F,
          -0.0056973263F, -0.004664691F, 0.017972859F, -0.033445556F,
          -0.13111357F, 0.004107412F, 0.117104754F, 0.0016332548F, -0.030574929F,
          0.10470285F, 0.08708792F, 0.053427335F, 0.013881851F, 0.07839816F,
          0.10811975F, -0.033495713F, 0.019431693F, 0.051576473F, -0.044373877F,
          -0.15028052F, -0.091429584F, 0.031290438F, 0.019042036F, -0.007079917F,
          0.035302322F, 0.034673605F, 0.018508686F, -0.017847322F, 0.008763441F,
          0.012082404F, 0.019482153F, -0.009357351F, 0.00662322F, 0.03452286F,
          -0.004354577F, 0.024212698F, -0.024611134F, -0.037218127F,
          -0.031250365F, -0.07885131F, -0.07662887F, -0.006090651F, -0.11470997F,
          -0.13623257F, -0.16395572F, -0.05432901F, -0.13327524F, -0.08862789F,
          -0.010934682F, 0.047943257F, 0.060178783F, -0.07441912F, -0.069468156F,
          -0.0059772814F, 0.023794856F, -0.067378335F, -0.00066841044F,
          -0.013383239F, -0.040230345F, -0.03317869F, -0.052250363F,
          -0.07401949F, -0.032126512F, -0.037195597F, -0.08646819F, 0.01025824F,
          -0.043895386F, -0.18392192F, -0.119223595F, 0.122038595F, 0.027798714F,
          -0.08727664F, 0.17553277F, 0.18924259F, 0.075338095F, -0.069485486F,
          -0.005275987F, 0.054676462F, -0.14129463F, -0.14940383F, 0.005512422F,
          -0.03629266F, -0.14840256F, -0.039807025F, 0.004031179F, -0.029192192F,
          0.09238895F, -0.28275794F, -0.5167497F, -0.08508242F, -0.038025975F,
          -0.46141315F, -0.27345008F, -0.019805644F, 0.031087596F, 0.12960464F,
          -0.05157113F, -0.20314358F, 0.015294256F, 0.12577522F, -0.12982155F,
          -0.00513594F, 0.048140045F, 0.07047777F, -0.004811573F, -0.03529742F,
          0.006171516F, 0.049237464F, -0.025701722F, 0.038708523F,
          -0.0028213016F, 0.03410545F, 0.08376409F, 0.034192443F, -0.051267527F,
          0.02905223F, 0.040314265F, -0.08094115F, -0.04247805F, -0.019253414F,
          0.0028270318F, -0.012137062F, 0.0536802F, -0.030845918F, -0.08569461F,
          -0.005477897F, 0.090620786F, 0.0053733583F, -0.030929184F, 0.08089884F,
          0.08175443F, 0.048292555F, 0.098119564F, 0.12022301F, 0.11765566F,
          0.093340576F, 0.11052079F, 0.1135011F, -0.018630581F, 0.011392457F,
          0.03692079F, -0.021925934F, -0.04184694F, 0.010359301F, 0.082498945F,
          0.006841286F, 0.00087102584F, -0.053434663F, 0.017740639F,
          0.036715556F, 0.042682745F, 0.06378977F, 0.009671687F, 0.071497895F,
          0.081669845F, -0.03617033F, 0.11767637F, 0.041281044F, -0.0278825F,
          0.049661342F, 0.009146379F, -0.03761803F, -0.020597244F, -0.09046855F,
          0.007985358F, 0.03431229F, 0.013118473F, -0.009851684F, 0.014862054F,
          -0.029050356F, -0.04333488F, 0.0025049148F, -0.016465789F,
          -0.022905033F, 0.059717566F, 0.024593018F, 0.012810207F, 0.016593613F,
          -0.0180274F, -0.029620212F, 0.021919543F, -0.00765235F, -0.04322417F,
          0.025803648F, 0.080565356F, -0.015380153F, 0.08187836F, -0.0058651343F,
          0.041637182F, -0.025670996F, -0.042394977F, -0.12499988F, 0.02504918F,
          -0.0427704F, 0.0013101434F, -0.03003886F, -0.040778838F, 0.042309888F,
          0.020095982F, 0.04957059F, 0.049016554F, 0.055569667F, -0.10435389F,
          -0.13834272F, -0.10687787F, -0.1557197F, -0.10492739F, -0.100460865F,
          -0.052822802F, -0.008820213F, -0.16382581F, -0.11622552F,
          -0.027050212F, -0.11195946F, 0.045672923F, 0.091007784F, -0.01815902F,
          0.06905892F, 0.09107461F, -0.016709086F, -0.16012342F, -0.030736092F,
          -0.1830346F, -0.18016969F, 0.05702386F, -0.001079454F, -0.002653834F,
          0.045240495F, 0.049544144F, -0.37951112F, -0.21287487F, -0.2894928F,
          -0.6567748F, -0.025944266F, -0.12609188F, -0.075260594F, 0.10543581F,
          0.00020257077F, 0.07388247F, 0.06761902F, 0.064217456F, 0.04178681F,
          0.04144657F, 0.0755102F, 0.036510706F, 0.0028796725F, 0.035976183F,
          0.057806212F, 0.0030653733F, 0.030165719F, -0.028014498F,
          -0.011680479F, 0.06837321F, 0.0012227506F, -0.009762209F, 0.052622285F,
          0.049569424F, 0.013328756F, 0.015864108F, 0.008680662F, -0.029710094F,
          -0.03957127F, -0.07305122F, -0.07059134F, 0.07226619F, -0.1426004F,
          0.049966443F, -0.08485471F, -0.07649337F, 0.033660617F, -0.009021514F,
          0.14302881F, 0.10587668F, 0.097088404F, -0.12991582F, -0.08999282F,
          -0.1002126F, -0.39292184F, -0.017471628F, -0.08269954F, 0.037442707F,
          0.14042622F, -0.009207592F, -0.011585114F, -0.0013610092F,
          -0.004985664F, -0.0011167418F, 0.023080328F, 0.009153156F,
          -0.005079273F, 0.013276173F, -0.031549826F, 0.14684394F,
          -0.0011723088F, -0.002741861F, -0.014269794F, 0.047762968F,
          0.04606693F, -0.192855F, 0.033300906F, 0.051900573F, 0.042162064F,
          0.015700573F, -0.06391488F, 0.112957716F, 0.009399044F, -0.09925412F,
          0.033845212F, 0.0178436F, -0.0008765665F, -0.010739288F, -0.06649674F,
          0.016712485F, 0.014255676F, -0.039353605F, -0.003913909F, 0.026940979F,
          0.0029741337F, 0.02576294F, -0.0029437859F, -0.08312708F, 0.029933061F,
          0.053969987F, -0.050715182F, -0.012516226F, 0.025954187F,
          0.0066132336F, 0.0012782409F, 0.09664009F, -0.07265063F, -0.22244215F,
          0.07462106F, 0.110409886F, -0.16184033F, -0.1343133F, 0.04643562F,
          -0.0068891593F, 0.04542778F, -0.006478744F, -0.016157964F,
          -0.04531849F, 0.04358294F, -0.00928601F, -0.023346424F, -0.0068559176F,
          -0.01202249F, -0.2220112F, 0.07357462F, -0.016798131F, -0.10852458F,
          -0.21582252F, 0.047832377F, -0.0950349F, -0.07296982F, 0.00094056857F,
          0.08172039F, 0.078208655F, 0.045160316F, -0.23950799F, 0.16804251F,
          0.044688173F, -0.122983426F, 0.02020569F, -0.051522106F, 0.13965702F,
          -0.11874431F, -0.0774327F, 0.03642212F, 0.039237354F, -0.04747361F,
          -0.10064844F, 0.026724491F, -0.2503613F, 0.23235734F, -0.112405054F,
          -0.23049672F, 0.11800537F, -0.034605276F, -0.06280745F, -0.276322F,
          0.17209403F, -0.099307306F, 0.15630226F, -0.07767426F, -0.22494695F,
          0.30571494F, -0.061677396F, 0.05285523F, -0.19680254F, 0.15890643F,
          0.11319838F, -0.3060538F, 0.0983252F, 0.17390683F, -0.3358721F,
          -0.00607386F, -0.01409801F, 0.03913924F, -0.1807727F, -0.03287048F,
          0.04523166F, -0.020977907F, -0.023820505F, 0.0015747466F,
          0.0059432224F, 0.015926776F, -0.024912199F, 0.013892518F,
          0.0075452826F, 0.07512232F, -0.052733127F, 0.1291131F, -0.26042393F,
          0.1684253F, 0.028809598F, -0.12493246F, 0.049198784F, 0.12615287F,
          -0.32211328F, -0.01166139F, 0.13504015F, -0.10509854F, 0.06424143F,
          0.05141002F, 0.24880685F, -0.29116985F, -0.020133473F, 0.09261909F,
          -0.038486872F, -0.07330369F, 0.041304406F, 0.03744968F, -0.012182226F,
          -0.06676914F, 0.048149824F, -0.022387806F, -0.07396128F, -0.022488562F,
          0.06761657F, 0.028935418F, 0.068982475F, 0.021934452F, -0.034312747F,
          0.022553328F, 0.016410932F, 0.05535885F, 0.01746022F, -0.007728229F,
          0.026365852F, -0.012596335F, -0.006250799F, 0.020973513F,
          -0.028611403F, 0.021623127F, 0.01313785F, 0.029156718F, -0.011476599F,
          -0.013174221F, 0.013790378F, -0.041428696F, -0.034025297F,
          -8.413361E-5F, 0.037756607F, 0.036413856F, 0.02083526F, -0.022093598F,
          -0.018649494F, -0.007536643F, -0.029447902F, -0.015090313F,
          0.0018010016F, 0.005687648F, 0.031848628F, -0.052646734F,
          -0.026664933F, -0.011914799F, -0.036735844F, -0.08788248F,
          -0.08385587F, -0.06380932F, -0.045616876F, -0.06428343F, -0.062852286F,
          0.05985424F, 0.02983666F, 0.011045215F, 0.021933148F, 0.023098495F,
          0.03970114F, 0.0011128805F, -0.022043074F, -0.04006456F, -0.038678713F,
          -0.0030358678F, -0.013860065F, 0.021283906F, 0.030227913F,
          0.008907244F, -0.10063628F, -0.105952896F, -0.07783096F, 0.046321884F,
          0.038416002F, 0.070259556F, 0.029154146F, 0.02777539F, 0.07464375F,
          -0.0779307F, -0.0063520307F, -0.0380518F, -0.014307934F, 0.040577088F,
          -0.025550956F, 0.0014854678F, 0.0495549F, 0.020996956F, -0.076645955F,
          -0.0073300796F, 0.0291162F, -0.3021957F, -0.38520828F, -0.3319697F,
          0.16444944F, 0.2745715F, 0.19454284F, 0.020054614F, 0.05848324F,
          -0.0011967593F, -0.06337634F, -0.035911944F, -0.02202688F,
          -0.07421584F, -0.025875924F, -0.042313084F, -0.045983545F,
          0.015238301F, -0.004296635F, -0.041034732F, -0.003315319F,
          -0.04014749F, -0.03861289F, 0.0116622F, 0.0006961524F, -0.0014634385F,
          0.0070408913F, 0.028941974F, -0.02336872F, -0.033044416F, -0.01905048F,
          0.0094646225F, -0.0044646543F, 0.0040042964F, 0.07409605F,
          -0.08569493F, 0.06507623F, 0.053178493F, -0.17596194F, 0.051137824F,
          -0.01113466F, -0.23847033F, 0.037542515F, -0.00244963F, 0.019924257F,
          0.04585011F, 0.028067552F, 0.01478499F, -0.021565968F, -0.04494244F,
          -0.07095492F, -0.068644755F, -0.025146158F, -0.0047769872F,
          -0.005912147F, -0.015811829F, -0.017560545F, -0.02761698F,
          0.031791605F, 0.03799621F, 0.02966931F, 0.016804757F, 0.09629187F,
          0.02238852F, 0.0001625969F, 0.091562346F, 0.021582864F, 0.040107053F,
          0.08537307F, 0.05893704F, 0.03356645F, -0.05980945F, 0.05221414F,
          0.011497267F, -0.06696635F, 0.019689055F, -0.005635872F, -0.061480816F,
          0.027116274F, 0.015957877F, -0.0070062205F, -0.024541726F,
          0.029892113F, -0.0052223466F, -0.018280528F, 0.06966379F, 0.016819265F,
          0.008304127F, -0.03574137F, 0.025188962F, -0.03966343F, -0.013457689F,
          0.019535558F, -0.043582615F, 0.034719065F, 0.04071694F, 0.00024157342F,
          -0.0064766877F, -0.029873488F, -0.020953054F, -0.055092927F,
          -0.09013928F, -0.09562309F, -0.05940952F, -0.030552914F, -0.054323036F,
          0.007827525F, -0.09559777F, 0.06504558F, 0.009257005F, -0.093420915F,
          0.032556497F, -0.012843374F, -0.085976616F, 0.029508017F,
          0.0020842638F, -0.09868867F, -0.04359668F, -0.019078907F, -0.10725993F,
          -0.08755688F, 0.019974194F, -0.083658315F, -0.052123975F, 0.019518372F,
          -0.051651996F, -0.008186874F, -0.0049314722F, -0.08371709F,
          -0.030177878F, 0.0052475096F, -0.07289305F, -0.022820344F,
          -0.042374037F, -0.18383406F, 0.047649343F, -0.055576917F, -0.25485903F,
          0.0707245F, -0.07506239F, -0.21800676F, 0.06809934F, -0.01937128F,
          -0.2698802F, -0.023194917F, -0.028226485F, -0.27818382F, -0.06401777F,
          -0.00011543413F, -0.24593131F, -0.027165813F, 0.044561345F,
          0.09958656F, 0.07415072F, -0.004043165F, 0.083539166F, 0.034145236F,
          0.024838619F, 0.12206108F, 0.083807886F, -0.021106321F, -0.22707543F,
          0.009526389F, 0.020186534F, -0.27449545F, -0.050937742F, -0.036505163F,
          -0.24415873F, -0.0006504838F, 0.036802147F, -0.019133285F,
          0.015491313F, 0.033765785F, -0.016678743F, 0.0135944765F, 0.010563457F,
          -0.024161467F, 0.012117959F, 0.07925987F, 0.10217524F, 0.08783004F,
          0.04489317F, 0.074647285F, 0.046839483F, 0.07900291F, 0.076467454F,
          0.08659563F, 0.037274558F, 0.10446777F, 0.08033032F, 0.048847366F,
          0.08780393F, 0.069409184F, 0.073911205F, 0.07358073F, 0.077741265F,
          0.023252985F, -0.00070034404F, 0.02339607F, 0.0070418604F,
          -0.004556952F, 0.014147379F, -0.013109865F, -0.022606937F,
          0.009554589F, 0.040371057F, 0.057885997F, 0.049106266F, -0.049219046F,
          -0.07782143F, -0.07563077F, 0.071401276F, 0.07271332F, 0.023728915F,
          -0.017947575F, -0.042582486F, -0.0427115F, -0.013110164F,
          -0.031710643F, -0.017604263F, 0.041899238F, 0.028490359F, 0.052766655F,
          -0.008514565F, -0.012852242F, -0.011195261F, -0.02223033F,
          -0.018352397F, -0.008219783F, 0.005360606F, 0.017824013F, 0.022506565F,
          -0.015360263F, -0.017921526F, -0.02076896F, -0.023575678F,
          -0.012681943F, -0.00042230735F, 0.0054735164F, 0.027753165F,
          0.037760288F, -0.013418449F, -0.01636331F, 0.06882833F, -0.012711828F,
          -0.09762432F, -0.06880759F, -0.04550582F, -0.144612F, -0.07888457F,
          -0.032745536F, -0.036133092F, -0.022905942F, 0.034964453F, 0.01874056F,
          0.019317033F, -0.0010842661F, 0.008371394F, -0.0010071506F,
          0.011061782F, 0.03147521F, 0.0077277725F, -0.040995736F, -0.09111513F,
          -0.08937779F, -0.17284901F, -0.2350712F, -0.2601261F, 0.059737176F,
          0.07811228F, 0.076642044F, -0.029780569F, -0.06821684F, -0.07741802F,
          -0.008653251F, -0.033887345F, -0.06003306F, -0.04183229F,
          -0.0005124794F, -0.030234031F, -0.041065782F, 0.036822733F,
          -0.021449856F, -0.006720088F, 0.034395356F, -0.0047556455F,
          -0.040941685F, 0.059779778F, -0.026712166F, -0.02697289F, 0.041801095F,
          -0.027526028F, -0.045376465F, 0.0101573365F, 0.014315378F,
          0.052629806F, -0.064329855F, 0.046170868F, 0.07194779F, -0.07066808F,
          -0.0074618487F, 0.0889077F, -0.091689244F, -0.0046380353F,
          -0.053740736F, 0.030103002F, -0.0078550745F, -0.36829326F,
          -0.39106628F, -0.32113892F, 0.15389808F, 0.3630814F, 0.20330034F,
          -0.00047940185F, 0.024183825F, -0.0048949537F, -0.0008461409F,
          0.0069629247F, -0.019345587F, 0.001808189F, 0.004226397F, -0.0263515F,
          -0.010336004F, 0.020502757F, -0.023078497F, -0.08655109F, -0.08706893F,
          -0.11233091F, -0.02697258F, 0.013659306F, -0.03716088F, 0.05248474F,
          -0.041405387F, 0.0074045127F, 0.006815083F, -0.046221565F,
          0.081798024F, 0.0068069994F, -0.08558804F, 0.070355006F, 0.02855666F,
          0.052565135F, 0.027417345F, -0.020314675F, -0.00806176F, -0.013365115F,
          -0.016419636F, -0.011537046F, -0.02557206F, -0.001455021F,
          0.016992996F, 0.048171323F, 0.017410723F, 0.052460406F, 0.062978156F,
          0.023319032F, 0.028820913F, -0.012799252F, 0.0147231F, 0.011468698F,
          0.045000788F, 0.044789106F, 0.053131018F, 0.08361051F, 0.0699779F,
          0.0813957F, 0.09614109F, -0.059784684F, -0.09354576F, 0.0027005377F,
          -0.10970632F, -0.041194677F, 0.067040116F, -0.029743796F, 0.07940178F,
          0.107928775F, -0.111765F, -0.12126803F, -0.0088167945F, -0.14116673F,
          -0.012366646F, 0.057331696F, -0.0031844846F, 0.10527235F, 0.05598414F,
          -0.02122496F, 0.006102528F, -0.0050923373F, -0.017493946F,
          -0.044575166F, -0.030685028F, -0.015198803F, -0.024729028F,
          -0.03866091F, 0.14965202F, -0.028341834F, -0.09042421F, 0.029791877F,
          -0.3416995F, 0.07196141F, -0.17442654F, 0.13551219F, 0.1843738F,
          0.017171951F, -0.024927529F, 0.009469234F, 0.012805438F, -0.0392906F,
          0.0011650184F, -0.003437344F, -0.00968664F, -0.0005316264F,
          -0.045792535F, -0.0115309F, 0.026699511F, 0.009378019F, 0.084917486F,
          0.10396837F, 0.032130595F, 0.08423484F, 0.07589891F, 0.017481888F,
          0.008471531F, 0.008274746F, 0.011557036F, 0.036719754F, 0.015681118F,
          0.0015002443F, 0.012317039F, 0.022188278F, -0.0063109933F,
          -0.033554938F, -0.007143473F, -0.018692603F, -0.041347407F,
          -0.006954095F, 0.007310939F, -0.030499985F, -0.029870579F,
          -0.021603404F, 0.0063882275F, 0.0018063841F, -0.016134733F,
          0.03207816F, -0.00530631F, 0.016726917F, -0.0051149647F, 0.0052808546F,
          -0.0034223995F, -0.008528895F, -0.010152355F, -0.019634876F,
          0.012057919F, -0.023494733F, 0.03439974F, -0.022355359F, -0.024942923F,
          0.09900374F, 0.10613556F, -0.015040263F, 0.088942416F, -0.063701995F,
          -0.12736273F, -0.02960587F, -0.16413496F, -0.041361436F, 0.0019493342F,
          0.0060042986F, -0.011653526F, -0.0062610973F, 0.005964189F,
          -0.008510898F, 0.0119852545F, 0.011813268F, 0.0003938919F,
          0.016409727F, -0.012840595F, 0.0077326447F, -0.00568826F,
          -0.0072678956F, 0.0075024893F, 0.012263305F, 0.01024302F, 0.022969248F,
          -0.010222047F, 0.036804788F, -0.012716515F, 0.03607745F, 0.041960064F,
          -0.0051311753F, -0.003822737F, -0.033387583F, -0.019129962F };

        static const float biasReformatted[32]{ 0.29696274F, 1.6026102F,
          0.026254892F, 0.073631525F, -0.2887566F, 0.19936985F, 4.2163815F,
          4.497778F, 0.37257576F, 0.2741859F, 1.0851988F, 0.42751944F,
          -1.8824761F, 0.86933494F, 2.4487655F, 1.7714332F, 0.3653438F,
          2.7911277F, 0.43311F, 1.9982179F, 0.78859246F, 3.503273F, 2.571065F,
          1.6973145F, 0.54011154F, 0.511622F, 2.6674256F, 1.31284F, 0.9849243F,
          1.6609602F, -0.33782935F, 1.4166672F };

        b_convolution(X[0], Z[0], reformattedAndTruncatedWeights[0],
                      biasReformatted[0]);
      }

      void c_conv2dDirectOptimizedColMajor(const float X[346112], float Z[692224])
      {
        static const float reformattedAndTruncatedWeights[18432]{ 0.08141631F,
          0.07700906F, 0.05417548F, 0.045497432F, 0.10337583F, 0.14116223F,
          -0.11739202F, -0.16274463F, -0.13900934F, -0.09027247F, 0.016471045F,
          -0.05824771F, 0.02823778F, 0.09898015F, 0.007288875F, 0.035912152F,
          -0.016372776F, -0.13838485F, -0.039769337F, -0.106651396F,
          -0.079532325F, 0.03470487F, 0.07606033F, 0.05354048F, 0.099067986F,
          -0.015033261F, 0.07323391F, -0.027463445F, 0.04789312F, 0.052654218F,
          -0.098426476F, -0.18072505F, -0.085806616F, -0.049526848F,
          -0.051607005F, 0.03956834F, 0.0049539055F, 0.0353246F, 0.011823655F,
          0.047471672F, 0.047343664F, -0.030530341F, -0.053410064F,
          -0.027825827F, 0.008717037F, -0.11120251F, -0.026890133F,
          -0.045806073F, -0.0946119F, -0.029270245F, -0.00892754F, 0.05854569F,
          0.09216966F, -0.027134137F, 0.2412613F, 0.28619125F, 0.38809398F,
          0.2766396F, 0.2162443F, 0.06383916F, -0.21727644F, -0.35163477F,
          -0.31396273F, 0.017131714F, -0.012332301F, 0.03583688F, 0.03065639F,
          0.001461162F, 0.06114705F, -0.07097946F, -0.1060827F, -0.008300951F,
          -0.038676072F, -0.011472302F, 0.0319168F, 0.002449689F, 0.041639186F,
          0.022773027F, -0.036355056F, -0.067690216F, -0.16671772F,
          -0.0031596434F, -0.07179571F, -0.1531601F, -0.10320735F, -0.025591047F,
          -0.042507354F, 0.033424728F, 0.083340734F, 0.045122832F, -0.011911106F,
          -0.015305665F, 0.03554948F, 0.02849658F, 0.08294713F, 0.044303667F,
          -0.010746539F, -0.00081358856F, -0.022216193F, -0.07076181F,
          -0.04460983F, -0.045802206F, -0.06002946F, 0.00043352248F,
          0.051988598F, -0.03641714F, -0.03984978F, -0.0043201894F, -0.14109907F,
          -0.17034602F, -0.04504815F, -0.008861322F, -0.07207465F, -0.001982685F,
          -0.0019441607F, -0.14926417F, -0.06773344F, -0.04136329F, 0.006116441F,
          0.054880347F, -0.07310537F, -0.19061464F, -0.06977887F, -0.010787857F,
          -0.023449684F, 0.10749951F, -0.07081205F, -0.05546295F, -0.05509601F,
          -0.008162308F, 0.005265488F, -0.020282835F, 0.05076807F,
          -0.0029240332F, 0.03291437F, -0.07308369F, -0.01370386F, 0.056479704F,
          0.11342847F, 0.047013268F, 0.02141979F, 0.036744162F, -0.07188436F,
          -0.0017053115F, 0.04362376F, 0.032194607F, 0.062696934F, 0.1010504F,
          0.108414456F, 0.0053134016F, -0.055936646F, -0.12249583F, -0.05526622F,
          0.006659673F, -0.06405099F, -0.07902622F, -0.18572098F, -0.23868291F,
          -0.0656794F, 0.093443535F, 0.058325037F, 0.13940722F, -0.011016245F,
          -0.015705867F, -0.0058316835F, -0.08381444F, -0.05879861F, 0.03599675F,
          -0.036577318F, 0.041726984F, 0.0041452735F, -0.0039042586F,
          0.032178298F, 0.032765828F, -0.01599077F, -0.0039096856F, 0.007819878F,
          -0.048452396F, -0.076227166F, -0.038002204F, -0.08083687F,
          -0.12893216F, -0.12551774F, -0.023919104F, -0.0509912F, 0.016905872F,
          -0.034558512F, 0.0032531053F, 0.059995055F, 0.0051954533F, 0.0192943F,
          -0.014007364F, 0.02212652F, 0.032048013F, 0.0040805624F, -0.094357535F,
          -0.10598296F, -0.051619977F, 0.045549035F, 0.077174395F, 0.04339226F,
          0.06627404F, 0.10648954F, 0.031765953F, -0.071809225F, -0.1575349F,
          -0.16199647F, 0.041362558F, 0.006148838F, -0.008825168F, -0.081728995F,
          -0.06611972F, -0.125169F, 0.00781542F, 0.0957047F, 0.07095422F,
          -0.050936807F, -0.16635756F, -0.18039966F, -0.14420739F,
          -0.0069363853F, 0.107893504F, 0.12462067F, 0.079261504F, 0.04643579F,
          0.010869563F, -0.022825593F, -0.037341177F, 0.016306158F,
          -0.031520683F, -0.032759067F, -0.040382467F, 0.0076674446F,
          0.07561334F, -0.035574116F, -0.04084537F, -0.04405724F, -0.016189758F,
          -0.0015839632F, 0.08068613F, 0.07442864F, 0.06734922F, 0.10451045F,
          0.047050595F, 0.028679572F, 0.03874643F, -0.04750845F, -0.07854906F,
          -0.050235093F, -0.040860943F, -0.004651951F, -0.020863913F,
          -0.13346507F, -0.16258451F, -0.22967082F, 0.031772573F, -0.0031213132F,
          0.012408375F, 0.031728808F, 0.022169365F, 0.07513493F, 0.083531246F,
          0.14991038F, 0.23816212F, 0.25931665F, 0.277921F, 0.039815173F,
          -0.38361752F, -0.6188045F, -0.4083395F, -0.09087595F, -0.1698291F,
          -0.21578173F, -0.09059163F, -0.032235645F, -0.021681122F, 0.078513585F,
          0.011001856F, 0.020493526F, -0.23647271F, -0.2713273F, -0.216807F,
          -0.041156325F, 0.02307164F, 0.093590185F, 0.22489089F, 0.21628715F,
          0.0981423F, -0.0075079314F, 0.007369955F, 0.010237536F, -0.09224075F,
          -0.070451744F, -0.044048294F, -0.017031934F, 0.025948038F,
          0.034604516F, -0.02259239F, 0.02488301F, 0.010035944F, -0.062751494F,
          0.0006600487F, -0.025001984F, -0.022605285F, -0.0008617079F,
          -0.022588333F, 0.038393877F, -0.101118945F, -0.058665033F, 0.04469904F,
          -0.08047813F, -0.06637031F, 0.06331778F, -0.04706613F, -0.008488575F,
          -0.07885088F, 0.013704714F, 0.028465932F, -0.14017375F, -0.021494152F,
          0.055535145F, -0.09056854F, 0.0114019F, 0.027327172F, -0.074977264F,
          0.05905018F, 0.052839648F, -0.11015256F, 0.06247506F, 0.05782586F,
          -0.13366473F, 0.02079035F, 0.05577885F, -0.014844418F, 0.008660113F,
          0.025278494F, -0.0764656F, -0.042017F, -0.0056029097F, 0.007168302F,
          0.035073537F, 0.021134194F, 0.031283338F, 0.053746983F, 0.005111603F,
          0.039318983F, 0.059026327F, 0.06303648F, 0.00793262F, 0.05682559F,
          -0.009851668F, -0.008708505F, -0.0028517202F, 0.00340697F,
          -0.00048958726F, 0.008858539F, -0.0124213565F, 0.0076745274F,
          0.014096452F, 0.013676529F, 0.03439916F, -0.0682319F, -0.10097075F,
          0.013308691F, -0.032768015F, -0.09406549F, 0.024580412F,
          -0.0025590488F, -0.069416165F, -0.012274875F, 0.018467866F,
          0.006139495F, -0.026522573F, -0.026008321F, -0.059707835F,
          -0.014435643F, -0.0064423317F, -0.0005571656F, 0.0018887195F,
          -0.061252087F, 0.021589562F, 0.030064968F, -0.10783357F, 0.0071899765F,
          0.048376523F, -0.078349866F, 0.020223858F, -0.035727434F, 0.04115568F,
          -0.039091405F, -0.11725302F, 0.0008032478F, -0.062224206F,
          -0.070480965F, -0.01089459F, -0.05981501F, 0.035241567F, -0.042860005F,
          -0.026161175F, -0.05063061F, -0.07807633F, 0.004615594F, -0.062083438F,
          -0.029275773F, -0.0013076586F, 0.030705253F, -0.017134171F,
          -0.0017208485F, -0.033676717F, -0.047457624F, -0.035064396F,
          0.0350973F, 0.012228757F, 0.0034588585F, 0.0058285203F, -0.094681785F,
          0.029337835F, 0.0513452F, -0.03114223F, -0.016802639F, 0.120121166F,
          0.039726656F, -0.057196617F, -0.030147849F, -0.0123339845F,
          0.025995975F, -0.013906742F, -0.052362848F, -0.00090973F, 0.02715882F,
          -0.038819492F, 0.009309018F, 0.055571627F, -0.027769959F,
          -0.00050469214F, 0.0871258F, 0.033558745F, 0.0097093275F,
          -0.017622085F, -0.014465814F, 0.017007837F, -0.050114524F, 0.02294333F,
          0.03959035F, -0.04171055F, -0.013026728F, 0.017294498F, 0.020760525F,
          0.024143266F, -0.0043089036F, -0.04236281F, 0.030324936F, 0.014967182F,
          -0.1461161F, 0.01252816F, 0.0030335118F, -0.037076604F, 0.049705546F,
          0.03500338F, 0.2840986F, -0.115609154F, -0.19131778F, 0.39278924F,
          -0.09396588F, -0.22063737F, 0.26107344F, -0.14767802F, -0.13624625F,
          -0.056669015F, -0.0090511525F, 0.062808864F, -0.09916202F,
          -0.05023672F, 0.026195027F, -0.15818678F, -0.050779562F, 0.049619112F,
          -0.33092126F, 0.07991331F, 0.14566907F, -0.33437735F, 0.21710414F,
          0.14081204F, -0.24906512F, 0.11798401F, 0.091346F, 0.0128443F,
          0.01931802F, -0.028131805F, 0.0114578735F, 0.012730085F, -0.03037587F,
          0.024598805F, 0.012636719F, 0.0013061289F, -0.0146360975F,
          0.024373906F, -0.00048833713F, -0.070079975F, -0.051040668F,
          0.003361007F, -0.021891525F, 0.00043761448F, -0.025800472F,
          0.00040311544F, 0.020824071F, 0.04548407F, -0.06252546F, -0.051176675F,
          0.03814569F, -0.024663795F, 0.031307828F, 0.025536524F, 0.20786774F,
          -0.045991257F, -0.08904795F, 0.15293777F, -0.10026268F, -0.029719116F,
          0.025320137F, -0.08147616F, 0.018274814F, -0.06687035F, 0.15323706F,
          -0.007024208F, -0.13731027F, 0.13677731F, -0.009154184F, -0.07636745F,
          0.06452352F, 0.020356417F, -0.02730651F, -0.013810648F, -0.06804604F,
          -0.008714812F, -0.0056855995F, -0.10464009F, 0.008677521F, 0.01748892F,
          -0.10428714F, -0.073974855F, 0.0006772059F, -0.013170161F,
          -0.07422764F, -0.05819916F, -0.02745813F, -0.04492756F, 0.008693612F,
          -0.037139323F, -0.034084924F, -0.0524407F, -0.04629365F, 0.10704086F,
          0.0568343F, 0.03674841F, -0.068897724F, -0.11012679F, -0.07148658F,
          0.013425496F, -0.036861934F, 0.019381637F, 0.027572038F, 0.009018893F,
          0.084495366F, 0.017967332F, 0.0013321217F, 0.0024882273F, 0.022912042F,
          -0.012507519F, 0.00034895306F, 0.0030095086F, -0.017898304F,
          -0.01607257F, 0.020504596F, 0.005631669F, -0.0011359435F,
          0.0052586733F, -0.03128573F, 0.068788856F, 0.046497088F, -0.035004843F,
          0.020933576F, 0.04330013F, 0.022177253F, -0.0031573295F, -0.042744555F,
          -0.019577354F, 0.005147358F, -0.01121604F, -0.012314829F,
          -0.040235057F, 0.019491138F, 0.061923794F, 0.032688044F, 0.007856688F,
          0.01481053F, 0.060656216F, -0.1553941F, -0.177806F, -0.041596133F,
          -0.089297675F, -0.1986682F, -0.037339825F, 0.10591879F, 0.13084486F,
          0.05701364F, -0.119045876F, -0.09164115F, 0.017653378F, -0.09229368F,
          -0.090052634F, 0.03729166F, 0.06713835F, -0.048482485F, -0.03442994F,
          0.06814498F, -0.0065839947F, -0.09212646F, 0.028017394F, 0.12664756F,
          0.12918578F, -0.041029334F, -0.057414103F, -0.02285048F, 0.009497269F,
          -0.05075266F, -0.06295834F, 0.01583755F, 0.037547633F, 0.04136201F,
          0.1934212F, 0.1878573F, 0.23922333F, -0.02711657F, -0.12958789F,
          0.04848948F, 0.066679545F, -0.004060608F, 0.011360724F, 0.0083598755F,
          -0.053962074F, -0.05014038F, -0.016826501F, 0.00067008554F,
          0.010957093F, 0.035530597F, 0.0055061365F, 0.058090817F, -0.043884996F,
          -0.13270614F, -0.1055393F, -0.0070582544F, -0.028275397F,
          -0.012281213F, 0.031073293F, 0.10885891F, 0.049884092F, -0.013252383F,
          -0.07121623F, -0.03357894F, -0.008927361F, -0.012166301F,
          0.0070246626F, -0.014074957F, -0.054024402F, 0.051502142F,
          0.042380698F, 0.042485792F, 0.051811755F, -0.015885808F, -0.079507954F,
          0.0048696334F, 0.025167674F, -0.043993272F, 0.03852945F, 0.010440213F,
          0.032551523F, 0.111989215F, -0.09955095F, -0.12820752F, -0.044661436F,
          -0.049586657F, -0.11693399F, -0.025437813F, -0.07734783F, -0.21615128F,
          -0.16837257F, -0.07796846F, -0.11476105F, -0.025960578F, -0.082421795F,
          -0.041647058F, 0.016359748F, -0.08993299F, -0.07488503F,
          -0.0031477916F, 0.049459804F, 0.003382392F, -0.02886017F,
          0.0139031345F, 0.014325527F, -0.04425652F, 0.17771609F, 0.31039625F,
          0.15091191F, -0.20956819F, 0.00016264858F, -0.005488038F,
          -0.055852715F, -0.32579607F, -0.3319984F, 0.04778573F, -0.0059088413F,
          -0.045139823F, -0.02268127F, -0.06663788F, 0.0014765512F, 0.053820662F,
          0.012860422F, 0.0076920507F, 0.057888724F, 0.04723785F, 0.028820666F,
          0.015730752F, -0.032186978F, -0.0714883F, 0.009643387F, 0.04488381F,
          0.031247197F, -0.068540476F, -0.06462077F, -0.09012989F, -0.030098462F,
          0.030784333F, -0.022535475F, -0.08788217F, 0.011965833F, -0.037272505F,
          -0.033163335F, -0.10784624F, 0.008902133F, 0.04783639F, 0.020426739F,
          0.04263227F, -0.022684386F, -0.087841064F, -0.05517914F, -0.11554463F,
          -0.066204175F, -0.035011876F, -0.040175438F, 0.01215995F, 0.121183366F,
          0.093826815F, 0.07707544F, 0.03245068F, -0.09455732F, -0.14532334F,
          -0.0582418F, 0.03477175F, -0.06397649F, -0.03234813F, 0.020030782F,
          0.09165054F, 0.041838426F, 0.09239364F, 0.24121861F, 0.15468916F,
          -0.058465503F, -0.19426678F, -0.09356865F, -0.018317483F,
          -0.048549302F, -0.009165933F, 0.01702284F, -0.016305434F, 0.015591418F,
          -0.009036023F, -0.013021758F, 0.0117553305F, -0.0059873755F,
          0.019241903F, 0.03443627F, -0.04139837F, 0.09634141F, 0.06150562F,
          -0.06484197F, -0.20965354F, -0.1831098F, -0.041620307F, -0.1090513F,
          -0.12630515F, -0.024425806F, -0.0044644256F, 0.032640465F,
          -0.076770544F, -0.039752577F, -0.01114183F, 0.018279083F,
          -0.021765957F, -0.008548318F, -0.26412812F, -0.46678886F, -0.37971258F,
          0.16418813F, 0.42063233F, 0.09691352F, -0.09446803F, 0.04787199F,
          0.09331863F, -0.15058897F, -0.31895134F, -0.19249526F, 0.08746343F,
          0.14012544F, -0.031059572F, 0.066995345F, 0.24218717F, 0.15449177F,
          -0.022210864F, -0.12041822F, -0.052588392F, -0.019202547F,
          -0.043841507F, -0.01642532F, 0.033980798F, 0.021496264F, 0.03832686F,
          -0.045451757F, -0.05666622F, -0.035344016F, 0.05355099F, 0.111992046F,
          0.0054074833F, -0.031956982F, 0.06141343F, 0.00979026F, 0.021574788F,
          -0.04665997F, 0.013312522F, -0.011063901F, -0.11182358F, -0.013592137F,
          0.054476507F, 0.049548957F, 0.07538479F, -0.056172896F, 0.15627894F,
          0.06170182F, -0.12944256F, 0.027523048F, 0.015393892F, -0.0010250038F,
          0.012943593F, -0.014835884F, -0.03508424F, -0.05259309F, 0.015293844F,
          -0.008742586F, -0.018352434F, 0.017180387F, 0.0064152484F,
          0.012584017F, 0.0732564F, -0.025739005F, -0.06065816F, -0.03876405F,
          -0.08826788F, -0.11567346F, -0.1349209F, -0.10840271F, -0.12466572F,
          -0.13425913F, -0.03152585F, -0.08562057F, -0.073549405F, -0.031757105F,
          -0.086346716F, -0.056282062F, 0.03550821F, 0.0062537803F, 0.035312176F,
          0.03815003F, 0.017684419F, -0.013144074F, 0.0125385495F, 0.02938049F,
          0.01667805F, -0.02076683F, 0.03029331F, 0.012925215F, -0.024905931F,
          0.012851707F, 0.012706933F, -0.017967802F, 0.0011332699F, 0.02404999F,
          -0.002997191F, -0.0041367738F, -0.0056575323F, 0.025613258F,
          -0.0009785945F, -0.002390235F, -0.006932868F, -0.017249249F,
          -0.004983503F, -0.03141361F, 0.0014484478F, 0.010252931F,
          -0.032074407F, -0.11307186F, -0.07939141F, -0.12288307F, -0.12125514F,
          -0.10317113F, -0.04015297F, -0.09084462F, -0.039323732F, -0.036541022F,
          -0.025591759F, 0.0019437793F, -0.050628576F, -0.04350213F,
          -0.040779192F, -0.030643381F, -0.057047013F, -0.02175009F,
          0.018571956F, 0.04439523F, 0.027242418F, 0.036038768F, 0.013903513F,
          0.014889282F, 0.031255774F, 0.012171534F, -0.010399247F, 0.0036007084F,
          0.0027457469F, 0.011915137F, 0.04318064F, 0.016429929F, -0.03159676F,
          -0.017776495F, 0.022262888F, -0.020408949F, 0.0060479906F,
          0.010023624F, 0.0049848617F, -0.052633274F, -0.12473648F, -0.13488382F,
          -0.07653519F, -0.10831813F, -0.15029657F, 0.034263514F, 0.010587916F,
          -0.0071316157F, 0.024708431F, 0.0071430514F, 0.008336734F,
          -0.02270074F, 0.02101013F, 0.026899058F, 0.051778555F, 0.061590247F,
          0.027672201F, 0.0448566F, 0.060997874F, 0.030741185F, 0.010590241F,
          0.06358662F, 0.054828145F, 0.050181776F, 0.003690118F, 0.023880143F,
          0.038822263F, 0.034718502F, 0.02640254F, 0.0271645F, 0.0149904825F,
          0.023655605F, -0.0016209971F, 0.0061850357F, 0.018511232F,
          -0.0041188253F, -0.0015329749F, 0.012792108F, 0.025081892F,
          0.013224083F, -0.005526489F, -0.005346473F, 0.052608263F, 0.01807388F,
          -0.011411908F, 0.009668114F, 0.049584504F, 0.05451175F, -0.04919722F,
          0.03246659F, -0.021007076F, -0.035860494F, 0.005169185F,
          0.00049449736F, 0.0002347598F, 0.019427951F, -0.014544467F,
          -0.0034010902F, -0.012826474F, 0.011427314F, -0.03416074F,
          0.006902158F, 0.015412385F, 0.01914199F, 0.011795578F, 0.0070811715F,
          0.034848243F, -0.032957714F, -0.017379282F, 0.0011831161F, 0.02410288F,
          -0.02149585F, -0.01769914F, -0.012157339F, -0.04059386F, -0.03392843F,
          -0.02400308F, 0.008790655F, 0.025310671F, 0.016062155F, -0.003335926F,
          0.001438814F, -0.0006310698F, -0.0064126723F, -0.012281754F,
          -0.019443413F, 0.0054749F, -0.005630952F, 0.017047426F, 0.034310136F,
          0.021122608F, -0.03207064F, -0.022587249F, 0.0074941325F, 0.01283797F,
          -0.0006384925F, -0.021046352F, 0.03096214F, 0.0013122548F,
          -0.0144030545F, 0.009908577F, -0.037697043F, -0.022958295F,
          -0.009221733F, -0.009173792F, 0.023309283F, -0.0014795706F,
          -0.024028154F, 0.012945946F, 0.039145358F, 0.030410305F, -0.01393788F,
          -0.003780949F, -0.05837469F, -0.08915537F, -0.070039086F,
          -0.061718453F, -0.10545243F, -0.08705017F, -0.05187964F, -0.09379137F,
          -0.07448386F, 0.00023375804F, 0.007915841F, -0.034381595F,
          -0.023635585F, -0.0043956623F, -0.056468617F, -0.021502098F,
          -0.05172896F, -0.036050335F, 0.013070714F, -0.0008923088F,
          -0.024299927F, 0.015226361F, 0.01744087F, 0.0022131335F, -0.013529589F,
          0.01642052F, 0.02275803F, -0.0004323196F, -0.03665835F, 0.02421916F,
          -0.008448226F, 0.023741998F, -0.029074665F, 0.011419372F, 0.023164967F,
          0.012488494F, 0.032830864F, -0.0059770476F, -0.0041121696F,
          0.008686671F, -0.0022251653F, 0.02614999F, 0.00077659066F,
          0.022349127F, 0.037841193F, -0.028175395F, 0.012420874F, 0.020348012F,
          -0.00024553473F, 0.029819261F, 0.016575975F, 0.042915814F,
          0.0127934255F, 0.025952732F, 0.023217293F, 0.017851373F, 0.008834586F,
          -0.02347438F, 0.010935158F, -0.0062620123F, 0.0029873543F,
          0.023776412F, 0.013324826F, -0.024664717F, -0.04011399F, -0.02068558F,
          0.020151183F, -0.0015481611F, -0.0282212F, -0.022345584F,
          -0.030177288F, -0.041843515F, 0.023952872F, 0.030952543F,
          -0.0066503175F, 0.03937322F, 0.005847145F, 0.015923202F, 0.016659752F,
          0.013539347F, 0.015003174F, -0.018102903F, -0.052935887F, -0.06961145F,
          -0.042859364F, -0.09779387F, -0.1022896F, -0.059326246F, -0.049731437F,
          -0.030993398F, 0.07084654F, 0.048116926F, 0.07036023F, -0.05575279F,
          -0.058070768F, 0.0064709694F, -0.03946755F, -0.004154429F,
          0.020561794F, 0.029281288F, -0.027455807F, 0.008591651F, -0.030218098F,
          -0.058835197F, -0.029521704F, 0.009224159F, 0.032956664F,
          0.00073275634F, -0.002958862F, -0.025311017F, -0.024696039F,
          0.02042121F, -0.042730648F, -0.014804613F, -0.011890232F, 0.031605147F,
          0.015214552F, -0.046580344F, -0.04710901F, -0.0062029148F,
          -0.014695126F, -0.026535124F, -0.030369744F, 0.0077507445F,
          0.036505252F, -0.026382633F, 0.0040485896F, 0.019453863F,
          -0.024602458F, -0.023195F, -0.072822765F, -0.07682982F, -0.032358307F,
          -0.027995147F, -0.022327965F, -0.0028273317F, -0.03683057F,
          -0.018709643F, 0.015657065F, -0.028895408F, -0.022567736F,
          0.008267361F, 0.022234911F, -0.0149471965F, 0.06334424F, 0.095863536F,
          0.029464765F, 0.11182712F, 0.116086386F, 0.0910997F, 0.006601313F,
          0.004542464F, 0.04125812F, 0.0067786416F, 0.033883747F, 0.068350434F,
          -0.028023317F, -0.022934705F, 0.012116227F, -0.00031489247F,
          -0.040061463F, -0.029837918F, -0.05317652F, -0.027767168F,
          0.007906571F, -0.0060683354F, -0.0523417F, -0.03437286F, 0.012484488F,
          0.005065241F, -0.0324997F, 0.019499384F, 0.059126183F, 0.006216465F,
          -0.016264057F, -0.008636802F, 0.002917469F, -0.034430232F,
          -0.044771302F, 0.017152002F, -0.029951587F, -0.014036771F,
          -0.036843486F, 0.00454817F, -0.019940622F, 0.03173736F, -0.0076615545F,
          0.032109994F, 0.045968063F, 0.07247343F, 0.13886297F, 0.063143626F,
          0.00039182784F, -0.018692924F, 0.008269139F, -0.07348119F, -0.1322125F,
          -0.044887945F, 0.08316117F, 0.013782018F, 0.041544493F, -0.017671742F,
          0.0032258886F, -0.016944258F, -0.049076244F, -0.0077161198F,
          -0.020479403F, -0.009054122F, -0.005272993F, 0.007836214F,
          -0.017379822F, -0.02616343F, -0.015436444F, -0.031381197F,
          -0.034944788F, -0.021640455F, -0.105890155F, -0.16505907F,
          -0.084530815F, -0.20375921F, -0.35255784F, -0.24997136F, -0.1116474F,
          -0.21673748F, -0.12917916F, 0.01654861F, 0.053826887F, 0.046311066F,
          0.03040641F, 0.058040448F, 0.039839573F, 0.045811642F, 0.008034861F,
          0.0064065317F, -0.004470607F, -0.026564755F, -0.03284696F,
          -0.00782337F, -0.005072408F, -0.0023501525F, -0.01713749F,
          0.026277259F, -0.0013862427F, 0.0121047655F, 0.060305778F, 0.03548838F,
          -0.07090522F, -0.07524543F, -0.017875709F, 0.01064213F, -0.059207395F,
          -0.016848568F, 0.04442994F, 0.012640841F, -0.014061422F, -0.023902679F,
          -0.06243251F, -0.063474275F, -0.002028126F, 0.008460836F,
          -0.037891798F, 0.07543573F, 0.00883783F, 0.0035899882F, 0.020609168F,
          0.022569818F, 0.04436867F, 0.008867567F, 0.080014914F, 0.08308119F,
          -0.0208938F, -0.14337543F, -0.060361452F, -0.06678287F, -0.10088646F,
          -0.03597586F, 0.0125694405F, 0.04568528F, 0.0045027514F, -0.043591376F,
          -0.033066966F, -0.004490601F, -0.092826776F, -0.11838441F, -0.0780547F,
          -0.060983855F, -0.09011613F, -0.08495701F, 0.030021928F,
          0.00057080365F, 0.024337256F, -0.042461634F, -0.025254117F,
          0.0021171211F, -0.047988165F, 0.019346338F, 0.022249896F,
          -0.041011576F, -0.016868144F, 0.011448717F, -0.0531194F, -0.15488985F,
          -0.110414185F, -0.040121377F, -0.13822892F, -0.083204925F,
          0.0045049684F, 0.015447383F, -0.013177734F, -0.010099313F,
          -0.012162996F, -0.009990965F, 0.02954965F, 0.007041619F, -0.010293719F,
          0.045212913F, 0.12077915F, 0.09440085F, 0.083534166F, 0.16155133F,
          0.07485869F, 0.06324353F, 0.08945795F, 0.018396117F, 0.028293835F,
          0.083052374F, 0.028461209F, 0.106258854F, 0.101500675F, 0.074163035F,
          0.06515231F, 0.10892637F, 0.032711074F, -0.013240005F, 0.007910801F,
          0.0108180735F, -0.01154381F, -0.0055651464F, -0.010061013F,
          0.0034227055F, -0.0073555927F, -0.0016600258F, -0.04109865F,
          -0.034379303F, -0.019604469F, -0.016689437F, -0.044550605F,
          -0.0035230073F, -0.00035307836F, -0.026743898F, -0.038576044F,
          -0.0057384614F, 0.0177492F, 0.0436642F, 0.01596103F, 0.0023021314F,
          0.06265858F, 0.022621388F, 0.021900166F, -0.0070059323F, 0.0119597F,
          -0.009888414F, -0.06775332F, -0.033538748F, -0.1186678F,
          -0.0057171434F, -0.0058436343F, 0.008258254F, -0.0016859486F,
          -0.010475683F, 0.012079118F, 0.016139755F, 0.009890464F, 0.013143255F,
          0.010623254F, 0.0051371804F, 0.0048943995F, -0.021266498F,
          0.008050458F, -0.034279566F, 0.0097400835F, 0.017284287F, 0.050895605F,
          0.014844481F, 0.005811736F, 0.13420114F, 0.08435929F, -0.050059114F,
          -0.005501338F, 0.022607706F, -0.03617261F, -0.04790554F, 0.0077128117F,
          0.011520323F, -0.032658905F, 0.004693258F, -0.10562668F, -0.24844643F,
          -0.05854136F, -0.021006431F, -0.27715874F, -0.10327812F, 0.04982148F,
          0.041805647F, -0.023774518F, 0.088072374F, 0.13590391F, 0.07500816F,
          0.04551507F, 0.14029814F, 0.08964157F, -0.0124510825F, -0.024360094F,
          0.042297974F, -0.048822284F, -0.08496997F, -0.11026931F,
          0.00026812125F, -0.057793338F, -0.092321925F, 0.0072818124F,
          0.04437537F, -0.0029784618F, -0.016654842F, -0.050409067F,
          -0.038057633F, 0.014959343F, 0.00518049F, 0.015809158F, 0.002190541F,
          0.04760469F, 0.03862013F, -0.17272781F, 0.21729873F, 0.20207334F,
          -0.22757806F, 0.12857649F, 0.30669117F, -0.22448957F, 0.07596273F,
          0.36398953F, 0.03047097F, -0.01766001F, 0.014590402F, 0.023192389F,
          0.0024984044F, -0.07951137F, -0.0060547343F, 0.00918547F,
          -0.071510024F, 0.0077872085F, 0.10654931F, -0.055143837F, 0.01952816F,
          0.10728942F, 0.061436493F, -0.011240397F, 0.02167734F, 0.03539935F,
          0.11243014F, -0.036367245F, 0.014679249F, 0.048106138F, 0.000212494F,
          -0.15288612F, -0.008234304F, 0.07772542F, -0.12657599F, -0.093929395F,
          -0.17188631F, -0.087434396F, -0.05170438F, -0.16470982F, -0.11872068F,
          -0.03236276F, -0.029128244F, -0.11133629F, -0.12059855F, -0.14253923F,
          -0.056358915F, -0.011456787F, -0.12375078F, -0.14749037F, 0.06260273F,
          0.029418977F, -0.08322242F, -0.045688808F, -0.12665744F, -0.09175596F,
          -0.0028873351F, -0.17941952F, -0.11788507F, 0.057682946F, -0.04139385F,
          -0.066298F, 0.10112064F, 0.06697106F, -0.011071109F, 0.054954413F,
          0.024141353F, -0.045754507F, 0.008978705F, -0.009415283F, -0.091263F,
          -0.03010473F, 0.011520532F, -0.023098594F, -0.07236327F, 0.016226131F,
          -0.031321146F, -0.025441902F, -0.036014125F, -0.05305718F, 0.02448276F,
          0.164409F, 0.008947771F, 0.04511128F, 0.106097296F, 0.021589702F,
          -0.015732208F, 0.037164327F, 0.05637085F, -0.13056135F, -0.09580357F,
          -0.073675066F, -0.009702568F, 0.020919416F, 5.674196E-6F, 0.015281332F,
          0.079938464F, 0.06207729F, 0.12010824F, 0.007232334F, 0.058378164F,
          0.021815171F, -0.13153683F, -0.09146881F, 0.03418902F, -0.09092911F,
          -0.08664148F, -0.07405618F, -0.05088273F, 0.048532568F, 0.011908283F,
          -0.11249193F, -0.036868595F, 0.016800504F, -0.030389935F, -0.01070205F,
          -0.036404833F, 0.049149003F, -0.040320914F, 0.020999333F,
          0.0047267512F, 0.054556623F, 0.014702066F, -0.008951975F, 0.09420772F,
          -0.13535948F, -0.20090787F, -0.06687272F, -0.085170425F, -0.16139068F,
          -0.105240405F, -0.021055073F, 0.07092391F, 0.07367068F, -0.047503263F,
          0.12992686F, 0.058776353F, -0.029217808F, 0.08880176F, 0.14351627F,
          -0.025320284F, -0.08624431F, 0.1730877F, -0.0072104516F, -0.010248827F,
          0.049612533F, 0.013265809F, -0.0011377111F, 0.032588013F,
          -0.00087389804F, 0.042487264F, 0.047386102F, -0.04114299F,
          -0.010698722F, -0.0021648603F, -0.029664507F, -0.23383683F,
          -0.16326466F, 0.028925415F, -0.09029342F, -0.10237448F, -0.050635207F,
          -0.11574982F, -0.040053535F, 0.031914018F, -0.056770176F,
          -0.037288763F, 0.065865815F, 0.022910325F, -0.007664535F, 0.001329907F,
          0.13834678F, 0.006754801F, 0.011503452F, 0.07434689F, 0.08093474F,
          -0.023484087F, -0.016065823F, 0.07379737F, -0.029276928F, -0.12205967F,
          -0.07765245F, -0.06957659F, -0.23114331F, -0.16604207F, -0.011718052F,
          -0.07291814F, -0.15353131F, 0.050556257F, 0.16439724F, 0.08253107F,
          0.01343189F, 0.17997971F, 0.15796073F, -0.0049411324F, 0.030170975F,
          0.12634377F, 0.04485306F, -0.1462026F, -0.13217735F, 0.040341347F,
          -0.052345596F, -0.18851492F, 0.023561923F, -0.0012372273F,
          0.0015884326F, -0.1529606F, 0.060764007F, -0.03523165F, -0.037976936F,
          0.12773934F, 0.08662125F, -0.011579491F, 0.07931586F, 0.15064795F,
          0.04319375F, -0.12096964F, -0.104748964F, 0.030939413F, -0.092148036F,
          -0.18166637F, 0.056595594F, -0.0050774734F, -0.09294628F,
          -0.003075164F, -0.20761645F, -0.11106091F, 0.04423769F, -0.14160709F,
          -0.13355349F, 0.10058861F, 0.079296716F, -0.00775966F, -0.044375435F,
          0.06871196F, -0.058356687F, 0.0071852384F, -0.028479997F,
          -0.023055827F, 0.045301206F, -0.09112752F, 0.023023948F, 0.047291033F,
          -0.0031871076F, -0.02493576F, 0.020977976F, -0.014805984F,
          -0.009790847F, 0.011051477F, -0.015801722F, -0.036157954F,
          -0.071382545F, 0.035789035F, -0.038757388F, 0.042082805F,
          -0.057553228F, 0.008240522F, 0.06892583F, -0.090777725F,
          -0.0005802026F, 0.06903284F, 0.056067746F, 0.009948408F, -0.111088425F,
          -0.011376037F, 0.007844942F, -0.023633685F, 0.09755492F, 0.037661925F,
          -0.0022538933F, -0.07087925F, 0.10293349F, 0.01771356F, 0.063442655F,
          -0.018656585F, -0.10534312F, 0.03652457F, -0.16413754F, -0.022410607F,
          0.030212754F, -0.055083994F, -0.0015155071F, 0.084813304F,
          -0.051626123F, 0.026241459F, -0.0017005783F, -0.07084422F, 0.09063974F,
          0.029804392F, -0.047181454F, 0.065033235F, 0.015913192F, -0.01503196F,
          0.05416072F, 0.01761278F, -0.0030621565F, 0.048346095F, 0.01274538F,
          -0.055045936F, 0.03995148F, -0.033047095F, -0.020393416F, 0.02294444F,
          0.017040992F, -0.00095293095F, -0.030095156F, -0.031175323F,
          0.07742226F, -0.102419145F, 0.1003918F, -0.10523955F, -0.04055997F,
          0.03665172F, -0.11797265F, -0.015288801F, -0.04717674F, 0.002391455F,
          -0.057709634F, 0.03659418F, -0.029932803F, -0.028420953F, 0.08416958F,
          -0.039388083F, -0.021928607F, 0.06987166F, 0.03362181F, -0.006119661F,
          0.30396187F, -0.04088452F, -0.016015887F, 0.108351715F, -0.09911746F,
          -0.08934286F, -0.007570877F, 0.050282247F, -0.021290747F, 0.046790324F,
          -0.051558733F, 0.035861578F, -0.03840767F, -0.10712124F, -0.051683687F,
          0.037683945F, -0.11633052F, -0.10889281F, 0.051568925F, -0.118873835F,
          -0.05464256F, 0.04368099F, -0.054026466F, 0.034371518F, -0.059232764F,
          0.020348042F, -0.035507593F, -0.05856678F, 0.04053414F, -0.01603473F,
          0.0037480735F, 0.000639902F, -0.056284748F, -0.04831628F, 0.12443648F,
          -0.14084367F, -0.0002482328F, 0.10058045F, -0.18655905F, 0.14158626F,
          0.038991336F, 0.028717343F, -0.020322429F, -0.04587909F, 0.019415773F,
          -0.044823103F, -0.021829253F, 0.0116403075F, 0.0048238845F, 0.0229804F,
          -0.002382155F, 0.0019234287F, 0.05413431F, -0.059554566F, 0.082138136F,
          -0.019535566F, -0.031253453F, 0.025274262F, -0.050848365F,
          0.120864816F, -0.11299285F, -0.012438514F, 0.11507551F, -0.30542162F,
          0.108652055F, -0.053550936F, -0.009607032F, 0.061904587F,
          -0.041413292F, -0.021894557F, 0.042819932F, -0.083145104F,
          0.0038531076F, -0.005254986F, 0.003413388F, 0.024089755F,
          -0.023155706F, 0.0024321398F, -0.17173547F, 0.14047407F, -0.08881295F,
          -0.0048540235F, 0.1198018F, -0.1545538F, 0.12902379F, -0.08712378F,
          -0.15267837F, 0.1843511F, -0.12245955F, -0.08422671F, 0.18704951F,
          -0.059403434F, -0.010210826F, -0.04716034F, -0.06061279F,
          -0.002595492F, 0.10520896F, -0.109079465F, 0.10714708F, -0.02464133F,
          -0.05167177F, 0.05991508F, -0.1261021F, -0.014503228F, 0.05810961F,
          -0.032740854F, 0.021263508F, -0.0054346565F, -0.049232706F,
          0.04450522F, -0.04724191F, -0.007678408F, -0.010012078F,
          -0.0004419297F, -0.0334143F, 0.07740594F, -0.059591908F, -0.029936917F,
          0.07496676F, -0.04117872F, 0.01795476F, 0.026697353F, -0.083515264F,
          -0.02162626F, -0.011573942F, -0.03162581F, 0.012759218F, -0.13855967F,
          0.061710007F, -0.061747067F, -0.042504568F, 0.055173766F,
          -0.033997815F, -0.09244244F, 0.030914025F, -0.094419144F,
          -0.009505052F, 0.025598383F, 0.024575252F, -0.0077286353F,
          -0.028744042F, 0.11080407F, -0.1506354F, 0.10706668F, 0.024057334F,
          -0.12718724F, 0.14709315F, -0.16267128F, -0.037284F, -0.017728195F,
          -0.12794061F, -0.038332682F, -0.056893572F, -0.1415675F, 0.0064439424F,
          -0.013320008F, 0.009031833F, -0.013398843F, 0.03505259F, -0.014227736F,
          0.02194467F, 0.08949647F, 0.005118992F, -0.00526643F, 0.009085406F,
          0.0197871F, 0.04187975F, -0.107531786F, 0.08939188F, -0.0504499F,
          0.044211376F, 0.02042853F, 0.00931231F, 0.0013597956F, -0.08894938F,
          -0.08258046F, -0.056067877F, -0.07620528F, 0.048835076F, -0.120960794F,
          -0.014980271F, 0.06515998F, -0.025807586F, 0.017870465F, 0.0020169907F,
          -0.028250137F, 0.057578083F, -0.011831434F, 0.06557919F, 0.03681803F,
          -0.027286831F, 0.08030012F, -0.03817482F, -0.050178304F, 0.021032238F,
          -0.049737778F, -0.015088547F, -0.0049733417F, -0.0035757434F,
          0.052351862F, -0.006937945F, 0.007642497F, 0.020107273F, 0.05171441F,
          -0.0018417762F, -0.024073133F, 0.0028523814F, -0.055898026F,
          -0.04278286F, 0.035977513F, -0.012966481F, 0.010227065F, 0.039476965F,
          0.06529238F, -0.011174706F, 0.06203333F, 0.02152306F, 0.02369395F,
          0.012302938F, 0.031233322F, 0.0154993F, -0.14886731F, -0.2287831F,
          -0.17910552F, -0.33190027F, -0.32100785F, -0.20146784F, -0.23649654F,
          -0.15659644F, -0.096241154F, 0.043896064F, 0.07101556F, 0.051006496F,
          0.06179985F, 0.004249063F, -0.020004814F, 0.03183812F, 0.01215061F,
          0.038620222F, 0.06542358F, 0.01434151F, 0.05800173F, -0.059709273F,
          -0.097880125F, -0.037993655F, -0.08080493F, -0.04009477F, 0.05041405F,
          -0.026506986F, -0.07457764F, -0.053108465F, 0.04665709F, -0.06396919F,
          -0.09336745F, 0.09437769F, -0.06055347F, -0.08646488F, 0.00044201882F,
          -0.018988326F, -0.043598454F, 0.019484354F, 0.0013596453F,
          0.0036319348F, 0.050778847F, 0.02392095F, 0.009590558F, -0.011769656F,
          -0.122230135F, -0.2471852F, -0.07498929F, -0.15462495F, -0.11818393F,
          -0.03852785F, -0.054086223F, 0.0273317F, 0.001178526F, -0.035259504F,
          -0.008977001F, -0.011603963F, -0.04474651F, -0.0013595265F,
          -0.036293987F, -0.05792983F, -0.006806325F, 0.015010056F, 0.05923128F,
          0.060111087F, 0.04586486F, 0.08136889F, 0.012693804F, 0.05651578F,
          0.08703517F, 0.017310513F, 0.041278962F, -0.070024036F, -0.076573566F,
          -0.042146284F, -0.11105549F, -0.08776616F, -0.03666467F, -0.04964859F,
          0.041157003F, -0.0070329155F, -0.19353624F, -0.17151852F, -0.14067927F,
          -0.2819105F, -0.16683F, 0.017082257F, -0.018243559F, 0.06749394F,
          -0.08875186F, -0.15692438F, -0.1548245F, -0.14716046F, -0.1121291F,
          -0.13440993F, -0.07801951F, -0.07794146F, -0.0656234F, 0.04224685F,
          0.024567783F, 0.0392614F, 0.0256577F, -0.0026893448F, -0.07231674F,
          0.0029985216F, -0.04213337F, -0.010665754F, -0.058929224F,
          -0.14770105F, -0.11891388F, -0.07717189F, -0.10406885F, -0.010131524F,
          -0.067615174F, -0.043824278F, 0.03579153F, 0.04684765F, -0.039168943F,
          -0.053487625F, -0.026555885F, -0.02839512F, 0.0012607151F,
          0.039796166F, 0.019210875F, 0.039634448F, 0.01806604F, 0.08857706F,
          0.1803671F, 0.08889028F, 0.097830676F, 0.0710006F, 0.0584961F,
          0.013130497F, -0.028931739F, 0.06890921F, -0.018952707F, 0.016140418F,
          0.01789745F, -0.038837966F, -0.0025432208F, 0.02813097F, 0.020325294F,
          0.03050967F, 0.008079862F, -0.014790201F, -0.028775359F, -0.03532361F,
          -0.010561203F, -0.029552465F, -0.00014675672F, -0.041907668F,
          0.006830262F, 0.06829239F, -0.03435471F, -0.00323283F, 0.0037412439F,
          -0.04202481F, 0.027422136F, 0.04026704F, -0.01951021F, 0.007579654F,
          -0.023261623F, -0.1334702F, -0.13545771F, -0.07964801F, -0.13418952F,
          -0.09098719F, -0.052776907F, -0.051642727F, 0.02410998F, 0.04324754F,
          -0.017239818F, -0.04035631F, 0.028101647F, -0.01468031F, -0.021635618F,
          0.02782057F, -0.013206776F, -0.013929334F, 0.048445754F, 0.0018306558F,
          0.07432086F, 0.037626836F, 0.0074858395F, -0.04521578F, 0.0033994752F,
          -0.03455258F, 0.0017849616F, 0.047826793F, 0.09882683F, 0.07656419F,
          0.07819527F, 0.09999585F, 0.086394526F, 0.02224431F, 0.049298976F,
          0.009246592F, -0.073520996F, -0.08619819F, -0.0775332F, -0.09577794F,
          -0.115498036F, -0.07436314F, -0.067706384F, -0.07036756F,
          -0.020074608F, 0.026483752F, 0.111778066F, 0.07360677F, 0.118578866F,
          0.104903415F, 0.05115149F, 0.07616891F, 0.02820115F, 0.027349602F,
          -0.08791842F, -0.09876739F, -0.17798154F, -0.135091F, -0.12618272F,
          -0.09435276F, -0.06497363F, -0.03293307F, 0.004103059F, 0.02953608F,
          -0.0013762612F, 0.02137593F, 0.02076072F, -0.018145865F, 0.04886381F,
          -0.025989663F, 0.0015145628F, 0.016630983F, 0.059938062F,
          -0.069863796F, -0.089864746F, 0.04313146F, -0.02302465F, 0.045972604F,
          0.11773305F, 0.075739674F, 0.08649648F, 0.027683483F, -0.048563994F,
          0.051522143F, 0.019287428F, -0.00874765F, 0.024938488F, -0.024494184F,
          0.018821219F, 0.033840504F, 0.0028409178F, 0.07609006F, 0.063944235F,
          0.03804353F, 0.06288411F, 0.06016479F, -0.013928993F, 0.021180762F,
          0.015782658F, 0.037372958F, -0.05660947F, 0.018822687F, 0.0665398F,
          -0.07138916F, 0.014047962F, 0.013566322F, -0.029493267F, -0.023904366F,
          -0.028613567F, -0.011719873F, 0.026581952F, 0.010986923F,
          -0.043490723F, 0.027165068F, -0.021006173F, -0.02704501F, -0.02229452F,
          0.038241617F, 0.073518135F, 0.043040767F, -0.09312846F, -0.10676764F,
          0.020078244F, 0.014085505F, -0.08412457F, 0.023856526F, -0.16182464F,
          -0.08433793F, 0.02554156F, -0.0038864722F, -0.095640644F, -0.09276683F,
          0.018543491F, -0.02541911F, -0.055357408F, -0.04229019F, -0.038574345F,
          0.009131498F, -0.1537162F, -0.19479616F, -0.06745187F, -0.057645306F,
          -0.09049064F, -0.07222706F, 0.006052435F, 0.0038855807F, -0.03467872F,
          0.020120602F, -0.054296173F, -0.07834348F, 0.0410631F, 0.036194462F,
          -0.0680207F, 0.07536235F, 0.0052372827F, -0.08943773F, 0.007053122F,
          0.10715046F, 0.058641817F, 0.042302486F, 0.04716282F, 0.13349678F,
          -0.01944734F, -0.028994042F, 0.025772447F, 0.020494832F,
          -0.00016130827F, 0.03447516F, 0.029281223F, -0.012522926F,
          -0.00445704F, 0.1032181F, -0.0040634875F, -0.042173807F, 0.1496777F,
          0.14347935F, 0.005018641F, -0.06285179F, 0.029308673F, 0.17090875F,
          -0.052425653F, -0.08156167F, -0.04321628F, 0.06952065F, -0.0700868F,
          -0.08978582F, 0.011569611F, 0.00442489F, -0.023249941F, 0.113649115F,
          0.08884686F, 0.03777274F, -0.011044831F, 0.030370338F, 0.06159648F,
          0.027039146F, -0.037998646F, 0.00074521586F, -0.018063752F,
          -0.18128628F, -0.04730413F, -0.11802011F, -0.2630977F, -0.1766377F,
          -0.0012426138F, 0.0025941995F, -0.10693295F, -0.12062201F,
          -0.14558111F, -0.124989904F, -0.021901933F, -0.19099893F, -0.15919353F,
          0.06712218F, -0.010376811F, -0.048863485F, 0.009305689F, 0.055413555F,
          0.036516473F, 0.04583538F, 0.07492092F, 0.02081665F, -0.039446365F,
          0.037920404F, 0.0066730445F, -0.11197633F, -0.04768176F, 0.1498137F,
          0.14910261F, -0.21122429F, -0.15296638F, 0.08608711F, 0.0852638F,
          -0.1182977F, 0.07801924F, -0.24338788F, -0.084849685F, 0.12379937F,
          0.028073795F, -0.08868777F, 0.028707888F, 0.10539398F, 0.042802308F,
          -0.030349676F, -0.062593624F, -0.060792547F, -0.056816496F,
          -0.068317086F, -0.032343697F, -0.025985591F, -0.1639691F, -0.09336209F,
          -0.11181587F, -0.14899394F, -0.07573453F, -0.065502204F, 0.024127355F,
          -0.053283468F, -0.07505949F, 0.041326113F, 0.10976922F, -0.03639588F,
          0.05024985F, 0.058024086F, -0.15639216F, -0.053354703F, 0.03922397F,
          0.0065465686F, -0.061922215F, 0.011559233F, 0.22207904F, 0.31248003F,
          0.12215861F, -0.08635884F, 0.18298474F, 0.2449748F, -0.15188937F,
          -0.20200834F, -0.053691454F, -0.054384764F, -0.0834551F, -0.073262535F,
          0.002327574F, -0.113264F, -0.012783748F, 0.039447807F, -0.029749952F,
          -0.04544558F, -0.17220245F, -0.28383523F, 0.060285274F, 0.14499386F,
          -0.25060248F, -0.23992537F, 0.12803851F, 0.11270056F, -0.05076281F,
          -0.04147913F, 0.021111865F, 0.040035393F, -0.034910798F, -0.028440177F,
          0.050492935F, 0.036490954F, -0.020987313F, -0.015671695F,
          -0.019672116F, 0.0026323935F, -0.044827342F, -0.015775487F,
          -0.1604954F, -0.15454854F, 0.07743358F, -0.050120782F, -0.006586034F,
          -0.011035468F, 0.012927012F, 0.043304F, -0.059486438F, 0.033355676F,
          0.09874213F, 0.008269053F, -0.005353304F, 0.054209117F, 0.20028603F,
          0.18335944F, -0.090181574F, -0.20413604F, 0.15490405F, 0.11486181F,
          -0.07764178F, -0.110363744F, 0.043023046F, 0.032380234F, 0.12099806F,
          0.0019539227F, -0.19776203F, 0.070406616F, 0.16963108F, -0.103512935F,
          -0.055816285F, 0.119976155F, -0.015600033F, 0.011080244F,
          -0.00075391924F, 0.00264617F, -0.011320168F, -0.0067129945F,
          -0.03495572F, -0.027733251F, -0.040952265F, 0.06475368F, 0.0005515403F,
          -0.02379842F, -0.019283121F, 0.15228532F, 0.11007135F, 0.0076778694F,
          0.027798815F, 0.066258706F, 0.059176315F, -0.08976119F, -0.08140592F,
          -0.10289614F, -0.08828313F, 0.0351605F, -0.014234426F, -0.10421491F,
          -0.030014602F, -0.15458964F, -0.10742268F, -0.048859295F, 0.019070305F,
          -0.12335755F, -0.099027246F, 0.023360845F, -0.03565994F, -0.041939165F,
          -0.060776025F, 0.022636564F, 0.013424214F, -0.054493595F, -0.06479373F,
          0.013595378F, 0.050868485F, -0.04495349F, -0.043743234F, -0.036392286F,
          -0.052376937F, -0.007934097F, -0.061032936F, -0.0780931F,
          -0.047051147F, -0.02480409F, -0.024825586F, 0.028790185F, 0.06112576F,
          0.056916002F, 0.008354159F, -0.07077434F, -0.07357524F, -0.05573138F,
          0.05054106F, 0.05502306F, -0.0059077074F, 0.00961465F, 0.011575866F,
          -0.008580129F, -0.020490274F, -0.030209063F, -0.019364597F,
          0.012360285F, 0.013251291F, -0.042997282F, -0.0130488975F,
          -0.05959721F, -0.031996615F, -0.0433837F, -0.052049167F, -0.03135724F,
          -0.004243938F, -0.0024159038F, -0.027361698F, -0.03889179F,
          0.025005305F, -0.011888385F, -0.0771052F, -0.009890942F, 0.044905357F,
          0.027653003F, 0.052332427F, 0.045501348F, 0.041002713F, 0.033201657F,
          -0.0022602598F, 0.010016531F, -0.06178248F, -0.076297574F,
          -0.011859224F, 0.032406695F, 0.0075353757F, -0.0698858F, -0.14976515F,
          0.03265096F, -0.0849784F, -0.25729683F, 0.02677443F, 0.014716484F,
          -0.090251744F, 0.08541805F, 0.00042657417F, 2.7229135E-5F,
          0.060143016F, -0.07722717F, -0.09617335F, 0.043269973F, -0.006327894F,
          -0.04736664F, 0.029480496F, 0.042570703F, 0.037772458F, 0.014082957F,
          0.021048507F, -0.007857291F, -0.05150712F, 0.017967112F, 0.021487614F,
          -0.016078593F, -0.070778504F, 0.0013993527F, -0.033311274F,
          0.0017847033F, 0.039165597F, -0.004136114F, 0.01015873F, 0.07067174F,
          -0.036637552F, -0.031161107F, -0.01660539F, -0.02725225F, -0.07733498F,
          -0.09290676F, -0.0384455F, -0.049816545F, -0.07888281F, -0.04300082F,
          -0.020641634F, 0.032541875F, 0.022933323F, -0.10538739F, -0.114003144F,
          -0.054507528F, -0.042664792F, -0.03622383F, -0.036385432F,
          -0.015496887F, -0.06937775F, -0.037394688F, -0.09451629F, -0.15150756F,
          -0.12258732F, 0.017498089F, -0.039328575F, -0.054652337F, 0.009891512F,
          -0.024660034F, -0.011249479F, 0.009056477F, -0.025125816F,
          -0.0054051788F, 0.018510427F, -0.0022463514F, 0.012170704F,
          -0.04141907F, -0.035047755F, -0.032546684F, -0.09108748F, -0.22620225F,
          -0.08617492F, 0.02198619F, 0.0028617869F, -0.020913787F, 0.007530447F,
          -0.025395593F, 0.025992548F, -0.060147148F, -0.0030369866F,
          0.041677885F, -0.06776745F, -0.015124557F, -0.0037810793F,
          -0.02255846F, 0.0025858067F, 0.00066796556F, -0.07775752F,
          -0.06851098F, 0.016026331F, -0.017703552F, 0.0077845007F,
          0.0011382577F, 0.028984575F, -0.010737322F, 0.0024618935F,
          -0.10065138F, 0.024590665F, -0.037826166F, -0.00032557172F,
          0.012622706F, -0.0008283237F, 0.018896608F, -0.014249499F,
          -0.023600543F, 0.022157665F, 0.016765025F, -0.01455031F, 0.023267003F,
          0.013319411F, 0.007221877F, 0.05563501F, 0.039942853F, 0.010751344F,
          0.05455275F, 0.07530725F, 0.007824039F, 0.047775928F, 0.016360054F,
          0.015780736F, -0.016343022F, -0.0036296132F, -0.0072331177F,
          -0.047882084F, -0.03959825F, 0.00031463426F, 0.008945508F,
          -0.012120091F, -0.049390197F, -0.0011374005F, -0.01166144F,
          -0.06793983F, 0.021096388F, -0.024947174F, -0.08103093F, 0.029823879F,
          0.019809857F, -0.0473963F, 0.014904015F, 0.0024346355F, 0.029052945F,
          -0.082417674F, -0.07983256F, -0.030017909F, 0.01563202F, -0.025554424F,
          0.001988782F, -0.046917096F, -0.02834323F, -0.017050434F,
          -0.091355525F, -0.08775644F, -0.045649186F, -0.0450354F, -0.020682534F,
          -0.057705563F, 0.03687297F, 0.002416658F, 0.009548794F, -0.025125831F,
          -0.031858888F, 0.021031586F, -0.017097123F, -0.01140401F,
          -0.006695264F, -0.027258538F, -0.0570586F, -0.05063567F, -0.066198125F,
          -0.12870856F, -0.056368835F, 0.04031328F, 0.03777086F, -0.049540665F,
          -0.036241706F, -0.014905285F, 0.004659957F, -0.07387481F, -0.09671835F,
          -0.024209913F, -0.06065942F, -0.08420011F, -0.030517615F,
          -0.021418381F, -0.044023838F, -0.021499544F, -0.015680578F,
          -0.043822553F, -0.025103075F, -0.005798579F, -0.0053458507F,
          -0.004057309F, -0.014817142F, -0.030015059F, 0.02184527F,
          -0.021995783F, -0.040658284F, 0.00871602F, -0.001150426F,
          -2.3538429E-5F, 0.01280862F, -0.026021224F, -0.08650618F, 0.04489875F,
          -0.14755054F, -0.20940682F, 0.0026352608F, -0.054365933F, -0.09984295F,
          0.00043360528F, -0.0342728F, -0.02369684F, 0.040474787F, -0.023712099F,
          -0.025298316F, 0.031430714F, 0.0047845985F, -0.006653781F,
          0.019340988F, 0.09440449F, 0.07121784F, -0.034762375F, 0.12632364F,
          0.08763469F, 0.006011451F, -0.015513315F, 0.013116986F, -0.06255586F,
          0.0573268F, 0.032892372F, 0.03024169F, -0.0661968F, -0.07446804F,
          -0.017564401F, -0.038499605F, -0.0026661395F, 0.06579477F, 0.01974102F,
          0.006067129F, -0.017612644F, 0.005817571F, -0.04551139F, -0.027634567F,
          0.014190559F, -0.03054528F, -0.005388795F, -0.07842178F, -0.007660787F,
          0.015033681F, 0.0036611268F, 0.08082098F, 0.013991722F, 0.023409171F,
          0.029444087F, 0.029375814F, -0.012018624F, 0.024941549F, 0.016253946F,
          -0.0061891815F, 0.019780928F, 0.03005737F, -0.01858919F, 0.030902361F,
          -0.029196428F, 0.037896164F, -0.007651225F, -0.021087144F,
          0.0111856675F, -0.03722251F, -0.046125416F, -0.062771395F,
          -0.020212343F, -0.03728568F, 0.01440652F, 0.048504457F, 0.05640759F,
          0.024892062F, -0.012724963F, 0.008140228F, -0.013800279F, -0.04467868F,
          -0.009155783F, -0.040653553F, 0.065057255F, 0.099797316F, 0.10863708F,
          0.21318862F, 0.14775078F, 0.10709029F, 0.09621789F, -0.024101563F,
          0.034790035F, -0.020490954F, -0.007944898F, -0.02274244F, -0.05227051F,
          0.00090984517F, 0.0137162255F, 0.02438023F, 0.07073906F, 0.022926182F,
          0.048876397F, 0.0285963F, 0.027475523F, 0.0012499365F, 0.015025197F,
          0.007074581F, -0.014240628F, -0.012000556F, 0.02610776F, 0.011421053F,
          -0.008308412F, 0.004133338F, -0.03943121F, -0.020839747F,
          -0.041706555F, 0.007990041F, 0.03299771F, 0.029302608F, 0.03008527F,
          -0.0060210866F, 0.026059495F, 0.025604974F, 0.021388348F,
          0.0062294006F, -0.008631289F, -0.0023356853F, -0.061018486F,
          -0.075186F, -0.047596067F, -0.08504914F, -0.12497591F, -0.06023569F,
          -0.041373137F, 0.0247082F, 0.040276684F, 0.030856756F, 0.01924059F,
          -0.047067117F, -0.040698703F, -0.08268033F, -0.09664679F, 0.029402211F,
          0.003920364F, 0.026574463F, 0.040378705F, 0.051168527F, 0.034170885F,
          0.00805334F, 0.031546924F, 0.039973818F, 0.016840424F, 0.023181798F,
          0.008562293F, 0.027151546F, 0.10158822F, 0.04059611F, 0.047060974F,
          0.21708243F, 0.07630513F, 0.18104064F, 0.19847606F, 0.05445784F,
          -0.035779227F, -0.04299211F, -0.0015028561F, -0.020145314F,
          -0.103765436F, -0.032966226F, -0.020174207F, 0.0062039983F,
          -0.015049126F, -0.01129579F, -0.0082131745F, -0.022520266F,
          -0.0020877942F, 0.02078715F, 0.0207562F, -0.025367454F, 0.00011560298F,
          -0.00742391F, -0.025765939F, -0.050093923F, -0.01439048F, -0.09776404F,
          -0.06439424F, 0.005016968F, -0.0015888084F, 0.044943128F, 0.03852441F,
          0.00930763F, -0.032110315F, -0.039679203F, -0.048727732F,
          -0.044447888F, -0.0007279547F, 0.0023015465F, 0.061906353F,
          0.03464551F, -0.008025348F, 0.07671509F, 0.10288891F, 0.11021162F,
          0.19376893F, 0.15437281F, 0.13271928F, 0.0777762F, 0.06493948F,
          0.09247202F, -0.014866503F, -0.043618806F, -0.02205306F, -0.06583494F,
          -0.037037436F, -0.029998401F, -0.03526622F, 0.060863364F,
          0.0050273594F, -0.097365715F, -0.07248699F, -0.14674702F, -0.23534904F,
          -0.070831396F, -0.12766384F, -0.07784836F, -0.012266961F,
          -0.037393287F, -0.0106883235F, -0.011310979F, -0.027163422F,
          -0.006581564F, 0.007136233F, 0.029847164F, 0.03766811F, 0.049522925F,
          0.026882581F, 0.044559535F, -0.021237493F, 0.029863102F,
          -0.0031901991F, -0.028261768F, 0.012530267F, -0.09617564F,
          -0.072424375F, 0.0011129819F, -0.03266361F, 0.0002970159F,
          -0.022257313F, -0.042864088F, -0.029157506F, -0.0329245F,
          -0.027352478F, -0.019396404F, -0.025121998F, -0.07235651F,
          -0.07977835F, -0.14973003F, -0.26291618F, -0.12893608F, -0.12118155F,
          -0.10604148F, -0.027084038F, 0.046834163F, -0.082928255F, -0.1084465F,
          -0.081213586F, -0.23069295F, -0.14922774F, -0.15621905F, -0.16125663F,
          -0.03954197F, 0.015135107F, 0.06922495F, 0.0550674F, 0.0006865229F,
          0.038271494F, 0.09203269F, 0.016476208F, 0.03331717F, 0.0041316017F,
          -0.07284592F, -0.14935462F, -0.035937477F, -0.12340384F, -0.1622602F,
          -0.06471357F, -0.07073849F, -0.013916905F, -0.039872404F,
          -0.008606567F, -0.015645016F, -0.02131953F, -0.04032499F,
          -0.051350307F, 0.05923753F, -0.023488225F, -0.040223625F, 0.048222803F,
          0.0126109095F, -0.015076909F, 0.02102751F, -0.018945487F, 0.032171488F,
          0.03265826F, 0.010437862F, 0.035928443F, -0.018422753F, 0.02925841F,
          -0.03905751F, -0.079080395F, 0.044362314F, -0.059408043F, -0.04301618F,
          -0.0012324635F, -0.023390112F, -0.010907001F, -0.08102116F,
          -0.17469324F, -0.06314079F, -0.0600732F, -0.14284122F, -0.1253488F,
          0.09291902F, 0.06556821F, -0.03874619F, 0.044284716F, 0.5109818F,
          0.5730761F, -0.41225404F, -0.07230388F, 0.2939467F, -0.39447564F,
          -0.5176907F, -0.19742219F, -0.009611256F, -0.03090482F, -0.03453241F,
          -0.0006163616F, 0.012897948F, -0.00074521627F, -0.007348155F,
          0.006785788F, -0.00901736F, -0.007817141F, 0.014640415F, 0.03351635F,
          -0.014838583F, 0.0007756733F, 0.016002366F, -0.019277373F,
          -0.033363752F, 0.00013287293F, -0.012488425F, 0.00025184738F,
          0.00865986F, -0.017004965F, -0.040705085F, 0.018480992F, 0.022134686F,
          -0.0011667501F, -0.00013932554F, -0.06043423F, 0.46585122F,
          0.49953964F, -0.36502367F, -0.1253704F, 0.23777448F, -0.18134688F,
          -0.51673585F, -0.3002775F, 0.09216323F, 0.07377052F, 0.0674242F,
          -0.01937879F, -0.010930018F, -0.007512794F, -0.027777016F,
          -0.012147477F, 0.009782472F, -0.021167822F, -0.24579751F, -0.21101567F,
          0.16251826F, 0.045639172F, -0.13082328F, 0.08275314F, 0.19247043F,
          0.1209229F, -0.006587432F, 0.05462423F, -0.0026711388F, -0.016240781F,
          -0.010423529F, 0.0058937953F, 0.0117499335F, 0.0019041634F,
          0.030932052F, -0.117364846F, -0.46077406F, -0.42632926F, 0.22830522F,
          0.10658384F, -0.15384415F, 0.10585468F, 0.24773134F, 0.13352464F,
          0.0056927544F, -0.0027795907F, -0.009890944F, 0.01778824F,
          0.0064430027F, 0.014196048F, -0.011017681F, 0.019944193F,
          -0.015146838F, -0.03974678F, -0.09966982F, -0.079749495F, 0.021504378F,
          -0.042011295F, -0.05902778F, 0.0014862006F, -0.0029884134F,
          -0.021801606F, -0.019461216F, 0.03383003F, 0.06305956F, -0.0020852836F,
          -0.033512264F, 0.0357519F, 0.035129327F, -0.009590608F, 0.03123833F,
          0.01715955F, -0.020744301F, -0.025206296F, 0.011806117F, 0.029053016F,
          0.0011204878F, -0.03018044F, -0.015365466F, -0.0013031275F,
          -0.0071707796F, 0.003841247F, 0.008380513F, 0.031845734F,
          -0.023770466F, 0.028344532F, 0.024051577F, 0.03561712F, 0.025682932F,
          0.036697235F, 0.009129222F, 0.0087074805F, 0.015027448F, 0.04109899F,
          0.0054755397F, -0.0145813F, 0.027819492F, 0.0041558733F, 0.013114711F,
          0.025454953F, 0.016829902F, -0.027091438F, -0.0043311142F,
          -0.015918624F, 0.00046482502F, 0.0013384188F, -0.020392846F,
          -0.031901225F, -0.036802556F, 0.0140358005F, 0.037188333F,
          0.00062493293F, 0.050902158F, 0.034522045F, 0.016475627F,
          -0.017969191F, -0.01139062F, -0.047846217F, 0.0038515653F,
          0.013554881F, -0.012679249F, 0.016156217F, -0.013362969F,
          -0.008215345F, -0.008752396F, -0.03561201F, -0.0264898F, 0.024947943F,
          -0.02003022F, -0.06199679F, -0.034512807F, -0.0008234312F,
          -0.033907145F, -0.044033304F, 0.0059806015F, 0.034125935F,
          0.046701342F, -0.03790794F, -0.0345709F, -0.004850052F, -0.096674696F,
          -0.08020465F, -0.052691706F, 0.02006224F, -0.002100742F, 0.0059677465F,
          0.014140123F, -0.0026567988F, -0.013056528F, -0.007040263F,
          -0.010045071F, -0.007387815F, -0.081887014F, -0.054351322F,
          0.06352199F, 0.006090056F, -0.031383846F, -0.017627737F, -0.025688356F,
          -0.011072026F, -0.056768365F, -0.014579602F, 0.0033904563F,
          0.0009892687F, -0.006274311F, -0.030453512F, -0.031471692F,
          -0.025284681F, -0.04768758F, -0.05424585F, -0.011140035F, 0.012134182F,
          -0.003856648F, 0.040304445F, 0.014231909F, -0.004417438F, 0.011773901F,
          0.014181505F, -0.0034138144F, -0.0529467F, -0.021298908F,
          -0.0060627637F, -0.028814858F, -0.068748206F, -0.017238963F,
          0.012571182F, -0.005941168F, 0.0053687952F, -0.006900482F,
          0.046517596F, 0.048277333F, -0.031276062F, -0.045876708F,
          -0.021278175F, 0.008803115F, -0.020708028F, -0.01769114F, 0.026167F,
          0.006885361F, -0.030250454F, 0.0011494545F, 0.029087102F, -0.00647769F,
          -0.006194868F, 0.0071913246F, 0.012277124F, 0.03451437F, 0.01817679F,
          0.01447161F, -0.02662124F, 0.00946002F, 0.01754595F, -0.017872993F,
          -0.012214798F, 0.007310059F, -0.009448836F, 0.02420637F, 0.025396466F,
          -0.03562739F, -0.051320516F, -0.055496212F, -0.0055402745F,
          -0.0019022608F, 0.023007497F, 0.012481467F, -0.0141815655F,
          0.030741027F, -0.000110607485F, 0.030026361F, 0.03153693F,
          -0.007938511F, -0.020830052F, 0.0012840816F, -0.04007402F,
          0.005773538F, -0.015636655F, -0.12126685F, -0.12299256F, 0.0045191282F,
          0.0612963F, -0.0038156195F, 0.021145301F, -0.13592473F, -0.00882655F,
          -0.035783827F, 0.0900231F, 0.18970303F, 0.13499695F, -0.15320815F,
          0.055233896F, -0.015283255F, -0.22791892F, -0.06668914F, -0.16790266F,
          -0.016496899F, 0.1867365F, 0.034797605F, -0.25095966F, -0.07938013F,
          -0.21376218F, -0.01540234F, 0.04743978F, -0.009722693F, 0.09181043F,
          0.03937074F, 0.0015952685F, -0.004868214F, 0.04436593F, -0.07605382F,
          0.0199321F, 0.03468971F, 0.025109094F, 0.04797571F, -0.0076178056F,
          0.030874414F, 0.0046841595F, 0.033559777F, -0.032232054F, 0.04464768F,
          0.02786311F, 0.049442735F, 0.0500982F, 0.07816505F, 0.02675079F,
          0.04198211F, 0.06594791F, 0.121454336F, -0.2899964F, -0.1195194F,
          -0.21223953F, 0.025994474F, 0.18736252F, 0.08005925F, -0.30691832F,
          -0.09469911F, -0.22841744F, -0.08789218F, 0.1181003F, -0.03573828F,
          -0.3340411F, -0.061350565F, -0.22724834F, 0.0059220116F, 0.23125978F,
          0.05516807F, -0.31180006F, -0.0375323F, -0.26015183F, 0.0007576153F,
          0.34718823F, 0.069130324F, -0.32394266F, -0.030144287F, -0.25598288F,
          -0.019264288F, -0.009581725F, -0.045568638F, 0.017538024F,
          0.059612915F, -0.03451217F, -0.031280387F, 0.0010686276F,
          -0.062360644F, -0.3252143F, -0.056981266F, -0.38052964F, -0.054949675F,
          0.25675306F, -0.11719882F, -0.32609817F, -0.014442424F, -0.37025878F,
          0.021539263F, -0.045750402F, -0.04749724F, 0.01389389F, -0.0739315F,
          -0.0615883F, 0.02139462F, -0.09933961F, -0.061702184F, -0.1304847F,
          0.19624722F, -0.20650564F, -0.013359126F, 0.27828676F, -0.13821056F,
          -0.21136549F, 0.13000949F, -0.3541115F, 0.007180759F, -0.019894447F,
          -0.080275625F, -0.023353884F, -0.05084619F, -0.15835367F, 0.038617693F,
          0.01919872F, -0.049355682F, 0.03826735F, 0.0015883822F, -0.04508508F,
          0.026717328F, 0.020214487F, -0.069886886F, -0.028113041F, 0.049611453F,
          -0.04228869F, -0.014307016F, 0.056405097F, -0.08240497F, -0.012123765F,
          0.082817025F, -0.038720064F, -0.021349875F, 0.033657108F, -0.0649458F,
          -0.15156695F, -0.07098296F, -0.04517411F, -0.052273657F, 0.13795988F,
          0.03917676F, -0.09686554F, -0.03609608F, -0.045854006F, 0.0050944756F,
          0.0016441214F, -0.030803846F, 0.011772759F, 0.020345453F, 0.009167298F,
          -0.040301777F, 0.007091692F, -0.044219505F, -0.08245732F, -0.14256431F,
          -0.066585496F, 0.07070095F, 0.07034048F, 0.08963921F, -0.13765922F,
          -0.1829594F, -0.059557676F, -0.13582987F, 0.03201827F, -0.15654606F,
          0.099047236F, 0.1965992F, 0.035500217F, -0.06976517F, 0.034190323F,
          -0.1353649F, -0.10846993F, -0.006348394F, -0.13791595F, -0.014730022F,
          0.041920513F, 0.007799866F, -0.13248412F, -0.032028556F, -0.12732261F,
          -0.25672364F, 0.073912054F, -0.17325096F, -0.05662825F, 0.27989167F,
          0.06744721F, -0.28918418F, 0.107976265F, -0.11286593F, -0.15334195F,
          -0.014187857F, -0.13629153F, 0.053153384F, 0.15180174F, 0.01234376F,
          -0.11062196F, 0.052627966F, -0.16934717F, 0.010883046F, 0.023804875F,
          0.009943259F, 0.06810944F, 0.11459673F, 0.07230132F, -0.010586796F,
          0.031594895F, -0.012909904F, 0.039987348F, -0.010177352F, 0.034267932F,
          0.029013386F, 2.470812E-5F, 0.010941731F, -0.071700655F, -0.012872822F,
          -0.0026226805F, 0.0143271F, 0.026857032F, -0.038804624F, 0.018043423F,
          0.029076146F, 0.00150897F, 0.017897988F, -0.0024230052F, -0.08448027F,
          -0.037911978F, 0.049118955F, -0.04521872F, 0.0073740785F, 0.12693165F,
          0.07387411F, -0.0032928723F, 0.04283409F, -0.124740876F, 0.13286662F,
          0.023279918F, -0.03313525F, 0.03600713F, -0.055752765F, -0.071530856F,
          -0.04990265F, -0.10969661F, -0.09922832F, 0.015290786F, 0.01874801F,
          -0.03157212F, -0.010714433F, 0.032031815F, -0.030744292F,
          -0.013181958F, 0.0027522128F, -0.058198858F, -0.02062051F,
          -0.0068046716F, -0.07798047F, -0.03900551F, 0.03475403F, -0.08104241F,
          0.01424594F, 0.037455294F, -0.016099084F, -0.10192493F, 0.0047238343F,
          -0.07724913F, -0.09848006F, 0.08970932F, -0.05107749F, -0.08820713F,
          0.05211615F, -0.035695534F, -0.0028308574F, 0.024864918F,
          -0.030138824F, -0.01958054F, 0.03784568F, -0.01691181F, -0.036943845F,
          -0.019285738F, -0.07171983F, -0.052493487F, 0.049237397F, -0.1101936F,
          0.019615347F, 0.10659249F, -0.060181487F, -0.15312006F, -0.051542517F,
          -0.20510851F, -0.036301464F, -0.02605164F, 0.025761971F, 0.07564147F,
          0.09850926F, 0.104058035F, 0.048593488F, -0.015385902F, 0.06742211F,
          -0.018888082F, -0.050691992F, -0.016229166F, -0.015605816F,
          -0.14524901F, 0.0765732F, 0.02528401F, -0.035342965F, 0.029210327F,
          0.22342472F, -0.014686969F, -0.16285904F, -0.061058283F, -0.051990632F,
          0.031823862F, -0.09272453F, -0.04934653F, 0.05355947F, -0.25142053F,
          -0.037415203F, 0.08984408F, -0.028623162F, -0.04654215F, -0.09216093F,
          0.20831402F, -0.0119501855F, -0.09462353F, 0.014494427F, -0.19108044F,
          -0.080949664F, -0.042823747F, -0.08271729F, 0.1794289F, -0.07393054F,
          -0.1363256F, 0.12417808F, -0.05143802F, -0.057238717F, -0.009239372F,
          -0.0034694208F, -0.019053793F, 0.11619399F, 0.003049731F,
          -0.079087056F, -0.098498166F, 0.32607177F, -0.3968249F, 0.22104414F,
          0.47673777F, -0.55456024F, 0.15459284F, 0.3188358F, -0.3597764F,
          0.17874792F, -0.09897694F, 0.019250993F, 0.011460194F, -0.018705666F,
          0.16885091F, -0.05871342F, -0.032467328F, 0.043937534F, -0.102547616F,
          -0.15108296F, -0.22878872F, -0.061377704F, 0.17403513F, 0.20211603F,
          0.0028581053F, 0.015035225F, -0.14718752F, -0.08520422F, -0.02870606F,
          0.041260716F, -0.069902875F, 0.015928498F, 0.10819556F, -0.21916194F,
          -0.069887124F, -0.03973908F, -0.02925958F, 0.20779994F, -0.094472334F,
          -0.14706458F, 0.007942959F, -0.031708032F, 0.040082425F, -0.12871978F,
          -0.028823907F, 0.13997743F, 0.07970358F, -0.09684381F, -0.12556194F,
          0.053786196F, -0.0044048373F, 0.0035380942F, -0.10551184F,
          -0.03784151F, -0.017901367F, -0.13714403F, -0.17603116F, -0.11273689F,
          -0.118233465F, 0.02817733F, -0.09257306F, -0.14463714F, -0.13704725F,
          -0.2736263F, -0.1396745F, 0.027286796F, 0.16392584F, -0.057576243F,
          -0.01733612F, -0.07084722F, 0.18062933F, 0.030086553F, -0.15426536F,
          0.053250317F, -0.04058263F, -0.0030843804F, 0.029509587F, -0.15318935F,
          -0.075909406F, 0.08532147F, -0.079093866F, -0.011081628F, -0.24450074F,
          -0.13018869F, 0.14699556F, -0.03182506F, 0.036153004F, 0.07385839F,
          0.064334564F, -0.1107357F, -0.1702413F, -0.106170654F, -0.10482912F,
          0.014040747F, 0.16989481F, 0.29558647F, 0.1849349F, -0.07047277F,
          -0.13670741F, -0.12028104F, 0.057204306F, 0.32704645F, 0.10079526F,
          -0.58272517F, -0.3485595F, -0.43869728F, 0.150011F, 0.27131832F,
          0.11303013F, 0.076513864F, -0.06590712F, -0.19256146F, -0.0026750662F,
          0.0139137525F, -0.088782705F, -0.036755096F, -0.021512162F,
          -0.059712294F, 0.14975438F, -0.07008515F, 0.03735682F, 0.080012105F,
          -0.1485659F, 0.023763536F, 0.007316306F, -0.122431666F, -0.024428338F,
          0.06906752F, -0.007992389F, -0.05950811F, 0.010723397F, 0.14436795F,
          -0.00010246526F, -0.18664365F, -0.0677237F, 0.009515916F, -0.04624491F,
          -0.20097621F, 0.057476874F, 0.20735775F, -0.040164616F, -6.1028495E-5F,
          0.17603119F, -0.07447529F, -0.14872395F, 0.08090259F, -0.09712252F,
          0.05890476F, -0.01573238F, -0.16729273F, -0.014678086F, 0.05962412F,
          -0.060195677F, 0.092324086F, -0.07503123F, 0.028032487F, -0.0681267F,
          -0.37345463F, -0.04694603F, -0.24007024F, -0.19712913F, 0.13805333F,
          0.007841911F, 0.1234905F, 0.08224012F, -0.11058145F, -0.12613246F,
          0.03964029F, 0.019119501F, -0.09635576F, 0.02588936F, 0.10332994F,
          -0.10419607F, -0.10193948F, 0.1427759F, 0.109525576F, -0.09966225F,
          -0.1392822F, 0.15454112F, 0.05556215F, -0.110542975F, 0.2648849F,
          0.038438305F, -0.12113891F, 0.073563665F, -0.11297664F, -0.11160789F,
          -0.1767013F, -0.13751198F, 0.12972584F, -0.13448228F, -0.108021885F,
          0.118437424F, 0.108666666F, -0.03241881F, -0.041156664F, 0.1806486F,
          -0.084683485F, -0.12627445F, -0.24309182F, 0.0763755F, -0.098715365F,
          -0.24437779F, 0.12073337F, -0.09817162F, -0.018487168F, 0.12776566F,
          -0.05199568F, -0.022688238F, -0.4070118F, 0.08573927F, 0.539809F,
          -0.0059440634F, 0.2526687F, -0.055650122F, -0.4972008F, 0.03758147F,
          -0.12844837F, 0.23318243F, -0.10603814F, -0.18202017F, 0.15847299F,
          -0.18186113F, -0.03742194F, 0.041672807F, 0.0041640154F, 0.1018558F,
          0.08136453F, -0.21230865F, -0.21155198F, 0.14424197F, 0.0565667F,
          -0.19905345F, 0.02011857F, 0.03420182F, -0.048269056F, -0.013327755F,
          0.05392089F, -0.12932162F, -0.04815641F, 0.06735487F, -0.05768844F,
          -0.0773136F, 0.021049552F, -0.45487234F, -0.38614988F, -0.28147554F,
          0.23292667F, -0.20168638F, -0.26080123F, 0.8632064F, 0.38968042F,
          -0.044257626F, -0.023063635F, -0.023444423F, -0.022449866F,
          0.0071083917F, -0.021691902F, -0.039784588F, -0.0034389105F,
          -0.016874293F, -0.004643659F, 0.019242434F, -0.006992093F,
          0.002397478F, 0.020419369F, 0.012472791F, 0.015684882F, -0.009476559F,
          -0.0071210074F, 0.019708848F, -0.026892597F, -0.021794302F,
          -0.020272827F, 0.0065436456F, -0.027480178F, -0.021315107F,
          -0.025025925F, 0.01140077F, 0.0038447846F, -0.27767003F, -0.42861113F,
          -0.42285958F, 0.13574797F, -0.22948618F, -0.30099607F, 1.000479F,
          0.43393657F, -0.011570126F, 0.056008827F, -0.014714533F, 0.04396621F,
          0.029773878F, 0.018488636F, 0.030794702F, -0.028500225F, -0.018564926F,
          -0.031079685F, 0.24128105F, 0.2156058F, 0.14588349F, 0.07176584F,
          0.14521007F, 0.053275064F, -0.5749955F, -0.0748471F, 0.081834085F,
          0.022279475F, -0.0035204822F, 0.011843344F, -0.005845238F,
          -0.03279343F, -0.024178302F, 0.022906385F, 0.051470846F, 0.022537766F,
          0.1512616F, 0.10452021F, 0.02415377F, -0.113240786F, 0.11795121F,
          0.1115603F, -0.29371056F, -0.16726285F, -0.21022528F, -0.0109452205F,
          0.01179164F, 0.00090704375F, 0.009855305F, 0.04741486F, -0.0072191134F,
          0.00048937934F, 0.017886106F, -0.014691416F, -0.04465831F,
          0.009526822F, -0.024943635F, -0.12239322F, -0.0006721698F,
          -0.023774378F, 0.15336625F, -0.028812394F, -0.02602751F, -0.024407336F,
          0.017144555F, -0.021388797F, -0.011306067F, -0.036922794F,
          9.512381E-5F, -0.010787706F, -0.023671478F, -0.004288051F,
          -0.009007295F, -0.013402566F, -0.03340663F, 0.00030165628F,
          0.009121717F, 0.0017112403F, -0.0034517855F, 0.0004925093F,
          -0.005175894F, 0.01970621F, 0.0039488636F, -0.0097610615F,
          0.004528956F, 0.0056904643F, 0.033187628F, 0.003412795F, 0.0014554356F,
          0.01889712F, 0.008094728F, -0.017575046F, 0.01427397F, 0.049565166F,
          -0.008650032F, 0.001869526F, 0.0024814575F, 0.0042787655F,
          -0.007735499F, -0.0055094743F, -0.017496109F, -0.009921745F,
          0.014928218F, -0.018295556F, -0.0069890837F, -0.00037780945F,
          0.030169498F, 0.0013439313F, -0.00026713195F, 0.010897627F,
          0.014447317F, 0.068054646F, 0.022706987F, 0.03387116F, -0.020798804F,
          -0.013252912F, -0.018827064F, 0.006173332F, -0.032287396F,
          -0.009633336F, -0.012568718F, 0.02396452F, 0.022301763F, 0.008370857F,
          0.010727765F, -0.013098749F, 0.0028288607F, 0.015265378F,
          0.0031851183F, 0.016905002F, -0.03931692F, -0.022635037F,
          -0.026359946F, -0.055231016F, -0.015205045F, -0.010681669F,
          -0.03405235F, -0.058856227F, 0.009772677F, -0.01596895F, -0.024171468F,
          0.05485833F, 0.035358317F, 0.0028803572F, -0.021685159F, -0.03071142F,
          -0.048545577F, 0.017378666F, 0.0047508604F, -0.005982906F,
          -8.912899E-5F, 0.010653001F, -0.010280919F, -0.05794015F,
          -0.055634104F, 0.06965279F, -0.077113904F, -0.10912897F,
          -0.00018774427F, 0.020655794F, -0.013734532F, -0.033364102F,
          -0.016440552F, -0.0065767164F, -0.012363554F, 0.009130712F,
          -0.014603937F, -0.0055644996F, -0.00056887884F, -0.0012387971F,
          -0.014012399F, 0.02546758F, 0.026013598F, 0.018746547F, 0.009211431F,
          -0.009528478F, -0.00033033523F, 0.0077423365F, -0.0038017996F,
          -0.02241107F, -0.010484168F, 0.030542903F, -0.012355388F, -0.07469675F,
          -0.00060836755F, 0.011922218F, 0.021049384F, 0.014415281F,
          -0.004753949F, 0.010213031F, 0.025881786F, 0.038695112F, -0.06502276F,
          0.0036518155F, -0.007347097F, -0.016129257F, -0.011390726F,
          -0.005828218F, 0.003494624F, -0.025257623F, 0.00410753F, 0.0070579615F,
          0.014090887F, -0.010460597F, 0.008745293F, 0.0024511097F,
          -0.005344831F, 0.014861149F, -0.009968516F, -0.004916268F, 0.03423345F,
          -0.019698482F, -0.013367012F, 0.024465378F, 0.0032945005F,
          0.0016383049F, 0.0068626967F, 0.028443135F, -0.0010791036F,
          -0.038415253F, -0.011839645F, -0.04218791F, -0.03571474F, 0.022493338F,
          -0.0030960352F, -0.040082563F, 0.0027383014F, -0.0037236535F,
          -0.005633684F, -0.028852835F, -0.005525761F, 0.011788984F,
          0.000121209836F, -0.0010515377F, -0.12893055F, -0.055627294F,
          -0.09540036F, 0.12749921F, -0.088939965F, -0.095427595F, 0.11901483F,
          -0.057898037F, -0.08162939F, -0.00085904816F, -0.017592363F,
          0.051443752F, -0.11359745F, -0.054212388F, -0.011444169F,
          -0.033685345F, -0.077568315F, -0.02478047F, 0.0460368F, 0.018807087F,
          -0.03485059F, -0.0010636017F, 0.012530595F, -0.03276972F, 0.022212375F,
          0.011784801F, -0.012617581F, 0.06056036F, -0.003778085F, 0.0009365107F,
          0.030506548F, 0.038203713F, -0.017341241F, -0.03186941F, 0.0074602696F,
          0.056668624F, 0.05311698F, 0.04797274F, 0.06694176F, 0.053764246F,
          0.093336485F, 0.057852797F, 0.14578979F, 0.08743611F, 0.07931946F,
          0.02192247F, -0.0140943155F, -0.01862392F, 0.01023919F, 0.01380139F,
          -0.031676568F, -0.065071076F, -0.047191747F, 0.0390461F, 0.017572073F,
          0.021233525F, -0.00092260627F, 0.0026799408F, 0.056141254F,
          0.01863379F, 0.03119036F, 0.0094791325F, -0.035000872F, -0.0057190186F,
          -0.067347795F, -0.023400446F, -0.004249888F, -0.07336706F, -0.0735379F,
          -0.024524705F, -0.052146494F, 0.0056849383F, 0.0026592542F,
          0.029853037F, 0.011103971F, -0.002862684F, -0.009169496F, -0.0127469F,
          0.02252428F, 0.0052572107F, -0.0028978316F, -0.060385715F,
          -0.07422656F, -0.08916988F, -0.049833763F, -0.14220704F, -0.08753799F,
          -0.053815514F, -0.099886395F, -0.07898921F, 0.017337333F, 0.043203376F,
          -0.0053404267F, 0.020340461F, -0.0363529F, 0.00938007F, 0.0034947535F,
          0.0055765207F, -0.006170229F, 0.10232467F, 0.04176127F, 0.029459277F,
          0.08899515F, 0.07079197F, 0.05801561F, -0.014715875F, 0.09047832F,
          0.06351017F, 0.023952497F, -0.07109252F, -0.056072906F, 0.012324073F,
          -0.01092936F, -0.05884737F, -0.027614122F, 0.031214932F, -0.039695185F,
          0.006093376F, -0.022977248F, -0.010351221F, 0.056172147F,
          -0.070339285F, -0.069775715F, 0.02254798F, -0.025718555F, -0.0543817F,
          0.032312915F, 0.04202294F, 0.056239553F, 0.065926366F, 0.013442143F,
          0.084663846F, 0.1027985F, 0.04384345F, 0.0020401157F, 0.03296085F,
          0.049393803F, -0.051898558F, 0.06706938F, -0.035995174F, -0.03646341F,
          -0.11413546F, 0.04851344F, 0.09900563F, -0.04144126F, -0.0031240636F,
          0.06796206F, 0.036813978F, -0.072535485F, 0.028319374F, 0.046243906F,
          0.072508104F, -0.042136542F, -0.05237607F, -0.02349537F, -0.051630136F,
          -0.062185958F, -0.118577994F, -0.021303061F, -0.08161709F,
          -0.07682133F, -0.0706162F, 0.0022391803F, -0.040395737F, 0.0057155294F,
          -0.09240985F, -0.047629245F, -0.058049064F, -0.052751962F,
          -0.06752428F, -0.023389785F, 0.0059930347F, -0.019114425F,
          -0.033225864F, 0.028767556F, -0.033073895F, -0.044875488F,
          0.015800972F, -0.0044494825F, -0.0061274725F, 0.015911656F,
          0.06599402F, -0.04646497F, 0.04343984F, -0.108298615F, 0.025144685F,
          -0.011756771F, 0.043398954F, 0.022119356F, -0.0071639065F,
          -0.04323044F, -0.030815775F, -0.032105602F, -0.017324666F,
          0.019025529F, -0.040235814F, 0.011656789F, -0.0043371557F,
          -0.018029716F, -0.037365757F, 0.028045323F, -0.037167966F, 0.09440077F,
          -0.027967228F, 0.044966478F, 0.0028059513F, 0.015197818F, 0.02629872F,
          0.027470348F, 0.00102323F, 0.008851979F, -0.016830022F, -0.009497634F,
          -0.007605452F, -0.010325865F, 0.018201126F, -0.002016142F,
          -0.03153956F, -0.03398718F, -0.08823456F, -0.11868494F, -0.065099165F,
          -0.12320916F, -0.07336814F, -0.0357229F, 0.06967931F, 0.074919775F,
          0.053070758F, 0.079842165F, 0.10775309F, 0.10295243F, 0.037791796F,
          0.07678804F, 0.06813395F, -0.10582196F, 0.034866955F, 0.062280204F,
          -0.026159406F, -0.03385459F, -0.026308177F, 0.055822697F,
          -0.007338888F, -0.082048595F, 0.054991607F, -0.04527698F, 0.048035152F,
          -0.018739304F, 0.09778614F, 0.036174204F, 0.062349953F, -0.0018118866F,
          -0.01093976F, -0.021682013F, 0.050983068F, 0.030120512F, 0.043726236F,
          -0.008631469F, 0.0039462848F, 0.051889893F, 0.01622167F, -0.02324766F,
          -0.08020507F, -0.02708937F, -0.05163638F, -0.099441774F, -0.06851487F,
          -0.08881527F, 0.0012534041F, -0.06462764F, -0.10683038F, -0.049275335F,
          -0.06982618F, -0.026060866F, -0.09002458F, -0.14967264F, -0.093433924F,
          -0.05819836F, -0.12459301F, -0.03927897F, -0.04669718F, -0.044844247F,
          -0.057608295F, -0.05126302F, -0.049405277F, -0.06770487F, -0.08404196F,
          -0.035217118F, -0.037119642F, 0.03538483F, 0.037140377F, -0.025209766F,
          0.06280272F, 0.059553616F, -0.005480549F, -0.012887188F, -0.020348094F,
          0.022667645F, -0.0016908243F, -0.024313228F, 0.06329833F,
          -0.003361938F, -0.019530963F, 0.051078174F, 0.0043102982F,
          -0.04328688F, 0.04839256F, 0.044006392F, -0.0007776479F, -0.007157701F,
          0.025771521F, -0.05163497F, -0.03973723F, 0.0426697F, -0.023927625F,
          -0.032988343F, -0.030814024F, -0.000785183F, -0.0032617298F,
          0.016677616F, -0.037655257F, -0.01907045F, 0.046526253F, -0.04942665F,
          -0.05595367F, 0.008099549F, -0.070787676F, -0.04343961F, 0.0044620126F,
          -0.047022283F, -0.046041567F, -0.011028096F, -0.06898596F,
          -0.013638047F, 0.06411249F, -0.004379108F, -0.07947995F, 0.013167794F,
          -0.030132215F, -0.045146253F, 0.00658781F, 0.05326758F, -0.031055948F,
          0.046906147F, -0.07530164F, -0.03371601F, 0.08186545F, -0.06641534F,
          -0.03377356F, 0.10275277F, -0.033846263F, -0.033951715F, 0.3120882F,
          -0.033672106F, -0.17256758F, 0.30470878F, -0.07304551F, -0.17779008F,
          0.24617851F, -0.022038046F, -0.07576545F, 0.031716716F, -0.03220563F,
          -0.0044125495F, 0.016861621F, -0.015718276F, -0.0033309609F,
          0.012769399F, 0.012232135F, 0.0038276971F, 0.017176619F, 0.049898595F,
          -0.04887552F, 0.030934695F, 0.0016833097F, -0.08597957F, 0.014599589F,
          0.022587152F, -0.031711955F, 0.0142466035F, -0.08383201F, 0.021401675F,
          0.04347222F, -0.05722535F, 0.04656986F, 0.05156452F, -0.076574504F,
          -0.001965148F, 0.01385094F, -0.012198832F, -0.114216335F, 0.054818153F,
          -0.009718947F, -0.09658305F, 0.13341132F, -0.006021741F, -0.05190766F,
          0.0025139162F, 0.08609325F, -0.046756104F, 0.003471722F, 0.014970098F,
          -0.08980823F, 0.0009184629F, 0.006609943F, -0.084579326F,
          -0.061023604F, -0.019059721F, -0.15218985F, -0.0626795F, -0.062127955F,
          -0.22958426F, -0.0045034317F, -0.0012135236F, -0.1874843F,
          -0.108642526F, -0.02735145F, 0.011144908F, -0.07621306F, -0.061922442F,
          0.00638478F, -0.035203032F, -0.028916754F, 0.04862335F, 0.03385582F,
          0.018283082F, -0.019676575F, 0.053815756F, -0.0114103025F,
          0.0009101023F, 0.010341698F, -0.018778702F, 0.01160458F, 0.047662504F,
          -0.061199278F, -0.05267214F, 0.068125136F, -0.12345281F, -0.009458685F,
          0.0629667F, -0.0107512055F, 0.00081647426F, 0.06987486F, -0.042415503F,
          -0.059183046F, 0.09443499F, -0.044107556F, -0.024082158F, 0.01065126F,
          -0.031044763F, -0.06767644F, -0.2842466F, 0.1203909F, 0.1008014F,
          -0.43441534F, 0.21833485F, 0.156627F, -0.28406468F, 0.1504108F,
          0.0667475F, 0.005861139F, 0.010372136F, -0.022603847F, 0.07419835F,
          0.011982628F, -0.038584992F, -0.027825972F, -0.067854665F,
          -0.05115298F, 0.017124109F, 0.019363923F, -0.01099093F, 0.011211687F,
          0.044588573F, 0.0054600667F, -0.076530986F, 0.030950103F, -0.00802298F,
          0.008796287F, -0.033219673F, 0.01789992F, 0.016266223F, 0.0010438254F,
          -0.0069202012F, 0.03326845F, -0.00635963F, -0.028680017F, 0.08721912F,
          -0.09846993F, -0.047464486F, 0.06243544F, -0.17063141F, -0.04656445F,
          0.092683F, -0.121573776F, -0.026373178F, -0.014097458F, 0.014288847F,
          -0.0076519134F, -0.012743079F, -0.0063919527F, -0.009284248F,
          -0.0025984133F, -0.008231922F, 0.01912092F, -0.07237469F, 0.07697383F,
          0.009031184F, -0.15485586F, 0.1612376F, -0.039720606F, -0.097744174F,
          0.10462962F, -0.034905892F, 0.016867423F, 0.08110371F, 0.030747749F,
          0.08396189F, 0.10192228F, 0.005916416F, -0.012618391F, 0.0050428146F,
          -0.032272354F, 0.0004169638F, 0.0035648416F, 0.007661632F,
          -0.025920238F, -0.00079088664F, 0.015740031F, -0.020160865F,
          0.027688524F, -0.020497365F, -0.0050532236F, 0.017896842F,
          -0.04861515F, 0.080170274F, 0.036606155F, -0.0046641766F, 0.09357468F,
          0.0025079604F, -0.050153803F, 0.08459082F, -0.064258635F,
          -0.048343968F, 0.025526235F, -0.082661234F, -0.05544055F,
          -3.521851E-5F, 0.013624844F, -0.048109896F, -0.07168013F, 0.08399826F,
          -0.0054673925F, -0.03972429F, 0.067163706F, -0.019594492F,
          0.0030763317F, 0.031436857F, 0.0067442926F, 0.22067845F, -0.16663073F,
          -0.054123048F, 0.21436353F, -0.32452834F, -0.09652366F, 0.12618643F,
          -0.13934368F, -0.06610535F, -0.11567527F, 0.07999365F, 0.013753284F,
          -0.1364401F, 0.091360904F, 0.044554852F, -0.10850036F, 0.01856694F,
          -0.0011057578F, 0.035343718F, -0.0262492F, -0.03255149F, 0.07919991F,
          0.014766146F, -0.030295994F, 0.07353564F, -0.025214463F, -0.053637363F,
          -0.031075466F, -0.01853561F, -0.014669535F, 0.107570484F, 0.08617844F,
          0.09118121F, 0.018418517F, -0.0418672F, 0.006587377F, 0.028182069F,
          0.06480898F, -0.019555643F, -0.026726628F, 0.031509034F, -0.009524325F,
          -0.028176395F, -0.0014270339F, -0.03693599F, -0.051216487F,
          0.0037733295F, 0.062793605F, -0.051109545F, -0.046001773F,
          0.034092724F, -0.07043562F, -0.025460768F, 0.030650975F, -0.016274178F,
          0.009039543F, 0.0031100488F, -0.032696F, -0.05629671F, 0.0015137919F,
          -0.004186236F, -0.012641132F, -0.003769221F, 0.06805829F, 0.029883584F,
          0.05942837F, -0.0043772017F, -0.03632746F, 0.032824345F, -0.0697063F,
          0.016525876F, 0.04441092F, 0.030419946F, 0.0040229363F, -0.026755186F,
          0.008627255F, -0.028944753F, -0.02344978F, 0.02346581F, -0.038392637F,
          0.012166329F, 0.058131706F, 0.0725321F, 0.05279741F, 0.06395974F,
          0.031958777F, 0.0010024878F, 0.11271507F, 0.098472364F, 0.004024134F,
          -0.028912593F, -0.03772415F, -0.054778017F, -0.0077706696F,
          0.017377237F, -0.008309509F, -0.042728793F, -0.023356197F,
          -0.058380313F, 0.04282203F, 0.018515317F, -0.011163427F, 0.03895289F,
          -0.04693433F, 0.0074684317F, 0.07995714F, 0.06569153F, -0.034050196F,
          0.005497581F, -0.060248736F, -0.011989007F, -0.0009994775F,
          -0.11689387F, 0.00634567F, -0.028793527F, -0.05932244F, 0.040838216F,
          0.07459999F, 0.017286586F, 0.053861454F, -0.011378964F,
          -0.00082505046F, 0.014596612F, 0.017224092F, 0.01827653F, 0.036410112F,
          -0.008132318F, -0.07335879F, 0.04134344F, -0.035608422F, -0.016507147F,
          0.008240687F, -0.08885995F, -0.097163536F, 0.0064006564F, -0.103539F,
          -0.14610781F, -0.07048468F, -0.07970557F, -0.17260939F, -0.050038002F,
          -0.054090288F, -0.17536023F, 0.03402221F, -0.014901875F,
          -0.0042098896F, -0.017726718F, 0.009438243F, 0.017119663F,
          0.029992232F, 0.010074027F, 0.02730654F, 0.03960291F, -0.09138829F,
          -0.14677784F, -0.048665036F, -0.05541883F, -0.12050401F,
          -0.0055859117F, -0.049896892F, -0.1246707F, 0.0044551515F,
          -0.06169531F, -0.022691531F, -0.036335133F, -0.014264059F,
          -0.10980471F, 0.0081323935F, 0.048236996F, -0.008606662F,
          -0.016367236F, -0.033604197F, -0.020383073F, -0.03275397F,
          -0.020860579F, -0.01582556F, -0.017721735F, 0.034587793F, 0.014973273F,
          0.020581204F, -0.24733073F, -0.18257867F, -0.04566873F, -0.17451185F,
          -0.16543181F, -0.022203507F, -0.12973873F, -0.122428104F, -0.06391268F,
          -0.009423141F, -0.027534427F, -0.0008342399F, 0.0011914071F,
          0.06556419F, 0.0076313308F, -0.07828196F, -0.039719988F, 0.049868274F,
          0.09518406F, 0.012827691F, 0.027529718F, 0.09526101F, -0.048232745F,
          0.07222527F, 0.013954382F, 0.011373562F, 0.019411469F, -0.06626819F,
          -0.038479812F, -0.0075274133F, -0.033837788F, -0.021977877F,
          0.0824574F, -0.06347134F, -0.005306206F, 0.077839196F, 0.029952383F,
          0.02718922F, 0.06585261F, 0.066094615F, 0.027742326F, 0.03328752F,
          0.09874965F, 0.04976375F, 0.013642107F, 0.004053036F, 0.021214776F,
          0.0026560922F, 0.015162136F, 0.02853808F, 0.033669617F, 0.007463462F,
          0.022081997F, -0.0038076614F, -0.2658557F, -0.30972576F, -0.1249489F,
          -0.32060406F, -0.38244137F, -0.16919431F, -0.33343533F, -0.37779748F,
          -0.20984279F, -0.033839088F, 0.04138091F, 0.055081926F, -0.03686698F,
          -0.021496372F, 0.06408904F, -0.04754507F, -0.035908844F, -0.011768056F,
          -0.07774883F, -0.043234557F, -0.060969073F, -0.048391033F,
          -0.09094386F, 0.010638226F, -0.02321037F, -0.087770246F, -0.035980497F,
          -0.0021365504F, -0.03488289F, 0.0054141525F, -0.07117645F,
          -0.056208033F, 0.037413854F, -0.12668307F, -0.06947889F, -0.04564429F,
          0.023560379F, 0.035789877F, 0.025474386F, 0.039102793F, 0.010561027F,
          -0.05657267F, 0.07394616F, 0.07301006F, 0.037606172F, -0.22532515F,
          -0.17923315F, -0.14047667F, -0.1881461F, -0.19555812F, -0.12457212F,
          -0.13123105F, -0.1630074F, -0.047471054F, -0.022339959F, -0.014823763F,
          -0.0026725F, -0.12417085F, -0.13633089F, -0.04332676F, 0.040130593F,
          0.024285154F, 0.043153793F, -0.1108581F, -0.07738919F, -0.09958352F,
          -0.15407078F, -0.07898071F, -0.12566459F, -0.11995554F, -0.16258948F,
          -0.07161812F, 0.04034228F, 0.05503518F, -0.0028575202F, -0.013684042F,
          0.009853764F, 0.030306324F, -0.005162139F, 0.0450208F, 0.052603554F,
          0.10328342F, 0.07936134F, 0.045967616F, 0.0042085396F, -0.009122882F,
          0.0034583667F, -0.07338377F, -0.051543392F, 0.0059311264F,
          -0.018387929F, -0.059229847F, -0.019654168F, 0.09763521F, 0.04111931F,
          -0.034078613F, -0.115853816F, 0.0018433252F, 0.01536071F, 0.003272709F,
          0.07319575F, 0.08870731F, -0.08814835F, -0.081007354F, 0.063656755F,
          0.043571547F, -0.12353159F, -0.038315598F, -0.07848655F, -0.08069359F,
          -0.033693295F, 0.099204734F, -0.10100033F, -0.16554452F, 0.038898952F,
          0.22857854F, 0.13668847F, 0.07197485F, 0.1014645F, 0.04561242F,
          -0.13263592F, -0.0132546425F, 0.0916593F, -0.044396397F, -0.110233806F,
          -0.05190919F, -0.07156914F, -0.09361938F, -0.013302312F, 0.14052564F,
          0.013507776F, -0.074947014F, -0.08400393F, 0.07223952F, 0.03779728F,
          0.24324456F, 0.25907356F, 0.08948478F, -0.33834407F, -0.0073089907F,
          0.28533918F, 0.07849415F, -0.19474335F, -0.16141495F, 0.010540111F,
          -0.0253334F, -0.013429928F, 0.004200143F, 0.055953108F, 0.00903072F,
          -0.03791646F, -0.027937798F, 0.025012696F, 0.06080837F, 0.007934158F,
          -0.021844482F, 0.00019391735F, 0.081316F, 0.07961664F, -0.0039262497F,
          -0.06756094F, 0.0036633208F, -0.01864806F, -0.039062236F,
          -0.062389974F, 0.031935558F, 0.009202357F, -0.041106842F,
          -0.047166944F, -0.052024677F, 0.082051925F, -0.0042141895F,
          0.09487503F, 0.094307356F, -0.128049F, -0.004623069F, 0.07244327F,
          -0.020166436F, -0.12334961F, -0.026801517F, 0.00905482F, -0.014788366F,
          -0.07699915F, -0.012716125F, 0.05165172F, 0.036142748F, -0.0031419084F,
          -0.08459384F, -0.008593366F, -0.081953585F, -0.0514457F, -0.06677906F,
          -0.070982896F, -0.07624365F, -0.060772628F, -0.12997557F, -0.1936028F,
          -0.06526032F, -0.040410504F, -0.08622988F, -0.06995363F, 0.10716393F,
          0.11377141F, -0.11626358F, 0.065185905F, 0.12338513F, 0.061503183F,
          -0.035874914F, -0.038902327F, 0.028926805F, -0.064273976F,
          -0.08282889F, -0.042204347F, 0.10000654F, 0.016612714F, -0.10305617F,
          0.08335708F, -0.21790218F, -0.24461375F, 0.15542187F, 0.31436506F,
          -0.052957434F, -0.09557667F, -0.06575344F, 0.14310412F, 0.09012823F,
          0.02332315F, 0.0010703352F, -0.00519165F, 0.096794404F, 0.08890443F,
          -0.07442182F, -0.07838047F, -0.0296217F, -0.041992433F, -0.02663277F,
          0.069473F, -0.0037768856F, -0.2015865F, -0.2530075F, 0.07609922F,
          0.07672973F, -0.029073574F, 0.010954414F, -0.011427317F, 0.008275689F,
          -0.06188345F, -0.07043724F, -0.068219945F, 0.0014051909F, 0.016663376F,
          -0.024431897F, -0.036620397F, -0.011048885F, -0.0067983866F,
          0.059357498F, 0.050307844F, 0.024108347F, -0.040884852F, -0.032581206F,
          0.04162462F, -0.0043400833F, -0.03384786F, -0.0468959F, -0.020089867F,
          -0.028511738F, -0.026120964F, 0.0015085393F, 0.03766923F, 0.06792834F,
          0.015080461F, -0.03452185F, -0.026299791F, -0.032624293F, -0.12787798F,
          -0.06642325F, 0.0140043665F, 0.13090871F, -0.041475512F, 0.06210903F,
          0.06508956F, 0.03162904F, -0.042941727F, 0.002572701F, 0.038893525F,
          -0.06162876F, -0.08858157F, -0.0082403645F, -0.010211737F,
          0.052466623F, 0.053703666F, 0.09018023F, -0.15313554F, -0.0936889F,
          0.022896303F, 0.10268667F, -0.05985876F, -0.037037812F, 0.003320114F,
          0.07036723F, -0.05335221F, -0.10904187F, 0.031279102F, -3.0144562E-5F,
          -0.10564158F, -0.022588102F, 0.008072489F, -0.16240296F, -0.112739824F,
          0.09138881F, 0.11240259F, -0.07705243F, -0.01485162F, 0.061098326F,
          0.058787074F, 0.024129532F, 0.070895575F, 0.040789098F, -0.10042666F,
          0.024319792F, 0.09310609F, -0.002716027F, -0.103387624F, 0.03223192F,
          0.011191867F, -0.09885651F, -0.041053966F, -0.00092200993F,
          -0.042843938F, -0.11064764F, -0.05386754F, -0.0056134094F,
          -0.04466621F, -0.01727425F, -0.13972571F, -0.07122695F, 0.05500556F,
          -0.02025469F, -0.14185871F, -0.031726934F, 0.0023962716F, 0.042897645F,
          0.23679948F, 0.15691715F, -0.037263144F, -0.23939058F, 0.09376072F,
          0.2208477F, 0.0054596323F, -0.22153205F, -0.12103631F, -0.017960442F,
          -0.15169138F, -0.11884496F, 0.002157453F, 0.00833046F, -0.05962512F,
          -0.047256347F, -0.0073745213F, -0.016026277F, -0.090433024F,
          -0.07433866F, -0.01736241F, -0.0166439F, 0.006276324F, 0.014480041F,
          0.009774346F, -0.049912047F, 0.034514498F, -0.053206544F, 0.07443251F,
          0.0016566918F, 0.006700361F, -0.03953113F, -0.018469034F, 0.03458451F,
          -0.0743369F, -0.0074546416F, 0.006978024F, 0.087173656F, -0.0259038F,
          0.033071637F, -0.020748835F, -0.071114115F, 0.08356384F, -0.0877976F,
          0.0060009183F, -0.19217409F, 0.12333386F, -0.05633808F, 0.028216822F,
          -0.008066974F, -0.037394736F, 0.08176198F, -0.09969337F, -0.068827644F,
          0.100386344F, -0.1512139F, -0.020047659F, -0.045948707F, -0.21314289F,
          0.046817288F, -0.07696664F, 0.005707782F, 0.052250892F, -0.0454675F,
          0.04804093F, 0.056414157F, -0.07166193F, 0.2271602F, -0.18306358F,
          0.04909281F, 0.11749132F, -0.013274199F, -0.08127505F, 0.100063436F,
          0.00842303F, -0.033690628F, 0.08135744F, -0.087534234F, 0.10176885F,
          -0.011141323F, -0.07823218F, 0.18932907F, -0.13191104F, -0.0079117045F,
          0.07909435F, 0.034454074F, 0.10775827F, -0.12474279F, 0.15163013F,
          0.036237862F, 0.05202926F, -0.014831684F, 0.0011080458F, 0.0040196204F,
          -0.00034276178F, 0.022682719F, 0.004760827F, -0.002446565F,
          0.029511712F, -0.012976367F, -0.16826335F, 0.08414137F, -0.145634F,
          0.06308573F, -0.011123441F, -0.07643453F, 0.1523445F, -0.08101922F,
          -0.13086247F, 0.09473772F, 0.025735516F, -0.09711294F, 0.17816447F,
          -0.10162012F, 0.038014177F, -0.003564636F, -0.08037545F, -0.05547292F,
          0.15708534F, 0.025843166F, 0.1458246F, 0.15565203F, -0.036549076F,
          0.13360924F, 0.035554945F, -0.09388418F, -0.099029355F, 0.05626737F,
          -0.084503576F, -0.10609281F, 0.18672214F, -0.10466659F, 0.015554082F,
          -0.031270314F, -0.03081506F, -0.09934201F, -0.014889936F,
          -0.045317285F, -0.02341794F, -0.16875264F, -0.05009719F, 0.055044353F,
          -0.11006102F, -0.008541998F, 0.121712774F, -0.09206477F, 0.03575787F,
          0.017352296F, -0.04284179F, 0.042287096F, -0.024586732F, 0.056678507F,
          0.045375865F, -0.0013470472F, 0.05645957F, -0.17107616F, 0.12129706F,
          -0.1290085F, -0.07761032F, 0.07431127F, -0.07058319F, 0.003540524F,
          0.040542115F, -0.13532989F, -0.008025042F, -0.0070593623F,
          -0.11558942F, 0.02761865F, 0.024771538F, -0.0038802493F, 0.0027998285F,
          -0.08874038F, 0.02096335F, -0.08188126F, -0.00935783F, -0.059040237F,
          -0.062306833F, 0.036812805F, -0.046437092F, -0.04984854F, 0.2519003F,
          -0.21855734F, 0.09181258F, -0.015058558F, -0.23667957F, 0.18825532F,
          -0.21431297F, 0.040273536F, 0.1035312F, -0.09257704F, 0.23558837F,
          -0.05896838F, 0.11022729F, 0.027386412F, -0.03909188F, 0.13427709F,
          -0.066293254F, -0.068841696F, 0.005838609F, 0.031504396F, -0.09678087F,
          0.16481283F, -0.12639645F, -0.0031958509F, 0.022186147F, -0.05776417F,
          0.02361061F, -0.013746759F, 0.040205456F, -0.046820074F, 0.19033028F,
          -0.11300688F, -0.008853811F, 0.18264025F, -0.15700442F, -0.037701394F,
          0.024159953F, -0.09894749F, 0.045779333F, 0.06868807F, -0.029035952F,
          0.05968274F, -0.061772633F, 0.06251723F, -0.0036740382F, 0.039657872F,
          0.06385519F, -0.034062717F, 0.06649447F, -0.040616475F, -0.072141565F,
          0.019647708F, -0.080529615F, -0.025821842F, 0.044090264F, -0.25357756F,
          0.09639483F, 0.0025399944F, -0.23313506F, 0.0804969F, -0.04148192F,
          -0.055365622F, 0.0082996255F, -0.13351427F, 0.05466151F, 0.040241502F,
          -0.08278699F, -0.070454754F, -0.012267956F, -0.04642514F,
          -0.066132754F, -0.009021211F, 0.034649484F, -0.16968726F, 0.07009765F,
          -0.17397591F, -0.08686368F, 0.1296176F, -0.17350742F, 0.09733896F,
          0.079909675F, -0.056125913F, 0.11963127F, 0.12980944F, -0.047570236F,
          0.20196146F, -0.009796609F, 0.018950732F, 0.030075355F, -0.09177147F,
          0.05050908F, -0.08858342F, 0.01647623F, -0.077722296F, -0.08374523F,
          0.035476245F, -0.07238264F, -0.0040322784F, -0.026521878F,
          -0.028445406F, -0.0833428F, 0.018307758F, -0.057560656F, -0.06435487F,
          0.034509305F, -0.033022746F, 0.019000858F, 0.014458541F, -0.040944945F,
          0.046738118F, -0.12683491F, -0.035651207F, 0.023874575F, -0.02145681F,
          0.035226647F, -0.056551203F, -0.08272678F, 0.06384981F, -0.03948994F,
          0.053979218F, 0.08441054F, -0.01957869F, 0.037264355F, 0.031984013F,
          -0.006087527F, 0.0045780116F, -0.090285875F, 0.1568658F, -0.045364533F,
          0.09815605F, 0.10101419F, -0.118098535F, 0.0891243F, -0.04029417F,
          -0.08559277F, -0.09625697F, -0.09903668F, -0.1302071F, 0.11583441F,
          0.030722767F, 0.029169362F, 0.11859791F, 0.092866585F, 0.11106693F,
          -0.059176136F, 0.023214074F, -0.043411173F, -0.09355786F, 0.008943984F,
          0.0549962F, -0.067581125F, 0.042091493F, 0.07686185F, -0.14212625F,
          0.041538198F, -0.022747511F, -0.12239567F, -0.004832824F, 0.041623276F,
          0.050390467F, 0.08568418F, 0.046226908F, 0.022221828F, 0.0037917248F,
          -0.00593428F, -0.30045748F, -0.3852984F, -0.022920702F, -0.10021626F,
          -0.2045956F, -0.13158786F, 0.038736567F, -0.024807874F, 0.18486843F,
          0.06712695F, -0.056831185F, -0.081125006F, 0.0022681796F, 0.04850361F,
          -0.0073981895F, 0.10204725F, 0.2525433F, 0.08296347F, -0.21742634F,
          -0.26059827F, -0.061799053F, -0.06194221F, -0.049679473F, 0.020696115F,
          0.11173003F, -0.23136911F, -0.59904164F, 0.27595234F, 0.35815954F,
          0.04843816F, 0.02509223F, 0.20177689F, 0.2860931F, -0.054970812F,
          -0.030467357F, -0.012852686F, -0.017120743F, -0.0071958634F,
          0.07590782F, -0.045834944F, -0.06033982F, -0.0041846475F, -0.06028585F,
          0.012535224F, -0.010899743F, 0.043666136F, -0.012972675F, 0.008453548F,
          0.034012478F, 0.0037033022F, 0.002595357F, 0.017498173F, 0.09315538F,
          0.2689328F, -0.115381934F, -0.16191697F, 0.17528027F, -0.08193106F,
          -0.23297411F, -0.20729765F, 0.020821232F, -0.0351671F, -0.029934356F,
          0.013738897F, -0.017015541F, 0.005144039F, 0.023725823F, 0.018715251F,
          0.04894216F, -0.02892231F, -0.083990514F, -0.1374473F, -0.0493536F,
          -0.050497554F, -0.018916944F, -0.052053023F, -0.023478841F,
          -0.012592927F, -0.069516614F, 0.012423773F, 0.11188888F, -0.2110166F,
          -0.26240218F, -0.24412185F, -0.21678141F, -0.1458796F, -0.18475032F,
          0.117125146F, 0.09032555F, -0.013437714F, -0.2188475F, 0.03431996F,
          0.08064695F, -0.1445549F, -0.08333882F, -0.049977604F, 0.07013045F,
          -0.0026722266F, 0.15284784F, -0.12708092F, -0.1393121F, 0.094792515F,
          -0.06417237F, -0.06259254F, -0.083611004F, -0.25379312F, -0.33875147F,
          -0.05970983F, 0.026727537F, 0.1785805F, -0.13557312F, -0.074134484F,
          0.017886048F, -0.018297574F, -0.018107036F, -0.038135115F,
          -0.0064709214F, 0.045693032F, 0.054138467F, 0.0047261785F,
          0.020497799F, 0.014660501F, -0.054367352F, -0.012416405F, -0.16248804F,
          -0.11950624F, -0.074654624F, 0.12904842F, -0.13623999F, -0.032151394F,
          0.155317F, -0.0008680301F, 0.102023594F, 0.06121111F, -0.029155834F,
          -0.13242842F, 0.06868424F, 0.14478835F, -0.047206804F, -0.0655393F,
          0.041188024F, -0.037851464F, -0.17679428F, -0.08489176F, -0.012343019F,
          -0.05283408F, -0.05014627F, 0.044154346F, 0.048174053F, 0.019915089F,
          -0.014388661F, 0.256907F, 0.25593898F, -0.13436885F, -0.21870351F,
          0.0129657695F, -0.02164791F, -0.08375674F, -0.088464394F, 0.12872855F,
          0.103850976F, 0.04033268F, -0.016193148F, -0.024210146F, 0.06184087F,
          -0.08480832F, -0.07916901F, -0.087222666F, -0.05562152F, -0.116783775F,
          -0.056183893F, 0.06513545F, 0.109372035F, 0.061619636F, 0.029311528F,
          0.07673948F, 0.044927318F, 0.05032556F, 0.051044382F, 0.0113223F,
          -0.14501545F, 0.028119821F, 0.06902221F, 0.002844555F, 0.07017788F,
          0.050324593F, -0.045465995F, -0.030943172F, 0.030382073F, -0.10714371F,
          -0.12296391F, 0.005589657F, -0.022230458F, -0.0051922826F, 0.04868317F,
          0.07209573F, 0.09642889F, 0.03339351F, -0.12059056F, 0.0076481085F,
          0.060730007F, 0.012422472F, -0.053559694F, -0.060801286F,
          -0.123451054F, -0.16684476F, -0.08272462F, -0.036866985F, -0.07122445F,
          -0.116282545F, 0.014812412F, -0.036601175F, 0.0009506085F, 0.06656946F,
          0.15339851F, 0.041993212F, -0.055375837F, -0.037137553F, 0.04037917F,
          -0.072316416F, -0.073112205F, -0.017249493F, -0.04845199F, 0.12848166F,
          0.07478254F, -0.0061236243F, -0.02412087F, -0.008024053F, 0.031137077F,
          -0.015327145F, 0.062764876F, 0.0164067F, -0.25609294F, -0.35773852F,
          0.1656702F, 0.26443258F, 0.09378089F, -0.022364404F, -0.0067174775F,
          0.100273326F, -0.06548573F, -0.035837248F, -0.07292281F,
          -0.0017482743F, -0.01962179F, -0.07340492F, -0.03801747F,
          -0.017794034F, -0.02758665F, -0.10204262F, 0.183991F, 0.30521378F,
          -0.29920092F, -0.37700436F, -0.04722329F, -0.14035611F, -0.2671825F,
          -0.109813355F, -0.025171926F, -0.017314665F, -0.01611228F,
          0.0030653311F, 0.021804644F, 0.012591324F, -0.030987656F, -0.00662719F,
          -0.0076230345F, 0.005138353F, 0.00034995808F, 0.0041892603F,
          0.0073359977F, 0.012198827F, 0.010267616F, -0.009298051F,
          -0.0014851164F, 0.0013575976F, -0.035665933F, -0.018807173F,
          -0.014369101F, -0.015793126F, -0.00545625F, 0.0048694857F,
          -0.022882905F, -0.017574381F, -0.012722515F, 0.014082994F,
          0.020137705F, 0.018110203F, 0.03423086F, 0.03350493F, 0.024546819F,
          0.07376772F, 0.037507486F, 0.018926647F, 0.04985816F, 0.027150223F,
          -0.020585358F, 0.015608474F, 0.043117184F, 0.010363199F,
          -0.0058923806F, -0.0030817525F, 0.033781033F, -0.00036694316F,
          0.0047231577F, 0.00028391893F, 0.0096924715F, 0.0054396493F,
          0.015348327F, 0.01162864F, 0.01998717F, 0.003771436F, -0.002390664F,
          0.0067698807F, 0.011825311F, -0.014271624F, -0.030116072F,
          -0.011875197F, 0.014444781F, -0.006318679F, -0.0020295004F,
          -0.009678398F, -0.008268911F, 0.0037698953F, 0.0059415535F, 0.0328663F,
          0.01642684F, -0.0042465674F, 0.0020303654F, 0.0062531843F,
          -0.015676862F, -0.03552025F, -0.025056494F, -0.046953667F,
          -0.007059171F, -0.010416507F, -0.0166882F, -0.048608337F, -0.04239453F,
          -0.0013038024F, 0.03296297F, -0.0010988421F, 0.009290149F,
          0.097734205F, 0.024265438F, -0.015492701F, 0.03843195F, 0.04053653F,
          0.003343456F, -0.005014709F, -0.0047259578F, 0.0029132252F,
          0.0047467686F, -0.00707683F, -0.016721394F, -0.00077957974F,
          0.009689638F, 0.0069373962F, 0.0201821F, 0.0044389274F, 0.03150051F,
          0.15183344F, 0.051045805F, 0.01783099F, 0.055380233F, 0.025049545F,
          0.005925252F, 0.023558477F, 0.032193895F, 0.07236319F, 0.040835775F,
          0.06570961F, 0.033087853F, 0.06005586F, 0.032902606F, -0.012974859F,
          -0.017696984F, -0.009328975F, -0.024380002F, -0.034477368F,
          -0.011308732F, -0.01998596F, -0.009784881F, -0.021869864F,
          0.011008273F, 0.020071896F, 0.018325845F, -0.029673165F, 0.012587038F,
          0.005514425F, -0.018960888F, -0.011387063F, -0.007890985F,
          -0.0051905373F, 0.0024557535F, 0.028669797F, 0.001111281F,
          0.036907732F, 0.0080698915F, 0.008831845F, 0.03142581F, 0.009975698F,
          0.0022367924F, 0.013349603F, 0.00513752F, 0.025959376F, 0.004577912F,
          0.006079249F, -0.014961034F, -0.004773393F, 0.007351093F,
          0.0066725314F, -0.007946062F, -0.0038293505F, -0.037476875F,
          0.00070396985F, 0.016440934F, -0.023514641F, -0.024019789F,
          -0.015341899F, 0.016928732F, 0.016580673F, -0.00681178F, 0.10511466F,
          0.031051496F, 0.0019926613F, 0.05800092F, 0.0368435F, -0.0024699247F,
          0.0016930848F, 0.010373843F, 0.0038837756F, -0.009583423F,
          -0.011253585F, 0.0019009812F, 0.027168825F, 0.028909305F, 0.009690293F,
          0.008616464F, -0.0047260164F, -0.014532688F, 0.037272993F,
          -0.0038398344F, -0.005019683F, 0.018215114F, 0.034344483F,
          0.019237956F, -0.0030668257F, 0.009591872F, -0.0029817154F,
          0.026740337F, 0.057639625F, 0.04550613F, 0.0043809065F, -0.0073017455F,
          -0.011266342F, -0.00843517F, -0.0040567555F, -0.0060884343F,
          -0.004513955F, 0.012500954F, -0.0004172073F, -0.023247547F,
          -0.0053775944F, 9.2452E-5F, 0.0059673404F, 0.012150718F, 0.035607282F,
          0.01933446F, 0.0112851085F, 0.039572116F, -0.014011313F, 0.028175939F,
          0.0052071027F, 0.0057564043F, 0.0128735695F, -0.0039985585F,
          0.0017652996F, -0.00014193842F, -0.009058221F, 0.007839817F,
          0.009818126F, 0.010940217F, -0.020593598F, -0.0038306199F,
          0.00028359453F, 0.00073379267F, -0.011860001F, -0.014408245F,
          0.0030084674F, -0.01774781F, -0.0065772394F, 0.03745772F,
          0.0052304342F, 0.002831521F, 0.015246331F, 0.015550013F,
          -0.0036281408F, 0.0054596574F, 0.01929943F, 0.01693256F,
          -0.0010989681F, -0.011814384F, -0.010773089F, -0.007657448F,
          -0.0051040575F, -0.013303861F, -0.027250694F, -0.022780905F,
          -0.019760907F, -0.0053890175F, 0.024073F, 0.0065591875F, 0.019218298F,
          0.0068060867F, 0.022967322F, 0.00019471084F, 0.0051813982F,
          0.0073271417F, -0.02953401F, -0.013640501F, 0.0067193657F,
          0.010329941F, -0.010867381F, -0.0054298034F, 0.026772622F,
          -0.027899282F, -0.002650545F, 0.001193633F, -0.0032860423F,
          0.01770347F, -0.008726789F, 0.008970275F, 0.020497268F, 0.008730957F,
          0.0095410785F, 0.01618875F, 0.020022083F, 0.0071816156F, -0.01707122F,
          0.0059819487F, 0.04195569F, -0.005963326F, 0.005564691F, 0.03240138F,
          0.006319478F, 0.060824465F, 0.057327867F, 0.06896029F, 0.06338499F,
          0.05195868F, 0.0075354925F, 0.0031387433F, 0.004988559F,
          -0.0017909098F, 0.029303845F, -0.037315167F, 0.05559314F, -0.08804667F,
          -0.15778941F, -0.06598515F, -0.017150244F, -0.11161112F, -0.014773297F,
          0.07410371F, 0.1019882F, 0.14537475F, -0.0057934057F, -0.030486606F,
          0.04690298F, 0.09019318F, -0.009668207F, 0.024857042F, 0.102144994F,
          0.011299118F, 0.09126968F, -0.01924456F, -0.07348748F, -0.021170124F,
          -0.010539117F, -0.0052238284F, 0.062381282F, 0.020467661F,
          0.006098282F, 0.18260306F, -0.08316304F, -0.04119726F, 0.009517259F,
          -0.005076957F, -0.068319924F, -0.07363899F, -0.015416278F,
          -0.023767905F, 0.062122915F, -0.16712965F, -0.16963282F, -0.11514886F,
          -0.10248134F, -0.10435043F, -0.057235055F, 0.0070064706F, 0.17526753F,
          -0.05794807F, 0.2842162F, 0.42254877F, 0.20609993F, 0.3007204F,
          0.43049717F, 0.24189012F, 0.083894484F, 0.12733121F, -0.0012952709F,
          0.078974105F, 0.13759597F, 0.0416758F, 0.036946665F, 0.09964248F,
          0.016296582F, 0.1362948F, 0.0067845467F, 0.043250434F, -0.044317752F,
          -0.04548769F, 0.026142418F, -0.021387612F, 0.013149176F, 0.030434918F,
          0.0050276476F, -0.0645892F, 0.10376922F, -0.14759861F, -0.1268848F,
          -0.016642872F, -0.22050634F, -0.19666597F, -0.07695581F, 0.000973849F,
          -0.049380727F, 0.04011106F, -0.20643619F, -0.1975456F, -0.13098529F,
          -0.16833848F, -0.19218414F, -0.16536301F, 0.0031879365F, -0.03021275F,
          0.052979607F, -0.076317355F, -0.12238494F, -0.06650033F, -0.014273523F,
          -0.10318009F, -0.07712569F, -0.04073757F, -0.007782728F, 0.036882196F,
          0.06130884F, 0.10773194F, 0.1945701F, 0.085734986F, 0.07992654F,
          0.13139625F, -0.008613467F, -0.13139921F, -0.024681456F, -0.14541173F,
          -0.27426356F, -0.06488412F, -0.20957534F, -0.250334F, -0.040993944F,
          0.014560955F, 0.084723294F, 0.027496781F, -0.05522696F, 0.03350338F,
          0.013321597F, -0.04350111F, 0.031056793F, 0.052527845F, 0.060043797F,
          0.0763914F, 0.017691845F, 0.036310628F, 0.10687966F, 0.03930481F,
          0.15746054F, 0.094984785F, 0.09914887F, 0.020005748F, 0.019133102F,
          0.02253069F, -0.038229138F, -0.013283155F, -0.00058744464F,
          -0.006460164F, -0.028122064F, 0.014168066F, -0.030219331F,
          -0.005986515F, 0.0133759165F, -0.008396715F, 0.07707883F,
          -0.003656235F, -0.04164883F, -0.038587842F, -0.029268892F,
          0.024076793F, -0.119922005F, 0.017763877F, -0.04055541F, -0.18931867F,
          -0.19187468F, 0.033988964F, -0.07640976F, -0.025923656F, -0.07985927F,
          -0.019380469F, 0.067785F, 0.04821647F, 0.07168976F, 0.09833055F,
          0.09264679F, 0.08093066F, 0.14006074F, 0.1089672F, -0.023455018F,
          0.09135767F, -0.12531352F, -0.20107734F, -0.057601303F, -0.118324615F,
          -0.25684178F, -0.03617001F, 0.08849611F, 0.018291289F, 0.10143184F,
          -0.047756076F, -0.09287587F, -0.004563177F, -0.14109464F, -0.12284336F,
          0.08893705F, 0.031232927F, -0.038141955F, 0.070597686F, -0.14921328F,
          -0.20118536F, -0.08234628F, -0.15293781F, -0.2165165F, -0.081043646F,
          0.1240902F, -0.00838161F, 0.046575475F, 0.001717914F, 0.048275623F,
          -0.03044232F, -0.041638486F, 0.0059502646F, 0.04062411F, -0.04773595F,
          -0.11683541F, 0.069486536F, -0.19995455F, -0.22566263F, -0.008207653F,
          -0.21539204F, -0.28755835F, -0.04445906F, 0.09082076F, 0.02836998F,
          0.0728138F, -0.0793337F, -0.014791232F, -0.009342294F, -0.14710847F,
          -0.09136611F, 0.03962506F, 0.08614855F, 0.07088707F, 0.07183027F,
          0.040978465F, 0.023348913F, -0.025304576F, 0.06545918F, -1.7298922E-5F,
          -0.10188903F, 0.024211207F, -0.036270566F, -0.02832546F, -0.19350222F,
          -0.21938597F, -0.15137947F, -0.20604442F, -0.17569722F, -0.01860651F,
          0.001011729F, -0.04713961F, 0.013575855F, -0.07363543F, -0.06749873F,
          0.0133823585F, -0.05047629F, -0.07648819F, 0.05648558F, -0.15290087F,
          -0.15628839F, 0.0354277F, -0.054844864F, -0.05139095F, 0.015163966F,
          -0.06805653F, -0.12849092F, 0.006218495F, -0.110120706F, -0.14671575F,
          -0.0117280325F, -0.06368239F, -0.087768525F, 0.013353387F,
          -0.05503322F, -0.09260451F, 0.015665805F, 0.003703652F, -0.09155431F,
          0.14023677F, -0.30039585F, -0.3194852F, -0.087644346F, -0.22414018F,
          -0.23985769F, -0.01887803F, -0.20888773F, -0.15050584F, 0.042890415F,
          -0.17191248F, -0.11486789F, 0.13084897F, -0.09233256F, -0.14748822F,
          0.1178486F, 0.43659812F, 0.022460174F, -0.29264256F, 0.44923466F,
          -0.12699002F, -0.43187183F, 0.1785944F, -0.10931951F, -0.27202287F,
          -0.035203833F, 0.026399419F, 0.0036149647F, 0.005594025F,
          0.0020483106F, -0.0016821485F, 0.010201532F, -0.015583941F,
          5.1444127E-5F, 0.041823525F, -0.0017667332F, -0.025150588F,
          0.016207758F, 0.0076662893F, -0.020369919F, -0.008310893F,
          0.012687376F, 0.004981805F, -0.008573249F, -0.010970114F, -0.02549085F,
          0.030798731F, 0.0024853216F, 0.0014898331F, 0.04819525F, -0.032125678F,
          0.004308989F, 0.4477984F, -0.04464623F, -0.39933816F, 0.54480594F,
          -0.03956065F, -0.4952396F, 0.25156838F, -0.18227513F, -0.37375775F,
          0.012529547F, 0.06335725F, -0.015600603F, -0.05054173F, 0.005677628F,
          -0.04687877F, 0.019013025F, 0.012808682F, -0.0009638985F, -0.16440119F,
          -0.037748333F, 0.119061165F, -0.34307402F, 0.025558136F, 0.2200659F,
          -0.13254115F, 0.079722926F, 0.14896989F, -0.02379818F, -0.022699235F,
          -0.021196796F, -0.03595524F, -0.042578638F, 1.4673231E-5F,
          -0.027389316F, 0.021649366F, 0.0057919156F, -0.4497813F, 0.041543983F,
          0.20313908F, -0.5271932F, 0.24769545F, 0.28522518F, -0.22338799F,
          0.05270199F, 0.11175123F, -0.01578465F, 0.00493862F, -0.03059706F,
          -0.009817323F, 0.004593469F, -0.006359769F, -0.005343634F,
          -0.049986057F, -0.009355134F, -0.08053629F, 0.029191343F,
          0.0006696402F, -0.09399269F, -0.00267098F, 0.040665258F, -0.0790664F,
          0.055714067F, 0.021660903F, -0.038258433F, 0.0120358085F,
          -0.016512511F, 0.05852874F, 0.01093665F, -0.049876492F, -0.005364082F,
          0.017988663F, -0.034753744F, -0.0003303721F, -0.0023095151F,
          -0.0038823741F, 0.011778136F, 0.0052349144F, 0.004690493F,
          0.0012624392F, 0.005040223F, 0.009129099F, -0.043798752F, 0.046038337F,
          -0.00913715F, 0.023534022F, 0.00070383213F, 0.0119128525F,
          0.018070072F, -0.014522887F, 0.0012128426F, -0.02109389F,
          -0.006972723F, 0.033182733F, -0.043442484F, 0.0026291045F,
          0.0016401664F, -0.022197526F, -0.00022294764F, 0.021776602F,
          0.011213407F, 0.012037893F, 0.006249215F, 0.014392477F, 0.021651959F,
          0.020742208F, 0.013692569F, -0.0015270005F, -0.007324524F,
          0.022477463F, -0.035510924F, -0.026351698F, 0.0056615975F,
          -0.031620078F, 0.017459786F, -0.032406706F, -0.04201041F, 0.014869506F,
          -0.008387722F, 0.025803402F, 0.0022858253F, 0.014403467F, 0.016007971F,
          -0.01578166F, 0.0075982767F, 0.010539912F, -0.008072241F, 0.044641577F,
          0.04199938F, 0.035608966F, 0.006758644F, -0.016715838F, 0.018727725F,
          0.031726554F, 0.041683953F, 0.01670543F, 0.008520457F, -0.03385775F,
          -0.008214364F, 0.008624448F, -0.025829583F, 0.049152173F, 0.04908954F,
          -0.009779851F, -0.03322126F, 0.02893834F, -0.011143286F, -0.009758393F,
          0.029901583F, -0.050317694F, -0.010756171F, -0.011831963F,
          0.0044999444F, 0.01629374F, -0.048203997F, 0.04942008F, 0.013194655F,
          -0.033368923F, 0.016416226F, -0.039329603F, -0.0018395972F,
          -0.1053452F, -0.052260213F, -0.0007302134F, -0.052152093F,
          0.019492922F, -0.022401648F, -0.018357325F, -0.0056353593F,
          0.0070824143F, -0.04731383F, -0.01641585F, -0.004518002F, 0.051210735F,
          0.0134186065F, -0.003963906F, 0.007353568F, -0.00055256643F,
          -0.0017779496F, -0.01626967F, 0.012934475F, 0.030908028F, -0.02716319F,
          -0.051416077F, -0.019655176F, 0.020871358F, -0.005713483F,
          -0.05314116F, 0.018640498F, -0.011691184F, 0.029867927F, 0.032630682F,
          0.001579966F, 0.031681735F, -0.003598663F, -0.024442246F,
          -0.010254603F, -0.008533192F, 0.018276757F, 0.005352799F,
          -0.0032156527F, 0.02257579F, 0.0009658656F, -0.007296457F,
          -0.00937633F, -0.026777072F, 0.0088405665F, -0.0033192409F,
          -0.025611816F, -0.010235389F, -0.024399634F, 0.035343643F,
          -0.032505397F, -0.020097645F, 0.0036374114F, -0.0072823246F,
          0.00794183F, 0.023686703F, -0.019969843F, -0.020461742F, -0.057475884F,
          0.0039337706F, 0.035510357F, 0.0038311905F, 0.023226691F,
          -0.025446009F, 0.006859601F, 0.0072041443F, -0.01528275F,
          0.00024802354F, 0.008493432F, 0.011087013F, 0.0050722472F,
          -0.020226173F, 0.022901611F, -0.005781564F, -0.0882727F, -0.003333043F,
          -0.010882767F, -0.07306444F, 0.030619122F, -0.0993036F, -0.034089923F,
          -0.042709753F, 0.06918106F, 0.032962304F, 0.01809824F, -0.045838002F,
          -0.08056511F, -0.06563414F, 0.07298965F, 0.020731859F, 0.040270157F,
          0.054703474F, -0.03391737F, 0.08940823F, 0.020834595F, -0.08523269F,
          -0.026904F, 0.050898332F, -0.06831849F, -0.012345148F, -0.08518841F,
          -0.05152735F, -0.09391031F, -0.06274133F, -0.02771727F, -0.072265945F,
          -0.056784127F, -0.046338152F, 0.04699686F, -0.081581324F,
          -0.120870486F, 0.010110052F, -0.12707527F, -0.098907806F, -0.1667048F,
          -0.1541623F, -0.12744462F, -0.18459487F, 0.065372966F, 0.04384318F,
          -0.025284499F, -0.031205628F, 0.035226196F, -0.0046611563F,
          0.0052623097F, 0.028350765F, -0.014503308F, 0.033980485F,
          -0.010326414F, 0.036423743F, 0.040888794F, -0.011037863F,
          -0.011419334F, 0.047856886F, -0.049180306F, -0.03959587F, 0.1487922F,
          0.2225489F, -0.018387435F, -0.067801364F, 0.048482507F, -0.14588793F,
          0.11638318F, 0.23871635F, 0.096233405F, 0.011321351F, 0.016116686F,
          0.052676387F, -0.0034166535F, -0.023272077F, 0.011169501F,
          -0.0135367345F, -0.030678893F, 0.043501325F, 0.059514374F, 0.06941314F,
          0.029563835F, 0.062701985F, 0.0677964F, 0.009711869F, 0.023900736F,
          0.09159655F, -0.043315392F, -0.05502784F, 0.012519632F, -0.038343564F,
          0.0071008983F, 0.05594111F, 0.07052306F, -0.08111976F, -0.0071059954F,
          0.00995208F, -0.1401617F, -0.13812104F, -0.12747003F, -0.13367184F,
          -0.17231807F, -0.17078659F, -0.13580376F, -0.1320742F, -0.14362738F,
          0.03806095F, 0.060163327F, 0.07505365F, -0.06781769F, -0.08920141F,
          9.3549796E-5F, -0.024446413F, -0.057726767F, -0.00060774526F,
          -0.10591838F, -0.15685625F, -0.046138354F, -0.18587163F, -0.26549047F,
          -0.21217602F, -0.0693243F, -0.0987527F, -0.039859507F, -0.14247665F,
          -0.1487589F, -0.09532792F, -0.16726032F, -0.20601694F, -0.23088004F,
          -0.16282383F, -0.17666613F, -0.19797969F, 0.0032037385F, 0.001812383F,
          0.05976827F, 0.0066661F, 0.020365186F, 0.019743936F, 0.04845246F,
          0.013948101F, 0.03935948F, -0.025194269F, -0.024955345F, -0.02161514F,
          -0.030004011F, -0.022578014F, -0.031231167F, 0.01993047F,
          -0.0060730726F, -0.026364613F, 0.08269088F, 0.122294925F, 0.06994675F,
          0.029103175F, 0.07051822F, 0.053597786F, 0.10141282F, 0.1814032F,
          0.09059036F, -0.07033281F, -0.1731572F, -0.009603208F, 0.14657807F,
          -0.058660947F, 0.21222174F, 0.037053056F, -0.12106182F, 0.12598863F,
          -0.020781266F, 0.049734943F, 0.005293444F, -0.02527871F, 0.008536662F,
          -0.040532306F, -0.005143559F, 0.0073570157F, -0.042075496F,
          0.042184934F, -0.03789682F, 0.022491142F, 0.03840226F, -0.030303476F,
          0.055648975F, 0.06314147F, -0.011048957F, 0.039671525F, 0.03365226F,
          -0.020312544F, 0.026758302F, 0.033815466F, -0.0101825595F,
          0.0031866992F, 0.022127677F, 0.00046127965F, 0.07806577F, 0.033757262F,
          0.04985156F, 0.0732393F, -0.005627009F, -0.0068337456F, -0.010546295F,
          -0.006852424F, -0.009121424F, 0.0040029394F, 0.065616496F,
          -0.0073891096F, 0.020633088F, -0.0024207244F, -0.074645855F,
          -0.027426025F, 0.033275276F, -0.008140432F, 0.045672365F, 0.062148657F,
          -0.013183403F, 0.10238818F, 0.07892805F, 0.032474566F, 0.09886379F,
          0.06418619F, -0.04294322F, 0.07556466F, -0.109878525F, -0.20717335F,
          -0.22227807F, -0.18952946F, -0.2376973F, -0.25702375F, -0.16908453F,
          -0.20463145F, -0.1335548F, 0.04925548F, -0.02096607F, 0.05061195F,
          0.034401204F, 0.024009619F, 0.07243766F, 0.038096987F, -0.00044508555F,
          0.04347594F, 0.03430526F, 0.040823553F, -0.050573587F, 0.06029909F,
          0.060121886F, -0.012855589F, -0.008950174F, 0.00799564F, -0.015494914F,
          -0.051996574F, -0.013332257F, -0.07506412F, -0.064348854F,
          -0.048177183F, -0.07718805F, -0.09459502F, -0.06274814F, -0.08659675F,
          0.05814526F, 0.0359604F, 0.065150216F, 0.05727822F, 0.04779655F,
          0.06256614F, 0.043300204F, 0.004429816F, 0.009749442F, 0.036245536F,
          0.18733452F, 0.004938709F, -0.1522251F, -0.035831127F, -0.1826498F,
          0.08588601F, 0.23516172F, -0.019490926F, 0.06560911F, 0.012120296F,
          0.08494984F, 0.07604912F, 0.023462495F, 0.103864945F, 0.03856455F,
          -0.006660782F, 0.11629276F, -0.03137273F, -0.02522732F, 0.0051102177F,
          -0.02915991F, -0.027931815F, 0.032951575F, -0.0021319694F,
          -0.030262377F, 0.07664175F, -0.063269414F, -0.0414537F, -0.022947012F,
          0.019533549F, -0.025457038F, -0.0011271193F, -0.024831265F,
          -0.05889647F, -0.041179936F, 0.03831574F, -0.01992652F, -0.02163733F,
          0.06358325F, -0.053160574F, -0.020324918F, 0.033297416F, -0.07141442F,
          -0.011851063F, 0.021216046F, 0.06586902F, 0.039823964F, 0.03849943F,
          0.018125858F, -0.009794296F, 0.062338855F, 0.0021470839F, 0.012483217F,
          -0.000568276F, -0.019636597F, -0.0051479945F, 0.015789965F,
          -0.04202034F, -0.03470012F, -0.015255487F, -0.026222598F, -0.02008746F,
          -0.072851874F, -0.05761614F, -0.046481352F, -0.05272117F, -0.10868284F,
          -0.02877781F, -0.06806765F, -0.016453955F, 0.03995868F, -0.04614589F,
          0.0021081879F, -0.015190163F, 0.019574074F, 0.049147274F, 0.013225234F,
          -0.00057936646F, -0.020002661F, 0.0018960589F, 0.08559496F,
          0.028623855F, -0.0033880977F, 0.0009514727F, -0.10541371F,
          -0.13060154F, 0.011036931F, -0.06691531F, -0.061861936F, -0.056838863F,
          -0.04166956F, 0.008896708F, -0.020915564F, 0.07177557F, 0.12722147F,
          -0.044185348F, 0.053268F, 0.063339025F, -0.014824619F, -0.061134055F,
          -0.06958585F, -0.016636912F, -0.06112008F, -0.065169625F, -0.05814812F,
          -0.043280695F, -0.0535084F, -0.04919303F, -0.051877037F, -0.008023377F,
          -0.009315294F, 0.031169236F, -0.024699723F, -0.025114184F, 0.03188394F,
          0.0062312074F, -0.066415615F, 0.0053840256F, -0.013256335F,
          0.0013525153F, -0.0040348014F, -0.041067224F, 0.006908293F,
          -0.01265303F, 0.026320338F, -0.026019175F, -0.06611839F, -0.09710659F,
          0.02094584F, -0.11574216F, -0.11269631F, -0.032396905F, -0.12389848F,
          -0.068837605F, -0.031292908F, -0.04044476F, -0.0018503213F,
          0.060248945F, -0.017627107F, 0.013812605F, 0.031035276F, -0.030208444F,
          0.025653666F, 0.011576256F, -0.020435723F, 0.00062828854F,
          -0.0041036354F, -0.03445579F, -0.032896783F, -0.022877423F,
          -0.03477144F, -0.030952925F, 0.027049014F, 0.0447777F, 0.020496465F,
          0.032246854F, 0.07327009F, 0.032516968F, 0.060814258F, 0.035603363F,
          0.07122067F, 0.04626921F, 0.06253204F, 0.06798134F, 0.012988801F,
          0.05347748F, 0.0701564F, 0.0007748035F, 0.023742734F, 0.041302454F,
          0.0010183875F, 0.03573037F, 0.0252003F, 0.015817838F, -0.0237449F,
          -0.050865926F, -0.016115038F, -0.0292119F, -0.0031875828F, 0.05972187F,
          -0.048089128F, -0.02248122F, 0.06947618F, -0.17349412F, -0.13308343F,
          0.08292052F, -0.23127764F, -0.14895849F, -0.048857316F, 0.015162358F,
          -0.013081116F, 0.048654824F, 0.06523802F, 0.038070966F, 0.015568282F,
          0.0578882F, 0.04466756F, -0.016742604F, 0.0012149651F, -0.0566216F,
          -0.009524743F, -0.10566602F, -0.1275135F, 0.030650489F, -0.10241186F,
          -0.041851096F, -0.011584899F, -0.027748752F, -0.02586743F, 0.0537313F,
          -0.028826723F, -0.06474047F, 0.054244664F, -0.036275398F,
          -0.051010676F, 0.0058893343F, 0.0027951233F, -0.046692178F,
          0.013395196F, -0.0783753F, -0.12584F, -0.0031129674F, -0.060920753F,
          -0.03462161F, 0.023233002F, -0.021845471F, -0.052852545F, 0.029843256F,
          -0.06462699F, -0.08111349F, 0.007867605F, -0.10330157F, -0.08826599F,
          0.03616406F, -0.03918343F, -0.047843546F, 0.050765846F, -0.08989646F,
          -0.040833432F, 0.050427124F, -0.062322233F, -0.002814514F,
          -0.070094414F, -0.03710752F, -0.0043746643F, -0.012286791F,
          -0.05470412F, -0.039059304F, -0.010766418F, -0.054219685F,
          -0.028823553F, 0.030958757F, -0.032746613F, -0.035461765F,
          0.030427549F, -0.0033304398F, 0.0126005225F, 0.012497675F,
          0.011688285F, -0.016373558F, 0.0010691455F, 0.06765403F, 0.042316906F,
          0.017077032F, 0.022735415F, -0.016472999F, 0.07073226F, 0.0073050265F,
          -0.0022162918F, 0.012095844F, 0.0067307362F, 0.010086242F,
          -0.0017650101F, -0.024045251F, -0.0018937816F, -0.016654884F,
          0.003748289F, -0.015178769F, 0.024458392F, -0.056263186F,
          -0.038600598F, 0.032064036F, -0.040027063F, -0.0061961897F,
          0.04869243F, -0.04355913F, 0.011761793F, 0.0068448647F, -0.028985199F,
          -0.05188518F, 0.030257259F, -0.029951613F, -0.009817195F, 0.016458109F,
          -0.019493187F, -0.026242992F, -0.0057905936F, -0.010511251F,
          -0.00042846362F, -0.00158973F, -0.051847536F, -0.03182693F,
          -0.022808405F, -0.06870235F, -0.05537276F, -0.021495821F, 0.018165572F,
          0.058228336F, 0.010681586F, 0.030985741F, 0.08042388F, 0.00010874549F,
          0.03690366F, 0.064475216F, -0.06638965F, 0.059077654F, 0.033929512F,
          -0.021164982F, 0.03385844F, -0.007547988F, -0.0039477884F,
          0.018894665F, 0.02365662F, 0.06261936F, 0.037180796F, -0.047404528F,
          0.03914406F, -0.020170754F, -0.06405103F, 0.018345598F, -0.02354422F,
          -0.033883832F, -0.0028417103F, -0.046406355F, 0.1428386F, 0.019028775F,
          0.072585225F, 0.115535535F, 0.039491285F, 0.05584284F, 0.051367298F,
          -0.017249199F, -0.10323125F, -0.19087014F, -0.047405757F, -0.25199836F,
          -0.24474095F, -0.14317764F, -0.2705603F, -0.06864972F, 0.017358394F,
          -0.019017452F, 0.113074005F, -0.0017789181F, 0.088233106F, 0.09860733F,
          -0.015683701F, 0.119522326F, -0.053339798F, 0.01675422F, -0.005440156F,
          0.07401767F, -0.031251F, -0.023207435F, 0.0020425038F, -0.06159025F,
          -0.020912815F, 0.047549F, -0.25781828F, 0.055893462F, 0.3044046F,
          -0.2234098F, 0.16192369F, 0.29496214F, -0.11587116F, 0.27568638F,
          0.1701202F, 0.082485944F, 0.06514317F, -0.055540714F, 0.062382836F,
          0.017801922F, -0.07914201F, 0.050151683F, -0.046181157F, -0.07399469F,
          0.00876476F, -0.0087518245F, 0.040207747F, 0.034419734F,
          -0.0090602245F, 0.043642834F, 0.0005089133F, 0.08757559F, 0.013660227F,
          0.0041789617F, -0.008610675F, -0.029060097F, 0.0008732132F,
          0.030506859F, -0.04064682F, 0.0092433095F, -0.010318949F, 0.03043902F,
          -0.028921442F, 0.021970533F, 0.15629025F, -0.037637085F, 0.11005955F,
          0.117022626F, -0.011758605F, 0.12758273F, 0.031524766F, -0.027587183F,
          -0.023191111F, 0.00649254F, 0.0131563265F, 0.0024932013F, -0.0867569F,
          0.008698919F, 0.035342094F, -0.00064790057F, 0.07274913F, -0.03688849F,
          -0.19075692F, 0.004729065F, -0.21647522F, -0.21958072F, -0.04352544F,
          -0.20093404F, -0.08134376F, 0.08095774F, 0.016513273F, -0.081206664F,
          0.027876612F, -0.06304663F, -0.0508646F, -0.051976215F, -0.13443874F,
          -0.020364668F, -0.027917866F, -0.059414104F, -0.009809286F,
          0.004915003F, -0.04278583F, -0.013161728F, -0.010702085F,
          -0.022921037F, -0.019326517F, -0.014171169F, 0.040448096F,
          -0.096192345F, -0.04886003F, -0.084598094F, -0.17796865F,
          -0.029024797F, -0.091607235F, -0.04425531F, -0.013693532F, 0.03894873F,
          0.07046011F, -0.025620783F, 0.0050839577F, 0.0017277972F,
          -0.047418147F, 0.0020226685F, -0.039761797F, 0.069325075F,
          -0.078137204F, -0.18732738F, 0.046443366F, -0.18530202F, -0.21096422F,
          0.11622624F, -0.14040229F, 0.016749775F, 0.022996921F, 0.019463051F,
          -0.06779607F, 0.0056221616F, -0.04122597F, -0.08434469F, -0.014969902F,
          -0.034681506F, 0.0887093F, 0.007786239F, -0.025907062F, 0.10205386F,
          -0.05186914F, -0.0045355903F, 0.099570766F, 0.035016853F, 0.08276059F,
          0.053121716F, -0.022685645F, 0.042857878F, 0.03857316F, -0.049095787F,
          -0.00984431F, -0.0072945375F, -0.082839005F, -0.040603314F,
          -0.036996126F, -0.015807742F, -0.062972195F, -0.009976797F,
          -0.05982337F, -0.1079516F, -0.034446515F, -0.057326477F, -0.01922407F,
          -0.001435712F, -0.027714454F, 0.042392623F, 0.032478444F, 0.007821492F,
          0.020071605F, 0.023693329F, 0.00023064604F, 0.0034680695F,
          -0.011108216F, 0.05501497F, 0.031526785F, -0.012074971F, 0.052922454F,
          -0.014369529F, -0.14902124F, 0.12287262F, -0.037304692F, -0.053464983F,
          -0.033807848F, -0.0043700584F, 0.004839135F, 0.07109969F, 0.070756234F,
          0.024168758F, 0.08271438F, 0.06740807F, 0.03323858F, -0.0007204756F,
          -0.0866136F, -0.09009362F, -0.019583894F, -0.15226349F, -0.080313526F,
          -0.071391575F, -0.101307325F, -0.023253977F, -0.04507199F,
          -0.024193976F, 0.155605F, -0.047657248F, 0.080018416F, 0.1376052F,
          0.011060614F, 0.1710598F, 0.09490307F, -0.081857316F, -0.14751117F,
          -0.11236816F, -0.0691815F, -0.18560529F, -0.1490322F, -0.108471654F,
          -0.17445454F, -0.05636715F, 0.031946674F, -0.006918716F, -0.1794321F,
          0.08830675F, -0.14810778F, -0.07497528F, 0.03225488F, -0.035959013F,
          0.061787494F, -0.019465491F, -0.046304833F, 0.13447818F, -0.047067042F,
          0.045999944F, 0.124122255F, -0.11489201F, 0.13450347F, 0.004639353F,
          0.043534182F, 0.010774192F, -0.16403034F, 0.04019084F, -0.12769951F,
          -0.12993424F, 0.0333286F, -0.050499722F, 0.028205384F, 0.061772794F,
          0.062021498F, 0.06551769F, 0.022229081F, 0.07166741F, 0.02849069F,
          0.009102679F, 0.03144127F, -0.00216485F, -0.09602156F, -0.14759897F,
          0.018839931F, -0.17606756F, -0.11315898F, 0.043716535F, -0.10320337F,
          -8.707937E-5F, 0.14202958F, -0.6158848F, -0.5292462F, -0.26271537F,
          -0.52123374F, -0.12125211F, 0.46092272F, -0.22126094F, 0.6250537F,
          1.2372807F, 0.004656999F, 0.03074697F, 0.00024546668F, 0.0025312055F,
          -0.0021961937F, -0.017892862F, -0.017081939F, 0.009570745F,
          0.014645435F, -0.0035414735F, -0.016880991F, -0.0035452568F,
          -0.009468151F, 0.021094143F, 0.03500163F, 0.009370341F, 0.025905889F,
          -0.024485152F, -0.03456296F, -0.002424866F, 0.0041428967F,
          0.013737214F, 0.00708547F, -0.013575946F, 0.0024083178F, 0.013797119F,
          -0.020766305F, 0.29348937F, 0.18092167F, -0.04008345F, 0.23902953F,
          -0.030486813F, -0.349829F, 0.07795982F, -0.22462432F, -0.42684466F,
          0.024300413F, 0.015543467F, 0.039982058F, 0.023414202F, 0.00183409F,
          0.0029165028F, 0.011628007F, 0.0018608151F, -0.036567487F,
          -0.18035871F, -0.18347111F, -0.023871813F, -0.27907804F, -0.10005809F,
          0.10060019F, -0.056219865F, 0.14639838F, 0.27003148F, 0.03859734F,
          -0.014452613F, -0.02084899F, 0.014994414F, -0.012000623F, -0.02475123F,
          0.026331808F, -0.034193873F, 0.016607694F, 0.008575348F, -0.072531216F,
          -0.10107889F, 0.054546136F, -0.049717683F, -0.018630788F,
          -0.009329948F, -0.082426496F, 0.05558512F, 0.0015485116F, 0.009838618F,
          0.009839625F, -0.005841102F, 0.016371958F, -0.0035294513F,
          0.030432066F, 0.0013921281F, -0.0029256262F, 0.006261955F,
          0.015867779F, 0.0053777527F, -0.007960262F, 0.03529155F, -0.06324288F,
          0.015093779F, 0.013513503F, -0.082296476F, 0.0038620324F, 0.007819793F,
          -0.00031905496F, -0.021679306F, 0.0071920953F, -0.017178427F,
          0.051630512F, -0.008066827F, -0.035018228F, -0.021859363F,
          0.0031974833F, 0.005636734F, 0.0012648203F, 0.0011609372F,
          0.009902942F, -0.0015434327F, -0.022613369F, -0.047176976F,
          0.012935669F, -0.013943325F, -0.013105884F, -0.020380773F,
          -0.015991889F, 0.0059915734F, -0.017031875F, -0.0077032796F,
          -0.00024196375F, 0.017297272F, 0.021542259F, 0.029267916F,
          -0.0052172495F, -0.042006247F, -0.017010221F, 0.0078029693F,
          0.0022992482F, -0.0055673607F, -0.044436086F, 0.0041954066F,
          -0.0043939133F, -0.02835826F, 0.011750327F, 0.008056717F, 0.01167415F,
          0.0020091021F, 0.015441935F, 0.0044045406F, -0.00576606F,
          -0.015595845F, 0.025961211F, -0.01457533F, 0.007500248F, 0.05386316F,
          -0.078521304F, -0.017738117F, -0.010938675F, -0.04650493F,
          0.024970636F, 0.0035416523F, 0.006573187F, 0.02201944F, 0.026373064F,
          0.011789857F, 0.0027598743F, -0.011553228F, -0.025067918F,
          -0.00058325473F, -0.025647849F, -0.00036549775F, -0.006953817F,
          -0.010513611F, -0.0053040436F, 0.00903908F, -0.034212194F,
          0.019290933F, -0.010810602F, -0.041518424F, 0.027116477F,
          -0.030947905F, 0.0026919171F, 0.03064561F, 0.044208456F, 0.004465142F,
          0.003625721F, -0.013481396F, -0.012174335F, 0.009963774F,
          0.0045765317F, 0.00077532F, 0.0156149445F, 0.009570004F, -0.4890347F,
          -0.2676105F, -0.27520633F, -0.29890773F, 0.07569544F, 0.17580967F,
          -0.101813875F, 0.4065661F, 0.6959902F, -0.020464499F, 0.007710143F,
          -0.020281117F, 0.03161991F, -0.04836425F, -0.043224134F, 0.031797864F,
          -0.03860455F, -0.053670224F, -0.03417879F, -0.009076995F,
          -0.0028581808F, -0.015921598F, 0.01593293F, -0.03790792F,
          0.0065559424F, 0.016627764F, -0.030656487F, 0.011981854F,
          -0.008318203F, -0.03672277F, -0.0053435084F, -0.010463359F,
          -0.056858983F, -0.028415473F, -0.007177941F, 0.0030807743F,
          0.006481322F, 0.0300647F, 0.017037285F, 0.0067941537F, 0.037496515F,
          -0.018509846F, 0.004204122F, -0.028292937F, -0.0023738146F,
          -0.020986198F, -0.013441669F, 0.003769319F, -0.0017751323F,
          0.011115806F, 0.009823601F, -0.016996767F, 0.0011016413F,
          0.0039957673F, 0.010296639F, 0.00023547343F, 0.0106487665F,
          -0.0010487707F, 0.0027446025F, 0.00986429F, 0.008342673F,
          -0.012019083F, -0.034283593F, -0.0017083377F, -0.02923572F,
          -0.028826805F, -0.046341036F, -0.042745437F, -0.048655637F,
          0.016356993F, -0.01063284F, -0.014535155F, -0.016383672F, 0.032187663F,
          -0.0021843915F, -0.012761443F, 0.008723605F, 9.225214E-5F,
          0.005952753F, -0.013094799F, -0.026706472F, -0.023826696F,
          -0.037191626F, -0.08965557F, -6.229881E-5F, -0.0048751375F,
          -0.014330289F, -0.0116245365F, 0.073870204F, 0.08998344F, -0.11505875F,
          -0.07876003F, -0.03456368F, 0.019453488F, -0.0428527F, -0.031874277F,
          -0.04120174F, 0.0097942185F, -0.063258216F, -0.10755646F, -0.23642923F,
          -0.10310865F, -0.2251295F, -0.48817453F, -0.1996586F, 0.029649694F,
          -0.1479386F, -0.10736494F, 0.010494255F, -0.027822282F, -0.009381967F,
          -0.017356426F, -0.052455626F, -0.010764318F, -0.01863845F,
          -0.014106197F, 0.00016787284F, 0.017666997F, 0.007798245F,
          0.041993815F, 0.03491871F, 0.047071733F, 0.034725964F, 0.0075231884F,
          0.047745325F, 0.005424777F, -0.03734466F, 0.011806937F, 0.04203978F,
          -0.028942946F, -0.0061314623F, 0.020484515F, -0.019994298F,
          0.03076885F, 0.009265022F, 0.045800056F, -0.15966134F, -0.055500608F,
          0.102604695F, -0.3350204F, -0.13871743F, 0.16967008F, 0.0039632306F,
          0.028704658F, 0.017047731F, 0.017015006F, 0.012907122F, 0.023142092F,
          0.02591669F, 0.042965956F, -0.019778632F, -0.0014104005F, 0.050222535F,
          -0.13640448F, 0.02586911F, -0.017909242F, -0.00900734F, 0.21005721F,
          0.081337616F, -0.10137986F, 0.095741436F, -0.015276866F, -0.029086992F,
          -0.020223476F, -0.053161297F, 0.10334783F, 0.00071850885F,
          -0.04116893F, 0.0026331353F, 0.026197772F, 0.002103633F, -0.054268416F,
          -0.048975848F, -0.16916575F, 0.06402006F, 0.17618553F, -0.059828047F,
          -0.076785944F, 0.049269155F, -0.052462377F, -0.009181038F,
          -0.019763859F, -0.028588422F, 0.014096606F, -0.012372834F,
          -0.007094612F, -0.023530914F, -0.009751293F, 0.017866522F, 0.05231817F,
          0.0011255646F, -0.010712179F, -0.061840616F, -0.17924966F,
          -0.08126515F, -0.040205784F, -0.106953256F, -0.08760262F, 0.051740732F,
          0.03693457F, 0.017742656F, 0.04397271F, 0.0967978F, 0.05547248F,
          -0.006409684F, -0.0018964641F, 0.02963204F, -0.013528915F,
          -0.028538352F, 0.004096297F, -0.0012148326F, -0.0060425997F,
          -0.018672258F, 0.02588914F, -0.0013088901F, 0.0032415907F,
          0.0152695095F, -0.031612664F, -0.050624423F, 0.057725374F,
          -0.002707522F, -0.06141098F, -0.016241541F, -0.015179199F,
          0.025896389F, -0.016512271F, 0.012909033F, 0.02190167F, -0.013931926F,
          -0.047337547F, -0.019415092F, -0.013598814F, 0.014773062F,
          0.00023909511F, -0.040393163F, -0.019608533F, -0.0028837912F,
          0.04497541F, -0.010590902F, -0.020364648F, -0.033231925F, 0.013144363F,
          0.04025904F, 0.009627752F, -0.014845205F, -0.0029359178F,
          -0.025873937F, 0.0037628554F, 0.048313264F, 0.004611096F, 0.028701583F,
          -0.018958353F, 0.002314745F, 0.011931492F, -0.016639706F,
          -0.0014002054F, 0.019215139F, 0.052838653F, -0.021182919F,
          -0.002524591F, -0.013883906F, -0.024824137F, 0.03140663F, 0.03129399F,
          -0.020253964F, -0.014235889F, 0.017783029F, -0.04172039F,
          -0.038992673F, -0.013786866F, -0.037573084F, 0.03532217F, 0.068909734F,
          0.0742061F, 0.1476354F, 0.07779267F, 0.043308165F, 0.10287517F,
          0.12111709F, -0.03271873F, -0.006828086F, 0.022554396F, -0.011893085F,
          0.044204183F, 0.058832195F, 0.0059227585F, 0.006340656F, -0.003376796F,
          0.025688712F, -0.012336905F, -0.088984445F, 0.019026734F,
          -0.033959348F, -0.13282628F, 0.03569399F, -0.04051811F, -0.11462793F,
          0.050615545F, 0.05406468F, -0.029812783F, 0.013886632F, -0.01153751F,
          -0.0027081517F, 0.0053627766F, -0.026327766F, 0.025476193F,
          0.03847449F, 0.035696182F, 0.025561461F, -0.00914972F, 0.021753274F,
          0.0227997F, 0.0025648265F, 0.0013285351F, -0.020926863F, -0.015060234F,
          -0.02192845F, -0.026745932F, 0.013745083F, -0.0625774F, -0.03510941F,
          -0.01633544F, -0.00030909904F, -0.040714275F, 0.012323304F,
          0.045215994F, 0.059423655F, 0.03081598F, 0.051242173F, 0.0056147627F,
          0.019567737F, -0.036776662F, -0.0065385364F, -0.035700772F,
          -0.002476378F, -0.014282613F, 0.012402857F, -0.028130934F,
          -0.02246675F, 0.005982578F, -0.015640767F, -0.02126193F, -0.057468988F,
          -0.020529334F, 0.014815318F, 0.0050683566F, 0.027579669F,
          -0.020795045F, 0.0036304365F, 0.037541024F, -0.0084220385F,
          -0.012298682F, 0.028197566F, -0.0045936F, 0.08823877F, 0.05711036F,
          -0.017820776F, -0.06671909F, -0.015912922F, 0.047123797F,
          -0.008017017F, 0.022815613F, 0.02384286F, -0.06562334F, -0.016271833F,
          0.018023606F, -0.006873036F, -0.009305421F, -0.02218854F,
          -0.006552377F, -0.034759145F, 0.005899947F, 0.009054344F, 0.011429448F,
          0.020039454F, 0.025887314F, 0.049295828F, -0.02675048F, 0.028921735F,
          -0.03400849F, 0.036732793F, -0.07351725F, -0.06628786F, -0.0064776777F,
          -0.02696006F, -0.08440332F, -0.013198136F, 0.031091953F, -0.018953385F,
          0.033885464F, -0.02617413F, -0.040140055F, -0.011967782F, 0.004875996F,
          -0.005564578F, -0.013385572F, -0.02083186F, -0.12751807F,
          -0.060036786F, -0.12289697F, -0.13796125F, -0.10591972F, -0.06358013F,
          -0.12469836F, 0.0488384F, -0.042827748F, -0.019892575F, -0.007144181F,
          -0.05775227F, -0.0001909455F, -0.06204436F, 0.016615279F,
          0.0073880283F, -0.0767637F, 0.017011372F, -0.06516009F, -0.040181275F,
          -0.08611757F, -0.17261754F, -0.13597377F, -0.10548079F, -0.18859683F,
          -0.08839592F, 0.033650286F, -0.0107050715F, 0.021913411F, -0.03390333F,
          -0.069606714F, -0.0630483F, 0.08147367F, 0.029885694F, 0.018712122F,
          -0.0981099F, -0.052263647F, -0.0074185245F, -0.11317495F, -0.10710205F,
          -0.07880001F, -0.12767896F, -0.086115025F, -0.057492204F,
          0.0070214537F, 0.012180668F, 0.011329331F, 0.0002487063F,
          -0.019350573F, -0.013286424F, -0.027458943F, -0.035476293F,
          -0.061121836F, -0.031764545F, -0.097003534F, -0.072481036F,
          -0.0010821006F, -0.088060826F, -0.11848723F, -0.0023436497F,
          -0.04540426F, -0.09178539F, 0.07494383F, -0.023368312F, 0.020295631F,
          -0.010827793F, -0.15331914F, -0.11689101F, 0.13335378F, 0.0062399223F,
          0.0062303436F, -0.0030651807F, 0.07123589F, 0.010643188F, 0.011988976F,
          0.0857813F, -0.006405497F, 0.017102757F, 0.04936107F, -0.021981293F,
          -0.005767411F, -0.16088067F, -0.11872996F, -0.042649206F, -0.18816946F,
          -0.15978709F, 0.03107119F, -0.11053883F, -0.10799067F, -0.094458535F,
          -0.19335432F, -0.11264537F, -0.12939039F, -0.25872996F, -0.25011942F,
          0.05044881F, -0.054744318F, -0.0759355F, -0.024534022F, -0.018792732F,
          0.010079043F, 0.0036960249F, -0.01914772F, 0.038696576F, 0.011945273F,
          -0.011185891F, -0.018550096F, -0.019802982F, 0.049691897F, 0.04695416F,
          -0.011752567F, -0.02768171F, -0.084657535F, 0.10752929F, 0.10902372F,
          0.049769334F, -0.11257236F, -0.11274371F, -0.10498227F, -0.055878833F,
          -0.15697995F, -0.124842085F, 0.02846843F, 0.021237148F, -0.05455906F,
          0.05403952F, 0.068612084F, 0.07022077F, -0.042151015F, 0.029088577F,
          0.035725787F, 0.027192019F, 0.02518055F, 0.044653643F, -0.011529327F,
          -0.049591407F, -0.006574877F, -0.18394963F, -0.16726224F, -0.13200957F,
          -0.041136455F, -0.07754917F, -0.03870685F, 0.015080171F, -0.020762648F,
          -0.034421988F, 0.016159277F, -0.02058627F, -0.10238256F, 0.03588526F,
          -0.017341033F, -0.069578625F, -0.050430838F, -0.069457375F,
          -0.06934653F, -0.09132782F, -0.13087226F, -0.114313684F, -0.02042559F,
          -0.15147027F, -0.11754318F, -0.100312784F, -0.18810055F, -0.083488315F,
          -0.18403608F, -0.22968858F, -0.16104057F, -0.032696612F, -0.16807117F,
          -0.06807276F, -0.04319115F, -0.12157142F, -0.031692266F,
          -0.0076479386F, -0.08251658F, -0.054409966F, 0.050097063F,
          -0.054898717F, -0.082847014F, 0.033161983F, 0.013729635F, 0.009891996F,
          0.05206899F, 0.033474408F, 0.022219153F, 0.005727066F, -0.020062108F,
          -0.036236268F, 0.041141767F, 0.017362872F, 0.039676912F, -0.11980367F,
          -0.06704302F, -0.08513855F, -0.01253738F, -0.023584532F, -0.02247426F,
          0.025915597F, 0.022880062F, 0.029560931F, 0.016218392F, 0.024706617F,
          -0.0034416614F, -0.013725422F, 0.0019729796F, -0.016361674F,
          0.01268074F, 0.039817195F, 0.07201466F, -0.013560385F, -0.022447605F,
          0.027066812F, 0.13210274F, 0.06527456F, 0.0972329F, 0.014359407F,
          0.06322622F, -0.020916216F, -0.06604206F, 0.05146741F, -0.028326621F,
          -0.04457885F, 0.015948076F, -0.0078073093F, 0.027562689F, 0.026449537F,
          0.009622827F, 0.005712002F, 0.044638697F, 0.047777172F, 0.040221415F,
          0.024785317F, -0.010370774F, 0.006952855F, -0.0007219937F,
          0.00073802896F, 0.028157532F, 0.0077279634F, -0.001983561F,
          0.020707263F, 0.0022826781F, 0.05593617F, -0.013057353F, -0.061101537F,
          -0.04894824F, 0.012044327F, -0.023631262F, -0.014831304F, -0.06801328F,
          -0.0886398F, -0.08901619F, 0.054111216F, 0.012929538F, 0.03373627F,
          0.0595995F, 0.04315039F, 0.061994918F, 0.055861473F, 0.0057699746F,
          0.04365194F, 0.022589417F, -0.085203774F, -0.0037138464F, -0.06970244F,
          -0.11983446F, -0.06251234F, 0.011888256F, -0.084846534F, 0.0381536F,
          0.014877381F, 0.014830706F, -0.006610986F, 0.002396622F, 0.04329901F,
          -0.017585209F, -0.025084455F, 0.021186858F, 0.017449671F, 0.02189517F,
          -0.012224062F, 0.025114361F, 0.031033212F, -0.033401288F,
          -0.007782385F, 0.0021134503F, -0.033079877F, -0.02216397F,
          -0.019915227F, 0.028473772F, 0.047826767F, -0.06200406F, -0.10865076F,
          -0.01562079F, 0.01201069F, -0.029507605F, -0.029851897F, -0.018637832F,
          0.013947426F, -0.09479654F, -0.06820498F, 0.0868996F, -0.14145564F,
          -0.05938556F, 0.026531817F, 0.016514184F, -0.024885157F, 0.027723577F,
          -0.026536023F, 0.035215005F, -0.061314777F, -0.01433729F,
          -0.008254501F, -0.0029929574F, -0.03846905F, -0.027640436F,
          0.059653096F, -0.10387146F, -0.042979736F, 0.083353326F, -0.027855596F,
          -0.036640882F, 0.011975618F, 0.027617145F, 0.05037188F, 0.018263333F,
          0.0036719244F, 0.07036884F, 0.058271382F, -0.0062121972F, 0.058174204F,
          0.074136056F, -0.05004152F, -0.00631851F, 0.006676495F, 0.012273104F,
          0.016372142F, -0.0058332547F, -0.013105764F, -0.009408383F,
          -0.006009178F, -0.05130682F, -0.13008142F, 0.14422394F, -0.115941465F,
          0.0082013635F, 0.02457411F, 0.05885615F, 0.039238792F, -0.10986825F,
          0.05672508F, -0.009251034F, 0.013251177F, -0.027670879F, 0.00957085F,
          -0.003911817F, -0.013198695F, -0.037319083F, -0.009941727F,
          0.010693372F, 0.012518939F, -0.012087638F, 0.018722929F, -0.023725536F,
          -0.04086927F, -0.036110662F, -0.052662697F, -0.009881956F,
          -0.07354654F, -0.04100372F, -0.006342372F, -0.00312681F, -0.021567143F,
          -0.059040993F, -0.0014209255F, -0.014574829F, -0.04849878F,
          -0.03568363F, -0.06834956F, 0.015798626F, -0.067785375F, -0.020785695F,
          -0.113007195F, 0.0070428555F, -0.021405347F, -0.122356765F, 0.0780872F,
          0.006242815F, -0.052686524F, -0.041606423F, 0.0026363945F,
          -0.009057588F, -0.025609888F, -0.05917518F, -0.030867694F,
          -0.037740283F, 0.069665246F, -0.034038976F, -0.07963006F, 0.03483297F,
          0.09554135F, -0.18529531F, -0.004576782F, 0.10372725F, -0.12517287F,
          -0.050931226F, 0.08472844F, -0.061887503F, -0.035707105F, -0.08620695F,
          0.109877065F, -0.031162906F, -0.11356348F, 0.060157754F, -0.05935502F,
          0.081371635F, -0.069734454F, -0.03970524F, 0.033915676F, 0.056879904F,
          -0.023440795F, -0.014756892F, 0.06519816F, 0.036043134F, -0.07742715F,
          0.043215055F, 0.11835121F, -0.17665496F, 0.021775533F, 0.081121296F,
          -0.02785091F, -0.0671086F, 0.039513517F, 0.023871569F, -0.0478858F,
          0.016244134F, -0.01778095F, -0.04586704F, -0.0028724791F, 0.00211452F,
          0.013075415F, -0.11749122F, -0.078630835F, 0.10897788F, 0.013367666F,
          -0.22103792F, 0.19284347F, 0.04501019F, -0.13641147F, 0.06987131F,
          -0.05291908F, 0.04553723F, -0.03117172F, -0.10721931F, 0.043306485F,
          -0.034555223F, -0.09629852F, 0.031787056F, 0.05878569F, 0.13019368F,
          -0.0598233F, -0.14930522F, -0.011925348F, 0.22187734F, -0.21017632F,
          -0.12068482F, 0.16727744F, -0.10632748F, -0.0050981725F, -0.034115855F,
          0.07873906F, 0.031695224F, -0.08116535F, 0.025072388F, 0.032935712F,
          -0.025180949F, -0.007014769F, 0.07030127F, -0.066751644F, 0.03722048F,
          0.056938484F, -0.015471682F, -0.057930876F, 0.008091094F,
          0.0035338714F, -0.078789294F, -0.00015471973F, -0.014499849F,
          0.05501075F, 0.007355547F, -0.077633694F, 0.024394838F, 0.025097659F,
          0.044592343F, -0.006085678F, -0.11785211F, 0.018272208F, 0.07030301F,
          -0.0909066F, -0.08407854F, 0.22045308F, -0.023849266F, -0.10234677F,
          0.09974611F, -0.026399508F, -0.044909947F, 0.053162124F, -0.014871956F,
          -0.10593146F, 0.029680906F, 0.04147501F, 0.010220416F, 0.035629433F,
          0.024654305F, 0.03721084F, -0.014228485F, -0.018152857F, 0.07213095F,
          0.03137749F, -0.004692315F, 0.01973108F, 0.003714451F, 0.021892909F,
          -0.053955328F, 0.027696395F, -0.006249869F, -0.020378F, 0.054042716F,
          -0.007747154F, -0.0036918577F, 0.0476065F, 0.038948856F, -0.05180415F,
          -0.04923661F, -0.027733715F, 0.03599685F, -0.009023335F, -0.08961513F,
          0.0027257814F, -0.013260777F, -0.0035843153F, -0.03229937F,
          -0.016275393F, 0.03317509F, -0.07189998F, -0.0685264F, 0.005654903F,
          -0.010746895F, -0.012199448F, 0.0047742035F, -0.0005175671F,
          0.016282164F, 0.015752098F, -0.0011739046F, -0.002422709F,
          0.012313574F, 0.01964932F, -0.010293617F, -0.013557821F, 0.005097718F,
          0.015415087F, -0.011514694F, -0.04492409F, 0.023319194F, -0.004953439F,
          -0.030097244F, -0.011327847F, 0.04191044F, 0.021399815F, 0.024747936F,
          -0.051050685F, -0.11391109F, -0.025689445F, 0.016375838F, -0.02552329F,
          0.0076204618F, 0.06193451F, 0.042742237F, 0.000143958F, 0.058012873F,
          0.047230665F, -0.030756831F, -0.0039558993F, 0.030530976F,
          -0.012852892F, -0.020554196F, 0.019086184F, 0.0075699207F,
          -0.022396402F, 0.0008367138F, -0.0049707517F, 0.0040247906F,
          0.0015348338F, -0.0104189515F, -0.046212442F, 0.01878543F,
          -0.024497105F, -0.015383424F, -0.012103361F, 0.008713046F,
          -0.027003381F, -0.011201645F, -0.008768867F, 0.021470625F,
          0.024521837F, -0.011184271F, 0.06273412F, 0.00230834F, -0.016510189F,
          -0.014181372F, -0.02383991F, -0.04692733F, -0.03566055F, -0.01976337F,
          0.033939634F, -0.02728161F, -0.04852562F, 0.014657057F, -0.012587398F,
          -0.035477713F, 0.022510808F, -0.009139042F, 0.036930967F,
          -0.0016055852F, 0.03848612F, 0.043182548F, -0.007633654F,
          -0.015073837F, 0.010907981F, -0.0381039F, -0.024005372F, -0.016550103F,
          0.01660123F, 0.021567944F, 0.07983098F, -0.01583267F, 0.032656178F,
          -0.013496827F, 0.019348236F, 0.003366884F, -0.049789518F, 0.024769241F,
          0.021850387F, -0.12387028F, -0.023995163F, 0.031750444F, -0.039805494F,
          0.008631253F, -0.0029333902F, -0.007842763F, -0.00039718917F,
          0.02466703F, -0.016938517F, -0.019932404F, -0.0014246487F,
          0.010013786F, -0.0088309115F, -0.02132217F, -0.052457217F,
          -0.04259834F, -0.040641997F, -0.042675994F, -0.06232572F,
          -0.050806288F, -0.06855289F, -0.08910708F, 0.048531517F, 0.06455285F,
          -0.040723093F, 0.041354313F, 0.2097164F, 0.033415917F, 0.011626383F,
          -0.041160937F, -0.05349052F, -0.01818663F, -0.012430214F,
          -0.013912096F, 0.010763952F, 0.009302759F, -0.0038502777F,
          0.015222341F, -0.0026434348F, 0.017300133F, 0.0021801277F,
          -0.014927514F, -0.0033726434F, 0.03590588F, -0.032537766F,
          -0.014304588F, 0.0060258457F, 0.030158753F, 0.027489958F,
          0.0003102721F, 0.014732682F, -0.0033748602F, -0.029645981F,
          0.038334206F, 0.006087506F, -0.01254084F, 0.0066051567F, -0.031635154F,
          0.006183293F, -0.0016307011F, 0.012747301F, -0.006426016F,
          0.030242652F, 0.021777047F, 0.016722035F, 0.013965992F, 0.012783063F,
          0.010937096F, -0.00597313F, -0.012833135F, -0.0074667037F,
          -0.015577621F, 0.0028960956F, 0.016470935F, -0.0076212855F,
          0.001977623F, -0.024557082F, -0.02123429F, -0.029264767F,
          0.00028696517F, -0.014403651F, -0.0411505F, 0.0043621627F,
          -0.014296187F, -0.026881507F, -0.020851677F, -0.0025540614F,
          -0.013968844F, -0.061301608F, 0.057170045F, 0.06944177F, -0.0346165F,
          -0.034751363F, 0.0053128307F, 0.07719693F, 0.12157506F, 0.019748466F,
          0.13772672F, 0.31851366F, 0.081623815F, 0.022741634F, 0.09693797F,
          0.023519564F, -0.014436373F, 0.010321138F, -0.008829752F, 0.07453528F,
          0.03788343F, -0.026094353F, -0.0022817003F, 0.0073400335F,
          -0.018722698F, 0.021405872F, -0.012056123F, 0.04456431F, -0.07333075F,
          -0.14236951F, 0.00087333284F, -0.0064603905F, -0.051151466F,
          0.030597307F, 0.008881609F, 0.0064833723F, -0.026816163F, 0.010498548F,
          0.013145063F, 0.005698319F, 0.0010552203F, 0.0026613898F,
          -0.018496752F, 0.004842004F, 0.019743692F, 0.020993588F, -0.008072198F,
          -0.016372481F, -0.0018472446F, -0.0025209733F, -0.003601715F,
          0.003149248F, 0.017379211F, 0.059187323F, 0.027865956F, 0.029992448F,
          0.03187499F, 0.0028675918F, 0.02414017F, 0.00323529F, -0.02039824F,
          -0.006116867F, 0.021442302F, 0.00954774F, -0.004106806F, -0.03623377F,
          0.006706925F, -0.03826664F, -0.016354946F, 0.025260238F,
          -0.0010096531F, -0.009120515F, -0.0070030014F, 0.04650134F,
          0.016274108F, 0.0145082055F, 0.008710893F, 0.014931511F, 0.008546293F,
          -0.015616409F, -0.0018331881F, -0.0127824815F, -0.02619062F,
          0.02012993F, -0.014357532F, 0.015481325F, 0.028656388F, -0.011489788F,
          -0.007102299F, -0.017548133F, 0.0034109203F, -0.0200114F, -0.0807639F,
          -0.013012759F, -0.026775215F, -0.029202972F, -0.0042312187F,
          -0.015776487F, 0.042854965F, -0.01380465F, 0.008075757F, 0.008259374F,
          0.029298117F, 0.00766618F, -0.011020995F, 0.007990009F, 0.04045281F,
          0.06676247F, 0.03782299F, 0.072060876F, 0.058671165F, 0.00848125F,
          0.026556594F, 0.018553087F, -0.03313801F, 0.03822574F, -0.08520194F,
          0.050114505F, -0.026175309F, -0.010526992F, 0.10787548F, -0.01663586F,
          0.012134125F, -0.016928067F, -0.0030165487F, -0.019962186F,
          -0.058937788F, -0.0029264768F, -0.023769252F, -0.048925277F,
          0.00022840199F, 0.017709779F, -0.005753007F, 0.05966044F,
          -0.0053564687F, 0.032848828F, -0.020079102F, 0.03185783F, 0.012718839F,
          -0.04715586F, 0.11778684F, 0.047285765F, -0.00037312583F, 0.078433216F,
          0.03235143F, 0.061211064F, 0.11916226F, 0.051893234F, 0.05703968F,
          0.011997798F, 0.011909399F, 0.01945315F, 0.051179256F, -0.045587067F,
          0.11082414F, -0.061407577F, -0.041891553F, 0.036285885F, -0.11891961F,
          0.11138236F, -0.036345154F, -0.02535192F, -0.0038710122F, -0.04532706F,
          -0.089666985F, 0.013681116F, 0.0060206684F, -0.002710665F,
          0.051342845F, -0.022329489F, 0.042813532F, 0.09785989F, 0.03107109F,
          0.13293193F, 0.084798396F, -0.0054245973F, -0.0017099383F,
          -0.03986698F, 0.017118858F, -0.041146662F, 0.044688225F, -0.05261276F,
          -0.11238934F, -0.015827017F, -0.022577733F, -0.09918944F,
          -0.0012593794F, 0.009160255F, 0.10152779F, 0.059566814F, 0.11461331F,
          0.009530544F, -0.13447517F, 0.027937412F, -0.0038355095F, 0.04524624F,
          0.036407072F, 0.0322902F, -0.022076149F, 0.0395814F, -0.08350822F,
          -0.045208935F, -0.044019178F, -0.059963144F, 0.061944254F, 0.07064523F,
          0.034223575F, -0.074647896F, 0.06931832F, -0.092888586F, 0.01919714F,
          -0.07521469F, -0.10738411F, 0.080283076F, 0.032928884F, -0.010904733F,
          -0.07678153F, 0.0925199F, -0.09554813F, -0.026222592F, 0.0050604497F,
          -0.092017464F, 0.010049292F, -0.02424685F, -0.19319251F, -0.034891106F,
          -0.12612332F, -0.2539719F, -0.00936642F, -0.120700076F, -0.20944108F,
          -0.014774106F, 0.04028637F, 0.07610892F, 0.020637404F, 0.071055725F,
          0.06564906F, 0.01620889F, 0.043008327F, 0.036715038F, -0.026940772F,
          -0.016276019F, -0.039132643F, -0.028201802F, -0.07312215F, 0.08637696F,
          0.009734223F, -0.0043095616F, 0.058869813F, -0.009676169F,
          -0.009603926F, -0.017354831F, 0.050861146F, 0.08191123F, 0.067522384F,
          0.005649537F, 0.11843173F, 0.051064156F, 0.012268914F, -0.19646432F,
          -0.32912174F, -0.11677257F, -0.21687853F, -0.36543173F, -0.14728542F,
          -0.13741422F, -0.20618583F, 0.012276121F, -0.057447728F, 0.06502854F,
          -0.010296678F, 0.055274095F, 0.2554257F, -0.025556462F, 0.105634734F,
          0.15225025F, -0.027698448F, -0.0023447429F, -0.086817764F,
          -0.012152669F, -0.05386919F, -0.22838238F, 0.07781566F, -0.13460712F,
          -0.101589635F, 0.082279585F, 0.042325385F, 0.02857373F, -0.029690094F,
          -0.002599229F, 0.13770372F, 0.03372723F, 0.033705298F, 0.101790205F,
          -0.018579148F, 0.069807716F, -0.07966525F, 0.0210652F, -0.037130278F,
          -0.10394931F, 0.064647004F, -0.03335635F, 0.003956082F, 0.10960538F,
          0.020051753F, 0.054957908F, 0.04150807F, 0.020233003F, 0.07084798F,
          0.118111305F, 0.029299816F, 0.07150213F, 0.026111567F, 0.07338153F,
          0.04711223F, 0.02711262F, 0.055999003F, 0.037969593F, 0.004667883F,
          0.023521015F, 0.0055687116F, 0.015750226F, -0.029475575F, 0.026109142F,
          0.020234277F, -0.028230311F, 0.022802448F, -0.035153866F,
          -0.0072916527F, -0.05378152F, -0.03351269F, 0.029422434F, 0.042668264F,
          -0.0408105F, 0.023160562F, 0.04028625F, 0.040789608F, 0.008155976F,
          0.068472534F, 0.05489777F, -0.012420234F, 0.053102408F, -0.059251618F,
          0.07216793F, 0.10116259F, -0.09352237F, 0.09928038F, 0.07622796F,
          -0.066579714F, 0.011186815F, -0.04025849F, -0.014963482F,
          -0.049430422F, -0.009022163F, 0.024746718F, -0.05976728F, 0.062579155F,
          0.12406054F, -0.1670461F, -0.1181352F, 0.036232963F, -0.10485558F,
          -0.21612018F, -0.078608505F, 0.022275964F, -0.11082556F, -0.110912114F,
          -0.012853999F, 0.0025679835F, -0.027704615F, -0.013313276F,
          0.022804635F, -0.019831227F, 0.018790314F, -0.034047376F, -0.03285065F,
          0.0024455073F, -0.036725268F, 0.06559754F, -0.12901673F, -0.04347142F,
          -0.011491172F, -0.07509198F, -0.025285846F, 0.09469828F,
          -0.0068880864F, 0.04876008F, -0.008717583F, 0.013242141F, 0.04269114F,
          0.022684524F, 0.0138955945F, 0.005265081F, 0.0052873595F, 0.076982655F,
          -0.08047069F, -0.066545725F, -0.02428129F, -0.16210996F, -0.02015702F,
          -0.036895595F, -0.007787249F, 0.09693595F, -0.06286679F, -0.027812172F,
          0.03592099F, 0.008416712F, -0.04686424F, 0.04065223F, -0.020479156F,
          0.004418658F, 0.02975149F, -0.021223506F, -0.031846117F, 0.025648488F,
          0.04185074F, -0.057937F, -0.0033200965F, 0.040113732F, 0.0062538404F,
          -0.013664053F, -0.047168475F, -0.005128412F, -0.017709643F,
          -0.010708735F, -0.020745125F, -0.0003851675F, 0.02610352F,
          0.042005535F, -0.0048959916F, -0.038111307F, 0.024789335F,
          0.024308441F, -0.013467184F, -0.015465849F, 0.0041012624F,
          -0.022702307F, 0.002901161F, 0.03357074F, 0.012185246F, 0.016777555F,
          0.02574181F, 0.037630238F, 0.050535195F, 0.0638777F, -0.06908355F,
          0.028208388F, 0.06952714F, -0.011346553F, -0.027568372F, -0.0651784F,
          0.09305629F, -0.022435032F, -0.08208665F, 0.02517384F, 0.0850466F,
          -0.06206018F, 0.1700356F, 0.09589599F, -0.09516528F, 0.077927545F,
          0.18268137F, 0.08308179F, -0.046855927F, 0.05429258F, 0.11338347F,
          0.006509061F, -0.0042457203F, -0.030152451F, 1.2960576E-5F,
          -0.013965201F, -0.07571252F, 0.04976975F, 0.027985007F, -0.024201717F,
          0.007673885F, -0.09276275F, -0.060316764F, 0.018105026F, -0.054323286F,
          -0.14227735F, 0.011323496F, -0.0003945147F, -0.039138746F,
          0.020478224F, -0.020027976F, -0.0031752521F, 0.058636293F, -0.0343268F,
          -0.055501424F, 0.0045748632F, 0.018855805F, -0.052003793F,
          0.035696242F, 0.042792458F, 0.015633114F, 0.01110635F, 0.04601745F,
          0.04861503F, -0.031417686F, 0.014894297F, 0.028923724F, -0.039543115F,
          -0.06352969F, -0.026090218F, 0.05939982F, -0.05443255F, -0.13515325F,
          0.058230404F, 0.09706612F, -0.15661378F, -0.061465833F, -0.13632275F,
          -0.048954032F, -0.034891818F, -0.20794125F, -0.1371376F, 0.12379061F,
          0.0063983733F, -0.057825938F, -0.06020902F, -0.026283827F,
          -0.007424844F, -0.035237905F, -0.10460218F, -0.0053094616F,
          0.021937903F, -0.035113115F, -0.033314902F, -0.014616337F, 0.08658949F,
          0.014202855F, -0.066814706F, 0.17408712F, 0.045816667F, 0.012672967F,
          0.12799768F, 0.06670689F, 0.008415559F, -0.07500642F, 0.019312538F,
          0.056744915F, -0.0803882F, -0.055339742F, 0.025183247F, 0.040757302F,
          -0.01878864F, -0.03799498F, -0.0021164159F, -0.0007942773F,
          -0.0045797257F, -0.031113368F, -0.013916994F, 0.029263303F,
          -0.018275201F, -0.0014823136F, 0.0062576286F, 0.01652498F,
          -0.0029161463F, -0.06618813F, 0.017302144F, -0.014761228F,
          -0.052026432F, 0.030303974F, 0.031224951F, -0.0043924917F,
          0.023857309F, 0.016599897F, 0.00859204F, -0.11000937F, 0.018346159F,
          0.068861224F, -0.028587222F, -0.026742196F, -0.14722979F, -0.14757067F,
          -0.0762046F, -0.06828412F, -0.30880487F, -0.19643271F, 0.0678059F,
          -0.12549154F, -0.14287515F, -0.1136899F, -0.059149384F, 0.015889773F,
          -0.032703128F, -0.04253132F, 0.0095240865F, 0.034317676F, 0.014773916F,
          0.003952064F, 0.029252121F, 0.09772189F, 0.08717224F, 0.028712098F,
          0.25304744F, 0.15923648F, 0.046942078F, 0.04233452F, 0.042460706F,
          0.008298388F, -0.028596278F, 0.006458602F, 0.024928166F, -0.029687207F,
          -0.01832711F, 0.017332805F, -0.009541797F, 0.011581793F, -0.008110322F,
          0.11023782F, 0.04995157F, -0.054502744F, 0.029413145F, 0.04217488F,
          0.053835593F, -0.008654511F, -0.011195851F, 0.035745036F, 0.028290138F,
          -0.005827722F, 0.033234566F, 0.07736456F, 0.073204555F, -0.011648294F,
          0.03189353F, 0.05576535F, -0.19608434F, -0.17229354F, -0.015910564F,
          -0.065629296F, -0.33437294F, -0.18320602F, 0.08183531F, -0.12754838F,
          -0.17243773F, -0.042530134F, -0.06277218F, 0.009121076F, -0.032270476F,
          -0.08111087F, -0.04871893F, -0.01991762F, -0.12708731F, -0.06459409F,
          0.007610056F, -0.062338192F, 0.012667921F, 0.031846654F, -0.06902083F,
          -0.012244145F, 0.04347297F, 0.0021774576F, -0.07215409F, -0.03389604F,
          -0.05954127F, -0.034031577F, -0.035888214F, -0.08900823F,
          -0.024473308F, 0.014333241F, -0.04611014F, -0.0051601185F,
          -0.035349037F, -0.054244313F, -0.0076132677F, -0.0061789197F,
          -0.019974584F, 0.024000969F, 0.022339009F, -0.048247747F, -0.03220533F,
          -0.0008302559F, 0.023187835F, -0.0042114262F, 0.0065012584F,
          0.036342155F, 0.03046779F, -0.0099419365F, 0.024235794F,
          -0.0024789525F, -0.007369101F, -0.034450248F, -0.057659477F,
          0.0073919524F, -0.030331545F, -0.02562113F, 0.015531981F, 0.03129018F,
          -0.0064828363F, -0.019400613F, -0.0050423136F, 0.017211638F,
          -0.054373838F, 0.023946501F, 0.043754246F, -0.07006789F, 0.0060910163F,
          0.007944504F, 0.00729612F, 0.066408955F, -0.05371134F, -0.019246338F,
          0.039815143F, 0.0009006789F, -0.040150415F, -0.016832156F,
          0.007162335F, -0.02865571F, -0.122237325F, -0.029028242F, 0.034633532F,
          -0.07154382F, 0.00841672F, 0.033493534F, -0.018454364F, -0.012890275F,
          0.045635458F, 0.08002017F, 0.027410738F, -0.14244132F, 0.09301929F,
          0.006256833F, -0.1451626F, 0.08327222F, -0.044627856F, -0.09305579F,
          0.15522382F, -0.010139996F, -0.085848935F, 0.065407835F, 0.032534726F,
          -0.025480775F, -0.013794771F, 0.012687317F, -0.069193445F, 0.08668462F,
          -0.049762215F, -0.04484017F, 0.06238076F, -0.0060318485F,
          -0.038654245F, 0.030821351F, 0.043348376F, -0.017760247F, -0.07305476F,
          0.21720217F, 0.032149345F, -0.06176462F, 0.15305471F, 0.037251F,
          -0.027480043F, 0.036479868F, 0.042160004F, 0.0053143958F,
          0.0064016692F, -0.010864551F, 0.0037968312F, 0.028359309F,
          -0.0036072165F, -0.035498682F, 0.020384684F, -0.064888544F,
          -0.08461905F, 0.07781349F, 0.050924703F, -0.4003682F, 0.04748528F,
          0.03542228F, -0.23600386F, 0.04498783F, -0.086759545F, 0.089624554F,
          -0.024181042F, -0.08212496F, 0.020874735F, -0.013005809F,
          -0.016891435F, -0.014528336F, 0.05556989F, -0.031585205F,
          -0.023337211F, -0.022935998F, -0.036267757F, -0.036682144F,
          -0.09713493F, -0.009092099F, 0.033278525F, -0.0698635F, -0.13013425F,
          0.05230885F, -0.014912994F, -0.075573504F, -0.05404091F, 0.0645872F,
          -0.05060888F, -0.16014038F, 0.10802636F, -0.15547091F, 0.016966324F,
          -0.22544068F, -0.14288479F, 0.043170907F, -0.1542297F, -0.14682727F,
          -0.09911226F, -0.059262175F, 0.0146313105F, -0.009328686F,
          -0.018554278F, -0.06483052F, -0.077808306F, 0.04211531F, -0.032866847F,
          -0.13281956F, 0.0074467F, 0.020600168F, -0.16675216F, 0.010998755F,
          0.08443151F, -0.108457856F, 0.011275389F, 0.07589167F, -0.075472586F,
          0.013989253F, -0.09530086F, 0.15063244F, -0.08750908F, -0.10490376F,
          0.05508906F, 0.0670505F, -0.12157406F, -0.09492228F, 0.09564547F,
          -0.09570418F, 0.09591871F, -0.0053795315F, -0.08469581F, 0.11309883F,
          0.025834592F, -0.13381918F, -0.036742635F, 0.027750278F, 0.23020957F,
          -0.35395882F, 0.034069765F, 0.30850783F, -0.3957257F, -0.084949054F,
          0.22456545F, -0.04863301F, -0.13449527F, -0.07041566F, 0.12795264F,
          0.037534058F, -0.10717047F, 0.18603411F, 0.037722282F, -0.03312097F,
          0.07500922F, 0.08577928F, 0.017057449F, 0.009621777F, 0.02415137F,
          -0.018191658F, 0.018476054F, -0.01846408F, -0.06310288F, 0.041864153F,
          0.027019951F, 0.059804376F, 0.0359633F, -0.09646513F, 0.008114913F,
          0.18062177F, -0.08041458F, -0.09055443F, 0.07506298F, -0.060984574F,
          0.031860873F, 0.023548799F, 0.11372869F, -0.06503653F, 0.12275735F,
          0.08393295F, -0.08086277F, 0.0019631924F, -0.015274064F,
          -0.0040185973F, 0.004120805F, -0.023429807F, -0.02371132F,
          0.026961504F, -0.0018295217F, -0.05537694F, 0.021878257F, 0.030958707F,
          0.08948107F, -0.118450254F, -0.00940003F, 0.09066737F, 0.043462623F,
          -0.12544958F, -0.0044771037F, 0.027503893F, -0.032513432F,
          -0.047458105F, -0.07463322F, 0.002489353F, 0.077874415F, -0.03046685F,
          -0.014122512F, 0.07657453F, 0.036192235F, 0.018283045F, 0.0136616845F,
          0.010566155F, 0.00039440935F, -0.057393845F, 0.007458577F,
          0.0012075728F, 0.017089112F, -0.031659294F, -0.017556036F,
          0.024966845F, -0.027417969F, -0.007067317F, 0.095706046F,
          -0.011143131F, -0.056228288F, 0.044884987F, 0.06462268F, -0.047422342F,
          -0.17442109F, 0.09479432F, -0.015310619F, -0.046247587F, 0.051181138F,
          0.006502662F, -0.07926157F, -0.10340023F, 0.13621867F, 0.051315334F,
          -0.061770663F, -0.06879117F, 0.033742677F, 0.06802211F, -0.09988313F,
          0.014118679F, -0.017988803F, -0.070949696F, -0.13639948F, 0.17371693F,
          -0.05598258F, -0.12985563F, 0.0037342997F, 0.12000276F, -0.15347007F,
          -0.06444734F, 0.039565228F, 0.03609655F, -0.026314784F, 0.06946952F,
          0.040575255F, -0.047691472F, 0.053403925F, -2.137096E-6F, 0.040509272F,
          -0.05339622F, -0.035259753F, -0.0060056113F, -0.02764959F,
          -0.02621421F, -0.015035415F, -0.048096996F, 0.00950007F, 0.03190843F,
          0.00884348F, -0.0684072F, 0.030348571F, -0.06940411F, 0.017282967F,
          0.10541163F, 0.07269451F, -0.034395725F, 0.13565296F, 0.052955728F,
          -0.023062382F, 0.01650624F, -0.11080862F, 0.0021441577F, 0.067804F,
          -0.039384805F, -0.007921599F, -0.016841756F, -0.039466567F,
          -0.04101065F, 0.09555866F, 0.020931877F, -0.03496614F, 0.11349284F,
          -0.022093715F, -0.013143075F, 0.039502755F, -0.11709202F, 0.12700301F,
          0.05373649F, -0.111666776F, 0.051940903F, 0.16965759F, 0.045425314F,
          -0.0080867065F, 0.0030582007F, 0.02253197F, -0.13531022F, -0.08117344F,
          -0.068962805F, -0.009906758F, 0.041356605F, -0.021140607F,
          -0.02350529F, 0.05423687F, -0.091061056F, -0.118995145F, 0.09002792F,
          -0.030950384F, -0.17481928F, 0.1891796F, 0.23569822F, -0.2642259F,
          -0.08606154F, 0.110612586F, 0.12607336F, 0.0035852448F, -0.039550614F,
          0.11433293F, 0.07089311F, -0.013336105F, 0.2224867F, 0.17195746F,
          0.030155607F, 0.030131975F, 0.03252947F, -0.044630907F, -0.004860082F,
          -0.057233505F, -0.017492209F, 0.027743397F, -0.018642545F,
          0.002088539F, -0.048157502F, 0.16903205F, 0.011608636F, -0.011995091F,
          0.065669276F, 0.098883264F, -0.027830625F, 0.043442275F, 0.055740625F,
          -0.030548109F, 0.11567487F, 0.013245573F, -0.14801778F, 0.091916874F,
          0.006628479F, -0.20165415F, -0.042204134F, -0.08575703F, -0.017951215F,
          -0.115836665F, -0.020885194F, -0.07396184F, -0.121204905F,
          0.011134467F, 0.025590027F, -0.0022785128F, -0.012702124F,
          -0.12824543F, 0.0147114815F, -0.06960352F, -0.12534533F, -0.045935176F,
          -0.020935608F, -0.13041164F, -0.09676473F, -0.02709697F, -0.08334632F,
          0.12503189F, 0.060591854F, -0.049485695F, 0.1627343F, 0.09799557F,
          -0.071394905F, 0.03730053F, 0.053431336F, 0.09692337F, 0.0552528F,
          -0.06903615F, 0.07707202F, 0.15324946F, 0.086411275F, 0.05620727F,
          0.057431046F, 0.105465464F, 0.050442614F, 0.0501937F, -0.0070393095F,
          0.035483073F, 0.057267185F, 0.09857611F, -0.07547186F, -0.07509998F,
          -0.06211167F, 0.1264375F, 0.25111073F, 0.040525384F, 0.010365553F,
          0.08576032F, 0.24279591F, -0.03697411F, -0.020058284F, 0.09382226F,
          -0.11130883F, -0.06770966F, -0.20342503F, 0.03934284F, 0.07203119F,
          0.025057847F, -0.07088415F, 0.011038983F, -0.101621665F, 0.16140698F,
          -0.03930439F, 0.15370105F, 0.10043833F, -0.08719066F, 0.08819281F,
          0.1276902F, -0.13920121F, 0.042635214F, -0.071016066F, 0.124947935F,
          0.038053405F, -0.06328356F, 0.16012906F, -0.022903413F, -0.042358525F,
          0.049535897F, -1.5388228E-5F, 0.013549733F, 0.06540798F, -0.009928912F,
          -0.012988985F, 0.03210667F, 0.06425246F, 0.013946585F, 0.04280061F,
          0.03154228F, -0.02889406F, 0.16987862F, -0.08186794F, 0.03162488F,
          0.05873581F, -0.09737757F, -0.04822662F, 0.05663429F, -0.08216687F,
          0.002800374F, 0.14242463F, -0.12219287F, 0.06886741F, 0.17719978F,
          0.06958863F, -0.05064132F, 0.13383266F, 0.04864308F, 0.017848164F,
          -0.017971722F, -0.045858823F, 0.03154942F, 0.0033928158F,
          -0.034115344F, 0.01598301F, 0.030139871F, -0.0457651F, -0.053145815F,
          -0.0066694557F, 0.021298537F, 0.045101408F, 0.051287893F, 0.011783269F,
          -0.060604874F, 0.05636731F, -0.044870373F, 0.023684133F, 0.101960234F,
          0.05154777F, 0.05096262F, -0.02151439F, -0.010912916F, -0.012579156F,
          0.00011129025F, 0.01221966F, 0.027556643F, 0.031041754F, 0.028763901F,
          0.11741782F, 0.09596029F, 0.1272652F, -0.018444551F, -0.020211563F,
          0.0547305F, -0.023065032F, -0.0022007362F, -0.01002322F, -0.08589392F,
          -0.10745355F, -0.090196885F, -0.07271341F, -0.12092597F, -0.16390066F,
          0.09770144F, 0.10573389F, 0.032587014F, 0.03866835F, 0.09142761F,
          0.08408403F, 0.011365124F, 0.07352727F, 0.1145697F, -0.08457205F,
          0.057394743F, 0.018814849F, -0.09947343F, 0.03522706F, 0.031651083F,
          -0.048708547F, 0.0034636762F, -0.010322991F, 0.010166144F, 0.02669933F,
          -0.12433503F, 0.080661215F, 0.097448386F, 0.06497406F, 0.05835665F,
          0.12050713F, 0.016971452F, -0.013804352F, 0.0026491259F, 0.033407107F,
          -0.11860219F, -0.016336061F, -0.052546937F, -0.096231535F,
          -0.029954886F, -0.015282443F, -0.12378128F, -0.037596494F, 0.06835288F,
          -0.16769086F, -0.06428098F, 0.014948482F, -0.1328055F, -0.03749769F,
          -0.08286849F, 0.022925192F, -0.043035205F, -0.05924519F, -0.039912477F,
          -0.06021657F, -0.044899054F, -0.1050495F, 0.0019997484F, 0.06876849F,
          -0.029454498F, -0.0714101F, 0.023608364F, 0.015326964F, -0.04127913F,
          -0.024884738F, 0.06561515F, 0.031390246F, -0.0057675987F, -0.06254029F,
          0.124873765F, 0.026080947F, 0.05930734F, 0.04438621F, 0.016495505F,
          -0.053010646F, -0.024493486F, 0.11096626F, -0.14736979F, -0.027221687F,
          -0.024585595F, 0.0523316F, -0.26435867F, 0.04780145F, 0.12719706F,
          -0.06686748F, -0.086061F, 0.063432895F, -0.12770274F, 0.06329955F,
          -0.19866355F, 0.046662975F, 0.15912995F, -0.002739208F, 0.18708596F,
          -0.097857915F, -0.0064967955F, -0.09888299F, 0.0786631F, 0.12625949F,
          -0.098935604F, -0.068174124F, 0.17167652F, 0.013097347F, -0.15071477F,
          0.24942294F, 0.050806835F, -0.17899278F, -0.11201259F, 0.05072076F,
          0.16384849F, -0.10939086F, 0.101598166F, 0.22800267F, -0.04320878F,
          -0.022522865F, 0.032087613F, 0.027269341F, -0.049709585F, -0.06339391F,
          0.053090326F, 0.023746273F, 0.032188207F, 0.12881128F, -0.13489105F,
          0.06947898F, 0.06679097F, 0.020054301F, -0.05555838F, 0.004583768F,
          0.20548758F, -0.029028911F, -0.075728424F, -0.07308618F, 0.042241942F,
          -0.01938015F, -0.0066796634F, 0.013439107F, 0.029936254F, 0.026505874F,
          -0.011395013F, 0.010558174F, 0.05733118F, -0.044825293F, -0.1020177F,
          0.23095275F, -0.014170617F, 0.05164588F, 0.09544255F, -0.0012790912F,
          -0.0663211F, -0.06354509F, 0.010977001F, -0.025573744F, -0.047398813F,
          -0.10753067F, 0.017397607F, -0.03499916F, -0.1795486F, -0.0375061F,
          -0.08923388F, -0.066805735F, -0.11283334F, -0.30308467F, -0.1585148F,
          -0.18836525F, -0.22099286F, -0.12869073F, -0.14147703F, -0.006751668F,
          0.0031972635F, -0.016844427F, -0.1713729F, -0.026761997F, 0.058886733F,
          0.024143558F, -0.09346882F, -0.0018244319F, -0.0065617654F,
          -0.08780103F, 0.016393019F, -0.16567226F, -0.026277684F, 0.03206078F,
          -0.0030095947F, -0.04771682F, 0.15634342F, -0.21781418F, 0.13105404F,
          0.23761341F, -0.023803549F, -0.3118846F, -0.06630962F, 0.32589576F,
          0.04827522F, -0.016448023F, -0.092448175F, -0.016523825F,
          -0.033132862F, -0.15525584F, -0.01731531F, -0.047270592F, -0.07278879F,
          -0.06062047F, 0.06447623F, 0.084655404F, -0.12658921F, -0.03879672F,
          0.06899759F, -0.11639351F, -0.28463122F, 0.12700987F, 0.102016695F,
          -0.10007581F, 0.006958356F, 0.089055374F, 0.020806838F, -0.018481348F,
          0.040066276F, 0.09906086F, -0.2971126F, 0.0906537F, -0.04032476F,
          0.010333789F, -0.12413784F, -0.048415113F, -0.10452083F, 0.07104325F,
          -0.023933677F, 0.0010850889F, 0.06742789F, -0.060519602F,
          -0.015779568F, 0.02298497F, -0.008740184F, -0.121282764F,
          -0.107611634F, 0.012432075F, -0.23220786F, -0.072707064F,
          -0.084071636F, -0.022604898F, 0.06610074F, 0.010490191F, -0.21165124F,
          -0.020988775F, 0.07164256F, -0.11637183F, -0.050727468F, 0.03348339F,
          -0.019507661F, -0.075336345F, -0.055205725F, -0.0041279774F,
          0.02882919F, -0.03479574F, 0.059495687F, 0.088626735F, 0.07403542F,
          0.06349012F, -0.00015975646F, -0.089782715F, 0.15404874F, 0.18129955F,
          0.037511576F, 0.014422844F, 0.057940215F, 0.016558023F, 0.02050251F,
          0.042678356F, 0.060339738F, 0.18130356F, 0.052130233F, 0.0031012208F,
          0.022134708F, 0.067197F, -0.0032806962F, -0.055999234F, -0.063993946F,
          0.019345982F, -0.13672225F, -0.013980927F, 0.091396056F, -0.020697981F,
          -0.08961554F, -0.055515077F, 0.079968035F, -0.05555238F, -0.041088626F,
          0.08205684F, -0.076479875F, -0.042313892F, -0.075780265F, -0.0678734F,
          0.06120875F, -0.15057719F, 0.06956722F, 0.1361889F, -0.24346857F,
          0.062293712F, 0.13245639F, 0.03641113F, -0.086627F, 0.0012823585F,
          0.026184095F, 0.009599838F, 0.022033205F, -0.041618988F, -0.1390523F,
          -0.039825536F, -0.03278002F, 0.110967845F, 0.05936171F, 0.021770298F,
          -0.069566816F, -0.09232298F, -0.15205802F, 0.17423372F, -0.033718728F,
          -0.03749073F, 0.094488874F, -0.08942761F, -0.01769877F, 0.023237461F,
          -0.03858773F, -0.14040701F, -0.20280536F, -0.13289608F, -0.087726876F,
          -0.049544215F, -0.10693594F, 0.10061374F, 0.1132746F, 0.10317976F,
          0.07170088F, -0.12402878F, 0.06676553F, -0.048769288F, -0.004546317F,
          -0.00481968F, 0.013108295F, 0.04352729F, -0.118771315F, -0.07224257F,
          0.023657754F, 0.013850864F, -0.00988196F, 0.015780283F, 0.011474821F,
          -0.005149414F, -0.007440917F, -0.035148397F, -0.05537323F,
          -0.018142909F, 0.071941994F, 0.01241327F, 0.01246431F, -0.16965769F,
          -0.18241885F, -0.113359824F, -0.10582834F, -0.24757846F, -0.20069374F,
          -0.014165843F, -0.1004948F, -0.08699864F, 0.07701255F, 0.01864963F,
          0.02353397F, 0.030426031F, 0.09281792F, 0.058910325F, 0.047224607F,
          0.049083665F, 0.05210319F, -0.049368933F, -0.055690113F, -0.028772546F,
          -0.13639133F, -0.27627596F, -0.13382524F, 0.016354123F, -0.06740135F,
          -0.106425464F, 0.07257655F, -0.01167562F, -0.045186024F, 0.020940613F,
          0.02625059F, 0.04999668F, -0.0048817447F, -0.02904834F, -0.0007390118F,
          -0.10935827F, -0.026534084F, 0.09572727F, -0.007080081F, -0.045488317F,
          0.044343542F, -0.013048211F, -0.053350862F, -0.003324725F,
          -0.051890526F, 0.0312943F, 0.076150194F, -0.023786599F, -0.06415059F,
          -0.0467285F, 0.059011642F, 0.07207595F, 0.017599778F, -0.100648746F,
          -0.16531193F, -0.084622085F, -0.08548499F, -0.19483271F, -0.2528511F,
          -0.020241773F, -0.011469812F, -0.08034797F, 0.0031718467F,
          -0.036680955F, -0.037370075F, 0.08524062F, 0.014712775F, -0.008225742F,
          -0.02328832F, -0.009778665F, 0.01464875F, -0.053743184F, -0.031441044F,
          -0.018000431F, -0.108792536F, -0.14532022F, -0.11862775F, 0.04726728F,
          -0.029511463F, -0.03424586F, -0.11374085F, -0.08090831F, -0.019602748F,
          -0.06223777F, -0.16905382F, -0.12321477F, 0.036687557F, -0.04226488F,
          -0.04288227F, -0.065652385F, -0.108651645F, 0.02361553F, -0.052026264F,
          -0.24981017F, -0.1365356F, 0.07499735F, -0.042418025F, -0.021890672F,
          0.096671544F, 0.068003796F, 0.008133804F, 0.106649466F, 0.17306276F,
          0.13021074F, -0.00044225965F, 0.06333141F, 0.07957383F, 0.1353538F,
          -0.004952713F, -0.017693615F, 0.061283372F, 0.118088596F,
          -0.0011242615F, 0.0001355538F, 0.03111671F, 0.050707683F, 0.11618112F,
          0.033795893F, 0.022180282F, 0.025514342F, 0.09551115F, 0.07815146F,
          0.04028232F, 0.06935215F, 0.0837664F, -0.069131725F, -0.017807428F,
          -0.0015231236F, -0.056747343F, -0.14477897F, -0.0805207F, 0.045184646F,
          0.034689117F, -0.018051865F, 0.10450678F, 0.017415695F, -0.022963155F,
          0.107346654F, 0.12563561F, 0.14258948F, -0.10199658F, -0.012421613F,
          0.03978712F, 0.016832761F, 0.06470999F, 0.029283429F, -0.041939482F,
          -0.06634212F, -0.026414583F, 0.040292464F, -0.017474806F, 0.003565492F,
          -0.047390413F, -0.064591706F, -0.01597798F, -0.0063038263F,
          -0.07808628F, -0.060769048F, 0.005204289F, -0.0029262041F,
          -0.05278733F, -0.052530218F, -0.07593797F, 0.044991504F, -0.09253533F,
          -0.16302039F, -0.005976969F, 0.01760243F, -0.11500747F, -0.00359347F,
          0.032836277F, -0.07744927F, -0.08472227F, 0.0174896F, 0.055868722F,
          0.0069147013F, -0.04406509F, -0.022012299F, 0.040231362F, -0.05583409F,
          -0.023244552F, 0.017458873F, -0.04037664F, -0.08837075F, -0.05091251F,
          0.10867047F, 0.055546474F, 0.047761973F, 0.056335222F, 0.008222948F,
          0.087456584F, 0.035362754F, -0.003353496F, 0.07103412F, -0.0072964747F,
          -0.048347548F, 0.03496174F, -0.12316953F, -0.037565522F, -0.04558322F,
          -0.13833676F, -0.14533487F, -0.13075912F, -0.043557618F, -0.08793287F,
          -0.10849829F, 0.10643812F, 0.070843086F, 0.034295462F, 0.06785711F,
          0.08765693F, 0.10053737F, -0.04271212F, 0.03917713F, 0.055307884F,
          -0.018569782F, 0.011466767F, 0.0028863165F, -0.13786703F, -0.13047622F,
          -0.043336537F, -0.016878998F, -0.11650576F, -0.13750659F, -0.08662089F,
          -0.018544609F, 0.0008266583F, 0.042890124F, -0.029488353F,
          0.012092981F, 0.048591934F, 0.119067356F, 0.06984239F, 0.034445573F,
          -0.006318508F, 0.011725503F, 0.043876737F, 0.03417792F, 0.025572509F,
          -0.0061837006F, 0.018263381F, -0.022046486F, -0.12333507F,
          -0.04217242F, 0.076639295F, -0.08023981F, -0.23527814F, -0.15010598F,
          0.16914542F, 0.048240118F, 0.020050837F, 0.04774333F, 0.019430961F,
          0.01227605F, 0.015230757F, -0.0021533584F, 0.035668045F, 0.027032921F,
          -0.05791614F, -0.063241005F, 0.010583257F, -0.009279141F,
          -0.030681401F, 0.008801466F, -0.03708863F, -0.041485526F, 0.020658648F,
          -0.024856756F, -0.046154477F, -0.0072887153F, 0.024925943F,
          0.049261663F, 0.013403198F, -0.00082584424F, 0.011815986F,
          -0.0043034204F, 0.007966601F, 0.04409957F, 0.072912775F, -0.018349687F,
          -0.016109888F, 0.05166363F, -0.0698364F, -0.011652212F, -0.007629461F,
          -0.06911329F, 0.0020409245F, -0.013562977F, -0.11006119F, -0.03756982F,
          -0.099352814F, -0.14821917F, 0.022135206F, -0.07557372F, -0.010571873F,
          0.04232915F, -0.0626011F, 0.01686686F, 0.0048235836F, -0.02562731F,
          0.00014434129F, 0.051178023F, -0.0035573922F, -0.053687762F,
          0.007352142F, 0.036249373F, 0.014099495F, 0.05786712F, 0.051774535F,
          0.031295452F, 0.08532013F, 0.06182552F, 0.015429454F, 0.04387212F,
          0.017843474F, 0.05581602F, -0.045020886F, 0.0808653F, -0.05424157F,
          -0.08760339F, 0.01559604F, -0.07492352F, -0.00879083F, -0.038127765F,
          0.03811706F, 0.10903338F, 0.033207297F, 0.11729775F, 0.04411481F,
          0.104815386F, 0.021468416F, -0.09916145F, 0.02363066F, -0.01568401F,
          0.004228307F, -0.00033388817F, 0.02486362F, 0.029031243F,
          -0.008269965F, 0.0024903659F, -0.042558473F, 0.01382878F,
          -0.043591347F, -0.08397784F, -0.011929642F, -0.06685775F, -0.09476106F,
          -0.019155128F, -0.053813443F, -0.034211703F, 0.049901795F,
          0.019418389F, -0.058935028F, 0.06953142F, -0.004458419F, -0.09772624F,
          0.064248875F, -0.039188445F, -0.104996234F, 0.009281539F,
          -0.027268296F, -0.05723864F, -0.0112009F, -0.065580636F, -0.007607474F,
          -0.029390424F, -0.022433035F, -0.0077613704F, 0.108943865F,
          0.016026799F, -0.07398454F, 0.15586568F, -0.0729545F, -0.08706305F,
          -0.030110983F, -0.1557745F, -0.1342982F, 0.036820974F, -0.12260371F,
          -0.04454482F, -0.033617653F, -0.17984237F, -0.01910598F, -0.050718077F,
          -0.10136592F, 0.037814848F, 0.020973034F, 0.037127655F, 0.030321127F,
          0.03590599F, 0.030039234F, 7.1105984E-5F, 0.020354204F, 0.009958548F,
          -0.009450279F, 0.06085119F, -0.18464808F, -0.08862389F, -0.0806363F,
          -0.31636506F, -0.11131086F, -0.12619904F, -0.1745825F, 0.028313937F,
          -0.01734982F, -0.01722615F, 0.017019328F, 0.02039375F, -0.014191238F,
          -0.017713761F, 0.029827768F, 0.0003809148F, -0.02205761F,
          -0.0022675113F, -0.075792074F, 0.0125337085F, -0.006184718F,
          -0.053143397F, 0.01457597F, -0.011005608F, -0.03865346F, 0.045941945F,
          0.012798745F, 0.048207972F, 0.03983347F, -0.027880445F, 0.0951254F,
          0.0051210932F, -0.02326515F, 0.09777004F, -0.014585175F, 0.05848673F,
          -0.0125865955F, -0.04029987F, 0.036689036F, -0.13350163F,
          -0.033319328F, 0.008823658F, -0.101804405F, -0.012850605F,
          0.007652882F, -0.25820655F, -0.14564645F, -0.19837378F, -0.39016697F,
          -0.13206455F, -0.20534909F, -0.20486532F, -0.026995277F, -0.07971548F,
          -0.117619656F, 0.034694064F, -0.17137827F, -0.05123891F, 0.099697635F,
          -0.13792618F, 0.03480507F, 0.10686301F, -0.012650987F, 0.07900421F,
          0.0363533F, 0.08305805F, 0.2047794F, 0.11926657F, 0.051006507F,
          0.06415653F, 0.009848091F, 0.046955653F, -0.009369957F, -0.014234297F,
          0.058240596F, -0.037943438F, -0.044769198F, 0.023687648F,
          -0.055683542F, -0.038577635F, 0.016727688F, -0.02112776F,
          -0.005740607F, -0.05985187F, -0.08030504F, -0.024793211F, -0.0455644F,
          -0.042480014F, 0.0067912587F, 0.028949298F, 0.006802907F,
          -0.0094928965F, -0.06475111F, -0.026543332F, -0.0009146284F,
          0.02030847F, 0.032669246F, 0.020035354F, -0.0062100245F, 0.041293517F,
          0.019514821F, 0.0038134032F, -0.045486707F, 0.02877176F, -0.023878738F,
          0.06645879F, 0.019026162F, 0.0031293598F, -0.028052354F, 0.035398122F,
          0.0033410038F, 0.16675846F, 0.03126471F, 0.009222879F, 0.031602383F,
          0.012258094F, 0.032922707F, 0.0033285143F, -0.0032820678F, 0.02396038F,
          0.0026096136F, 0.011832495F, 0.035273414F, -0.010787433F,
          -0.0119229015F, 0.05009072F, 0.04656392F, 0.055514302F, 0.048923466F,
          0.083872095F, 0.0770873F, 0.05929404F, 0.064522624F, 0.045830064F,
          0.019123025F, -0.090028755F, 0.016230997F, 0.0040365816F, 0.030534614F,
          0.043200478F, -0.010705016F, -0.08611663F, 0.021510037F, -0.06001625F,
          -0.059851635F, -0.022942284F, -0.054118153F, -0.09947189F,
          -0.039034057F, -0.046110954F, -0.0344231F, -0.02769614F, 0.07115342F,
          -0.013482946F, -0.03188869F, 0.048882764F, -0.056469485F,
          -0.026310267F, -0.01971435F, -0.011131146F, -0.029196894F,
          0.023548938F, 0.009213136F, 0.02805161F, 0.0024380758F, 0.0251057F,
          -0.009932792F, -0.07485376F, -0.051182505F, 0.04203722F, -0.04597226F,
          -0.0736464F, 0.03894142F, 0.083295874F, -0.042528193F, -0.048867762F,
          0.0014589026F, 0.07939096F, 0.040489763F, 0.04341658F, 0.11781381F,
          0.048104774F, -0.12900095F, 0.025312832F, 0.048947327F, 0.0034669456F,
          -0.052758455F, 0.027537517F, -0.01620584F, -0.014197059F, 0.010800816F,
          -0.011685343F, -0.08905886F, -0.12631561F, 0.05725282F,
          -0.00017856802F, -0.060637362F, 0.0011245998F, 0.012611434F,
          0.016021488F, -0.049208753F, 0.04036503F, 0.06860586F, -0.04437255F,
          -0.07673885F, -0.0052449256F, -0.011643121F, -0.07382244F,
          0.012400671F, 0.13986239F, 0.0041321837F, -0.100352824F, -0.05164456F,
          0.08874508F, -0.0026900817F, 0.14279158F, 0.1642846F, 0.06711456F,
          -0.12172942F, 0.04273375F, 0.15977131F, 0.04216205F, -0.07639913F,
          -0.05870707F, -0.039793644F, -0.057916686F, -0.018629968F,
          0.0020810375F, -0.0039851144F, -0.016261706F, 0.0046988023F,
          0.017249297F, 0.03800688F, 0.067092225F, -0.029925393F, 0.017162384F,
          -0.008243147F, 0.1581735F, 0.018643863F, -0.027714454F, -0.10617494F,
          -0.03231457F, 0.020466598F, -0.002468064F, -0.036263328F, 0.043806057F,
          0.03353401F, 0.0022599201F, -0.10589874F, 0.013698635F, 0.062001977F,
          0.022658631F, 0.04235922F, 0.005063909F, -0.10155504F, 0.03355154F,
          0.07685638F, -0.04583674F, -0.08553422F, 0.025219463F, 0.03701256F,
          0.070728615F, 0.055177227F, -0.07665653F, -0.06116179F, 0.0418819F,
          -0.012043659F, 0.007507372F, -0.008357683F, -0.049248092F,
          -0.015535406F, -0.030556377F, -0.035356816F, -0.11899307F,
          -0.09606992F, -0.025169507F, -0.03605265F, -0.062290244F, -0.07840092F,
          -0.06919307F, -0.053980928F, 0.15993436F, -0.0067936177F, -0.16140288F,
          0.043843333F, 0.16418597F, 0.081972025F, 0.013596585F, -0.11515148F,
          -0.09907852F, 0.045230065F, 0.10479695F, -0.003340059F, -0.09842718F,
          0.034474898F, 0.13879403F, 0.053348348F, -0.11892709F, -0.04124263F,
          0.053931575F, 0.05055312F, -0.031098086F, -0.0029526558F, 0.056466628F,
          0.028080434F, 0.051595602F, 0.06912003F, 0.015307952F, -0.07835355F,
          -0.032706596F, 0.033854395F, 0.022756513F, -0.033383157F,
          -0.026804114F, -0.035185028F, -0.09057331F, -0.04443593F, 0.028883748F,
          0.018107763F, -0.025409106F, -0.05902858F, -0.023931425F, 0.065462284F,
          0.011833701F, -0.018486617F, -0.026376238F, 0.0012426964F,
          -0.033689458F, 0.0039581405F, 0.021574058F, 0.023317521F, 0.027910987F,
          0.008077309F, 0.07545308F, 0.13061285F, -0.1616448F, -0.207285F,
          -0.020526564F, 0.12874998F, 0.057338703F, -0.06908041F, 0.082467996F,
          0.07202837F, -0.02580984F, -0.050228693F, 0.13654041F, 0.0802832F,
          -0.068661414F, -0.11353207F, 0.022776881F, 0.010655746F, -0.06913902F,
          -0.06463955F, 0.09357116F, 0.13618076F, -0.023833137F, -0.038494054F,
          -0.026994942F, 0.053044964F, 0.014072299F, 0.032238793F, 0.048244152F,
          -0.0136722F, -0.015702669F, 0.01951376F, -0.041818477F, -0.047398876F,
          -0.008126505F, -0.069287375F, 0.033748783F, -0.026250342F,
          -0.08748735F, -0.15072018F, -0.067875795F, -0.021811951F,
          -0.016012842F, -0.062838666F, -0.0047869794F, -0.027770871F,
          -0.02658731F, -0.041946217F, -0.011069895F, 0.032465957F, -0.084419F,
          -0.11384787F, -0.0037721323F, -0.06797877F, -0.019396903F,
          -0.0022311031F, 0.03808993F, -0.12874885F, -0.08384397F, 0.099490754F,
          0.1543738F, -0.08362648F, 0.0042612357F, 0.13320583F, 0.059874285F,
          -0.16299318F, -0.08053816F, 0.08978899F, 0.036913227F, -0.17025743F,
          -0.09829882F, -0.0892175F, -0.1481481F, -0.056158386F, 0.1138106F,
          -0.012407456F, -0.0985458F, -0.039050344F, 0.13222052F, 0.012922392F,
          -0.00879317F, -0.12462869F, -0.1159492F, -0.102218494F, -0.002339239F,
          0.028006501F, -0.017476713F, -0.06458394F, -0.011816509F, 0.11433131F,
          0.082146496F, 0.018570403F, -0.20437947F, -0.078577556F, 0.075593956F,
          0.013995302F, -0.1507394F, -0.09827492F, -0.043068457F, -0.09122795F,
          -0.030813212F, 0.0684362F, 0.07290268F, -0.033634905F, 0.015797548F,
          -0.018450107F, 0.07213949F, -0.040191367F, -0.0026161547F,
          -0.0023198463F, -0.010124967F, 0.01123099F, 0.026028588F,
          -0.047232714F, -0.017052662F, 0.021163907F, 0.085535824F, 0.017243793F,
          0.091741845F, -0.02957208F, -0.09842478F, -0.047321014F, -0.02151705F,
          -0.06881399F, 0.059637353F, 0.023929872F, 0.02005852F, -0.023278395F,
          -0.03953744F, -0.010199216F, -0.037764885F, 0.04055027F, 0.046211246F,
          0.029805291F, -0.100366496F, -0.06514845F, 0.102345616F, -0.004524697F,
          -0.0052686096F, -0.028982768F, 0.1154954F, 0.10107338F, 0.109371416F,
          -0.05279665F, -0.11091496F, -0.05674981F, 0.10274537F, -0.053984687F,
          0.09389989F, -0.03552815F, -0.085883886F, 0.058331963F, 0.0081468765F,
          0.062747404F, 0.102189675F, -0.13380592F, -0.03292514F, 0.04730991F,
          -0.02149833F, -0.08623742F, -0.1247429F, 0.009334759F, 0.00949943F,
          -0.025777336F, 0.00673414F, 0.026661422F, -0.021472005F, 0.008681122F,
          -0.01211851F, -0.01182493F, 0.15167494F, 0.12885204F, 0.0654451F,
          0.016027767F, -0.045376692F, 0.047259297F, 0.09527071F, 0.14085579F,
          0.19080842F, 0.036619045F, 0.005585559F, -0.0014738942F, 0.09100202F,
          -0.02565158F, -0.06001796F, 0.008142137F, -0.04487813F, -0.011821945F,
          -0.001986202F, -0.014786955F, -0.01593339F, 0.04315342F, -0.046814684F,
          -0.16086835F, 0.06391449F, 0.044858903F, 0.00388979F, -0.01244719F,
          -0.0138990935F, -0.006338005F, 0.059381258F, -0.0068682334F,
          -0.014919879F, -0.0078202505F, -0.02790422F, 0.019727409F,
          0.009527513F, -0.08593411F, -0.015883345F, -0.084055595F, -0.07313372F,
          0.06433549F, 0.05016297F, 0.012421483F, 0.002133675F, -0.1129207F,
          -0.058768895F, 0.05573351F, 0.06448083F, 0.08502185F, 0.09447747F,
          -0.06255924F, -0.13070823F, -0.064176254F, -0.04249575F, -0.18937129F,
          -0.05991903F, -0.033714794F, -0.133097F, -0.0034116246F, -0.06716898F,
          -0.14205886F, -0.07206028F, -0.01002396F, -0.01136744F, -0.041977566F,
          -0.03389204F, -0.08145836F, -0.0051657674F, -0.06289029F,
          -0.044457424F, -0.008813114F, -0.070844054F, -0.29821208F,
          -0.37635887F, -0.24933319F, -0.0867195F, 0.054771513F, 0.14674501F,
          0.28346816F, 0.13570645F, 0.06979292F, -0.007860892F, -0.021258939F,
          0.049004942F, -0.079201244F, 0.0025025366F, -0.019740177F,
          -0.043614548F, 0.029580902F, -0.008048359F, -0.053352524F,
          -0.060545094F, -0.15855716F, -0.118705556F, -0.04553433F, 0.0324593F,
          0.07408468F, 0.029608315F, -0.13500355F, 0.047267463F, -0.07262387F,
          0.043574855F, -0.008264768F, -0.03047321F, -0.073558725F,
          -0.039559767F, -0.089702524F, 0.064253956F, -0.027262222F,
          -0.053705946F, -0.0022909988F, -0.059340272F, -0.0406251F,
          0.028550994F, 0.029866891F, 0.05057077F, -0.063358106F, -0.119683415F,
          -0.23942716F, 0.035501737F, 0.020192832F, 0.013101544F, 0.10859107F,
          0.15712155F, 0.14095841F, 0.013351693F, -0.050844003F, -0.04975461F,
          0.038235612F, 0.11458682F, 0.13929814F, -0.13715567F, -0.14300746F,
          -0.05664976F, 0.03189173F, 0.034624856F, 0.0971961F, -0.0042212345F,
          0.079422295F, 0.09056317F, -0.1247459F, -0.075439334F, -0.059246626F,
          0.09836475F, 0.047309805F, -0.027471408F, -0.041063864F, -0.10294111F,
          -0.06859847F, 0.013489004F, 0.019933768F, 0.046864938F, -0.026342914F,
          -0.0667085F, -0.052338153F, -0.09495154F, 0.06329519F, 0.110966735F,
          0.03867812F, -0.01351007F, -0.12870397F, -0.03190932F, -0.009772659F,
          -0.019570509F, 0.020689752F, 0.116403334F, 0.023155656F, -0.049406547F,
          -0.032566123F, -0.06805329F, 0.13093534F, 0.21778908F, 0.17616227F,
          -0.08338437F, -0.2093122F, -0.13380562F, -0.10756708F, -0.047282796F,
          -0.02126342F, -0.02573711F, 0.1683111F, 0.15323111F, 0.17892322F,
          0.17477684F, -0.08236654F, -0.29826027F, -0.47090113F, -0.25818F,
          0.03552419F, -0.054203577F, -0.015242319F, 0.012367103F, -0.06940169F,
          -0.029738108F, 0.03170587F, 0.04565236F, 0.0454136F, -0.012860009F,
          0.12330138F, -0.010142333F, 0.0054916414F, -0.03077238F, -0.074468836F,
          -0.08123908F, -0.037975304F, -0.008111989F, 0.0777681F, -0.0051151393F,
          0.04281362F, -0.17604212F, -0.29814762F, -0.11965476F, 0.061637983F,
          0.07791954F, 0.08371151F, -0.045273893F, 0.041500352F, -0.07375103F,
          0.029799234F, 0.009393376F, 0.055788856F, 0.011208421F, 0.07508461F,
          0.010339727F, -0.03268237F, -0.025252586F, 0.06264745F, 0.05081593F,
          0.012853034F, -0.042975947F, 0.013347156F, -0.054721452F,
          -0.053543802F, 0.047731847F, 0.041795157F, 0.024767844F, -0.006407959F,
          -0.015636768F, -0.03107571F, 0.014629202F, 0.01797424F, -0.013637528F,
          0.029930882F, 0.049584642F, 0.020423409F, -0.00580628F, 0.012435267F,
          -0.016832914F, -0.027173597F, -0.02238366F, -0.023967018F,
          -0.019014765F, -0.026235081F, 0.00021815348F, -0.0069697783F,
          -0.050522592F, -0.05296363F, -0.0014595203F, 0.003756125F,
          -0.025325444F, 0.014638834F, 0.0053495932F, -0.016947124F,
          0.005962674F, -0.0017990028F, 0.0192265F, -0.016025586F,
          -0.0015779878F, 0.02447064F, -0.018725319F, -0.0039588823F,
          0.005666898F, -0.01830561F, -0.020652877F, 0.011585121F, 0.0048064757F,
          0.012771401F, 0.02053947F, 0.021625854F, 0.02083563F, -0.020318996F,
          -0.0076653864F, 0.0011204475F, 0.0023620594F, -0.014982373F,
          0.0068875323F, -0.025428606F, -0.08265158F, -0.047977287F,
          -0.021425448F, -0.061137408F, -0.018641626F, 0.016881872F,
          -0.03070248F, -0.005516797F, 0.021095898F, 0.007605027F, 0.009498214F,
          -0.0037535997F, -0.022769384F, 0.011095411F, -0.045829162F,
          -0.03100456F, -0.025944378F, -0.029219126F, 0.017924055F, 0.052315906F,
          0.044600774F, 0.014095359F, 0.006024074F, 0.010552012F, 0.016756855F,
          0.0311873F, 0.041974243F, 0.00043090733F, -0.017348949F, -0.040292505F,
          0.013669396F, 0.010234237F, -0.04352413F, -0.00019377713F,
          0.020464424F, -0.01373394F, 0.022002695F, 0.023362368F, 0.032398462F,
          0.017998401F, 0.0010189994F, 0.03622224F, 0.008416723F, 0.014046791F,
          0.0065420065F, 0.0020780817F, 0.0020623219F, 0.033648197F,
          -0.030077973F, -0.026507955F, -0.06521095F, -0.023200307F,
          -0.04128415F, -0.09721382F, 0.100068115F, 0.09774248F, 0.113040954F,
          0.065609075F, 0.054354873F, 0.07310375F, 0.021586716F, 0.029836098F,
          0.017261231F, 0.035132904F, 0.005820141F, 0.031730507F, 0.024735989F,
          0.026172576F, 0.032906067F, 0.008275703F, 0.0030503366F, 0.042464558F,
          0.0005826323F, 0.026186204F, -0.039600883F, -0.013616613F,
          -0.038443685F, -0.015745621F, 0.0003684344F, -0.008450221F,
          0.024157843F, -0.07181466F, -0.03834208F, -0.018971998F, -0.07540085F,
          -0.08412009F, -0.012675547F, -0.087357186F, -0.04126277F, -0.0436851F,
          0.03399019F, 0.013600046F, 0.025201164F, -0.0054509626F, -0.006673424F,
          0.00075105217F, 0.01453068F, 0.053819805F, 0.019037083F, -0.002843336F,
          -0.01838037F, 0.018920813F, 0.023072198F, -0.05385956F, -0.018474277F,
          0.0066834893F, -0.016514992F, -0.016118018F, -0.010627677F,
          -0.0117060505F, -0.028550508F, -0.016852528F, 0.0036374484F,
          -0.071869254F, -0.017457748F, -0.0006834108F, -0.03750122F,
          0.02025301F, -0.031910077F, -0.0024232059F, 0.036054865F, -0.06334758F,
          -0.06295357F, -0.007300199F, 0.014764036F, -0.019023439F, 0.034208726F,
          0.010783185F, 0.043891747F, 0.0039841435F, -0.058028214F, 0.034997534F,
          -0.0133783445F, 0.00016508428F, -0.029470501F, -0.004647876F,
          -0.0020519055F, 0.009213708F, -0.019686533F, -0.03830266F,
          0.001345655F, 0.017836865F, 0.03278219F, 0.005129857F, 0.027387109F,
          0.02664399F, 0.016966455F, 0.026526874F, 0.01675886F, -0.015040459F,
          0.034313403F, 0.032057784F, 0.011368365F, -0.015332757F, -0.032824967F,
          -0.011575254F, 0.0066482886F, -0.08735521F, -0.010076906F,
          0.0047247764F, -0.034444883F, -0.039210714F, 0.0024481425F,
          0.010146051F, 0.036394704F, 0.03205559F, 0.015064352F, 0.017166564F,
          0.040743522F, 0.013296739F, -0.0055897622F, 0.018384404F,
          0.0028046952F, 0.061333988F, 0.015752746F, -0.00026659743F,
          -0.024801176F, -0.006682592F, 0.03045506F, 0.020830393F, 0.0076864758F,
          -0.031111214F, 0.0040080375F, -0.0110714175F, -0.006063817F,
          0.036469348F, 0.021119654F, 0.0021976226F, -0.024151158F, 0.039061904F,
          0.036125947F, 0.008583064F, 0.010193794F, 0.034455772F, 0.002232737F,
          -0.0015517025F, 0.014175147F, 0.013834515F, 0.026205027F,
          -0.019012969F, 0.0077507785F, 0.022500666F, -0.010373174F,
          0.0095523065F, 0.008522835F, -0.0052246964F, -0.023410173F,
          0.017377602F, 0.066805474F, -0.022660132F, 0.004025871F, 0.08805059F,
          0.015001454F, 0.012825214F, 0.07510704F, 0.003252164F, 0.027628172F,
          -0.012410305F, 0.018670656F, 0.046802133F, 0.028402554F, 0.020994341F,
          0.012844309F, 0.021922592F, 0.026338391F, 0.014589444F, -0.01742343F,
          0.0074459505F, 0.007401095F, 0.014851805F, -0.008299562F,
          0.0046853023F, 0.013805972F, -0.0446454F, -0.080159105F, -0.064605154F,
          -0.03694888F, -0.03750325F, -0.108946316F, -0.1452281F, 0.060185175F,
          0.003442365F, 0.020354236F, 0.33441812F, -0.48912728F, -0.969306F,
          0.5775591F, -0.0133382855F, -0.46196932F, 0.5095422F, 0.3910525F,
          0.10374314F, 0.0062061506F, -0.013175217F, -0.018005192F,
          -0.015952004F, -0.031721234F, -0.03642662F, -0.010240989F,
          -0.019294165F, -0.031619675F, 0.0031019673F, 0.018192576F,
          -0.010390958F, 0.019462612F, 0.04516287F, 0.0031935943F, 0.0011533052F,
          -0.0071023023F, 0.020053009F, -0.008538069F, -0.048271917F,
          -0.025662482F, -0.055502094F, -0.028819848F, -0.027538536F,
          0.011502408F, 0.009609787F, 0.0051907785F, -0.1905845F, 0.17771058F,
          1.2223257F, -0.33422083F, -0.19532451F, 0.17990665F, -0.3323138F,
          -0.33277968F, -0.28628194F, 0.04124504F, -0.0079816785F, -0.021065673F,
          0.031866934F, -0.028653335F, -0.038478903F, 0.003591694F, 0.021201396F,
          0.028406119F, 0.21355826F, -0.2683101F, -0.8048004F, 0.27146122F,
          0.03633169F, -0.30920765F, 0.35599318F, 0.23260029F, 0.1399931F,
          0.017465286F, -0.0078048646F, -0.02131195F, 0.0033530977F,
          -0.025478123F, -0.029855428F, 0.02227957F, -0.024778817F, 0.000179063F,
          -0.044876017F, 0.19483334F, 0.24036482F, -0.19645987F, -0.0950486F,
          0.1544989F, -0.16126816F, -0.15437892F, -0.104696356F, -0.0129496055F,
          -0.006468225F, 0.025295246F, 0.011773281F, -0.013242968F,
          -0.0022937474F, -0.0061216075F, -0.0029365183F, -0.015540494F,
          -0.050537154F, 0.01774465F, -0.0073034023F, -0.017995417F,
          -0.020866726F, -0.0468588F, -0.029955389F, 0.026544152F, 0.001878957F,
          0.0037993798F, 0.022211945F, -0.049602598F, -0.004907275F,
          0.013295897F, -0.031001477F, -0.077657394F, -0.052583292F,
          -0.018109972F, 0.005083752F, 0.0030133303F, 0.0014014667F,
          -0.011342854F, 0.03703261F, -0.01736696F, -0.004067291F, -0.024094328F,
          -0.008337614F, -0.022864519F, -0.013706861F, -0.026443144F,
          0.012647079F, -0.039910655F, -0.016955562F, 0.018404135F,
          -4.156989E-5F, -0.0015024015F, 0.02142575F, -0.00492159F,
          -0.012316145F, -0.0062272605F, -0.03682276F, 0.042925477F,
          -0.010704611F, -0.0124643855F, -0.000907406F, 0.01855442F,
          0.020181023F, 0.010208014F, -0.02030584F, -0.0016634329F,
          -0.013975942F, -0.01584243F, -0.006123948F, -0.008012214F,
          -0.0145033635F, -0.040617917F, -0.023455992F, 0.012200208F,
          -0.0019204986F, -0.008432548F, -0.02079607F, -0.0010613094F,
          -0.003382531F, 0.0011760094F, 0.019281114F, 0.0008792688F, 0.02526653F,
          -0.0006037837F, 0.0014213116F, -0.010004192F, 0.03696956F,
          0.021511868F, -0.015664564F, 0.0019473687F, -0.03144415F,
          -0.0023011842F, 0.011191432F, -0.008209811F, -0.0046261754F,
          0.005515653F, 0.00578768F, -0.021508565F, 0.018711807F, -0.016784828F,
          -0.015295509F, -0.0037189627F, -0.035866063F, -0.006136001F,
          -0.01463192F, 0.004930987F, 0.024437977F, -0.01013861F, 0.005035669F,
          -0.0099383285F, -0.0057003894F, -0.022676276F, 0.031139502F,
          -0.0016184747F, 0.008106758F, 0.086232595F, -0.2663818F, -0.45305747F,
          0.28956348F, 0.019121846F, -0.23672783F, 0.23134828F, 0.14421706F,
          0.0046314066F, -0.03877506F, -0.023812924F, -0.0032036612F,
          -0.03274364F, -0.054967392F, -0.044962697F, 0.008690443F, -0.0196351F,
          -0.01009132F, -0.017462319F, 0.0022723798F, -0.0134076765F,
          -0.02808381F, -0.017937783F, 0.004184814F, 0.019320073F, 0.006175993F,
          -0.0051999493F, 0.004551003F, 0.011288112F, -0.01329418F,
          0.0118176285F, 0.011068136F, 0.0010953458F, 0.0035000332F,
          -0.024846517F, 0.003036879F, -0.0043475796F, -0.0008605468F,
          0.0045911483F, -0.0331808F, -0.003763339F, -0.0031100647F,
          -0.0032171786F, -0.018283686F, -0.017752716F, 0.016496524F,
          0.005335759F, 0.0072623976F, -0.009948909F, -0.004872076F,
          0.013169105F, -0.011348617F, -0.021464659F, 0.0069777607F, 0.03755491F,
          0.0015964942F, -0.040072337F, 0.024218976F, -0.00031369226F,
          -0.0077638915F, 0.03729295F, 0.0021801349F, -0.01789393F,
          -0.042552996F, 0.014715096F, -0.01935753F, -0.023499584F,
          -0.017410716F, -0.057358768F, 0.003052879F, -0.026762128F,
          0.025793992F, -0.030531803F, -0.0028991243F, -0.0024296283F,
          0.015820844F, 0.03024516F, -0.009616297F, -0.011882836F,
          -0.0055304077F, 0.010427041F, -0.07885935F, 0.09961523F, 0.19943912F,
          -0.04746844F, -0.103913836F, 0.012969995F, -0.07634806F, -0.057448536F,
          -0.058984343F, 0.05102439F, 0.017955361F, 0.06591885F, 0.064559884F,
          0.028467102F, 0.019560961F, 0.050480705F, 0.013592679F, 0.046213124F,
          0.09557249F, -0.0016121301F, 0.0032588765F, 0.033620585F,
          -0.051746853F, -0.04659395F, 0.05242022F, -0.038476348F, -0.07696882F,
          0.023884963F, -0.13256644F, -0.06740472F, -0.024919676F, -0.071861796F,
          -0.04261442F, -0.048424657F, -0.030949768F, -0.011859912F,
          -0.037554763F, -0.10085379F, -0.014186539F, -0.00079923467F,
          -0.17417993F, -0.017919665F, -0.0213178F, -0.0911533F, -0.056502014F,
          0.19566385F, 0.08305809F, -0.087337755F, 0.1346726F, 0.07217985F,
          -0.09719293F, 0.12775344F, 0.028531821F, -0.098787025F, 0.03921311F,
          -0.040649362F, 0.062106688F, -0.027597673F, -0.14502147F,
          -0.044424787F, 0.017467344F, -0.0549383F, 0.0521923F, -0.047026675F,
          0.19222035F, 0.07114119F, -0.0068670716F, 0.23101412F, 0.0783993F,
          -0.019943085F, 0.16948886F, 0.076957494F, 0.05191494F, -0.061682895F,
          -0.08580413F, 0.104495004F, -0.03582229F, -0.011898127F, 0.039905343F,
          -0.083441086F, -0.10043036F, 0.032032818F, 0.011871132F, -0.046724234F,
          -0.05037559F, -0.04813394F, -0.010268838F, 0.04521424F, -0.01972644F,
          -0.024488326F, 0.027017364F, -0.020251704F, 0.08433291F,
          -0.0055815317F, -0.113504626F, -0.012521393F, -0.004481104F,
          -0.1090564F, -0.003814654F, 0.017857775F, 0.025954636F, 0.03407359F,
          0.07227777F, 0.02616453F, 0.054420676F, 0.03261007F, -0.027649833F,
          0.024822246F, 0.055016644F, -0.050092924F, -0.039353546F, -0.05641144F,
          -0.06903639F, -0.0005115828F, -0.07401708F, -0.034867555F, 0.0291628F,
          0.14172962F, -0.13593438F, -0.19950119F, 0.08330363F, -0.12251667F,
          -0.21203513F, 0.08269584F, -0.05229474F, -0.152398F, 0.067230545F,
          0.041961554F, 0.032681186F, 0.021329764F, 0.022618245F, 0.09511367F,
          0.008188469F, 0.0347401F, 0.014485804F, -0.032520242F, -0.113194436F,
          -0.112953596F, -0.09137275F, -0.20443049F, -0.19703889F, -0.034259316F,
          -0.09617594F, -0.15494855F, 0.08876449F, -0.015829084F, -0.064322114F,
          0.043346938F, -0.0066519002F, -0.08969231F, 0.11969951F, 0.066829614F,
          0.009989125F, 0.14218238F, 0.09682661F, 0.05054941F, 0.038433235F,
          0.0012603264F, -0.035231397F, 0.06532825F, 0.059382018F, 0.011190879F,
          -0.10567976F, -0.044345606F, 0.0029023546F, -0.15646341F, -0.0610979F,
          -0.024102798F, -0.1541645F, -0.05617955F, 0.015375616F, -0.041106414F,
          -0.07103082F, 0.05796109F, -0.062215783F, -0.074845396F, 0.04454642F,
          -0.050087824F, -0.07456708F, 0.044329233F, 0.01921442F, 0.087057106F,
          0.014819093F, 0.022525417F, 0.10427003F, 0.061678812F, -0.026901847F,
          0.039293073F, -0.021397453F, 0.1925517F, -0.10560073F, -0.025371172F,
          0.18116026F, -0.16006531F, -0.096608706F, 0.17237525F, -0.14633028F,
          -0.08655574F, 0.06300302F, -0.109598935F, -0.10282608F, 0.10105681F,
          -0.06749737F, -0.04914598F, 0.07178752F, -0.09725272F, -0.13589905F,
          -0.00050439703F, 0.044708192F, 0.003927225F, 0.008129979F,
          0.052315015F, 0.039433528F, -0.020635432F, 0.0038072127F,
          -0.0048628836F, 0.07003332F, 0.0034695452F, 0.046963274F,
          -0.028901719F, -0.19676444F, -0.15052173F, -0.0028489833F,
          -0.043348443F, 0.046996508F, 0.03595777F, -0.031258475F, -0.028976036F,
          0.04148679F, -0.055651013F, 0.051580887F, 0.012304505F, -0.009540025F,
          0.015831266F, -0.017973838F, -0.0018580733F, -0.048319913F,
          -0.07842856F, -0.07130383F, -0.10976658F, 0.0049824077F, -0.026475182F,
          -0.0759656F, -0.09117729F, -0.12353069F, -0.06079463F, -0.1084652F,
          -0.22438483F, -0.13730796F, -0.10982326F, -0.15791963F, -0.09153332F,
          0.00077409094F, -0.011873954F, 0.0015029851F, -0.023193771F,
          0.005540005F, 0.019927371F, 0.010996892F, 0.055966467F, 0.041596096F,
          0.057207305F, -0.07108271F, 0.031554945F, 0.120208986F, -0.038186837F,
          0.08087558F, 0.06711145F, -0.020574411F, 0.033910666F, 0.02444927F,
          0.09332121F, 0.007021372F, 0.012991761F, 0.092883915F, 0.016275615F,
          0.043436423F, 0.0721172F, 0.0040535247F, -0.031504624F, -0.07816051F,
          -0.025462352F, -0.02343006F, -0.0556576F, -0.00029128388F,
          -0.01855609F, 0.010329187F, 0.05256412F, 0.09654757F, -0.15611945F,
          -0.06364577F, 0.010488807F, -0.19980092F, -0.107119985F, -0.027253484F,
          -0.15061341F, -0.04272273F, -0.13735098F, -0.14562756F, -0.06627403F,
          0.028703682F, -0.034299705F, -0.005548603F, 0.12979501F, 0.13000672F,
          0.056691296F, 0.026823606F, 0.010040909F, 0.00063204835F, 0.00414638F,
          -0.033930134F, -0.03380473F, -0.0069984617F, -0.035214182F,
          -0.023539266F, -0.033142105F, 0.062057342F, 0.0009899961F,
          -0.010985818F, 0.059389077F, 0.023599321F, 0.0008218941F, -0.04535846F,
          -0.013060163F, 0.06815566F, 0.04701272F, 0.057546746F, -0.028257983F,
          0.043578167F, 0.0056431154F, -0.09135482F, -0.051108222F,
          -0.051528443F, -0.039813574F, -0.12848096F, -0.028752567F, 0.06409592F,
          0.06774309F, -0.016463514F, -0.0017483518F, 0.069082685F, 0.03305105F,
          0.04415674F, -0.018702608F, -0.0107254F, -0.025523521F, 0.0583059F,
          -0.013589468F, -0.069265306F, 0.029462235F, -0.06782912F,
          -0.062371556F, -0.14097978F, -0.038819455F, 0.14801762F, 0.093735114F,
          -0.0068461266F, 0.18865387F, 0.26221064F, 0.23039976F, -0.0020094486F,
          -0.047181F, -0.02511901F, -0.042872023F, -0.04817616F, 0.010397192F,
          0.03686306F, 0.002676085F, 0.06635125F, 0.05268928F, -0.07490124F,
          -0.1691621F, -0.050501503F, 0.06583429F, 0.07711301F, 0.0032618495F,
          0.04600384F, 0.029637923F, -0.013047393F, -0.016008954F, 0.06044418F,
          -0.0541071F, 0.003520033F, 0.009266503F, -0.01875674F, -0.004046948F,
          -0.07335384F, 0.055675656F, 0.015488656F, 0.018792434F, 0.023942415F,
          0.007358426F, -0.07750603F, 0.035089083F, 0.024249347F, -0.03612583F,
          0.024587188F, -0.10650139F, -0.049456175F, -0.006031814F, 0.099718995F,
          -0.02800498F, -0.049058855F, 0.043254174F, 0.0028553726F, -0.16942768F,
          -0.28858203F, -0.25112996F, -0.036400706F, -0.010987269F, 0.034569833F,
          0.03470842F, 0.048177443F, 0.09355965F, 0.12521252F, 0.051959965F,
          0.018307058F, -0.0016076498F, -0.08276862F, -0.013615835F,
          -0.036859997F, -0.14748803F, -0.13102596F, 0.07907393F, 0.082650624F,
          0.04179141F, -0.08927967F, -0.062411766F, -0.066195235F, -0.046565983F,
          -0.003100324F, -0.05094043F, 0.035764396F, 0.09180732F, 0.014863767F,
          -0.0641293F, -0.12603334F, -0.051305413F, 0.028639484F, 0.061091326F,
          0.08887796F, -0.11764164F, -0.19826171F, -0.12422761F, 0.07599887F,
          0.18168698F, 0.07580444F, 0.05044615F, 0.06796659F, 0.044461794F,
          0.11101902F, 0.23142944F, 0.015182472F, -0.08931384F, -0.10472982F,
          0.04286969F, -0.14077096F, -0.29522863F, -0.1643616F, -0.0684419F,
          -0.063324966F, 0.03903229F, -0.045059368F, 0.02519802F, -0.04413412F,
          0.01549316F, 0.031094365F, 0.0060030706F, 0.020248339F, 0.023207799F,
          0.015486854F, -0.0006300821F, -0.062333513F, -0.004692455F,
          0.049217746F, 0.028284168F, 0.040254947F, -0.10006981F, -0.050816905F,
          0.026146024F, -0.007971868F, -0.069707364F, -0.018242905F,
          0.015197182F, 0.00499538F, 0.03135705F, -0.07760179F, -0.05899095F,
          -0.038054246F, 0.04893431F, 0.044631768F, 0.0063128946F, 0.041225053F,
          0.031291798F, 0.019963698F, -0.050763357F, -0.02177207F, -0.02399126F,
          0.042865466F, -0.017518107F, -0.027361862F, 0.06676647F, 0.01856507F,
          0.015144551F, -0.02670818F, -0.03178275F, -0.018571153F, -0.008885113F,
          -0.117537506F, 0.042934548F, -0.016745634F, -0.27499616F,
          0.0066854157F, 0.014881052F, 0.07146965F, 0.124410264F, -0.08906486F,
          -0.057881664F, -0.009490736F, -0.08665277F, -0.01848374F,
          -0.032850597F, 0.025100945F, 0.00090530637F, 0.059667073F,
          -0.038856782F, -0.0065061986F, 0.004260258F, -0.06339464F,
          -0.03371356F, -0.042122435F, 0.03225613F, 0.032850645F, 0.06501986F,
          -0.020493744F, -0.07370319F, -0.13196114F, -0.008473676F, 0.018717136F,
          -0.03806937F, -0.024172453F, 0.010772671F, 0.0047446135F, -0.06212871F,
          0.06337889F, 0.061560705F, -0.048486553F, -0.016618134F, 0.018263875F,
          0.033243217F, 0.012060566F, 0.013505566F, -0.021378882F, -0.15867619F,
          -0.061035264F, 0.0066307015F, 0.0018729551F, -0.010979115F,
          -0.46245104F, -0.6822493F, -0.29981166F, 0.15486294F, 0.2296275F,
          0.008134328F, 0.13844533F, 0.24600558F, 0.1645553F, 0.008706883F,
          0.035345443F, 0.016964493F, -0.02811485F, -0.105835006F, 0.0021720387F,
          0.05086132F, -0.011465386F, -0.00049852545F, 0.058204606F, 0.04678409F,
          0.064381234F, -0.01551231F, -0.019218067F, -0.01798024F,
          -0.0154361185F, -0.01348334F, -0.032901756F, 0.06495748F, 0.025407005F,
          -0.03603316F, -0.003979909F, -0.1449438F, -0.14264786F, -0.052899394F,
          -0.02671477F, 0.00038096798F, -0.21244237F, -0.310769F, -0.07227687F,
          -0.15177055F, -0.017080007F, 0.15900442F, 0.11466224F, 0.22973238F,
          0.12244767F, 0.044491768F, 0.055102345F, 0.0075039538F, 0.04503773F,
          0.04909537F, 0.008689937F, 0.013583749F, 0.01594575F, -0.020823842F,
          -0.0055681067F, 0.018058106F, -0.007956485F, 0.066087335F,
          0.011911609F, 0.03924039F, 0.017215978F, -0.026071116F, 0.022295283F,
          -0.027379144F, 0.019014044F, 0.020217221F, -0.0111891525F,
          -0.020954994F, -0.006765628F, 0.008001257F, -0.053793613F,
          -0.0061170803F, -0.27072182F, -0.3909985F, 0.02309813F, -0.28131527F,
          -0.007272323F, 0.30462766F, 0.012354751F, 0.3190491F, 0.20302778F,
          -0.029471194F, -0.065868616F, -0.029787542F, -0.1180945F, -0.06749344F,
          0.10098736F, 0.019415144F, 0.100079075F, 0.14277203F, -0.09624757F,
          0.13700172F, 0.106569745F, 0.17423946F, 0.19658712F, -0.058411986F,
          0.0402567F, -0.16057552F, -0.22244436F, 0.0020545262F, 0.017823804F,
          0.028457988F, -0.035024196F, 0.042753395F, 0.020373961F, 0.041790042F,
          0.038840942F, -0.009251762F, 0.023521677F, 0.2902707F, -0.008699655F,
          0.26429614F, 0.18391614F, -0.27506423F, -0.034357015F, -0.42548507F,
          -0.3337366F, -0.023332898F, -0.028007708F, 0.006010165F,
          -0.0074335095F, 0.013187441F, 0.02803075F, 0.0113199325F, 0.030686589F,
          0.017316954F, -0.012464023F, 0.08899181F, -0.01498482F, 0.11643034F,
          0.034541413F, -0.06671316F, -0.101005405F, -0.21986203F, -0.115791395F,
          0.023945551F, -0.06795473F, 0.00054700946F, 0.02803138F, -0.0395497F,
          -0.0024903473F, -0.014361059F, -0.005753035F, -0.051748466F,
          0.0054886723F, 0.020161126F, -0.0025463162F, -0.018305203F,
          0.012647482F, -0.009212262F, -0.01985468F, 0.009869672F, -0.02041389F,
          0.043670252F, -0.021788716F, -0.024069311F, -0.09193576F, -0.12892167F,
          -0.026749844F, -0.021142045F, 0.01389173F, -0.0035167458F,
          -0.007957514F, -0.009739308F, -0.031074224F, -0.024575666F,
          -0.06250358F, -0.040847514F, -0.03372753F, -0.058963224F,
          -0.0035998041F, 0.017929006F, -0.022809865F, -0.01031528F,
          -0.035520174F, -0.0504699F, -0.02859282F, 0.0078055253F, 0.025385212F,
          0.008333085F, -0.009429921F, 0.007110659F, -0.013387225F,
          -0.00023618767F, -0.03765172F, 0.009016097F, -0.05017927F,
          -0.0008999211F, 0.034375105F, 0.014100204F, -0.039194677F,
          -0.057506554F, -0.00014149929F, -0.023745893F, -0.032098047F,
          -0.008509525F, -0.0456855F, 0.03716503F, 0.0078118346F, -0.072243646F,
          -0.025863877F, -0.05074868F, -0.1394953F, -0.01078988F, -0.0124403015F,
          0.022202857F, 0.020585932F, -0.09455015F, -0.08803486F, -0.052522678F,
          -0.09524949F, -0.14478001F, -0.043179303F, 0.010340033F, 0.09241678F,
          -0.0041702893F, 0.0010402504F, 0.013811155F, 0.011141141F,
          -0.011176099F, 0.031055924F, -0.021231566F, -0.03341916F,
          -0.0048742723F, -0.020037051F, 0.029497424F, -0.06596553F,
          -0.073915005F, 0.01831149F, -0.14022715F, -0.056512307F, 0.010058012F,
          0.023850225F, 0.1037927F, -0.0127172405F, -0.0054701464F, 0.023645997F,
          -0.012480841F, 0.027733264F, -0.001843241F, 0.03690911F, 0.03202888F,
          -0.0025830676F, -0.022429522F, -0.005848913F, 0.0008588506F,
          -0.004775341F, -0.0095606195F, 0.006590119F, -0.029173149F,
          0.020642215F, 0.014948144F, 0.014979826F, 0.04223437F, -0.0005067483F,
          0.048681036F, 0.077695504F, 0.014718346F, 0.020319726F, 0.024655329F,
          0.0051483572F, 0.0029178304F, -0.004414121F, -0.007906427F,
          0.02504807F, -0.024694262F, 0.02021333F, -0.00560852F, -0.027126703F,
          0.012771134F, 0.020360494F, -0.0076217894F, -0.03919229F,
          -0.012420976F, -0.005598794F, -0.053581834F, 0.015817462F,
          0.005243878F, -0.019292343F, -0.036060885F, -0.031426195F,
          -0.034652524F, -0.045414824F, -0.048685364F, -0.016594134F,
          -0.032047078F, 0.03186484F, -0.0013835747F, 0.003568837F, 0.024129046F,
          0.04217593F, -0.042588543F, -0.031647902F, 0.024526589F, -0.04166763F,
          0.036095764F, 0.013000442F, 0.015008064F, 0.037530027F, 0.00028548343F,
          0.022788607F, 0.010088072F, -0.01430808F, -0.018496888F,
          -0.0064831157F, -0.014546773F, -0.23527622F, -0.05626939F,
          0.029590005F, 0.032535754F, 0.40403056F, 0.009756847F, 0.10366839F,
          -0.047904953F, -0.26568022F, -0.030844295F, 0.011047932F, -0.02528869F,
          -0.0024366418F, 0.02548311F, -0.02096468F, -0.0015934219F,
          0.027547028F, 0.0052041383F, 0.0014955862F, -0.03297083F,
          -0.027761301F, 0.0012860736F, 0.00293341F, 0.009560937F, 0.023920745F,
          0.026568215F, -0.004162748F, -0.10743246F, -0.037683353F,
          -0.024176233F, -0.06974697F, -0.04712155F, -0.036556724F,
          -0.014318669F, 0.021119608F, 0.048735417F, -0.008202581F, 0.011223664F,
          0.079089835F, -0.06464379F, -0.035795346F, 0.043116447F, -0.058687642F,
          -0.05017714F, 0.04637683F, 0.058523305F, 0.07097258F, 0.052750643F,
          -0.22542724F, -0.17094636F, -0.07337961F, -0.2621945F, -0.16473866F,
          0.02180504F, 0.0795319F, 0.0033298142F, 0.13214085F, 0.04388065F,
          -0.013002124F, -0.005086034F, -0.028630447F, -0.057359394F,
          -0.033723626F, -0.08564471F, 0.04105404F, -0.0860586F, 0.013618518F,
          0.13265362F, 0.11682022F, 0.018821891F, 0.062406823F, 0.022955898F,
          -0.048059627F, -0.009726423F, -0.073963165F, 0.026042115F, 0.05720249F,
          -0.00119584F, 0.0069177924F, 0.02316612F, -0.05956697F, -0.12509078F,
          -0.23859888F, -0.18994915F, -0.34955913F, -0.46248743F, -0.42733258F,
          -0.25718126F, -0.37013167F, -0.34659347F, 0.034795903F, 0.0171418F,
          0.031341527F, -0.0076323445F, -0.026730463F, 0.0033659935F,
          -0.009067289F, 0.020572832F, -0.0045841266F, -0.023981731F,
          -0.048178867F, 0.004762322F, -0.0970494F, -0.13611743F, -0.07603967F,
          -0.11273808F, -0.064025834F, -0.05457849F, 0.002368832F, 0.06691467F,
          0.044228204F, -0.15631434F, -0.030157141F, 0.013970674F, -0.036615107F,
          0.06983253F, 0.06404997F, -0.17639063F, -0.22144924F, -0.08619434F,
          -0.393119F, -0.35263807F, -0.14533347F, -0.30823162F, -0.25942102F,
          -0.065274656F, 0.027874863F, -0.0043197135F, -0.01338877F,
          0.008808364F, -0.026194848F, 0.013865314F, 0.05930914F, 0.079843275F,
          0.061611224F, 0.016523376F, 0.09174038F, 0.07267878F, 0.15277332F,
          0.021389209F, 0.01918388F, 0.17333634F, 0.07053967F, 0.107066944F,
          -0.119142115F, -0.13581438F, -0.030292988F, -0.1024468F, -0.09407434F,
          -0.08686215F, 0.044064365F, -0.047222205F, -0.039508156F, -0.13928966F,
          -0.1582121F, -0.10763005F, -0.24741992F, -0.29454753F, -0.23601884F,
          -0.1815064F, -0.20074913F, -0.18027125F, -0.044819444F, -0.028862962F,
          0.014612379F, -0.026407465F, -0.12934098F, 0.03137155F, -0.07978428F,
          -0.0937784F, 0.08019899F, -0.034914922F, 0.081590876F, -0.009789497F,
          -0.06753307F, -0.02205096F, 0.023957953F, -0.103481375F, -0.06504051F,
          -0.02983675F, 0.043658167F, 0.09745397F, 0.08255043F, -0.09052367F,
          -0.05403372F, 0.0317764F, -0.06354291F, -0.02202262F, 0.012022375F,
          -0.17508474F, -0.04834271F, 0.07133407F, -0.020025931F, 0.007934824F,
          0.026448015F, -0.042762756F, 0.051984616F, 0.07322081F, -0.09291699F,
          0.04519917F, 0.0820519F, -0.13714632F, -0.008199126F, 0.080453396F,
          -0.029067084F, -0.003436551F, 0.0075375857F, 0.008766281F,
          0.026680691F, -0.033987492F, 0.02037483F, 0.019126622F, 0.002202363F,
          0.03579439F, 0.017957287F, 0.023624795F, 0.09913824F, 0.016933328F,
          0.05928642F, -0.0012353532F, 0.06065438F, 0.012190999F, -0.07654298F,
          -0.058590196F, 0.07609362F, 0.03914417F, 0.07884769F, 0.04925427F,
          0.101905145F, 0.05735154F, -0.012405756F, 0.04228695F, 0.012856597F,
          0.021365643F, -0.054251894F, -0.0139791565F, 0.10930343F, 0.0488976F,
          0.066577196F, 0.1009967F, -0.034393825F, 0.03891003F, 0.07514947F,
          0.09227267F, 0.050490886F, 0.014191035F, 0.13399482F, 0.082181096F,
          0.034563843F, 0.050167307F, 0.09158168F, 0.07339191F, -0.10140306F,
          -0.032743365F, -0.029168166F, -0.20161761F, -0.12853992F,
          -0.093406245F, -0.046591334F, -0.023017228F, -0.06669219F,
          0.0033196253F, 0.047388498F, 0.0076768165F, 0.08599405F, 0.035429727F,
          0.013530952F, 0.042607505F, 0.041619338F, 0.029927082F, -0.014476107F,
          -0.025783729F, -0.118370734F, -0.14253472F, -0.07527373F, -0.13995746F,
          0.016172465F, -0.02834922F, -0.09538984F, 0.00017141544F,
          -0.0026366068F, 0.042502843F, 0.06995159F, 0.018202307F, 0.0010931295F,
          -0.034902167F, 0.0045636958F, 0.04588912F, 0.0694728F, 0.04379054F,
          -0.004603679F, 0.1118363F, 0.059540607F, 0.04766191F, 0.067446716F,
          0.08944794F, 0.08883985F, 0.09001663F, -0.018513283F, 0.0018571385F,
          0.10287272F, -0.032219365F, -0.075875536F, 0.07301143F, -0.11407505F,
          -0.05111588F, -0.057315726F, 0.012212703F, -0.03566879F, -0.018846905F,
          -0.008850892F, -0.032363772F, 0.018543513F, 0.008931995F, 0.033794884F,
          0.04783875F, 0.029287642F, -0.091033734F, -0.030811874F, 0.38744423F,
          -0.0053459276F, -0.17728621F, -0.022469083F, 0.060233284F,
          -0.09333387F, -0.110290885F, 0.059687674F, -0.16111113F, -0.09382661F,
          0.10441382F, -0.013726891F, 0.10718128F, 0.04073543F, -0.0035744444F,
          0.01576566F, -0.19683139F, 0.08081559F, -0.11200462F, -0.075893715F,
          -0.04245832F, -0.067320816F, 0.39729732F, -0.023656487F, -0.07446478F,
          -0.06454951F, -0.11578769F, -0.014195873F, 0.11962106F, -0.071868435F,
          0.020006634F, 0.09889081F, 0.12580107F, 0.10414941F, 0.08097763F,
          0.3061687F, -0.041292876F, -0.30125257F, 0.28436828F, -0.09816099F,
          -0.4056929F, 0.02400213F, 0.04332381F, 0.06823339F, 0.0022605567F,
          -0.0413136F, -0.051395297F, 0.028440611F, -0.034997366F, -0.15439461F,
          0.011943862F, -0.11160917F, -0.14540498F, -0.02274857F, -0.1152596F,
          -0.11234407F, 0.036326505F, 0.12157354F, -0.034297593F, -0.012761731F,
          -0.05134871F, 0.019442838F, -0.06651891F, -0.025244996F, -0.021382531F,
          -0.057417646F, -0.029869461F, 0.087979324F, 0.036837403F,
          -0.013807767F, -0.09367454F, 0.033613753F, 0.054043606F, -0.10338858F,
          -0.022918269F, 0.03309233F, -0.024294324F, -0.051621612F, -0.14843455F,
          -0.17958046F, -0.10161057F, 0.078612946F, -0.027210431F, -0.13980836F,
          0.058118444F, -0.08384664F, -0.15603928F, -0.102473296F, -0.088620685F,
          -0.17635114F, -0.024927335F, 0.088238254F, -0.16553646F, -0.10657334F,
          -0.041389134F, -0.06200181F, -0.016005063F, -0.024364952F,
          -0.17610419F, 0.017114634F, 0.033720765F, -0.02438304F, 0.12046418F,
          0.039414953F, -0.016807089F, 0.121393315F, 0.124463685F, -0.088106304F,
          -0.17044166F, -0.099635474F, -0.06477953F, 0.037123136F, 0.06523954F,
          -0.09663414F, -0.08246754F, -0.0063743275F, -0.19548304F, 0.07405346F,
          0.3358172F, 0.12442445F, -0.029045178F, 0.046544332F, 0.016654786F,
          -0.064938575F, -0.062239412F, 0.0072722216F, 0.063380554F, 0.10963929F,
          0.00747183F, 0.059433572F, 0.07738429F, 0.040641144F, 0.14705946F,
          0.101297945F, -0.16301587F, 0.099646576F, -0.07391202F, -0.06142145F,
          -0.113374874F, -0.036101487F, -0.049650066F, -0.071458116F,
          -0.0473803F, -0.05650511F, -0.016215796F, -0.12128474F, -0.045995302F,
          0.05231088F, -7.79642E-5F, 0.033534415F, -0.009750654F, -0.013578748F,
          0.11672921F, -0.11225765F, -0.058060087F, 0.10568836F, -0.11081884F,
          -0.0050366945F, -0.019372454F, -0.1182606F, -0.112445824F,
          -0.034949172F, 0.08662997F, 0.07615916F, -0.120399766F, 0.015854757F,
          0.122254744F, -0.06529767F, -0.16807154F, -0.08806159F, -0.015501219F,
          -0.015514852F, 0.049507804F, 0.008532951F, 0.26425368F, 0.06267593F,
          0.05144633F, 0.061900064F, 0.023129629F, 0.09418507F, -0.0035748982F,
          -0.09846638F, 0.09782626F, -0.09573945F, -0.04934952F, 0.04427469F,
          -0.0013208011F, -0.0076651527F, -0.048267268F, -0.06920177F,
          0.03431211F, -0.3432633F, -0.035048068F, 0.0072972467F, -0.044640653F,
          -0.026744554F, 0.0012842919F, -0.06917885F, 0.047403116F, 0.034699373F,
          -0.1344353F, -0.013566297F, 0.008160956F, -0.043808572F, 0.013638334F,
          0.105572194F, -0.1400134F, -0.087093785F, 0.1320474F, -0.0077545363F,
          0.16528384F, -0.07949876F, -0.009455681F, 0.01882233F, 0.015326889F,
          0.037812896F, 0.122248545F, 0.0066052713F, -0.0493584F, -0.17412658F,
          0.07431281F, -0.06963742F, -0.0710875F, -0.03923529F, -0.042507265F,
          0.0599681F, 0.06658243F, -0.0014127482F, 0.15611134F, 0.047703177F,
          0.0061591757F, 0.014538346F, 0.06845131F, -0.20522024F, 0.0650033F,
          0.050352074F, -0.13339235F, 0.11685422F, 0.02815092F, -0.030516313F,
          -0.064777195F, 0.033427972F, 0.16310185F, 0.025080206F, -0.21071969F,
          0.09898901F, 0.0058444263F, -0.12619571F, -0.1095819F, -0.004323947F,
          0.064842515F, -0.23050357F, 0.06974274F, 0.1150225F, -0.04386962F,
          0.062429782F, 0.056354616F, -0.054131504F, -0.092543155F, -0.11931246F,
          -0.16460225F, 0.14555414F, 0.10411743F, -0.19695409F, -0.06741936F,
          0.018006645F, -0.04525295F, -0.04553724F, -0.018575491F, -0.019353664F,
          -0.04425788F, 0.0045770598F, 0.022390418F, 0.017833175F, 0.00673995F,
          0.053721454F, 0.057586495F, -0.04223664F, 0.077824816F, 0.0634705F,
          -0.047934353F, -0.023692569F, -0.036822032F, -0.13872002F,
          -0.010941667F, -0.017137868F, -0.02464774F, -0.014761107F,
          -0.029280607F, 0.010856958F, -0.012135353F, -0.0020860725F,
          -0.033234514F, 0.058409072F, 0.041998256F, 0.036295332F, -0.029999018F,
          0.017653896F, 0.038279966F, -0.042138904F, -0.016938455F, 0.027372057F,
          -0.053734504F, -0.030761344F, -0.05725946F, -0.024320886F,
          -0.0064273276F, -0.021976747F, -0.01494404F, 0.006649858F,
          0.0052406103F, -0.011676143F, -0.049227912F, 0.00845172F,
          -0.056145523F, -0.1483056F, -0.04320595F, 0.010016841F, -0.04991396F,
          -0.043728452F, 0.011932968F, -0.028677853F, -0.06852698F, 0.040574048F,
          0.059134964F, -0.015882265F, 0.022470051F, 0.026507124F, 0.005842212F,
          0.023284608F, 0.037508342F, -0.069331676F, 0.07689812F, 0.07918683F,
          -0.04863916F, 0.005079147F, 0.0006629279F, -0.08662431F, -0.039689444F,
          -0.027710434F, -0.018691823F, 0.035150874F, -0.03976547F,
          -0.080308594F, 0.017603692F, -0.01200523F, -0.040106747F, -0.08302416F,
          -0.06440446F, 0.042346835F, -0.091715224F, -0.14780691F,
          -0.0146552725F, -0.03573618F, -0.08364999F, 0.051267702F,
          -0.0021449407F, -0.0018276281F, -0.0308598F, -0.0036153898F,
          -0.008229048F, -0.0103729805F, -0.015033169F, -0.009107642F,
          0.011197969F, -0.020973455F, -0.046030436F, -0.010750017F,
          -0.14481126F, -0.17146581F, -0.05178246F, -0.111265995F, -0.12062764F,
          -0.033052184F, -0.0037363388F, 0.009769162F, 0.055958785F,
          -0.09153031F, -0.059183143F, 0.05779555F, -0.029741684F,
          -0.0013227543F, 0.05090432F, -0.022317525F, -0.0053742817F,
          -0.02433457F, -0.0014149137F, 0.0014002193F, -0.006917763F,
          0.01277031F, 0.029808912F, -0.013207628F, -0.065075405F, -0.035209782F,
          -0.054939173F, -0.033579264F, -0.038481474F, -0.0121206045F,
          -0.016192634F, -0.022135725F, -0.036659572F, -0.033130027F,
          0.022866912F, 0.050169554F, -0.023810295F, -0.006340517F,
          -0.018462004F, -0.024401579F, 0.016775995F, -0.020051049F,
          0.00046529068F, -0.017424222F, -0.01007583F, 0.0068376022F,
          -0.011401591F, -0.011283497F, -0.0023598215F, 0.007140239F,
          0.01281772F, -0.0788893F, -0.03065287F, 0.025819067F, -0.09865916F,
          -0.046903826F, -0.028374163F, -0.043579437F, -0.01128684F,
          -5.796052E-5F, 0.010756622F, 0.0332631F, -0.035989814F, -0.02703468F,
          0.0087859295F, 0.053395115F, -0.008121496F, 0.0012283705F,
          -0.007534974F, 0.022631804F, 0.011202088F, 0.03958964F, 0.0123731755F,
          -0.04116891F, 0.06910974F, 0.0011255433F, -0.018015375F, 0.018978577F,
          0.00066422F, 0.0035795504F, 0.042822752F, 0.05134071F, 0.07774611F,
          0.10124858F, 0.053531416F, 0.060219366F, 0.070013225F, -0.05043309F,
          -0.05156317F, -0.071120344F, -0.02057264F, -0.033896893F,
          -0.021581937F, 0.015605712F, -0.013955766F, -0.012903528F,
          -0.024972834F, -0.020162398F, -0.056551535F, 0.036759265F, 0.03656781F,
          -0.029182088F, -0.0024229318F, 0.016357157F, -0.015414154F,
          0.07333491F, 0.0780708F, 0.0039672917F, -0.012394255F, 0.0064965454F,
          0.04009718F, -0.008558333F, -0.02310776F, 0.018565191F, 0.0013393296F,
          0.0041605937F, -0.0331411F, -0.0051126406F, -0.0032524723F,
          0.014323046F, -0.011133229F, -0.017895408F, -0.007276375F,
          -0.009094421F, -0.006622002F, -0.023587583F, -0.023812117F,
          -0.036902174F, 0.017568326F, 0.011562874F, -0.008304669F, -0.02048517F,
          0.0064584296F, -0.0014097368F, 0.0026994345F, -0.045820553F,
          -0.011985442F, -0.00942671F, 0.00063786295F, -0.021412635F,
          -0.027369263F, -0.033606116F, -0.016509691F, 0.033623163F,
          -0.020077039F, -0.015362728F, -0.004794965F, -0.0043924577F,
          -0.020932537F, -0.026119519F, -0.0426121F, -0.022362119F, -0.03059359F,
          -0.02861497F, -0.029308919F, -0.0048673055F, 0.006030182F,
          -0.0076851733F, -0.011921807F, 0.017419498F, -0.023533428F,
          -0.015708774F, 0.018622208F, 0.00031925613F, -0.0056567F, 0.004691757F,
          0.01834745F, 0.03548858F, -0.019493587F, 0.027519412F, 0.0210299F,
          -0.041692875F, -0.028668914F, -0.016836444F, -0.010098959F,
          0.018080918F, 0.004490389F, 0.005571845F, -0.0048396927F, 0.032454364F,
          -0.016820455F, 0.0042737937F, 0.03697613F, -0.017084695F, 0.007634094F,
          -0.0030362892F, 0.06110347F, 0.013347222F, -0.025458222F, 0.05571886F,
          -0.014795983F, -0.026725015F, 0.03341302F, 0.012166674F,
          -0.0014108991F, -0.045339823F, -0.033967543F, 0.10672157F,
          -0.03647739F, -0.042134903F, 0.06343837F, -0.044213858F, -0.03208786F,
          -0.016110362F, -0.069001436F, 0.015544566F, 0.06435196F, -0.07218451F,
          0.100264214F, 0.02526052F, -0.01635935F, 0.0077988706F, 0.06547493F,
          -0.052788652F, -0.0066901525F, -0.016032526F, -0.043163247F,
          -0.055097852F, -0.036905065F, 0.009722616F, -0.030402552F,
          -0.038332216F, -0.027673751F, -0.08476648F, -0.14823173F, 0.016511144F,
          -0.063318625F, -0.09000218F, -0.101461135F, -0.1027873F, 0.010250975F,
          -0.041536763F, -0.044745993F, 0.09857661F, -0.03800469F, -0.07007411F,
          0.03666354F, 0.0066933124F, 0.042984184F, -0.02473413F, 0.13508694F,
          -0.021904688F, -0.21547619F, 0.21686581F, 0.02587298F, -0.23207526F,
          0.21204603F, 0.02781697F, -0.0571864F, -0.047901206F, -0.0031873968F,
          0.040238827F, -0.05063648F, 0.02123572F, 0.088183746F, -0.08270694F,
          -0.004006643F, 0.07541725F, 0.0153027205F, 0.036192592F, 0.007368681F,
          0.014670907F, -0.01133836F, 0.036825184F, 0.005403595F, 0.0055584363F,
          0.040001314F, -0.07978155F, -0.0018705989F, 0.03900157F, -0.078641884F,
          0.033390213F, 0.10283648F, -0.109662175F, -0.034637015F, 0.107475504F,
          0.011563607F, -0.0466345F, -0.019265553F, 0.02516739F, 0.03472683F,
          -0.014189419F, 0.018842187F, -0.0016261407F, -0.009429085F,
          -0.046396576F, -0.033649653F, 0.10206717F, -0.04042798F, 0.011980085F,
          0.2587928F, -0.054313526F, 0.007944507F, 0.10471169F, -0.16475166F,
          -0.058980137F, 0.033110846F, -0.16602868F, -0.057325285F, 0.047002442F,
          -0.13289441F, -0.032307968F, -0.013580023F, -0.013134401F,
          0.015476199F, 0.018461546F, -0.035216287F, 0.009608911F, 0.019958472F,
          0.020074835F, -0.0009125114F, 0.04252409F, -0.076565035F,
          -0.0010711959F, 0.08275153F, -0.08447243F, -0.018514905F, 0.030477222F,
          -0.085086584F, -0.051419023F, 0.023133017F, -0.07846479F, -0.09889754F,
          -0.009650114F, 0.015610599F, -0.028422534F, -0.16075501F,
          -0.026957026F, -0.015026145F, -0.10761792F, -0.00096213206F,
          -0.0041970536F, 0.028110102F, 0.027983742F, 0.017547619F,
          -0.027796084F, -0.009511726F, -0.011619454F, 0.012921154F,
          0.077472314F, 0.0362318F, -0.10178012F, 0.06480788F, 0.011755518F,
          -0.17098767F, 0.07173818F, -0.02124207F, -0.20029457F, -0.01710269F,
          0.00033645416F, 0.026567172F, -0.033213433F, 0.054298226F, 0.13336538F,
          -0.033548616F, 0.031934407F, 0.055980776F, -5.1928266E-5F, 0.11451903F,
          0.03103864F, 0.0014649231F, 0.114713944F, 0.04129716F, 0.031688906F,
          0.043024693F, 0.033053353F, 0.011514402F, -0.0668054F, -0.06663708F,
          -0.04612755F, -0.046840984F, 0.008134892F, -0.017465247F,
          -0.042428337F, 0.022011504F, -0.086389154F, -0.039634865F, 0.10116107F,
          -0.09367438F, -0.0030865853F, 0.051039238F, -0.08395229F,
          -0.030984193F, 0.08602489F, 0.013026261F, 0.013480316F, -0.04152466F,
          0.015938133F, 0.03421003F, -0.061957773F, -0.012094378F, 0.0017349079F,
          -0.0053686523F, 0.0047069746F, 0.04131695F, -0.092909984F,
          0.018701961F, 0.043062862F, -0.05301014F, -0.041269954F, -0.030525448F,
          -0.041827902F, -0.0016583831F, -0.017722363F, -0.025150018F,
          -0.043193575F, -0.024137145F, 0.0038556457F, 0.0015340515F,
          0.013137947F, 0.026918557F, -0.0038226335F, 0.040047128F, 0.021653255F,
          -0.028470905F, 0.04609331F, 0.05713256F, -0.0081963325F, -0.05634196F,
          0.03778764F, -0.03737788F, -0.044777714F, -0.035603072F, -0.06251505F,
          -0.048058074F, 0.04406962F, -0.08716072F, 0.009758061F, 0.08374621F,
          -0.036200434F, 0.023296567F, -0.0067108483F, -0.004969053F,
          -0.0037515853F, 0.03276541F, -0.011993034F, -0.017328454F,
          0.046034705F, -0.0022704802F, 0.010487034F, 0.008869652F, 0.011749576F,
          0.011432616F, -0.019720342F, -0.033929165F, 0.018587733F,
          -0.027641635F, -0.01938991F, -0.023524776F, 0.020022243F, 0.09507505F,
          0.097039714F, -0.17932294F, -0.0073856255F, -0.033370078F, 0.06151611F,
          0.042709596F, -0.027224487F, -0.053064696F, -0.031784367F,
          0.0018986166F, -0.05568456F, 0.034220997F, 0.008898784F, -0.03690957F,
          -0.107379854F, -0.050524164F, 0.15226704F, -0.15833189F, -0.0592223F,
          0.15122218F, -0.09731978F, -0.01664208F, 0.06419602F, 0.13338485F,
          0.075159214F, 0.07875298F, -0.042775515F, -0.05618309F, -0.011796311F,
          -0.01381392F, -0.02213867F, -0.039922927F, -0.03053226F, -0.04584283F,
          0.10783471F, -0.056639906F, -0.06645274F, 0.039383307F, -0.042595338F,
          -0.0013298806F, 0.028247233F, -0.041564483F, -0.034451634F,
          -0.02834007F, -0.119342476F, -0.27130875F, -0.06563767F, -0.067976035F,
          -0.14234585F, -0.06123766F, -0.028814288F, 0.039046917F, 0.01483116F,
          0.13472168F, 0.16274256F, 0.10397647F, 0.115840256F, 0.1729292F,
          -0.03325818F, 0.07666043F, 0.0034219238F, 0.002524656F, -0.056701977F,
          -0.28290573F, -0.061706554F, -0.07144675F, -0.20731397F, -0.022405947F,
          0.0044933776F, -0.0071848393F, 0.03239618F, -0.0104717435F,
          -0.032160703F, -0.03807835F, 0.037963547F, 0.06938089F, 0.016219819F,
          0.3172337F, 0.14926766F, -0.11353808F, 0.24317062F, 0.07948582F,
          -0.11980092F, 0.11074673F, -0.003960326F, -0.14606959F, -0.08306425F,
          -0.02776501F, 0.035967834F, -0.022763738F, 0.031425435F, 0.05969045F,
          -0.006852496F, -0.010073963F, 0.019839512F, 0.03699712F, 0.04303664F,
          0.056863356F, 0.052131005F, 0.056841638F, -0.05294585F, -0.005908916F,
          -0.07375229F, -0.16788875F, -0.045391344F, 0.007762094F, -0.023550589F,
          0.006790039F, -0.0042204224F, -0.021027021F, -0.03078701F, 0.01605517F,
          0.07831305F, -0.056412477F, -0.07697485F, -0.058695413F, -0.09998289F,
          -0.18208654F, 0.00021814692F, -0.08623621F, -0.1253811F, 0.006742694F,
          0.0067951437F, 0.1001813F, 0.045583252F, -0.1425619F, -0.15371439F,
          -0.09850548F, -0.03678618F, -0.045451183F, -0.06876138F, -0.113355584F,
          -0.072097555F, 0.031377915F, -0.119815096F, -0.12824129F,
          -0.064607844F, -0.04821607F, -0.10443017F, -0.1126311F, -0.1426219F,
          -0.009290505F, 0.014906992F, 0.017027725F, 0.15454583F, 0.05761984F,
          0.09452464F, 0.11891517F, 0.04223832F, -0.079411834F, 0.011686426F,
          0.008563717F, 0.0062239408F, 0.02899738F, -0.026685225F, -0.026036238F,
          0.06379066F, -0.0010364387F, 0.046955884F, 0.015680099F, 0.015887251F,
          0.115122244F, 0.18822898F, -0.061995406F, 0.07904518F, 0.07811913F,
          -0.038155485F, 0.04208226F, 0.043337803F, 0.02901257F, -0.00044453703F,
          -0.0778013F, -0.10591696F, -0.037947793F, -0.14747179F, -0.07864325F,
          -0.17820634F, -0.0397025F, 0.024316616F, -0.20671675F, 0.010363012F,
          0.08657817F, -0.036713038F, 0.089638405F, 0.17637141F, 0.033954814F,
          0.034869052F, 0.0003172527F, 0.01745687F, 0.004436147F, -0.04379211F,
          0.004147078F, 0.018202858F, -0.034843057F, 0.028706774F, 0.010946367F,
          0.044939395F, 0.04500558F, -0.012012125F, 0.0018684987F, 0.008724486F,
          0.017335681F, -0.048110567F, -0.0067546675F, 0.032805383F, 0.1040686F,
          -0.15761231F, -0.166025F, -0.010135765F, -0.02589323F, -0.11082901F,
          -0.01886045F, 0.108863786F, 0.017513491F, 0.042847943F, 0.13574818F,
          0.10860509F, -0.027561218F, 0.015292322F, 0.029057454F, -0.08201618F,
          0.029692391F, 0.010211789F, 0.03799606F, 0.013177455F, -0.060105033F,
          -0.028866487F, 0.011915037F, -0.0016412582F, 0.017386492F,
          0.0039063077F, -0.021524338F, 0.019032827F, -0.22353478F, -0.15854368F,
          0.061468445F, -0.1620997F, 0.037050907F, 0.110135294F, -0.017494919F,
          -0.027749779F, -0.02522258F, -0.116995655F, -0.13198574F, 0.008718966F,
          -0.03283505F, -0.0559094F, 0.03931107F, -0.01538277F, 0.05604241F,
          0.013516814F, 0.05586585F, 0.15528786F, 0.028169755F, 0.01754793F,
          0.11119316F, 0.016549692F, -0.018817255F, 0.020008745F, 0.011307739F,
          -0.15055904F, -0.16900979F, -0.0021866858F, -0.10158818F, -0.10545708F,
          -0.0039517228F, 0.097222954F, 0.08293044F, 0.04351639F, 0.19325933F,
          0.2040959F, -0.00017804588F, 0.05740826F, 0.050951306F, -0.12750898F,
          -0.25787133F, -0.15130138F, 0.0005504822F, -0.1716734F, -0.06547129F,
          0.030375352F, -0.009721353F, -0.0209023F, 0.029270876F, 0.14957203F,
          0.12653892F, 0.08614578F, 0.094183F, -0.03349811F, -0.03272514F,
          -0.0050203847F, -0.19611286F, -0.18156886F, -0.2369616F, -0.09719414F,
          0.012907894F, -0.19109727F, -0.10163071F, 0.053290747F, -0.02522685F,
          -0.037991364F, 0.039602675F, -0.045664813F, -0.021636056F,
          0.027628304F, -0.10059067F, -0.08868849F, 0.05479204F, -0.11195692F,
          -0.04574034F, 0.046613246F, 0.038134195F, -0.0023101373F, 0.009619325F,
          -0.07842828F, -0.24737608F, -0.017002907F, -0.09560898F, -0.13927814F,
          0.1369678F, 0.016453173F, 0.0110436445F, 0.013294692F, -0.002796667F,
          -0.00058159773F, 0.03439797F, -0.016898563F, -0.037141275F,
          0.008064403F, 0.04427203F, 0.06650086F, -0.012571413F, 0.06413802F,
          -0.019845687F, -0.08306496F, -0.07668555F, -0.066487685F,
          -0.029684989F, 0.007778853F, -0.010306545F, -0.011689159F,
          -0.091458455F, 0.10652752F, 0.099688515F, -0.0276767F, 0.034258224F,
          0.016476817F, 0.02326594F, 0.059020974F, 0.045488454F, 0.067560606F,
          -0.03369077F, -0.072403036F, -0.07163264F, -0.08173384F, 0.05789721F,
          0.0031089108F, 0.03238677F, 0.019280538F, 0.0275774F, -0.06479191F,
          -0.046396654F, 0.07503341F, -0.09912972F, -0.070569F, 0.0802977F,
          0.08866902F, 0.14680146F, 0.15792255F, 0.07038177F, 0.15817444F,
          0.12962984F, -0.47318336F, 0.14951442F, 0.044573747F, -0.0011175481F,
          -0.023127893F, -0.015034913F, -0.13371404F, -0.083418824F,
          0.0024061364F, -0.07913341F, 0.018120414F, -0.017578516F,
          -0.027280236F, -0.12637097F, 0.007732497F, 0.0032744547F,
          -0.025521567F, 0.008112807F, 0.17932361F, 0.059463914F, -0.012086402F,
          -0.072411016F, -0.16404213F, -0.097455226F, 0.055562276F,
          -0.063586295F, -0.1128844F, 0.049005285F, 0.15259276F, 0.00031137056F,
          0.010875917F, -0.02009829F, 0.0329952F, 0.04795364F, -0.07236791F,
          -0.036051616F, -0.023056896F, 0.059200905F, 0.0023790062F,
          -0.13021056F, -0.101956755F, 0.0072621563F, -0.08030048F, 0.021831911F,
          -0.07095595F, 0.070615605F, 0.037751436F, -0.04531093F, -0.1930106F,
          -0.057083204F, 0.0028177616F, -0.4255392F, -0.3294236F, 0.112660095F,
          -0.15780865F, -0.37245294F, 0.00023967908F, 0.021771323F,
          -0.0016471785F, -0.0452975F, 0.07337558F, 0.0024841728F, 0.0021478522F,
          0.019484602F, -0.030826977F, -0.03888894F, 0.013908601F, -0.024044568F,
          -0.0457211F, 0.16200331F, 0.015396508F, -0.009082964F, -0.30050614F,
          -0.1508506F, -0.013472055F, -0.07987588F, 0.043923493F, -0.07970032F,
          -0.04909551F, 0.07485765F, -0.00016813695F, -0.20441566F,
          -0.019633835F, 0.02741567F, -0.033247553F, -0.024529155F,
          0.0052690404F, 0.03501677F, 0.036687694F, 0.013291413F, -0.10212093F,
          0.045214303F, 0.014823383F, -0.010698082F, 0.041002505F, -0.047530074F,
          0.17797522F, 0.016191516F, -0.019277759F, -0.028693076F, -0.046412483F,
          -0.037952792F, -0.18841757F, -0.04277075F, -0.10870802F, -0.21068309F,
          -0.09454557F, -0.027424674F, 1.5668421F, -0.10958296F, -0.033203553F,
          -0.04826149F, -0.02868481F, 0.046345916F, -0.012949335F, -0.09926418F,
          0.06910732F, -0.06747149F, -0.13494705F, 0.067267396F, -0.01669015F,
          0.05602403F, 0.0052760355F, -0.23555107F, -0.016195925F, 0.082525305F,
          -0.03630378F, 0.15453334F, -0.0067955395F, -0.034303304F,
          -0.027403846F, -0.06279305F, -0.12011927F, 0.09494659F, 0.044894103F,
          -0.022915062F, -0.025030872F, 0.023401033F, 0.04582888F,
          -0.0010958137F, 0.0140105495F, 0.046296667F, -0.04237765F,
          -0.035747215F, -0.0060449927F, -0.06501525F, 0.019756531F,
          -0.028020112F, 0.02491671F, 0.029208673F, -0.0069511784F, 0.04091644F,
          -0.13962506F, -0.06544734F, -0.020098872F, 0.012789798F, 0.046896674F,
          -0.0005576804F, 0.016161323F, 0.03803069F, -0.0288831F, -0.0785932F,
          0.012361807F, 0.006796103F, -0.015034009F, 0.01801104F, 0.05994352F,
          -0.044683944F, -0.17660952F, 0.08404953F, -0.066241875F, 0.20147097F,
          -0.030268138F, 0.026845692F, 0.024917006F, 0.014984237F, 6.0167276E-5F,
          0.11774326F, -0.13830283F, -0.12669559F, -0.09182971F, -0.024089282F,
          -0.011445015F, -0.0063735843F, 0.035954926F, -0.02186312F,
          -0.014865607F, -0.004056596F, 0.00030880273F, -0.08679351F,
          -0.10101462F, -0.07375452F, -0.03480325F, -0.031624366F, -0.14802235F,
          0.0105869705F, -0.0942351F, 0.035257474F, -0.006863515F, -0.051888835F,
          -0.042033438F, -0.026774025F, -0.056425184F, -0.010569832F,
          -0.04258873F, -0.021196652F, 0.047947623F, -0.06410319F, 0.068481624F,
          0.0056723636F, -0.0063676275F, 0.021367611F, -0.014624915F,
          -0.043334972F, -0.011040756F, -0.06468064F, 0.010033539F,
          -0.018604979F, -0.018269721F, -0.09419982F, -0.13287626F, 0.06257066F,
          0.010326712F, -0.11409613F, -0.011905347F, 0.090746626F, 0.06946564F,
          -0.062344063F, -0.08473892F, -0.047068503F, -0.028006922F,
          -0.04595132F, 0.012018459F, -0.020224988F, -0.0039175926F,
          -0.029628996F, -0.050066024F, -0.032645203F, -0.06692162F,
          -0.023028374F, 0.007484137F, -0.004443066F, -0.010108144F,
          -0.0031518894F, -0.07166871F, 0.024595726F, -0.033545546F,
          -0.020958776F, -0.042601474F, -0.02847577F, 0.02530493F, -0.023345314F,
          0.0069360603F, 0.020702507F, -0.025929715F, 0.026739059F, 0.03812324F,
          0.037445154F, 0.024545975F, -0.012833553F, 0.029692207F, 0.0056531895F,
          0.020651119F, 0.00020072391F, -0.0025526462F, -0.00087968266F,
          -0.009659784F, 0.012348476F, 0.049872074F, -0.019522736F, 0.058850504F,
          0.0340948F, -0.08986671F, 0.007076219F, -0.0051603224F, -0.023503462F,
          0.038888387F, 0.06474258F, -0.07501628F, -0.018095914F, -0.09889009F,
          0.06595177F, 0.015554916F, -0.031681795F, 0.009333002F, -0.023852453F,
          -0.068742186F, 0.012584654F, 0.012415828F, -0.007183932F, 0.03626305F,
          0.020948684F, 0.05523235F, -0.0071179224F, -0.138195F, -0.13050066F,
          0.016242482F, -0.04536102F, -0.007552349F, -0.012582219F,
          -0.037449345F, -0.004163622F, 0.07186303F, -0.0035791076F,
          -0.014728917F, -0.0025505428F, -0.011958117F, 0.038101625F,
          -0.09141865F, -0.18790211F, -0.14918093F, -0.15551117F, -0.33985624F,
          -0.23404971F, -0.11132919F, -0.0837873F, -0.007087765F, 0.050400775F,
          -0.003687862F, -0.016955113F, -0.0045545157F, -0.00502478F,
          0.00890879F, -0.034789525F, -0.0046802005F, -0.010130399F,
          0.019503588F, 0.0053032995F, -0.001410798F, 0.02900883F, -0.016089791F,
          -0.013813351F, -0.030649671F, -0.023077626F, 0.005392575F,
          -0.0035118756F, 0.030331025F, 0.038316764F, 0.04136846F, 0.07458828F,
          0.08106709F, 0.048288286F, 0.049344085F, 0.047515962F, -0.049576398F,
          -0.00091417565F, 0.010690009F, 0.0026172062F, -0.022907913F,
          -0.034878533F, 0.03247963F, -0.0056747086F, -0.0105254F, 0.029935505F,
          0.0038173457F, -0.06195901F, 0.008714985F, 0.017374787F, 0.008165041F,
          0.009373851F, -0.014871745F, 0.017233372F, -0.06438169F, -0.03235349F,
          -0.022626793F, 0.00094831333F, -0.032104813F, -0.03298404F,
          -0.018402629F, -0.043048535F, 0.015065905F, -0.009183583F,
          0.0016574787F, -0.00024157656F, 0.018207422F, 0.035964888F,
          0.04119677F, -0.035821635F, -0.012966553F, -0.017647952F,
          -0.006620484F, -0.033105325F, -0.027505616F, -0.029234743F,
          -0.03327857F, -0.051002797F, 0.0103445705F, -0.03874036F,
          -0.029587973F, -0.009749194F, 0.0031971333F, 0.027783692F,
          0.023128944F, -0.0022626724F, 0.031019261F, -0.002870257F,
          0.011557198F, 0.02433074F, 0.02901526F, 0.02527336F, 0.036485717F,
          -0.00045893187F, 0.0032554264F, 0.024861043F, 0.00028734945F,
          0.03551366F, 0.027985243F, 0.027818773F, 0.006066972F, -0.04507086F,
          0.032683052F, 0.0024411913F, -0.056661684F, -0.0571201F, -0.028583718F,
          -0.022620488F, 0.013989771F, 0.054015193F, 0.020639367F, 0.051974934F,
          0.057495452F, -0.005466803F, 0.007298595F, 0.009956892F, 0.0032384587F,
          -0.052037787F, -0.060991563F, 0.019597802F, -0.075357854F,
          -0.117626704F, -0.02832449F, -0.034560017F, -0.03329556F, 0.032793142F,
          -0.005636473F, 0.016906405F, -0.024182599F, -0.027269218F,
          -0.035183877F, -0.037643373F, 0.015601146F, -0.006206202F,
          -0.012688478F, 0.0036815575F, -0.00089966535F, -0.012876963F,
          -0.040714707F, 0.018886818F, 0.015428658F, -0.026181862F, 0.027460149F,
          0.0039251456F, -0.009257415F, 0.002224373F, -0.018124446F, 0.0646964F,
          0.008332336F, -0.08570647F, 0.014633658F, -0.005409577F, -0.022655949F,
          0.048063226F, 0.0133349085F, 0.011300506F, 0.008454196F, 0.06408653F,
          0.007247655F, 0.005195561F, 0.015319765F, -0.019725544F, -0.038023334F,
          -0.012433566F, 0.0016588048F, 0.014542352F, -0.020684838F,
          -0.022717064F, -0.001304276F, -0.038324144F, 0.010722971F,
          -0.059932444F, -0.037643038F, 0.0026038708F, -0.0026516274F,
          0.009654652F, -0.004654861F, 0.015976997F, 0.021149721F, -0.020157838F,
          -0.014991721F, -0.032784395F, -0.011970529F, 0.018208181F,
          -0.002664852F, -0.02073655F, -0.014375882F, -0.025170056F,
          -0.026929555F, -0.020904599F, 0.029560424F, 0.025707383F, -0.02003341F,
          0.0076193647F, 0.031297985F, -0.0064783837F, -0.00059462583F,
          0.0016927766F, -0.018056469F, -0.052407082F, -0.007628381F,
          -0.06462355F, -0.0974455F, -0.036303516F, -0.03867199F, -0.011478252F,
          0.04822404F, 0.046921253F, 0.032887816F, 0.030794883F, -0.07219658F,
          -0.07626436F, -0.06316149F, 0.065394364F, 0.002126099F, 0.0055983476F,
          -0.04692752F, -0.033785466F, -0.008174385F, 0.050697826F, 0.071017645F,
          -0.050947696F, 0.1027992F, -0.010537901F, -0.07893504F, -0.09051303F,
          -0.11334296F, -0.015088819F, -0.13307822F, -0.16795652F, -0.15164039F,
          -0.11190863F, -0.15280807F, -0.07296166F, 0.034261383F, 0.11487118F,
          0.0054954877F, 0.008218959F, 0.046039585F, -0.021865748F,
          -0.008725932F, 0.043221015F, -0.019138932F, -0.023891527F,
          -0.12921387F, -0.09748798F, -0.122246176F, 0.044412993F, 0.08204263F,
          0.16990669F, 0.20843762F, -0.043367166F, -0.05391813F, -0.041967403F,
          -0.023419775F, -0.005927377F, 0.07665295F, 0.047577016F, 0.009310102F,
          0.0047365353F, 0.0063780593F, 0.1771285F, 0.1718301F, 0.020738605F,
          0.044379912F, -0.1628532F, -0.16738617F, -0.10096732F, 0.032367792F,
          0.015270555F, 0.0069083436F, -0.008631245F, -0.057071056F,
          0.008389288F, -0.06470268F, -0.05390038F, 0.025617527F, 0.054799795F,
          0.049324546F, 0.07267623F, -0.0031607621F, -0.11639912F, 0.022761727F,
          -0.08650605F, 0.0032064237F, -0.11875195F, -0.057775397F, 0.052089278F,
          -0.033484343F, -0.12654884F, -0.060366597F, -0.04241658F, 0.078586236F,
          0.14072411F, 0.02356804F, 0.042018812F, -0.023242239F, -0.032701835F,
          -0.038172506F, 0.00665109F, 0.034056555F, 0.121669605F, -0.042384647F,
          0.051234964F, -0.02063902F, -0.05588348F, -0.06629102F, -0.190103F,
          -0.05566899F, -0.1477446F, -0.13985269F, -0.05464616F, 0.06182891F,
          0.14570369F, 0.0011860467F, -0.06452554F, -0.043045234F, 0.004498779F,
          0.004784219F, 0.05782814F, 0.011025784F, -0.052393295F, -0.19509906F,
          -0.1405939F, -0.026705116F, -0.003249957F, -0.0077702096F,
          0.015988262F, -0.065344095F, -0.0007046658F, -0.0017770721F,
          -0.053319015F, 0.018185062F, -0.05841106F, -0.02047398F,
          -0.0015222635F, 0.07813148F, 0.058989923F, 0.06313269F, 0.035236362F,
          -0.02758484F, 0.041972376F, 0.032448303F, 0.0056419754F, 0.044533685F,
          0.012941958F, -0.0889008F, -0.018561801F, -0.048056267F, 0.042639006F,
          0.036395006F, -0.048868075F, -0.03209337F, 0.0474974F, -0.0875684F,
          -0.020442968F, -0.005459331F, 0.026953261F, 0.035346605F, -0.03161914F,
          -0.0038890636F, 0.0426802F, 0.048854403F, 0.09201002F, 0.03942955F,
          -0.031364758F, -0.104265116F, -0.10452447F, -0.06988605F, -0.01374944F,
          0.0053715324F, -0.002206539F, 0.05717478F, 0.10847885F, 0.012614745F,
          0.045415703F, -0.03566428F, -0.038160272F, -0.019218212F,
          0.0017700304F, 0.016400555F, 0.09781868F, 0.064376645F, 0.044381842F,
          -0.034298364F, -0.015703682F, -0.045220118F, -0.0653943F, 0.029689334F,
          0.12123947F, 0.2143493F, 0.31276983F, 0.10819717F, 0.0025380803F,
          -0.19708383F, -0.19817731F, -0.017917767F, 0.06608769F, 0.009755574F,
          -0.01973742F, -0.08402188F, -0.02310296F, 0.0574753F, 0.014512208F,
          0.04343319F, -0.015537163F, -0.03545575F, 0.002114289F, -0.040855788F,
          -0.021695592F, -0.020353362F, 0.08803771F, 0.034061864F, -0.016374862F,
          0.012317766F, -0.009312942F, -0.08544149F, -0.11167799F, -0.08398343F,
          -0.04484616F, -0.041857164F, -0.07248509F, -0.10348832F, -0.031046975F,
          0.046748497F, -0.021370038F, 0.058494058F, 0.15264869F, 0.03796777F,
          -0.030971214F, -0.024880406F, -0.011760345F, 0.0040496318F,
          -0.04333826F, -0.06472566F, -0.060075242F, -0.14126779F, -0.009593791F,
          -0.017568754F, -0.0059515033F, 0.0066178977F, -0.00028137522F,
          0.032033972F, 0.044235714F, 0.09257656F, 0.16176431F, 0.012403202F,
          0.030853083F, 0.0066403165F, -0.04742876F, 0.012612573F,
          -0.0068003074F, -0.014851824F, 0.016201517F, -0.026456365F,
          0.035287406F, -0.0026425587F, 0.019616408F, 0.059307285F, -0.00142458F,
          0.085809015F, 0.05387273F, 0.019233858F, 0.07670156F, 0.034949612F,
          0.015593758F, -0.03980651F, 0.006033849F, 0.10835283F, 0.074304216F,
          0.022677626F, -0.08713014F, -0.3589909F, -0.15299731F, -0.02153706F,
          0.11751403F, 0.060784627F, -0.07229156F, -0.031730894F, -0.036179863F,
          -0.044346437F, -0.028966103F, -0.0071358653F, -0.02798937F,
          -0.071943864F, -0.055545032F, -0.05313574F, 0.0055028535F,
          0.0058596595F, 0.02451039F, 0.04816686F, -0.024600897F, -0.028529955F,
          -0.043932628F, -0.08316473F, -0.0123350145F, 0.05224605F,
          0.0065199137F, -0.09620897F, -0.07916684F, -0.07487168F, -0.04679111F,
          -0.064257056F, 0.058898922F, 0.06460362F, -0.0014782879F,
          -0.021085927F, -0.025015684F, -0.032785535F, -0.022266524F,
          -0.008292641F, -0.001573374F, 0.03625675F, -0.09104307F, 0.06184843F,
          0.00028613862F, -0.11490994F, -0.10962427F, -0.0026360473F,
          0.005362023F, 0.0034565337F, -0.014573956F, 0.030703321F,
          0.0065461877F, -0.02844271F, 0.093358114F, 0.06783177F, -0.024226645F,
          0.0025266076F, -0.04242562F, 0.0021557848F, -0.056035582F,
          0.060768902F, 0.13311483F, -0.0063094157F, 0.025290096F, 0.08626145F,
          0.0045539676F, -0.11825195F, -0.09840876F, 0.14190772F, 0.15942104F,
          0.06662432F, -0.035961278F, -0.029702276F, -0.057256255F, -0.08096563F,
          -0.1253878F, -0.11610182F, -0.32592642F, -0.30541822F, -0.24648613F,
          0.07205354F, 0.07719393F, 0.12899446F, 0.24597253F, 0.3013688F,
          0.30909586F, 0.10100902F, 0.12731588F, 0.08488918F, -0.019341553F,
          -0.04218027F, -0.026644217F, -0.055736594F, -0.12648176F, -0.07341113F,
          -0.0214033F, 0.08066116F, 0.08640487F, -0.027019914F, -0.06348297F,
          -0.13884638F, 0.039985135F, -0.18125373F, -0.204781F, 0.09233184F,
          0.11680937F, 0.06693287F, -0.025491718F, 0.014876619F, -0.0006647994F,
          -0.08778549F, -0.12660964F, -0.11638477F, -0.019540312F, 0.04464408F,
          0.11102727F, -0.04889998F, 0.094180234F, 0.11037644F, -0.09854228F,
          -0.05745446F, 0.003292925F, -0.029952053F, 0.11685907F, 0.08605582F,
          -0.051898696F, 0.048813898F, -0.04105493F, 0.045201343F, -0.07617654F,
          -0.1257046F, 0.1442086F, 0.082421176F, 0.12634511F, -0.09425005F,
          -0.22852035F, -0.1688356F, -0.23547184F, -0.29701608F, -0.29346487F,
          -0.030355811F, -0.0641864F, -0.03777558F, -0.010190577F, -0.05337339F,
          -0.07066761F, -0.008187321F, -0.02502365F, -0.016142717F, -0.06462876F,
          -0.11169862F, -0.03861763F, 0.018110594F, -0.04187F, 0.09968025F,
          0.03833245F, 0.028235262F, 0.04206159F, -0.027491765F, 0.00081219594F,
          -0.047561612F, -0.025936715F, -0.22285336F, -0.13775234F, 0.028018609F,
          -0.031066312F, -0.020466466F, 0.07189634F, 0.20943171F, 0.1407549F,
          -0.021142557F, -0.0931131F, -0.07197144F, 0.010504421F, -0.15959471F,
          -0.07694787F, -0.15671399F, -0.51220894F, -0.15886192F, 0.039918676F,
          0.047738153F, 0.15892951F, 0.09800719F, 0.24369735F, 0.124432445F,
          0.07243765F, 0.11120433F, 0.0556862F, 0.037802357F, 0.09868568F,
          -0.00033352667F, -0.093332335F, -0.19268255F, -0.037083738F,
          -0.04252047F, -0.120431475F, -0.010871484F, 0.019902885F,
          -0.0040907455F, 0.06801646F, 0.028705208F, 0.06975102F, -0.0022374461F,
          -0.043215588F, 0.116796285F, 0.033338685F, -0.09498571F, -0.029614996F,
          -0.0074639837F, -0.03197432F, -0.06605716F, -0.07156424F, 0.011878499F,
          -0.017823528F, -0.061492708F, -0.01846195F, -0.0026369258F,
          -0.024075484F, 0.032092236F, 0.0177125F, 0.0075107473F, 0.022346301F,
          -0.030826194F, -0.01307929F, -0.028716857F, -0.045864172F,
          -0.034103323F, 0.043828066F, 0.04207586F, 0.028928595F, 0.006148079F,
          -0.069036454F, 0.027969873F, 0.10145802F, 0.058310386F, 0.09475554F,
          0.019870006F, 0.04723086F, -0.017996684F, -0.062420703F, -0.004099627F,
          -0.0044957567F, -0.03111245F, 0.060410336F, 0.039307583F, 0.046684314F,
          0.023945969F, -0.022096004F, -0.0032846313F, -0.08705933F,
          -0.04371635F, 0.05546256F, -0.009573602F, -0.0038301288F, 0.040926944F,
          0.049125414F, -0.001388755F, -0.06358711F, -0.041251507F,
          -0.050762717F, -0.056563407F, 0.0010411916F, 0.10018581F, 0.042205624F,
          0.055430725F, 0.08711229F, 0.065425806F, 0.10291182F, 0.01201433F,
          0.11176661F, 0.040697135F, -0.081256114F, 0.025588717F, -0.042644847F,
          -0.08276562F, -0.021346632F, -0.05278954F, -0.016072009F,
          -0.0033996925F, 0.0015450917F, -0.034736663F, 0.005917543F,
          -0.020700846F, 0.008524933F, 0.092872985F, 0.26750794F, 0.09161457F,
          -0.12081518F, -0.24782255F, -0.1562911F, -0.08786705F, -0.21881872F,
          -0.040906604F, 0.0043896264F, -0.014754336F, 0.034758706F,
          -0.012833641F, 0.009844628F, -0.035186533F, -0.01259608F, -0.00890846F,
          0.003901907F, 0.09041615F, 0.16166429F, 0.08832333F, -0.0639471F,
          0.006131785F, -0.024748165F, -0.09330266F, -0.18660633F, -0.1370531F,
          -0.0136908535F, -0.064613216F, -0.034026355F, 0.036020704F,
          0.10228393F, 0.025691481F, -0.03761363F, 0.07700735F, 0.036713514F,
          -0.047457024F, -0.0051769577F, -0.014991139F, 0.045571107F,
          0.08530173F, -0.02494244F, -0.05168747F, 0.027595019F, -0.025609555F,
          0.117322296F, -0.067240655F, -0.0047146194F, 0.037552968F,
          -0.063488305F, -0.0035443867F, -0.07026324F, -0.010585456F,
          -0.027458433F, 0.028322745F, 0.07394123F, 0.041228026F, 0.029117767F,
          0.06094984F, -0.006265833F, 0.098168515F, 0.016266184F, 0.0011514372F,
          -0.052093882F, -0.10949269F, -0.0779178F, 0.12230288F, -0.010090811F,
          -0.0072194184F, -0.051762674F, 0.021512665F, 0.015672157F,
          -0.019825414F, -0.01852312F, -0.0759505F, 0.020274682F, 0.10914405F,
          0.0032219267F, -3.8250117E-5F, -0.013052379F, -0.06916108F,
          -0.12655495F, -0.06563083F, 0.062159926F, -0.16276756F, -0.01815354F,
          0.13663438F, -0.033688482F, 0.021416482F, 0.12278593F, 0.010921391F,
          -0.00020049667F, 0.06904372F, -0.019364184F, -0.06799129F,
          -0.027276076F, 0.038477536F, -0.032061532F, 0.029653652F,
          -0.025953325F, -0.07895434F, 0.009424314F, 0.13899153F, 0.21370845F,
          0.10009749F, 0.08837438F, -0.044459797F, 0.008199682F, 0.037256006F,
          0.015363414F, -0.08156697F, 0.07635818F, 0.068924606F, -0.071955495F,
          -0.054984335F, 0.018739529F, -0.07143653F, 0.060182482F, -0.050248865F,
          -0.0093766805F, -0.023113502F, -0.015132123F, -0.0080876F,
          -0.019807043F, 0.041984595F, 0.02394602F, 0.0525739F, -0.06329337F,
          -0.03221089F, 0.34619418F, 0.08463092F, 0.0048506455F, 0.10938834F,
          0.01989434F, -0.09659943F, -0.1558454F, -0.20780659F, -0.22876179F,
          -0.07511166F, -0.2052821F, -0.1832745F, -0.07206225F, -0.19049212F,
          -0.21294013F, -0.12919101F, -0.09778702F, -0.07418986F, -0.06907124F,
          -0.08863667F, -0.12021215F, -0.0011834342F, -0.06098335F, -0.06097721F,
          0.122301646F, 0.12471798F, 0.028480126F, 0.1452213F, 0.10481032F,
          -0.0077759386F, 0.1288509F, 0.010748173F, -0.020633178F, -0.10934285F,
          -0.012853305F, 0.094026476F, -0.042246185F, -0.030395648F, 0.03875771F,
          -0.04336068F, -0.06722241F, -0.02672969F, -0.093596056F, -0.081565104F,
          -0.09051176F, 0.04304247F, 0.09661741F, 0.07582639F, -0.11387291F,
          -0.053873755F, -0.062635124F, 0.15924862F, 0.13828084F, 0.013618284F,
          -0.081927605F, -0.022842139F, -0.15293911F, 0.013401861F, 0.02089764F,
          -0.09661323F, -0.048324283F, -0.07465256F, -0.03428941F, 0.103905536F,
          -0.086146854F, -0.009726157F, 0.07258162F, -0.022650644F, 0.012677746F,
          0.07767448F, 0.034507286F, 0.0049140574F, 0.09921379F, 0.0830837F,
          0.027315471F, 0.07313089F, 0.016792292F, 0.053999215F, 0.045042146F,
          0.007528717F, -0.049892727F, 0.17269574F, 0.16477029F, -0.0332614F,
          0.029045016F, 0.16764373F, 0.016496947F, -0.063337825F, 0.008550031F,
          -0.051594265F, 0.12222111F, 0.21650116F, 0.06841532F, 0.051805194F,
          0.03423969F, 0.001968395F, 0.008788223F, 0.012459972F, -0.039190087F,
          0.07069414F, 0.103924714F, -0.039959967F, -0.0038302972F, 0.063051134F,
          -0.032599375F, 0.025037352F, 0.002642601F, -0.0483493F, -0.06619182F,
          -0.021573665F, -0.118262544F, 0.036064945F, 0.04048884F, -0.11675855F,
          -0.030762395F, -0.1042249F, -0.011216835F, -0.022289833F,
          -0.033946022F, -0.061383225F, -0.014170468F, 0.026781272F,
          -0.014075997F, 0.047797117F, 0.035189502F, 0.026485717F, 0.1421807F,
          0.041655228F, 0.048797976F, 0.13541318F, 0.0051063546F, -0.022029843F,
          0.16462556F, 0.070859924F, 0.06396979F, 0.16895983F, 0.076166965F,
          -0.021476978F, 0.049492534F, 0.102637395F, 0.008502149F, -0.080434956F,
          -0.106413215F, 0.015543084F, 0.015950406F, -0.12261378F, -0.03593424F,
          0.023496138F, -0.10397203F, -0.058218233F, 0.061602194F, -0.004131442F,
          -0.088636994F, 0.0355597F, 0.014890796F, -0.06678635F, 0.028124087F,
          0.026805773F, -0.01337845F, -0.24097739F, -0.18937854F, -0.09337724F,
          -0.10174205F, 0.0047787707F, 0.08291126F, -0.1487979F, -0.14544241F,
          -0.049735688F, 0.047912065F, 0.09426079F, -0.0058585303F, 0.060515262F,
          0.002678359F, -0.028222676F, 0.04900384F, -0.008469351F, -0.043982442F,
          0.07031536F, -0.1173262F, -0.07703069F, 0.10958634F, 0.0087942295F,
          -0.108809695F, -0.055492163F, -0.019610826F, -0.067307666F,
          0.005901315F, 0.022869708F, -0.0389124F, 0.06949121F, -0.0137404725F,
          -0.12101073F, 0.07334173F, -0.08588508F, -0.13079755F, -0.1823574F,
          -0.042731162F, 0.14233793F, -0.37781742F, 0.0070659495F, 0.361347F,
          -0.385748F, -0.017933303F, 0.46053082F, 0.00269833F, -0.03604222F,
          0.020222368F, -0.019359471F, -0.021865927F, -0.01821033F,
          -0.028185455F, -0.013897908F, 0.014401667F, 0.017412717F,
          0.0021409816F, -0.024015794F, 0.0319524F, -0.004124758F,
          -0.00027270714F, 0.0053574117F, 0.047374815F, -0.009308085F,
          -0.046438985F, 0.0044933185F, 0.010900369F, -0.0039405655F,
          0.0069768503F, 0.017939826F, 0.0021716235F, -0.025778143F,
          -0.010046345F, -0.2948254F, -0.03377729F, 0.1542754F, -0.40324748F,
          0.0028601948F, 0.3538153F, -0.39121667F, -0.07794739F, 0.54004675F,
          0.0052474863F, 0.01404795F, -0.011802397F, 0.022470335F,
          -0.0057077724F, -0.06229384F, 0.046536155F, 0.004406747F,
          -0.042919286F, 0.10512354F, 0.04472308F, 0.019767407F, 0.06445841F,
          0.018616902F, -0.119966164F, 0.19644514F, 0.0730725F, -0.33612186F,
          0.006991132F, -0.019699406F, -0.012762862F, -0.013494366F,
          -0.04815648F, -0.024809036F, 0.006797969F, -0.079703934F, -0.05174971F,
          0.014205939F, -0.07573216F, -0.12743652F, 0.2632344F, 0.08704362F,
          -0.31956092F, 0.30122298F, 0.07985241F, -0.35459375F, -0.009696261F,
          -0.010588519F, -0.005331905F, 0.01035028F, 0.014690277F,
          -0.0063741524F, 0.024512287F, 0.016932815F, -0.035358645F,
          -0.017391812F, -0.0001499851F, -0.011507836F, -0.0047420324F,
          0.009433864F, 0.012789492F, -0.053804655F, 0.019264178F, 0.10916178F,
          -0.025824215F, -0.032984164F, -0.014858284F, -0.06377649F,
          -0.047162108F, -0.016178567F, -0.06430962F, 0.00325094F, 0.002887428F,
          -0.0093403375F, 0.004345512F, 0.013216953F, 0.011181695F, 0.01731709F,
          -0.0064162835F, -0.010048398F, 0.0158806F, 0.008850171F, -0.029154789F,
          -0.00071121135F, -0.012099328F, -0.015124944F, 0.012935236F,
          0.005405801F, 0.023615709F, 0.0031982074F, -0.023918593F, 0.010272733F,
          -0.0010898114F, 0.028136022F, -0.009865138F, 0.009435322F, 0.01597122F,
          0.008306252F, 0.011884657F, 0.018403955F, -0.019336762F, 0.008999806F,
          0.00915496F, 0.003360795F, 0.0069301347F, 0.014593236F, 0.0055482495F,
          0.018020479F, 0.000872663F, -0.010557907F, -0.001323036F,
          -0.005469404F, -0.02018408F, -0.01162877F, -0.017512007F, 0.003081277F,
          -0.024861012F, -0.032129183F, 0.009168429F, -0.01763029F,
          -0.0013554796F, 0.012166219F, -0.0260556F, -0.015218118F, 0.01103511F,
          0.0031045354F, -0.004642484F, 0.01440568F, 0.03239972F, 0.038679868F,
          -0.008361212F, 0.016875543F, 0.026043776F, 0.023832886F, 0.051915206F,
          0.02563586F, -0.033517353F, -0.025798827F, 0.015285392F, 0.01297634F,
          0.09862789F, 0.00998943F, 0.13461508F, -0.010437938F, 0.007051791F,
          -0.014331143F, -0.0046111485F, -0.005261342F, 0.032272805F,
          -0.062157232F, -0.016501244F, 0.02424184F, -0.059288315F,
          -0.00068789005F, -0.020195132F, -0.047739033F, 0.035802804F,
          0.019009925F, -0.111523F, 0.004681656F, 0.019264849F, -0.10821393F,
          0.019945027F, -0.008820357F, -0.003969114F, -0.004705837F, -0.0099368F,
          0.009922164F, -0.014776421F, 0.0058499314F, -0.0013985813F,
          -0.037618697F, 0.0038238512F, 0.0048455815F, 0.022806698F,
          -0.0043975264F, 0.011766134F, 0.034768265F, 0.01556323F, 0.006974516F,
          0.012755184F, 0.017115012F, -0.015746549F, -0.04191039F, 0.0077091125F,
          0.0017651768F, -0.028800305F, -0.0096437335F, 0.00033184412F,
          0.006998156F, 0.0019290556F, -0.0110854255F, -0.0022728555F,
          0.04680624F, -0.00031274307F, -0.02169525F, -0.023203148F,
          -0.04019119F, 0.032388423F, -0.013108308F, 0.008357028F, -0.004788738F,
          -0.013558357F, 0.0068600066F, 0.00064064376F, -0.031013064F,
          -0.007705873F, 0.017287998F, -0.016464857F, -0.020106737F,
          -0.001105555F, -0.017455965F, -0.010548745F, -0.006732671F,
          -0.01891021F, -0.0043623997F, 0.009376155F, -0.010699989F,
          -0.017983217F, 0.0060533998F, 0.012208566F, -0.0033928095F,
          -0.013201111F, 0.002991187F, 0.012467694F, -0.011357946F, 0.00785361F,
          0.02624187F, 0.00613777F, -0.00977338F, -0.0047544F, 0.005783734F,
          -0.013727985F, 0.011873791F, 0.0041492213F, -0.088662036F,
          -0.04579366F, -0.024820609F, -0.18540093F, 0.092134595F, 0.016946329F,
          -0.07323022F, 0.11174778F, -0.07804001F, -0.053465854F, -0.043887608F,
          -0.042614087F, -0.041955378F, -0.07189596F, -0.044202324F,
          -0.066021144F, -0.12629578F, -0.09250314F, 0.23257755F, 0.43882442F,
          0.71781516F, -0.0050407536F, -0.04683837F, -0.0463828F, -0.2762606F,
          -0.5198604F, -0.46246237F, -0.0012918664F, -0.0021193265F,
          -0.019576134F, 0.014445076F, 0.029525897F, -0.0254393F, 0.008941329F,
          0.0030925982F, -0.03549733F, 0.01521681F, 0.034133215F, 0.027998125F,
          0.0024729308F, -0.019226361F, 0.007951009F, -0.02701291F, 0.014533059F,
          0.0018487655F, -0.005454553F, 0.0010539597F, 0.013840586F,
          0.0047349874F, 0.00076415745F, -0.026052391F, -0.0077976184F,
          -0.018745942F, -0.03928131F, -0.34531245F, -0.40048072F, -0.57757854F,
          -0.19896069F, -0.1042258F, -0.17854708F, 0.01577427F, 0.2911752F,
          1.2708408F, 0.025606344F, 0.049586847F, -0.020072615F, 0.048343986F,
          0.012822059F, -0.017152417F, 0.035541132F, 0.024589023F, -0.022094576F,
          0.2497296F, 0.31108838F, 0.31838334F, 0.13673586F, 0.083533056F,
          0.01103588F, 0.0141285425F, -0.29082772F, -0.94562864F, 0.019289326F,
          -0.00024101879F, -0.028676929F, 0.0029086494F, -0.008783829F,
          -0.015472889F, 0.045593496F, 0.007344344F, -0.009371037F, -0.19045998F,
          -0.05027022F, 0.0682146F, -0.23456818F, -0.06377993F, 0.21562861F,
          -0.20864601F, -0.07881633F, 0.38066235F, -0.0114312945F, 4.3621632E-5F,
          0.023143794F, 0.020449758F, 0.02469161F, 0.035486337F, 0.025614714F,
          0.019084485F, 0.0015558652F, 0.027524278F, 0.021005563F, 0.018254878F,
          -0.0035939717F, 0.03010951F, 0.0049059545F, 0.013770728F, 0.012704297F,
          -0.01263227F, -0.030337306F, -0.047873035F, -0.03202537F, 0.02185335F,
          -0.036868077F, -0.04837437F, 0.056119237F, 0.01021782F, -0.080983415F,
          0.025348581F, -0.005856245F, -0.0040373635F, -0.022025358F,
          0.0033929264F, 0.007364309F, 0.0008605704F, -0.01105137F,
          -0.0112623125F, 0.016153378F, -0.02318655F, 0.0075193383F,
          0.009600642F, 0.0061376095F, -0.030629065F, 0.010756939F, 0.019032171F,
          -0.0054753684F, -0.008032683F, -0.010872694F, -0.015832525F,
          -3.643382E-5F, 0.015803518F, -0.01266302F, -0.030816186F, 0.01825036F,
          0.037683908F, -0.018458402F, -0.009809098F, 0.015128277F, 0.013043025F,
          -0.029378533F, -0.011581761F, -0.0008380095F, 0.025686944F,
          -0.011002743F, 0.051004592F, 0.030302381F, -0.010426093F, 0.009803869F,
          0.0047851326F, 0.01222464F, 0.020256938F, -0.042360567F, -0.050238732F,
          0.008187635F, -0.022211576F, 0.005580492F, -0.0064676637F,
          0.013550479F, -0.018806709F, 0.011802081F, 0.0155645525F,
          -0.013185715F, -0.027805217F, -0.0058265734F, 0.0064057414F,
          0.003583744F, -0.009712871F, -0.016214188F, -0.011942628F,
          -0.015683888F, -0.01352557F, -0.011113733F, 0.029525928F, 0.07272881F,
          -0.04357124F, 0.023505628F, -0.026320087F, 0.024300123F, 0.013322788F,
          0.01864624F, -0.043537844F, 0.006798599F, -0.004989325F, 0.016740253F,
          -0.0038997426F, 0.00042849703F, -0.008177831F, -0.037728176F,
          0.005834129F, 0.074352324F, 0.23718792F, 0.4001451F, -0.09649501F,
          0.02885187F, 0.18476464F, -0.27451092F, -0.36358023F, -0.33126462F,
          -0.03290305F, -0.012399568F, -0.020646125F, -0.0032495535F,
          -0.037604574F, -0.059728514F, -0.01010614F, -0.070532314F,
          -0.032459423F, 0.0132225985F, -0.02214762F, -0.013976462F, 0.03678255F,
          -0.0018582465F, -0.031739086F, -0.0023516982F, 0.0014483159F,
          -0.011587342F, 0.013662576F, 0.014408958F, -0.0012987274F,
          -0.0073866975F, -0.050885588F, 0.0031384644F, -0.0074965153F,
          -0.01026021F, -0.016851436F, 0.005484227F, -0.0073786764F,
          0.0044594044F, -0.020141734F, -0.023103477F, -0.044772062F,
          0.027286153F, -0.02701494F, -0.01748101F, 0.01641903F, -0.024627743F,
          -0.01276708F, -0.009497611F, 0.0026315087F, 0.021434624F,
          0.00023120602F, 0.007328479F, 0.0058463397F, 0.03194882F,
          -0.009167109F, -0.031213986F, 0.021170655F, -0.019379385F,
          0.014399201F, -0.00046986982F, -0.012542932F, 0.011818707F,
          -0.014261983F, -0.045485493F, -0.038164865F, -0.017598381F,
          -0.022476882F, -0.03222276F, -0.018916542F, -0.019727618F,
          -0.054654997F, -0.018463628F, -0.022441037F, 0.006760082F,
          0.012904678F, -0.013912649F, -0.040594637F, -0.021442525F,
          -0.020104768F, -0.054541003F, -0.13332595F, -0.08587368F, 0.018408207F,
          -0.14673269F, 0.0060408507F, 0.15472606F, -0.1041475F, 0.07060001F,
          0.15628807F, 0.030412812F, -0.017170634F, -0.05976595F, -0.023251854F,
          -0.04075209F, 0.05584974F, 0.035287336F, 0.09069248F, 0.07095753F,
          -0.029646613F, 0.06082333F, 0.0030173175F, 0.06405353F, -0.08410757F,
          -0.04614815F, -0.00084534194F, 0.0016074639F, 0.016629018F,
          0.24470681F, 0.1830435F, -0.08438051F, 0.023889916F, -0.075334035F,
          -0.1050146F, -0.14629808F, -0.09119723F, 0.007615825F, -0.32347408F,
          -0.10509377F, 0.022340685F, -0.07314861F, 0.08132253F, 0.09553132F,
          0.1273184F, 0.09867406F, 0.012839368F, 0.06604308F, 0.13130562F,
          0.05605883F, 0.17287919F, 0.045313798F, -0.15952866F, -0.075962976F,
          -0.15818864F, -0.00949586F, -0.1291156F, 0.083162315F, 0.050213348F,
          0.08151849F, -0.10259026F, 0.006067393F, -0.0043944013F, 0.012484365F,
          -0.012068284F, 0.011638454F, -0.10896463F, 0.10333467F, -0.09577069F,
          0.13341057F, 0.11765506F, 0.13490829F, 0.041642107F, -0.036587104F,
          0.04288849F, -0.042556047F, -0.027488062F, -0.026550608F, 0.035042148F,
          0.007125026F, 0.022861991F, 0.041096494F, 0.0016440828F, -0.12790687F,
          -0.18341354F, 0.011908356F, 0.06973702F, 0.083241604F, 0.03263698F,
          0.06467389F, 0.037753113F, -0.0013416221F, -0.18887345F, 0.100599915F,
          0.067006946F, 0.15290822F, 0.023716168F, -0.1409804F, -0.004949925F,
          -0.06949786F, -0.00980113F, 0.11249351F, 0.15091024F, -0.0808402F,
          0.07982962F, -0.06524814F, -0.13266958F, -0.14870405F, -0.14469689F,
          -0.055062637F, 0.0046321317F, 0.07118904F, -0.020244258F, 0.021104984F,
          -0.09852151F, -0.05102051F, -0.07157215F, -0.03432087F, 0.025957724F,
          -0.17416044F, -0.13502555F, -0.053036626F, -0.0518156F, -0.07685896F,
          -0.02041763F, -0.04323388F, -0.021544758F, -0.046391506F, -0.25721443F,
          -0.14835322F, 0.08260423F, -0.14643772F, 0.039953485F, 0.12313971F,
          0.08261108F, 0.08950491F, 0.021475675F, 0.00118383F, 0.041864347F,
          0.010416005F, 0.0027980811F, 0.061364796F, -0.0346076F, 0.021253098F,
          -0.027110964F, -0.024418756F, -0.19056535F, -0.21719164F, 0.011112362F,
          -0.14516102F, -0.01586402F, 0.12640934F, 0.029839499F, 0.12569825F,
          0.0074356743F, -0.0870598F, -0.070660464F, 0.0006501864F,
          -0.041092716F, -0.013953109F, 0.032702975F, 0.08948976F, 0.053340774F,
          0.0064444747F, 0.04902505F, -0.014762717F, 0.022259615F, 0.0029394764F,
          0.015043948F, -0.029015223F, -0.017032724F, 0.019692913F, -0.03236174F,
          0.022582326F, 0.06977432F, -0.0075692483F, 0.012562812F, -0.03274586F,
          -0.049864087F, -0.020709498F, -0.03959353F, -0.009794327F,
          0.010416827F, -0.013105838F, 0.058654305F, 0.05774207F, 0.07881004F,
          -0.035877414F, -0.00856963F, -0.020881554F, -0.03394014F, 0.09065577F,
          0.105224796F, -0.05432225F, 0.08704244F, -0.05299082F, -0.084338024F,
          -0.065799095F, -0.056134753F, 0.015238666F, -0.11667502F, -0.08158696F,
          0.007899559F, -0.09984591F, -0.0016538997F, 0.057329614F, 0.06835914F,
          0.071346685F, -0.0014801013F, 0.009597363F, -0.008670889F,
          -0.009631334F, 0.0040737647F, 0.008362188F, -0.012751508F,
          0.019037576F, 0.009957317F, 0.0036245214F, -0.010500187F, 0.047790833F,
          0.049775098F, 0.041607715F, 0.0002539082F, -0.020300645F, 0.000805823F,
          -0.043195497F, 0.00930621F, 0.30362007F, 0.18817544F, -0.021442339F,
          0.16841401F, -0.070494495F, -0.1837249F, -0.16573158F, -0.1474265F,
          -0.021608952F, -0.15062277F, -0.10276741F, 0.027990252F, -0.05549687F,
          0.07190041F, 0.042831045F, 0.052835003F, 0.074153386F, -0.004931808F,
          0.24831195F, 0.23005576F, -0.081603564F, 0.15047151F, -0.029405776F,
          -0.13666062F, -0.17813735F, -0.15670894F, -0.025530599F, -0.2806446F,
          -0.16520566F, 0.055667184F, -0.040203255F, 0.07870807F, 0.09985166F,
          0.13618042F, 0.11956628F, 0.04808174F, -0.009787659F, -0.020159408F,
          -0.012843172F, -0.05705954F, -0.013590417F, -0.0022398627F,
          -0.035768103F, 0.018466251F, 0.021680515F, -0.08697213F, -0.14603467F,
          0.0037943833F, -0.085237496F, 0.04399395F, 0.06674359F, 0.11963035F,
          0.025665712F, 0.045298357F, -0.05527781F, 0.038115315F, 0.005945228F,
          -0.011817483F, -0.008161668F, -0.04374777F, -0.0063270237F,
          -0.04167454F, -0.008658316F, 0.25054473F, 0.12131112F, -0.07966348F,
          0.09696329F, -0.08880987F, -0.14360076F, -0.16687241F, -0.15276256F,
          -0.004642085F, -0.185728F, -0.070736386F, 0.005957943F, -0.12941286F,
          -0.15020494F, -0.092049316F, 0.0696998F, 0.06879847F, 0.048283704F,
          0.25659257F, 0.33034125F, 0.16380768F, 0.13719091F, -0.06510257F,
          -0.1991047F, -0.11453919F, -0.4144793F, -0.22792923F, 0.05710932F,
          0.074732505F, 0.022590498F, 0.08620937F, 0.065898806F, -0.014911687F,
          -0.009039251F, -0.007066661F, -0.055129424F, -0.01579764F,
          -0.004166933F, -0.012465599F, 0.03791835F, -0.018024854F, 0.026915442F,
          0.0035344046F, 0.0428135F, 0.034749977F, -0.0115731815F, 0.002662543F,
          -0.018527938F, -0.077951595F, -0.049958892F, -0.008306361F,
          -0.010626638F, -0.08608472F, -0.03485062F, 0.21812432F, 0.28810647F,
          0.17119119F, 0.109565996F, 0.011334756F, -0.16104177F, -0.040840555F,
          -0.4431572F, -0.33163148F, 0.055562764F, -0.015992647F, -0.021190176F,
          -0.105855875F, -0.0826918F, 0.08477702F, -0.028681241F, 0.08204025F,
          0.0974851F, 0.03507825F, -0.15746038F, -0.15223718F, -0.11883378F,
          -0.22921324F, -0.013665763F, 0.10664711F, 0.21962982F, 0.23112677F,
          -0.021171875F, -0.09781785F, 0.010074727F, -0.08261258F, 0.00886019F,
          -0.001289418F, 0.06480973F, 0.1223686F, 0.0061745658F, -0.2530608F,
          -0.4870792F, -0.14914863F, -0.2851311F, 0.08346163F, 0.20486915F,
          0.0453217F, 0.42400518F, 0.23955305F, 0.003906164F, -0.0028500808F,
          -0.023469072F, -0.019084696F, 0.00024341763F, 0.01372254F,
          0.017793752F, 0.011344508F, 0.00196802F, -0.009151311F, 0.037920304F,
          0.02713214F, 0.100488216F, 0.07312168F, 0.041444883F, -0.07848377F,
          -0.12007752F, -0.026047416F, 0.03796262F, -0.012275772F, -0.018416401F,
          0.029580953F, -0.12491464F, -0.06429111F, -0.11679452F, -0.1547901F,
          -0.051697463F, 0.030588755F, 0.030785415F, 0.00075577677F,
          -0.0148382895F, -0.040370826F, 0.008951471F, -0.014365636F,
          -0.0026804449F, -0.022085208F, 0.026646486F, -0.02068283F, 0.03249738F,
          -0.08093974F, -0.18677826F, -0.083542764F, 0.0015463245F,
          -0.008849298F, -0.0060697165F, -0.0055759493F, -0.0010591128F,
          0.006839736F, -0.028294966F, -0.014191448F, -0.008851967F,
          -0.081536524F, -0.025925715F, 1.9055513E-5F, 0.038592547F,
          -0.012867268F, 0.0018716234F, -0.027436692F, -0.050156035F,
          -0.022062283F, -0.026298456F, -0.016774617F, -0.005566602F,
          -0.005342668F, 0.0127867665F, -0.05568972F, 0.021299481F,
          -0.061540566F, -0.0082068825F, -0.014414811F, 0.009574122F,
          0.06089501F, -0.014667979F, -0.017890876F, 0.004649524F, -0.066046044F,
          -0.014779681F, -0.014568521F, -0.034680884F, 0.004000302F,
          0.017983742F, 0.00081751955F, -0.058053277F, -0.06342705F,
          -0.025742762F, -0.090648785F, -0.010793739F, -0.048952032F,
          -0.0055560474F, 0.028897487F, 0.06327473F, -0.15436177F, -0.08755669F,
          -0.14225529F, -0.03309423F, 0.0694083F, 0.032773387F, 0.18414283F,
          0.054143064F, -0.042526174F, 0.0035632995F, -0.017121246F,
          -0.05555105F, 0.011218712F, -0.016475515F, -0.013226953F,
          0.0027688611F, -0.0096609F, -0.054324996F, -0.02186329F,
          -0.0023366865F, -0.030277569F, -0.044958152F, -0.05595249F,
          0.08788644F, 0.0029488492F, 0.015895868F, 0.0038943018F, -0.003853118F,
          0.02658359F, -0.033532772F, 0.03364976F, 0.04599503F, -0.006968587F,
          0.019394867F, 0.029785385F, -0.077087F, -0.034140762F, 0.013154135F,
          0.004595652F, 0.020407354F, -0.0042319573F, -0.012294904F,
          0.00091703533F, 0.008095619F, 0.035606902F, 0.00467614F,
          0.00055809616F, 0.012587774F, 0.04154475F, 0.044515062F, -0.0336534F,
          0.007015481F, -0.0254068F, -0.03591822F, -0.0017336404F, 0.016291568F,
          0.05231575F, 0.02767616F, 0.013892706F, 0.07767166F, -0.03859929F,
          -0.03696458F, 0.026251204F, 0.0404383F, 0.024565222F, 0.035579946F,
          0.039326236F, 0.010504284F, 0.0070094615F, 0.019303069F,
          -0.00038490552F, 0.033972517F, -0.024949338F, -0.031139532F,
          0.031359363F, -0.0011494977F, -0.01931212F, -0.013874311F,
          -0.03474853F, -0.033583265F, 0.015580556F, 0.012385648F,
          -0.0026227478F, -0.065546624F, -0.10673379F, 0.001895885F,
          0.022141252F, 0.04218397F, 0.03988165F, -0.015199526F, -0.019273277F,
          0.017129594F, 0.009022147F, 0.026842544F, 0.029007109F, -0.017913472F,
          -0.036110308F, -0.0012476336F, 0.06839665F, -0.039048582F,
          -0.13501562F, -0.05153814F, -0.23185317F, -0.10456696F, -0.07701358F,
          0.1057631F, 0.11291148F, -0.009714177F, -0.034791015F, -0.017764498F,
          0.036992095F, 0.04251563F, 0.023742F, -0.030085692F, 0.007332805F,
          -0.02499045F, -0.042098183F, -0.014995571F, 0.013417669F,
          -0.035035364F, -0.03526201F, 0.03967967F, -0.008336944F, -0.014983452F,
          0.0059322193F, 0.035747238F, -0.0045661307F, -0.012780851F,
          0.03591458F, 0.049584836F, 0.008994324F, -0.028646719F, 0.059476897F,
          -0.003553237F, 0.08821191F, 0.09794614F, 0.025436567F, 0.0107450215F,
          0.091933966F, 0.089622565F, 0.046626803F, -0.054402277F, 0.06280921F,
          0.045541745F, -0.0021924735F, -0.074878335F, 0.07313373F, 0.06462791F,
          -0.024240691F, 0.021732451F, 0.01680853F, 0.043841F, -0.02467706F,
          -0.014161812F, 0.02387496F, -0.02340388F, -0.06862979F, 0.07718648F,
          -0.0073921103F, -0.007358332F, -0.05252026F, 0.03942884F, 0.11126226F,
          0.021898674F, -0.0025256702F, 0.10718741F, -0.07329288F, 0.017492937F,
          0.061382335F, 0.048897695F, -0.009679599F, -0.01450954F, 0.058914844F,
          -0.013821233F, -0.047949363F, 0.0026431864F, -0.011050662F,
          -0.046286296F, -0.002201528F, -0.036097F, 0.0031464545F, 0.02563208F,
          -0.0010544523F, -0.0392433F, -0.04866173F, 0.016240055F, -0.027918417F,
          -0.047386315F, -0.047309883F, -0.02618901F, 0.016657803F, 0.013749226F,
          -0.054487318F, 0.032208752F, -0.015573775F, -0.002616192F,
          0.024392828F, 0.018624326F, -0.017477376F, 0.004178438F, 0.092965186F,
          0.08287207F, -0.031049933F, 0.024886286F, 0.06208327F, 0.026790392F,
          0.10296941F, 0.04178006F, 0.016837068F, 0.058299236F, 0.0015729645F,
          0.03682792F, -0.04230698F, 0.018295769F, -0.028189294F, -0.01772325F,
          -0.065649495F, -0.042427585F, -0.027724195F, -0.09738658F,
          -0.053005043F, 0.0038106877F, -0.07307183F, -0.09310178F, -0.03187529F,
          0.008644247F, 0.03880132F, -0.120266035F, -0.10459129F, -0.012151706F,
          -0.05072746F, -0.1635943F, -0.041596707F, 0.07394041F, 0.049206592F,
          0.027737826F, 0.068500996F, 0.1061668F, 0.05268905F, 0.13794534F,
          0.07487672F, -0.035591945F, 0.01066305F, 0.009860384F, 0.03663309F,
          0.059534848F, -0.055713482F, -0.056296498F, -0.0048677158F,
          -0.026043752F, -0.07552972F, 0.048801698F, 0.05282267F, 0.014993336F,
          0.061514303F, 0.09027434F, -0.0060943495F, -0.013729067F,
          0.0011096309F, 0.014946994F, 0.03704694F, -0.056446034F, 0.014843274F,
          0.10720034F, 0.047159843F, 0.024959771F, 0.06962376F, 0.056576494F,
          -0.0059268125F, -0.016317794F, -0.02317955F, -0.010913369F,
          -0.016271342F, -0.036316346F, 0.08132233F, -0.021423666F,
          -0.010707378F, 0.024167879F, -0.012249928F, -0.004697512F,
          0.021425223F, -0.06502175F, 0.06056499F, -0.004224608F, 0.061434783F,
          0.030536324F, 0.036799125F, -0.016418066F, -0.054743636F, 0.04302002F,
          -0.0028794336F, 0.06745367F, -0.0012509013F, 0.056107942F,
          0.037826262F, 0.018406816F, 0.039610714F, 0.10068226F, 0.034484647F,
          -0.017772403F, 0.04810815F, 0.051692612F, 0.03696241F, -0.0049194945F,
          0.0239217F, -0.0063254917F, 0.007314022F, -0.0050038104F, 0.018744512F,
          0.029110655F, -0.0045144274F, -0.001962844F, -0.0042089615F,
          0.009680603F, 0.0026242856F, 0.018728852F, -0.054616213F,
          -0.0038574652F, -0.069784835F, -0.010394209F, -0.059632532F,
          -0.17216231F, -0.07051156F, -0.03019692F, -0.17392959F, -0.12695546F,
          -0.041701317F, -0.1217692F, 0.00970357F, 0.05904689F, 0.060564723F,
          0.027431183F, 0.013784841F, 0.037184265F, 0.026705965F, 0.010712907F,
          0.026402576F, 0.039261933F, 0.0064534876F, -0.0106359F, -0.014459091F,
          0.06443044F, -0.033454336F, -0.010947158F, 0.05355166F, 0.04342794F,
          0.050044157F, 0.10402321F, 0.14498636F, 0.055025574F, -0.0014215966F,
          0.017866792F, 0.0071811555F, 0.005290043F, -0.014061141F,
          0.0045950874F, 0.015381731F, -0.0067521874F, 0.020052178F,
          -0.15486544F, -0.25978205F, -0.14137124F, -0.17726621F, -0.2977706F,
          -0.1858018F, -0.120038606F, -0.18639459F, -0.21147813F, 0.03459313F,
          0.0740698F, 0.027730616F, -0.029027022F, 0.07338768F, -0.05504393F,
          -0.028290402F, -0.026291205F, 0.0106998645F, -0.20685251F,
          -0.28260714F, -0.15362957F, -0.26497957F, -0.37417263F, -0.24014625F,
          -0.18167284F, -0.26756862F, -0.1785268F, 0.031737536F, -0.0031895754F,
          0.00011782501F, 0.003901185F, -0.0025004135F, 0.017247522F,
          0.017727342F, 0.03993409F, 0.031641092F, -0.046633605F, 0.031898763F,
          -0.0052891644F, 0.08322659F, 0.06540275F, -0.13432461F, 0.031653836F,
          -0.09106861F, -0.03374587F, -0.122262776F, 0.016755981F, 0.080328666F,
          -0.025314586F, 0.03326665F, -0.04201069F, 0.009196833F, -0.03958814F,
          -0.026668193F, 0.10483768F, 0.15130085F, 0.11693995F, 0.18273379F,
          0.14747176F, 0.05354511F, 0.12163114F, -8.5382735E-6F, -0.023422968F,
          -0.014794545F, -0.082322404F, -0.028162122F, -0.13645813F,
          -0.23027907F, -0.032672163F, -0.07663789F, -0.039392136F, 0.03952118F,
          -0.018124605F, 0.016650967F, 0.090933785F, 0.020360276F, 0.12421627F,
          0.11607537F, 0.14243749F, 0.16363874F, -0.037070855F, -0.2157155F,
          -0.09012152F, 0.036256485F, -0.06377066F, 0.1263392F, 0.0091905845F,
          0.04072078F, 0.016026825F, -0.02802467F, 0.12535821F, 0.018486371F,
          -0.06399332F, 0.046426903F, -0.03678897F, -0.05554897F, 0.056604642F,
          -0.02288467F, 0.032196674F, -0.047171738F, 0.040333014F, 0.062224545F,
          -0.023068476F, -0.029254617F, -0.019538332F, 0.0027313458F,
          -0.017457657F, -0.008755035F, -0.13123503F, -0.07690994F,
          0.0017267324F, -0.11446308F, -0.039565004F, -0.026001172F,
          -0.017184071F, 0.1340052F, -0.027079301F, -0.14909634F, -0.107927315F,
          0.007226616F, -0.09608441F, 0.115757406F, 0.031139614F, 0.07379695F,
          0.046523403F, -0.14362478F, 0.019492885F, 0.15836027F, 0.062803514F,
          0.023507902F, 0.16341144F, -0.023992669F, 0.12998681F, 0.15006012F,
          0.020814039F, -0.063713655F, 0.0501357F, 0.03227109F, 0.00055398315F,
          0.04450534F, -0.04497784F, 0.014847253F, 0.062140685F, -0.070763454F,
          -0.09735538F, 0.03238403F, -0.03628771F, -0.06425295F, -0.06744391F,
          -0.13310125F, -0.052635625F, -0.060757205F, -0.10958429F,
          -0.024951838F, -0.047334332F, -0.032452337F, -0.11272271F,
          -0.13111432F, -0.009665147F, -0.026552F, 0.019980384F, 0.010331239F,
          0.026046576F, 0.049684502F, 0.029199144F, 0.028041437F, 0.013244534F,
          0.020616818F, -0.04377473F, 0.014710644F, -0.0019888387F, 0.05828344F,
          -0.10568348F, -0.05872555F, -0.12765998F, -0.2773017F, -0.07318597F,
          -0.04915934F, -0.074099876F, -0.03495614F, -0.022392835F,
          -0.018156001F, -0.03385158F, 0.030058505F, -0.06333795F, -0.081400245F,
          -0.063578196F, -0.048552513F, -0.10979777F, 0.078212574F, 0.03058447F,
          -0.0073023243F, -0.012104571F, 0.047234464F, 0.043916937F,
          -0.057467435F, 0.019537728F, 0.0711893F, -0.07375156F, -0.050558537F,
          -0.040960785F, -0.03814505F, 0.008949598F, -0.08760707F, -0.008527866F,
          -0.009383564F, -0.08096896F, -0.01407463F, 0.0794772F, 0.062048394F,
          0.05467729F, 0.060965646F, 0.046999983F, 0.022337668F, 0.041260663F,
          0.060243174F, -0.04937968F, 0.057631988F, -0.024032362F, 0.10357376F,
          0.09640354F, -0.06833743F, 0.05837129F, 0.017127445F, -0.14863172F,
          -0.032597F, -0.06664419F, -0.14977084F, -0.053647295F, -0.14772615F,
          -0.15491179F, -0.01813174F, -0.0077935425F, -0.062211152F, 0.02004288F,
          -0.002086386F, 9.4024734E-5F, 0.025686173F, -0.026104862F,
          -0.026991272F, 0.034784406F, 0.0037538984F, -0.01832316F, -0.07309152F,
          -0.06851653F, -0.033273235F, -0.046644334F, 0.009139484F, 0.045410346F,
          0.032766085F, 0.07745984F, -0.0020641813F, 0.030141598F, 0.047768854F,
          0.14699765F, 0.23858695F, 0.21907856F, 0.08893789F, 0.051338226F,
          -0.03971104F, 0.016978223F, -0.058081012F, -0.09485416F, -0.07367306F,
          -0.10436409F, -0.12987758F, -0.07551229F, -0.11622051F, -0.061888073F,
          -0.08265442F, 0.072447374F, 0.15704435F, 0.09168729F, 0.13985078F,
          0.2576161F, 0.17175092F, 0.22826958F, 0.13578144F, -0.009471527F,
          -0.035489824F, -0.0926773F, -0.0025632808F, -0.08902959F, -0.18276544F,
          -0.016700044F, -0.06798937F, -0.030373236F, 0.009650193F,
          -0.028647918F, 0.05426809F, -0.009039305F, -0.01124772F, -0.019119628F,
          -0.06163358F, -0.07047562F, -0.045756426F, -0.023606077F, 0.07471367F,
          0.020025974F, -0.033962555F, 0.054573495F, 0.054818243F, -0.012605024F,
          0.08454509F, 0.0055872393F, 0.0013768247F, -0.07201758F, -0.026899694F,
          0.048072506F, -0.035839986F, 0.0122732315F, -0.0055654743F,
          -0.06797958F, -0.0039385427F, 0.034194186F, -0.10250405F, 0.10751777F,
          0.031380076F, 0.10010911F, 0.2395912F, 0.045239024F, 0.035453744F,
          -0.008514466F, -0.13127138F, -0.022393558F, -0.06193671F,
          -0.030608715F, 0.014049424F, -0.03071054F, -0.05613151F, -0.04549434F,
          -0.027123004F, -0.002858347F, -0.03110064F, 0.03218284F, 0.05530865F,
          0.0045011635F, 0.0077263354F, -0.06918467F, 0.052443575F, -0.0373337F,
          -0.024606312F, -0.12905766F, -0.08927063F, 0.051481586F, 0.092893176F,
          0.11572703F, -0.06371821F, -0.02102687F, -0.052522942F, -0.01706004F,
          0.03156736F, 0.037582334F, -0.02549889F, -0.06076278F, -0.21958609F,
          -0.07026248F, -0.0046794578F, -0.052733567F, 0.05181857F, -0.01227215F,
          -0.0789013F, -0.065612525F, -0.037873894F, 0.045643497F, 0.0714977F,
          0.03360596F, -0.00030730502F, -0.03974787F, -0.037418734F,
          -0.026323693F, 0.022188298F, -0.009972683F, 0.038387466F, 0.01022422F,
          0.043389607F, -0.028905766F, -0.06313321F, 0.18055044F, 0.13900208F,
          -0.06432814F, 0.05646851F, -0.1377125F, -0.025027309F, -0.044579938F,
          0.117074266F, 0.105991565F, 0.024414405F, 0.0073676137F, 0.012717872F,
          -0.019179828F, -0.055048276F, -0.016794112F, -0.0102959005F,
          0.019310065F, 0.0010978224F, 0.086670816F, 0.14756471F, 0.041676342F,
          -0.013081125F, -0.17448398F, -0.14870815F, -0.054015204F, 0.0326951F,
          0.014367198F, -0.025036292F, -0.10606024F, 0.014205963F, -0.07603325F,
          0.051916093F, 0.08814751F, 0.035333328F, 0.026455555F, -0.07427024F,
          -0.10462319F, -0.044861287F, -0.0067074727F, -0.03958874F, 0.01902314F,
          -0.08067502F, 0.008386593F, -0.028833883F, -0.024033602F,
          -0.011405272F, -0.14681627F, -0.07020628F, -0.022815341F, 0.06459119F,
          0.017691385F, 0.01612356F, -0.040325146F, 0.021511916F, -0.06388596F,
          0.035449136F, -0.023634052F, 0.038261626F, -0.08839183F, -0.043846175F,
          -0.003503922F, -0.09428727F, -0.043471865F, 0.025226373F,
          -0.043218423F, -0.03339637F, 0.010035212F, -0.120948F, -0.0065155304F,
          -0.059779994F, -0.040796336F, -0.0042377F, -0.08999618F, 0.10573377F,
          0.12607917F, 0.1272275F, 0.060474645F, -0.2385441F, -0.04853528F,
          -0.124467164F, 0.04446599F, -0.0041173254F, 0.016592102F,
          0.0077316696F, -0.0070796087F, -0.11975111F, -0.04795158F,
          -0.018898265F, -0.009455149F, -0.0006223912F, 0.034438554F,
          0.13422507F, 0.0061747283F, 0.010048653F, -0.12906954F, -0.10394642F,
          -0.006316464F, -0.024518242F, 0.047283128F, -0.058990866F,
          -0.06296445F, -0.09155173F, -0.025192022F, 0.0026592503F, 0.021626124F,
          -0.0064067356F, 0.01360619F, -0.040794086F, -0.047089636F,
          -0.06822984F, 0.024254179F, 0.050320473F, 0.14605579F, 0.00482944F,
          -0.044186592F, -0.13406868F, -0.011650705F, 0.0001425802F, 0.22594124F,
          0.26630998F, 0.12524526F, -0.0032340903F, -0.18695721F, -0.15204063F,
          -0.2182259F, -0.012378244F, -0.021242296F, 0.13390167F, -0.012065322F,
          -0.04808685F, -0.17011121F, -0.07041115F, -0.058593214F, 0.0462393F,
          0.017864794F, 0.11026553F, -0.070091814F, -0.17685087F, -0.07093223F,
          -0.234382F, 0.04437928F, 0.038423944F, 0.20572679F, 0.118022166F,
          -0.03560174F, -0.010198297F, 0.012022754F, 0.0038862266F, 0.016573606F,
          -0.038151708F, 0.020714676F, -0.024749774F, -0.04209699F,
          -0.013970548F, 0.0092522055F, -0.08096624F, -0.07822776F,
          -0.041827865F, 0.021585107F, 0.027988544F, -0.0024738717F,
          -0.09992475F, -0.078847416F, 0.07182985F, 0.08214393F, 0.06617746F,
          0.11984913F, 0.011788582F, 0.030922359F, 0.040593628F, -0.023146247F,
          0.09820123F, -0.1057339F, -0.10663696F, -0.1594918F, -0.08870017F,
          0.1702178F, 0.03342659F, 0.086027764F, 0.047402803F, -0.04019368F,
          -0.06662766F, -0.061637197F, -0.2201139F, 0.053024497F, 0.17864244F,
          0.07447802F, 0.16230057F, -0.050402414F, 0.04571125F, -0.017244501F,
          0.01789362F, 0.002935829F, -0.108779974F, 0.038899794F, 0.013834197F,
          0.029853288F, 0.011241318F, -0.043620843F, 0.022631755F, -0.038776513F,
          -0.037520763F, -0.10981327F, -0.05840112F, -0.0330923F,
          -0.00044027893F, -0.013409031F, 0.12624276F, 0.10262875F, -0.0314609F,
          -0.09061195F, -0.31257343F, -0.08065561F, -0.030442873F, 0.071568415F,
          0.06438426F, -0.01281182F, 0.02284531F, 0.08887408F, 0.0981339F,
          0.055046294F, -0.03973017F, -0.041620802F, -0.010451668F, 0.017084245F,
          -0.114447705F, -0.0016165037F, 0.057389017F, 0.047712013F,
          0.099389836F, 0.015525996F, 0.02982657F, -0.04537949F, -0.07730025F,
          0.04258773F, 0.05678506F, 0.023836602F, 0.013496551F, 0.0185374F,
          0.0112489015F, -0.012679771F, 0.0011054483F, -0.013854546F,
          0.010811919F, 0.0193528F, 0.019524911F, 0.038357243F, 0.022020623F,
          -0.002239474F, 0.0181541F, -0.00055732456F, -0.030422855F,
          -0.07136133F, 0.025417902F, 0.052732844F, -0.009556424F, 0.05029119F,
          0.023208046F, 0.11196847F, 0.048042607F, 0.042306498F, -0.054877535F,
          -0.06432362F, -0.07410665F, -0.03858688F, -0.09391965F, -0.08086241F,
          0.014868381F, -0.021002093F, -0.062080152F, 0.07814973F, 0.012347421F,
          0.027448036F, 0.108674504F, -0.054616135F, -0.041555557F, 0.011125719F,
          -0.078527555F, -0.049129035F, -0.026550652F, -0.02999494F,
          0.042761683F, -0.045209005F, -0.05772755F, 0.04767826F, 0.0291519F,
          0.036562935F, 0.047849085F, 0.035282813F, 0.03228977F, -0.022773677F,
          0.061977684F, 0.050138403F, -0.04590881F, 0.008255727F, -0.035431262F,
          -0.075918175F, 0.017341467F, 0.010924273F, -0.0064846836F,
          0.054106202F, 0.04467293F, 0.012479518F, 0.013186253F, 0.023851732F,
          0.021608138F, -8.779054E-5F, 0.0031235504F, 0.04446388F, -0.03167886F,
          0.0061148726F, 0.052381948F, -0.036154978F, 0.0057016485F, 0.03762448F,
          0.012603133F, -0.03179402F, -0.04255999F, -0.009801011F, -0.06565254F,
          -0.02710461F, 0.017134055F, -0.013966365F, 0.0070069707F, 0.01713537F,
          -0.02000164F, -0.015223002F, -0.014305456F, -0.04388101F, -0.04282687F,
          -0.0116015645F, -0.031134382F, -0.036649603F, 0.04668754F,
          0.018300952F, -0.017318118F, 0.07909816F, 0.04465826F, -0.008730406F,
          0.09146676F, 0.06661227F, 0.0059613576F, 0.0061274385F, -0.041968938F,
          0.0066986703F, -0.03880657F, -0.10067921F, -0.028033903F,
          -0.00043021515F, -0.000663703F, -0.0074594268F, 0.019559298F,
          0.023771575F, -0.010180004F, -0.0036908127F, 0.020391967F,
          0.027653325F, -0.013318066F, 0.0147552015F, 0.01717738F, 0.024431294F,
          0.008531226F, -0.028381284F, -0.009428684F, -0.008058875F,
          0.013504359F, 0.0042354534F, 0.035904255F, 0.009728508F, 0.053250227F,
          -0.022661837F, 0.010721018F, 0.06990599F, 0.040195104F,
          -0.00052821485F, 0.091067754F, 0.08326593F, 0.039824132F,
          -0.021589462F, -0.031915378F, -0.015184982F, -0.027039869F,
          -0.03906183F, 0.0075095855F, 0.022963354F, -0.02019871F, -0.011539504F,
          -0.013289792F, 0.00066863006F, 0.034905847F, 0.012385189F,
          0.025488306F, 0.0007733785F, -0.014201029F, 6.0909064E-5F,
          0.0009115255F, 0.049992774F, 0.0037635164F, -0.03959303F,
          -0.029477134F, -0.011141146F, -0.0015671694F, -0.017148051F,
          -0.0058859736F, -0.049033385F, 0.0030634706F, 0.0044571143F,
          -0.029167421F, -0.0036101171F, 0.011537735F, -0.024907228F,
          0.013940695F, 0.01663386F, -0.05038669F, -0.0041461363F, -0.09003912F,
          -0.09769106F, -0.11331953F, -0.20742173F, -0.110652275F, -0.09142485F,
          -0.108338505F, -0.080577016F, 0.011856077F, 0.01673459F, -0.020335779F,
          0.03309488F, 0.008091887F, 0.027893404F, 0.046087008F, 0.034656335F,
          0.046905663F, 0.040801115F, 0.025707774F, -0.0064208386F, 0.04948644F,
          0.036721203F, -0.0046656043F, 0.02478847F, 0.011903351F, -0.017590024F,
          -0.029752798F, 0.0032222667F, -0.010004366F, -0.036515545F,
          -0.021934986F, 0.03474447F, -0.00017181534F, 0.04117403F, 0.05053574F,
          -0.0136372475F, -0.012495259F, 0.0049611228F, -0.058836088F,
          -0.045620747F, -0.026854327F, -0.025355829F, -0.045116287F,
          0.0019412201F, -0.0016713166F, -0.009067057F, -0.022322064F,
          -0.0077359984F, -0.0169243F, 0.015358649F, -0.0044783517F,
          0.0046639126F, 0.027276156F, 0.025862277F, 0.0077315555F,
          -0.004937813F, 0.02727558F, -0.0142772775F, -0.03028741F, 0.006754755F,
          -0.004200424F, -0.0083410125F, 0.04299215F, 0.0075523243F,
          -0.03566359F, 0.013230979F, -0.013003831F, -0.010470811F,
          0.0047418294F, -0.004484032F, -0.017346112F, -0.051944703F,
          0.0057380963F, -0.020444188F, -0.064637505F, -0.0190323F, 0.017526684F,
          -0.0504926F, -0.023759022F, -0.00034974958F, -0.006867286F,
          -0.013663441F, 0.009063532F, 0.014925266F, 0.029570121F,
          -0.0043968023F, 0.04557577F, -0.0038018788F, -0.017106103F,
          0.010605222F, -0.03755942F, 0.009257607F, 0.009142802F, -0.014132197F,
          0.024554377F, 0.036466118F, 0.035442248F, -0.006243882F, -0.011809416F,
          0.009442281F, 0.014066589F, -0.038460184F, -0.010222174F, 0.011950376F,
          0.026384898F, 0.033942163F, 0.03201467F };

        static const float biasReformatted[64]{ 2.2947145F, 0.74867475F,
          1.0806046F, 4.8313894F, 1.5267246F, 2.1865807F, 0.9179493F, 3.6330745F,
          2.312155F, 6.245309F, 0.6191735F, 2.2355654F, 8.076201F, 2.7458282F,
          0.9430614F, 2.5835135F, 0.8361256F, 6.722781F, 1.1315362F,
          -0.52392733F, 2.9932225F, -0.9115094F, 0.7898221F, 1.680153F,
          0.20430592F, 3.2781312F, 2.3354216F, 3.2463188F, 1.3427238F, 9.508898F,
          1.2815928F, -0.22104928F, -0.3881451F, 2.856192F, 0.21467507F,
          -7.276886F, 2.6060967F, 1.9934636F, 4.1873207F, 0.30999768F,
          1.1706898F, 0.64323485F, 3.045703F, 3.585679F, 1.2729455F, 1.9448892F,
          1.7570822F, 1.9066379F, 4.775065F, 0.96661925F, 0.36272752F,
          0.54871845F, 4.907811F, -0.3512156F, 1.1257185F, -1.8325667F,
          0.57059103F, 2.876153F, -0.14725602F, 2.081345F, -1.5006952F,
          -4.2101884F, 1.2320235F, 0.47929436F };

        c_convolution(X[0], Z[0], reformattedAndTruncatedWeights[0],
                      biasReformatted[0]);
      }

      void conv2dDirectOptimizedColMajor(const float X[519168], float Z[2768896])
      {
        static const float reformattedAndTruncatedWeights[432]{ 4.15475F,
          -0.21744274F, -1.7739631F, 4.621684F, -2.4649665F, -3.96453F,
          2.6516328F, -3.427341F, -2.4524896F, 2.5824802F, -0.633782F,
          0.4593323F, 1.9763868F, -3.1398866F, -1.2011368F, -0.6369768F,
          -3.5482478F, -0.4780612F, -1.8773222F, 0.43975067F, 0.49014527F,
          -1.5090984F, 0.20880924F, 0.27472782F, -1.9582257F, -0.2910864F,
          -0.111589F, -3.7154074F, -1.75297F, 4.315141F, -0.8665625F, 2.7187803F,
          3.5720434F, 5.918529F, -3.132957F, -4.4355965F, 1.8176445F,
          -6.8437786F, 5.510972F, -5.3137407F, 3.8561075F, -2.4625492F,
          7.302033F, -3.5257297F, 6.3293F, 6.3161325F, 3.2478187F, -3.5183918F,
          -0.015870396F, 14.970987F, -7.0828476F, -6.997233F, -0.05741568F,
          1.5656918F, 0.8947923F, 0.8706552F, 0.2550694F, 0.94723433F,
          0.7512263F, 0.19581819F, 0.1887101F, 0.42504203F, 0.056730352F,
          0.26511416F, 0.4346227F, 0.14914913F, 1.0999478F, 0.67895746F,
          0.28938344F, 0.47107697F, 0.88851404F, 0.21068501F, -1.1946704F,
          -1.5020316F, -0.33562675F, -0.78132737F, -2.2627769F, -0.85184896F,
          -0.831185F, -1.2061363F, -0.6252349F, 0.96257496F, 3.1449528F,
          1.8155432F, 0.8271684F, 3.3933609F, 2.516235F, -0.6886726F, 0.7265783F,
          0.8782999F, -0.5821285F, -2.8088918F, -1.8810258F, -0.357392F,
          -2.9012043F, -2.421301F, 0.8325602F, -0.6263082F, -0.8109172F,
          -0.29920486F, -0.51504105F, -0.07033196F, -0.23116809F, -0.51276004F,
          -0.2257607F, -0.035111964F, -0.1693022F, -0.16709764F, -6.779034F,
          0.84826845F, 6.247691F, -9.102249F, 0.40092996F, 7.8968005F,
          -6.509769F, 0.9461779F, 4.7787375F, -7.5841885F, 0.7185177F,
          4.9479985F, -8.599098F, 2.696948F, 7.461342F, -4.6467633F, 1.568555F,
          3.7384257F, -3.8974833F, 1.5164798F, 3.005081F, -6.273049F, 1.6596197F,
          4.0634284F, -3.273463F, 1.4902865F, 2.2862144F, 26.91817F, 8.987194F,
          1.5173047F, 3.5731173F, -3.3462794F, -6.4552374F, -7.5696917F,
          -12.159109F, -17.652714F, -13.141581F, 0.0011468627F, 1.9309211F,
          -1.7181842F, 4.0185747F, 2.5493438F, 3.85168F, 4.45517F, 3.6873102F,
          -14.553491F, -9.850318F, -3.2038887F, -3.0477293F, -1.3716369F,
          4.3166256F, 3.6470785F, 8.733766F, 16.128977F, 6.446351F, -2.1623924F,
          -3.3056366F, 12.0313835F, 7.9161143F, -5.8088665F, 1.8730973F,
          -9.462493F, -7.0740285F, 3.4990153F, -2.5820699F, -3.2793233F,
          8.421325F, 9.0449915F, -3.7336428F, 1.0961701F, -7.728992F,
          -5.2503133F, 1.771544F, -2.3320208F, -1.5718275F, 4.613422F, 6.063185F,
          -3.167758F, 2.530469F, -4.3249173F, -2.818479F, 1.3285106F,
          -2.6271589F, 0.7571656F, -8.197072F, -1.4230238F, 1.8336012F,
          0.11563853F, -2.093777F, -0.80197906F, 2.1330957F, -1.8154942F,
          -0.89100933F, -9.582119F, 0.469802F, 1.4338636F, 1.2449813F,
          -0.96161205F, -1.0333172F, 2.160272F, -2.212492F, -0.5890623F,
          -6.65512F, 0.25981975F, 1.66537F, 4.239527F, -1.1707222F, -2.2922971F,
          -4.4335237F, -4.217666F, 1.2893252F, -0.23414959F, -12.54577F,
          -15.970953F, 6.557909F, 15.843604F, 11.91183F, 0.5028392F, 3.6747894F,
          8.543182F, -1.8988355F, -12.91544F, -12.707935F, 1.1926435F,
          7.9913445F, 3.6329172F, 2.6914232F, 6.0486403F, 6.7879653F,
          0.33278832F, -1.9437226F, -1.3393949F, -5.7737274F, -5.1242F,
          -5.7407317F, -6.414722F, -8.579987F, -5.6868606F, 1.1855531F,
          1.0533166F, 2.3786688F, 4.8405232F, 7.4059F, 3.976885F, -6.044401F,
          -8.48781F, -4.796477F, 1.8049091F, 1.4598991F, 2.760293F, 3.9789953F,
          6.8079147F, 2.2982183F, -4.281777F, -6.34286F, -4.524757F, 2.2356694F,
          1.8071955F, 2.6907058F, 2.426083F, 4.594817F, 1.7332705F, 4.986245F,
          0.066969894F, -5.7011766F, 7.163981F, 0.6105673F, -7.0565286F,
          4.69723F, 0.24096978F, -5.035774F, 4.3001356F, 0.058220506F,
          -4.8203645F, 6.6184826F, 0.39984587F, -6.545138F, 3.6699047F,
          0.15305063F, -3.8304245F, 3.5577247F, 0.39119267F, -4.1594954F,
          4.6562886F, 0.81525016F, -4.9113154F, 2.7843668F, 0.18847018F,
          -3.131401F, 5.739264F, 9.201627F, 6.196036F, -1.1032554F, -3.2915561F,
          -0.8142396F, -4.806652F, -6.654615F, -4.2514114F, 5.8704896F,
          9.107284F, 4.3818936F, -1.1995698F, -3.2327569F, -0.69717354F,
          -4.9337263F, -6.343983F, -3.0337353F, 4.211053F, 6.6866884F,
          3.9656246F, -0.24866366F, -1.660731F, -0.06264284F, -3.7590501F,
          -5.2991977F, -3.600497F, -1.6991484F, -3.2689805F, -1.5496098F,
          -2.0962791F, -3.85899F, -3.414242F, -1.6412337F, -2.2907367F,
          -1.5552415F, 1.7604725F, 0.7946382F, 0.98239917F, 1.6499985F,
          0.586368F, 0.075914636F, 0.22281086F, 0.56993365F, 0.4246F, 1.2349956F,
          1.1477005F, 1.2505919F, 2.0082703F, 1.8232054F, 1.4257127F,
          0.47035217F, 2.1207063F, 2.0606124F, -0.44354793F, -0.68741983F,
          1.0425907F, 2.298346F, -7.8635793F, 4.935215F, 4.465019F, -10.606957F,
          6.927896F, 0.42773873F, -0.70666367F, -0.17555983F, 4.0853634F,
          -8.241037F, 3.7455635F, 6.0181794F, -11.419729F, 6.237487F, 0.6295688F,
          0.016644869F, -0.76118994F, 2.369684F, -3.7402349F, 0.65638804F,
          4.1783533F, -6.6802382F, 3.108009F, -6.7199244F, 0.17637667F,
          5.8389034F, -8.671436F, 0.46686256F, 8.569196F, -5.0345F, 0.7448651F,
          4.6385026F, -5.1006227F, 0.77636F, 4.168942F, -7.6453853F, 0.7385409F,
          7.0355077F, -3.9256454F, 0.97065043F, 2.720795F, -3.905561F,
          0.9842723F, 3.6532943F, -5.6021204F, 0.85892344F, 5.077779F,
          -3.150271F, 0.6907241F, 1.9110261F, -0.3586526F, -1.0339311F,
          0.2644048F, 1.4257483F, -1.2589452F, -1.940589F, 0.9448387F,
          0.12447523F, -1.184548F, -1.6027186F, -1.9748026F, -1.3456103F,
          -1.6549468F, -2.7533417F, -2.8751657F, -2.4295769F, -1.3468146F,
          -1.3494521F, 1.6043985F, 2.6503825F, 1.9385072F, 2.460373F, 3.6217291F,
          2.6301212F, 0.4131043F, 2.542974F, 2.2343366F };

        static const float biasReformatted[16]{ 3.0205722F, -14.385002F,
          0.66348433F, 0.72302395F, -4.7087903F, 2.3703384F, -5.5467625F,
          3.4648867F, 1.1081994F, 0.8245135F, 0.88959795F, 1.0843929F,
          1.3636358F, -0.16059971F, 1.629756F, 1.9013643F };

        convolution(X[0], Z[0], reformattedAndTruncatedWeights[0],
                    biasReformatted[0]);
      }

      void d_conv2dDirectOptimizedColMajor(const float X[173056], float Z[346112])
      {
        static const float biasReformatted[128]{ 0.8480756F, 5.368713F,
          4.767747F, 2.6387382F, 0.91814315F, 2.3875012F, 1.9474213F,
          0.19964886F, 1.0487162F, -0.4514954F, 0.3613608F, 1.1274443F,
          -3.3007092F, 3.8650014F, 0.50680506F, 3.4901423F, 2.2302313F,
          1.4460531F, 1.7188064F, 1.2193577F, 2.7793472F, -5.7964425F,
          -4.7964134F, 0.47219765F, 0.20236385F, -0.4025936F, 1.2854774F,
          -0.84067404F, 0.46429902F, -0.10694432F, 2.4104247F, 1.5085659F,
          1.3607484F, 0.5784147F, 1.1564782F, -1.5395039F, -2.9297006F,
          2.3402047F, 3.5244162F, 3.5048347F, 1.5612764F, 0.90362006F,
          0.06613803F, 0.8267971F, 0.5200181F, 1.3544308F, -0.24258256F,
          1.8944937F, 0.7430099F, 1.4184893F, 1.0606213F, -1.1799284F,
          0.49789727F, -2.173774F, -0.59911215F, 5.0696135F, 1.2916714F,
          1.016531F, -0.57220364F, 1.6751156F, -3.608335F, -0.73804045F,
          -0.107361555F, 0.6058307F, 0.1488418F, -0.18456352F, 0.8895514F,
          -0.74341786F, 1.1283488F, 0.16480923F, -2.845357F, 2.3445587F,
          1.2547965F, 2.082278F, -0.40445754F, -3.5532856F, 1.3505411F,
          0.655373F, -0.9818385F, 1.3918991F, 0.76175344F, 1.5183569F,
          -1.8400631F, 0.7813442F, 0.3373134F, 8.434666F, 0.49568367F,
          -0.22247076F, 1.0240887F, 0.18353677F, -1.2211453F, 0.9224159F,
          -0.23667051F, 2.7973733F, 1.3865786F, 0.030849934F, 1.024495F,
          -0.8956393F, 4.1628675F, 1.4230206F, 2.4584203F, -4.216916F,
          -0.072594166F, 1.018352F, 2.32086F, 3.5050073F, 0.5265305F,
          0.45474958F, 2.653791F, -2.0852904F, -0.6317569F, 1.6180326F,
          2.4703393F, 1.3910398F, 1.0553948F, 0.66194063F, 1.7013441F,
          -6.563445F, 0.67904377F, 0.9954572F, 0.68682736F, 1.0545871F,
          1.8116026F, 1.9924049F, 0.13129854F, 2.8658094F, 3.052352F, 3.6017153F
        };

        static float reformattedAndTruncatedWeights[73728];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(reformattedAndTruncatedWeights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_3676393.bin", 73728);
        }

        bufferInitialized = true;
        d_convolution(X[0], Z[0], reformattedAndTruncatedWeights[0],
                      biasReformatted[0]);
      }

      void e_conv2dDirectOptimizedColMajor(const float X[86528], float Z[173056])
      {
        static const float biasReformatted[256]{ -0.4562807F, -0.7868844F,
          0.45128322F, -0.6542226F, 0.6742965F, 0.14244199F, 0.17220044F,
          0.640748F, -0.46889257F, -2.0420797F, -0.79727817F, -0.90175843F,
          0.34696645F, 0.34105647F, -0.37594128F, -0.8879955F, -1.4610443F,
          0.17885101F, -1.4291744F, -1.929589F, 1.090581F, -0.8156787F,
          0.2542547F, 0.64759636F, -2.591034F, -0.028421402F, -1.4133722F,
          0.4102589F, -1.3670977F, -0.025044918F, -0.8946104F, 2.4776711F,
          -1.3294096F, -1.0416908F, -1.1571836F, 1.3091328F, -0.20100993F,
          -1.2721127F, 1.2877724F, -0.908242F, 1.6810883F, 0.28614974F,
          -0.9062785F, -0.35222363F, -0.038202763F, -0.8655678F, -0.48041475F,
          0.49218965F, -0.29154515F, -0.06951499F, -0.6670656F, -0.2700225F,
          -1.126744F, -0.016178012F, 0.7572849F, -0.49728703F, 0.58730555F,
          -0.81576645F, -2.0070581F, -0.3489809F, -1.0326627F, 0.640645F,
          -6.647994F, -0.6287216F, -0.5114082F, -0.35021472F, 0.38873994F,
          -0.98151517F, -1.1003959F, -1.5839705F, -1.197649F, -1.2658906F,
          -0.6083057F, -4.610384F, -0.65511274F, -1.5839982F, -1.2121507F,
          -0.6458629F, -0.2495408F, 0.6564131F, 0.13081002F, 1.027004F,
          0.38050747F, -0.9579543F, -0.56526995F, -0.18028986F, -0.103019774F,
          1.5038948F, 0.29349852F, -0.4760693F, 0.57445085F, -0.3550154F,
          0.38868642F, -0.3505323F, 3.1823044F, -1.2934592F, -0.96636117F,
          -0.87568665F, -0.9509235F, 0.4215269F, 0.30429852F, -0.29366875F,
          -1.5260735F, 0.291808F, 0.11567849F, -0.15156126F, 0.05627072F,
          0.3388672F, -0.12668496F, 0.79513687F, -0.6215796F, -2.0381997F,
          -0.5459465F, -1.0772449F, 0.27813601F, 1.1520505F, 0.469787F,
          -2.8945255F, -0.9035201F, 0.36082864F, 0.026935339F, 0.6573448F,
          -0.83916295F, -0.41743612F, -0.52991366F, -0.07826078F, 0.5370641F,
          0.7817677F, 0.043385983F, 0.915815F, 0.025009215F, -0.7292385F,
          -0.85246974F, 0.41990972F, -1.5289068F, -1.1783539F, -0.2806692F,
          -0.12158203F, -0.06486714F, -0.5342989F, 1.4942373F, -1.5442076F,
          -1.5130757F, -6.0665154F, 0.016750216F, -0.5612117F, -0.80136716F,
          -1.1870413F, -0.47093618F, 2.0344524F, 1.2296734F, 0.6800113F,
          0.5594704F, -1.09656F, 0.8167778F, -1.8547593F, 0.5521138F,
          -1.4495888F, -0.4110813F, -2.545685F, -1.3041472F, -0.09415835F,
          0.5917171F, -1.0095594F, -0.0819509F, 0.4465689F, -1.1411438F,
          -0.5279249F, 0.004019022F, 0.009142637F, -0.16586566F, -3.848639F,
          -0.054660797F, -1.4163704F, 0.1841929F, 1.4203941F, -2.128891F,
          -0.9585794F, -0.81917304F, -1.1549041F, 0.17640138F, -0.17406404F,
          -1.5279024F, 0.38962173F, -1.0299641F, -0.962296F, -0.5178189F,
          -1.0817735F, -0.76538277F, 1.5194513F, -1.109234F, -0.76258826F,
          -1.3037789F, -1.312373F, -0.574921F, -0.24406123F, -1.1505308F,
          -1.5948873F, 0.1569671F, -0.6517068F, -2.0937946F, 1.3365273F,
          -1.0294635F, 0.9710724F, -0.017334104F, -1.2969954F, -0.89764524F,
          0.055248857F, 1.468345F, -0.10499036F, -0.3731289F, 0.48030472F,
          0.7294041F, -0.23688972F, 1.7848353F, -0.13829589F, 0.8092495F,
          0.030893326F, -1.5556405F, 0.5047884F, 0.25014913F, -0.039325F,
          -0.9147825F, 0.34542274F, -0.51629245F, 0.98945725F, 0.43075526F,
          0.40926027F, -0.50150883F, 0.7694042F, 0.11692631F, 0.26369405F,
          -0.58858037F, -0.5386901F, -0.37160408F, 1.1190312F, 1.4723132F,
          -1.7207726F, 0.19516015F, -3.3077524F, 0.8605001F, -1.3128076F,
          0.19466758F, -0.0025029182F, -0.35336506F, 0.7366344F, -2.8067868F,
          -1.4758952F, 0.9535707F, -0.31600344F, 0.27550417F, -0.9715723F,
          0.74634933F, -0.093253136F, -0.10062337F, -0.38350374F };

        static float s_Weights[294912];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(s_Weights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_4066852.bin", 294912);
        }

        bufferInitialized = true;
        e_convolution(X[0], Z[0], s_Weights[0], biasReformatted[0]);
      }

      void f_conv2dDirectOptimizedColMajor(const float X[43264], float Z[86528])
      {
        static const float biasReformatted[512]{ -1.4120386F, 0.02237463F,
          -0.49060708F, -1.0975916F, -1.7435689F, -3.3384712F, -0.20367134F,
          -0.55548215F, -0.96113044F, -0.8632604F, -1.1656783F, -1.2100128F,
          -0.4873984F, -1.8772047F, -1.7392832F, 0.08140707F, -0.49344695F,
          -0.7621585F, -0.13946736F, -0.3613516F, -0.27578628F, -0.2584666F,
          -1.0797883F, -0.66452026F, -1.9161699F, -1.0884416F, -0.6993902F,
          -2.0072744F, -0.48360062F, -1.4838322F, -1.4035677F, -1.7053425F,
          -2.5731766F, -2.7644908F, -0.9137056F, -1.5241735F, -0.99675477F,
          -1.1283368F, -1.7773548F, -1.8091199F, -1.7119721F, -0.88201094F,
          0.12257326F, -1.7831161F, -0.81988716F, -0.6402644F, -1.2112027F,
          -1.7500638F, -1.2047901F, -0.48482937F, -0.78161764F, -1.459139F,
          -0.27002418F, -0.7509957F, -2.7502403F, -2.2222886F, -0.9397394F,
          -1.0988284F, -0.06910342F, -1.1836902F, -0.3526281F, -1.7786891F,
          -0.87337977F, -0.2851597F, -2.7049484F, -0.45697236F, -0.3745917F,
          -1.4106107F, -0.69059336F, -0.11134875F, -1.1500361F, -1.5470331F,
          -1.091344F, -0.9410089F, -0.7619312F, -2.2837906F, -0.95498896F,
          -1.3400893F, -0.6958852F, -0.4547721F, -0.5080937F, -1.2155219F,
          -1.409794F, -1.4130529F, 0.28499007F, -1.4257969F, -2.3168876F,
          -0.31138086F, -1.2503234F, -1.1912205F, -0.83765316F, -0.8129761F,
          0.47078001F, -0.63791484F, -1.9164865F, -1.6107879F, 0.06401384F,
          -1.177387F, -1.9073896F, 0.17094153F, -1.9354455F, -1.4564216F,
          -0.6550095F, -0.92409647F, -1.0031532F, -0.017117262F, -0.8546551F,
          -0.9523581F, -1.0659907F, -0.74348235F, -0.6793378F, -1.2308086F,
          -0.5114703F, -1.2587881F, 0.27996862F, -1.4646243F, -0.62154603F,
          -1.2144065F, 0.054162502F, -0.6880347F, -0.89690435F, -1.6853992F,
          -1.45369F, -0.41635877F, -0.6513264F, -2.4447868F, -1.6032717F,
          -0.10521978F, -0.59330136F, -1.6714673F, -0.9133337F, -0.63370705F,
          -1.1698103F, -1.5035884F, -1.9542131F, -0.45958036F, -0.7706414F,
          -0.26857638F, -0.498335F, -1.8442338F, -2.1487632F, -0.06188202F,
          -2.5403922F, -1.0226101F, 0.22801447F, 0.055657744F, -1.6432662F,
          -2.3949745F, -0.4789127F, -1.0840868F, -0.76292956F, -1.1394522F,
          -3.8549228F, -1.9875054F, -0.83138F, -0.40380812F, -0.9660773F,
          -2.3439999F, -1.1114005F, -0.8201643F, -1.4469955F, -1.454665F,
          0.21516341F, -1.3090396F, -1.4267042F, -1.2014174F, 0.1121763F,
          -0.14189982F, -2.437736F, 0.39130092F, -1.0192493F, -0.16012406F,
          -1.3506289F, -1.6794372F, -2.3598104F, -1.5536597F, -0.59693307F,
          -1.6133689F, 0.31557554F, -0.4836477F, -2.48989F, -0.78030396F,
          -0.20069885F, -1.5254178F, -3.086124F, -0.63360465F, -1.6873609F,
          -1.08307F, -0.21696615F, -1.1307509F, -3.2226906F, -1.0497875F,
          -0.57247365F, -2.2409222F, -0.94020104F, -0.87307924F, -0.2667973F,
          -0.43612272F, -2.2617095F, -1.5249941F, -0.954388F, -1.3837342F,
          -2.0065155F, -0.81239104F, -1.1762037F, -1.2094667F, -0.5556015F,
          -1.1954288F, -1.3244987F, -1.1925231F, -0.5289319F, -0.99578226F,
          -0.50181234F, -1.4949328F, -0.69462025F, -2.2291808F, -1.0212064F,
          -3.1793494F, -0.73903555F, -1.1391917F, -1.3537087F, -2.2765393F,
          0.3424685F, -1.6802721F, 0.50167614F, -0.5317733F, -1.093617F,
          -1.4063239F, -1.5893186F, -1.2599896F, -1.8072236F, -2.2662196F,
          -1.0124148F, -0.3314281F, -1.6455224F, -1.644769F, 0.19916704F,
          -0.7042041F, -1.5542042F, -0.71290976F, -2.4484546F, -0.92563367F,
          -1.346642F, -1.4148375F, -0.8750577F, -2.0054507F, -0.49498558F,
          -0.7709633F, -0.8921703F, 0.05818808F, -0.5058818F, -1.5058866F,
          -1.3094088F, -1.8475397F, -1.2753181F, -2.8752642F, -1.7689835F,
          -1.215788F, -2.0367508F, -1.2935338F, -2.551055F, -0.59607935F,
          -2.5182188F, -2.9103823F, -1.8208492F, -1.8546758F, -1.4688853F,
          -1.4892187F, -0.029513836F, -1.4481958F, -1.8568227F, -1.019173F,
          -1.5106715F, -0.9574659F, 0.22181022F, -0.77168435F, -1.1059065F,
          -0.9243843F, -0.78656566F, -1.10338F, 0.83110523F, -2.168738F,
          -0.8267668F, -1.4686399F, -0.66610575F, -1.1181929F, -0.72401136F,
          -0.61245596F, -1.6979027F, -0.98293674F, -1.3817258F, -1.3827792F,
          -1.9793922F, -1.0899814F, -0.6523402F, -1.3888984F, -1.2887499F,
          0.446424F, -2.2200506F, -1.2205104F, -1.0502803F, -0.75466883F,
          -0.83304864F, -1.0348115F, -0.34154326F, -0.48855138F, -0.3694316F,
          -1.6007457F, -1.3806441F, -0.6402586F, -2.4731045F, -1.0733916F,
          -0.18877202F, -0.79542625F, -1.3633316F, -1.5094787F, -0.8163817F,
          -1.4586242F, -0.697134F, -1.060221F, -1.0138487F, -0.6419716F,
          -1.6278783F, -0.9875635F, -1.278961F, -0.8878233F, -0.9677786F,
          -1.1614647F, -0.80476356F, -0.44430837F, -2.6318598F, -1.6352973F,
          -1.6383767F, -1.0170952F, -1.8658984F, -0.12358481F, -1.4971353F,
          -2.4304864F, -1.0325401F, -1.3487725F, -0.42914534F, -0.82733005F,
          -1.1134841F, -0.30565023F, -0.47529042F, 0.007146597F, -0.756729F,
          -2.974772F, -0.74360347F, 0.325166F, -1.1405897F, -0.58411455F,
          -1.4545987F, -0.08706027F, -0.9470396F, -0.44553348F, -0.0071178675F,
          -1.2898686F, -0.27713174F, -1.7179588F, -1.6531311F, -1.954805F,
          -0.5591749F, -0.889245F, -0.43098044F, -1.9658451F, -1.6084532F,
          -0.46965772F, -0.33131814F, -0.6997243F, -0.62442076F, -0.3324536F,
          -1.0360566F, -2.7357724F, -1.1200864F, -1.4971001F, -0.19438094F,
          -0.73299825F, -0.8491932F, -0.7833873F, -1.1259739F, -1.2580463F,
          -1.275444F, -0.686584F, -1.1170247F, -1.8520024F, -0.28165412F,
          0.69214463F, -0.7955935F, -1.3640134F, -1.2322377F, -1.4398332F,
          -0.63946784F, -1.6452943F, -0.95413136F, -0.617602F, -0.2485463F,
          -0.96539474F, -1.1956981F, -1.0819132F, -0.37991175F, 0.055906534F,
          -0.7125379F, -0.9636879F, 0.1992017F, -1.3370183F, -1.826406F,
          -0.59908855F, -0.62634575F, -1.088121F, -1.9829917F, -1.4198141F,
          -1.2868185F, -1.5323902F, -0.9417784F, -0.9646028F, -1.1905956F,
          -0.8944604F, -2.2665808F, -0.18931186F, -2.3950927F, -0.42295074F,
          -1.9072785F, -0.3195721F, -0.6881765F, -2.3191314F, -1.0181507F,
          -0.24740398F, -0.7231403F, -1.9293828F, -1.8846025F, -0.55864596F,
          -1.3071135F, -0.62900805F, -0.82109606F, -0.6039696F, -1.3993193F,
          -2.2226057F, -2.5077982F, -0.9496299F, -0.55119616F, -0.18902886F,
          -1.6538234F, -2.365759F, -0.014912248F, -2.7562559F, -1.3648274F,
          -1.5212357F, -0.2004857F, -0.01001364F, -0.08050549F, -0.67098343F,
          -2.475264F, -2.009112F, -1.3173931F, -1.2709607F, -1.7828846F,
          -1.5059268F, -2.9581695F, -0.9887935F, -0.7360373F, -0.325781F,
          -1.2898422F, -1.0832976F, -1.1914234F, -0.9906676F, -0.6422696F,
          -0.059893012F, -0.59963095F, -2.2511163F, -1.9854176F, -0.5880537F,
          -1.0025371F, -1.2011214F, -0.5476566F, -1.1656722F, -1.8646792F,
          -1.4241234F, -0.8214452F, -1.788075F, -0.60759497F, -1.5393546F,
          -0.8607373F, -1.551841F, -0.31110835F, -1.2371459F, -1.2422541F,
          -0.55829597F, -1.5774368F, -1.3722935F, -3.0591593F, -0.14469242F,
          -1.7374369F, -1.149811F, 0.14882582F, -1.2197211F, -0.98677075F,
          -0.5488028F, -0.78620076F, -0.64981025F, -0.30048788F, -0.97075856F,
          -0.7042918F, -1.8498342F, -0.8520346F, -0.9974691F, -1.2977145F,
          -0.17071491F, -1.143539F, -1.2056631F, -0.49188894F, -1.2312697F };

        static float s_Weights[1179648];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(s_Weights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_4066861.bin",
            1179648);
        }

        bufferInitialized = true;
        f_convolution(X[0], Z[0], s_Weights[0], biasReformatted[0]);
      }

      void g_conv2dDirectOptimizedColMajor(const float X[86528], float Z[173056])
      {
        static const float biasReformatted[1024]{ -3.1831644F, -1.4111307F,
          -3.0776296F, -2.5585787F, -1.0349566F, -2.3858316F, -3.2626228F,
          -2.8205228F, -2.0844765F, -1.6441748F, -1.9953487F, 0.042580605F,
          -1.4602028F, -1.4418579F, -2.1320379F, -1.9351549F, -1.5251701F,
          -1.8960757F, -0.7439878F, -2.3087068F, -3.1349273F, -3.100839F,
          -2.2165751F, -2.5568662F, -4.01375F, -4.065041F, -1.9176143F,
          -2.6025336F, -2.11562F, -3.0578074F, -1.5158216F, -3.6900356F,
          -1.6313038F, -3.4838831F, -2.0530813F, -1.7558849F, -3.2462661F,
          -1.5470799F, -1.4397178F, -0.73522305F, -2.028833F, -0.5591531F,
          -3.8360677F, -2.8732448F, -2.1575403F, -0.6521778F, -2.1296787F,
          -3.7738504F, -4.2883005F, -2.4458613F, -3.6992538F, -2.9343798F,
          -2.2855935F, -3.1153843F, -2.195409F, -1.9439026F, -2.6093879F,
          -1.6423936F, -0.6457133F, -2.8388102F, -3.6026566F, -3.258649F,
          -2.3207443F, -1.7726716F, -3.3588746F, -4.031503F, 0.6659725F,
          -1.1995816F, -2.0061882F, -0.85763216F, -2.5761409F, -2.5037422F,
          -2.6176147F, -2.16461F, -1.4477117F, -1.5816026F, -4.0066795F,
          -2.0250125F, -0.9444258F, -3.6645162F, -3.398288F, -4.68829F,
          -2.196754F, -0.74297774F, -2.8454387F, -2.4864438F, -2.0493698F,
          -0.6366167F, 0.15701246F, -2.591055F, -3.7163424F, -1.623869F,
          -1.3308439F, -2.6458058F, -2.2088053F, -0.81422186F, -3.4593885F,
          -1.5992701F, -2.3381743F, -2.0106773F, -1.7974432F, -3.3523054F,
          0.09923053F, -1.8338596F, -3.5266173F, -2.5739694F, -4.4477754F,
          -2.2141044F, -1.44658F, -2.721028F, -2.7359266F, -1.9463115F,
          -2.3624027F, -1.8203547F, -0.83616304F, -4.0129423F, -3.267131F,
          0.34910917F, -1.2360785F, -1.5174179F, -0.51803553F, -0.6550137F,
          -2.2149146F, -2.8258586F, -3.0131037F, -1.9934448F, -2.881792F,
          -1.5488179F, -1.163127F, -1.5736285F, -2.4616573F, 0.63776565F,
          -1.9692314F, -2.3270097F, -2.442645F, -0.46496868F, -1.7031853F,
          -0.5147902F, -3.130798F, -3.0807064F, -3.5028458F, -1.6821396F,
          -4.042984F, -0.9940858F, -1.5873675F, -2.460132F, -1.6286211F,
          -1.6155992F, -2.7725701F, -3.032087F, -1.0123985F, -1.5240667F,
          -2.1879673F, -1.6974713F, -4.1775346F, -0.9938921F, -2.6068184F,
          -3.1689644F, -3.9851706F, -2.8361285F, -3.8725712F, -1.7543331F,
          -4.4701138F, -4.317276F, -1.2463559F, -2.0160642F, -0.5026629F,
          -1.5435419F, -1.4906204F, -2.5604446F, -0.1672256F, -1.5433478F,
          -2.2187467F, -2.0117435F, -1.9508605F, -2.4429836F, -1.5507605F,
          -1.1812239F, -2.6945796F, -2.5803168F, -0.55958354F, -1.1562366F,
          -1.6269853F, -1.9499385F, -2.5714884F, -2.147362F, -1.0306028F,
          -1.8703475F, -1.067447F, -2.4075477F, -2.9422019F, -1.806031F,
          -3.998335F, -2.9806306F, -3.374753F, -1.9941752F, -3.6530554F,
          -0.97662234F, -0.61641455F, -3.5692866F, -0.27024126F, -4.6589556F,
          -3.049962F, -1.58608F, -2.0974672F, -3.1496603F, -0.48573923F,
          -1.4496825F, -2.2280269F, -1.7367232F, -4.6938143F, -2.7862444F,
          -3.770606F, -1.869147F, -2.014762F, -1.7066551F, -0.9213784F,
          -1.3597794F, -1.7458472F, -3.1973867F, -2.4032083F, -1.5849953F,
          -1.6677196F, -3.0811906F, -4.4571667F, -0.91460204F, -2.0543518F,
          -1.6994125F, -2.2492378F, -1.9644873F, -4.2662373F, -2.45815F,
          -3.5667264F, -2.2707522F, -3.6223843F, -3.793471F, -2.4611688F,
          -2.0812838F, -1.1129024F, -4.387943F, -2.6106355F, -2.349917F,
          -4.0455823F, -2.880017F, -0.76081896F, -1.1769485F, -1.2580297F,
          -3.0692198F, -0.46800804F, -1.5483167F, -2.3076034F, -0.9444237F,
          -1.0047029F, -2.8992214F, -2.4177628F, -2.942972F, -1.9101025F,
          -1.7931309F, -3.3203664F, -2.9018922F, -4.860543F, -0.19664955F,
          -4.1686044F, -1.6171582F, -2.6280165F, -1.6334956F, -2.9036975F,
          -1.615063F, -1.618542F, -3.3328402F, -1.9974381F, -0.7484343F,
          -2.5602055F, -1.0954506F, -3.8113143F, -3.6235962F, -2.4248874F,
          -2.6890514F, -1.5072596F, -1.8564669F, -2.9846451F, -3.9960198F,
          -1.271898F, -3.1066658F, -2.6036866F, -3.9125962F, -3.0253565F,
          -1.8594083F, -2.8129914F, -0.8318671F, -2.4894607F, -3.0442133F,
          -1.1357583F, -1.491029F, -1.8569081F, -3.3210998F, -3.861054F,
          -2.3494115F, -2.3981674F, -0.87233424F, -1.810146F, -3.6305883F,
          -1.3651886F, -1.4393663F, -3.0034695F, -2.8954568F, -3.2415261F,
          -1.2665844F, 0.33458638F, -2.682519F, -0.9781797F, -3.6609814F,
          -2.9054127F, -1.3885186F, -3.118966F, -1.407889F, -1.9040345F,
          -1.5709009F, -2.1577497F, -2.2754562F, -0.5344814F, -2.4753332F,
          -2.3851867F, -2.7863915F, -2.5233936F, -3.952495F, -1.8405564F,
          -0.9893862F, -3.5030665F, -0.8277255F, -0.8896539F, -2.0627387F,
          -2.750294F, -1.217319F, -0.4203968F, -1.5233351F, -3.4239628F,
          -2.742591F, -0.20731997F, -1.977478F, -1.0008252F, -1.420679F,
          -2.9339583F, -1.4765514F, -2.8472905F, -1.7227453F, -1.4880841F,
          -1.5105811F, -1.7993988F, -1.804836F, -1.2907977F, -3.3480656F,
          -2.8698564F, -1.3968606F, -1.4065936F, -3.6915379F, -1.46703F,
          -1.8521279F, -2.1559181F, -2.1583107F, -0.9115155F, -1.0903015F,
          -1.6885915F, -3.5631957F, -0.5079315F, -5.976775F, -4.0015454F,
          -2.606798F, -2.8303158F, -2.3354895F, -1.5268856F, 1.5154449F,
          -2.7864745F, -3.7890167F, 0.5056329F, -0.7932254F, -4.956998F,
          -2.364411F, -2.3051677F, -0.47938967F, -0.29369664F, -0.9448073F,
          -2.039244F, -3.0605266F, -1.2923287F, -4.254617F, -2.3318305F,
          -2.5658536F, -3.4497807F, -1.225462F, -2.7373757F, -2.4348078F,
          -3.4969754F, -3.273696F, -1.7043324F, -2.7953122F, -2.0541863F,
          -2.183515F, -2.7653196F, -2.7252786F, -1.0274075F, -0.8984575F,
          -1.7225738F, -1.4559026F, -1.276459F, -3.3271484F, -1.7226039F,
          -1.9006158F, -2.8219354F, -3.6516013F, -2.2868633F, -0.68735003F,
          -2.7096715F, -1.8127816F, -3.9819152F, -2.879276F, -3.217161F,
          -2.8826244F, -2.4757917F, -1.8684087F, -3.0097544F, -1.5685569F,
          -2.1588097F, -2.9769585F, -1.9848099F, -2.46076F, -1.4916917F,
          -2.4774451F, -2.8877463F, -3.2081866F, -1.8832105F, -2.5690243F,
          -2.939471F, -5.190272F, -1.7301145F, -1.0261624F, -2.2976336F,
          -2.3540447F, -1.358001F, -0.78981876F, -2.9429677F, -3.4810667F,
          -3.2341297F, -1.5184774F, -2.0499096F, -3.6024969F, -2.3568962F,
          -3.9019728F, -1.4837372F, -2.6681108F, -1.7236879F, -2.7012663F,
          -0.9394531F, -1.2894192F, -1.7029262F, -4.8192444F, 0.23900008F,
          -0.74852276F, -2.393457F, -1.0094082F, -1.6620839F, -1.285555F,
          -1.1526191F, -2.1334298F, -0.4979099F, -0.72410524F, -0.84826326F,
          -2.0498633F, -0.95344234F, -2.175295F, -0.9648044F, -1.81091F,
          -1.4824522F, -3.0045824F, -3.0436795F, -2.207561F, -2.0980399F,
          -2.140327F, -2.0855136F, -5.6053867F, -2.2926004F, -0.75127864F,
          -3.0064366F, -2.2227437F, -0.6613184F, -1.3488069F, -1.9367402F,
          0.50780296F, -1.8967049F, -0.21382713F, -4.134313F, -2.5426564F,
          -3.1039803F, -0.8579123F, -3.2181368F, -1.0544026F, -2.1027954F,
          -2.7510972F, -3.3986697F, -2.4146006F, -1.6594968F, -3.963823F,
          -4.320286F, -0.8531022F, -1.4862603F, -2.4712417F, -0.27869654F,
          -3.6149657F, -2.1923337F, -2.534358F, -3.3415682F, -2.2655191F,
          -2.4433684F, -0.4998095F, -0.8390013F, -2.0558786F, -2.449298F,
          -2.8293688F, -1.2192606F, -1.9164348F, -2.1305516F, -3.3811393F,
          -0.9222572F, -3.1979792F, -2.6145332F, -3.2703154F, -4.4832606F,
          -0.8883865F, -1.6318775F, -3.2998774F, -2.268712F, -2.3965187F,
          -1.6427393F, -1.7033365F, -3.241191F, -1.0929247F, -1.3957672F,
          -1.5925684F, -1.3633541F, -1.900265F, -0.35112858F, -2.5794065F,
          -2.9191613F, -1.6978284F, -3.1348186F, -2.417589F, -3.689644F,
          -2.3658676F, -0.4545045F, -2.027186F, -4.2432575F, -0.76517653F,
          -3.417439F, -1.9081079F, -1.3287721F, -0.8565911F, -1.7839985F,
          -3.1847937F, -2.2547688F, -2.0758958F, -0.6768131F, -2.1794987F,
          -3.0771985F, -0.015329838F, -2.4842372F, -3.1239386F, -0.9320977F,
          -2.5777085F, -1.4504378F, -0.29826808F, -2.1130784F, -1.3502542F,
          -1.5253576F, -1.6708984F, -1.5504158F, -1.7786022F, -2.7948232F,
          -1.9218463F, -3.256166F, -1.7855008F, -2.7565868F, -1.5644318F,
          -1.9804254F, -1.1684604F, -2.4483562F, -4.4269075F, -1.061409F,
          -1.452387F, -2.889841F, -1.7248926F, -2.482339F, -2.6360555F,
          -3.186226F, -2.4842725F, -1.7120838F, -1.8622346F, -2.5832148F,
          -4.523857F, -1.1283326F, 0.0042214394F, -3.122984F, -1.835279F,
          -2.5134184F, -0.409096F, -0.662025F, -3.345718F, -2.4223742F,
          -4.5917683F, -3.0653763F, -0.8787856F, -2.1764908F, -3.5957937F,
          -2.4576313F, 0.27618766F, -1.430042F, -2.2038121F, -2.5787997F,
          -2.3097615F, 0.03765273F, -1.1025052F, -1.0066774F, -2.5408702F,
          -2.9873202F, -3.7688527F, -1.2251801F, -2.8030093F, -3.2273765F,
          -0.13151813F, -2.9843469F, -0.3292265F, -2.8107016F, -3.5696032F,
          -1.1167092F, -1.5063217F, -3.169518F, -1.524677F, -3.2700195F,
          -2.2263222F, -0.8304889F, -2.3183873F, -3.0823095F, -2.0029583F,
          -1.9697332F, -4.5051055F, -1.837045F, -4.574289F, -1.4635155F,
          -4.776903F, -1.893936F, -4.7445836F, -2.0192046F, -1.5283325F,
          -1.346447F, -2.3318708F, -2.6787117F, -2.4083195F, -1.8683326F,
          -1.0946748F, -2.461067F, -2.5461502F, -1.9786546F, -0.7876377F,
          -2.5530372F, -3.213523F, -2.0011063F, -5.9404087F, -1.6591249F,
          -1.5732844F, -1.7657567F, -1.5377858F, -2.0227385F, -2.9184418F,
          -2.8163447F, -1.8875748F, -1.0348911F, -1.9201986F, -2.3440926F,
          -2.0932126F, -4.5308104F, -3.0453343F, -4.1916447F, -2.9317186F,
          -2.1126254F, -1.7510462F, -1.1386603F, -2.3992207F, -0.33429193F,
          -2.4404833F, -3.7106638F, -1.6420243F, -0.22537863F, -2.160778F,
          -3.1861901F, -2.1893234F, -1.6511335F, -3.1760845F, -0.716408F,
          -1.3375878F, -1.8368235F, -1.8783814F, -2.0087852F, -3.1758366F,
          -1.8887825F, -2.9501424F, -2.2285008F, -2.6316745F, -1.6370304F,
          -2.922201F, -1.0845354F, -2.7130537F, -3.3730187F, -1.3094774F,
          -1.7679671F, -0.9133234F, -2.1846972F, -1.1747882F, -1.5379009F,
          -2.3112164F, -4.1185937F, -0.82960856F, -1.3403027F, -1.6506791F,
          -2.4307206F, -3.1112726F, -3.6655307F, -0.97204626F, -0.83968246F,
          -0.09257054F, -3.0458267F, -2.598576F, -2.8223217F, -2.0892797F,
          -3.0790534F, -1.7170652F, -2.6316335F, -0.7901261F, -0.62686276F,
          -3.1871052F, -0.8819394F, -1.5336854F, -1.8748814F, -2.499478F,
          -1.2757462F, -2.0217984F, -3.1556687F, -1.8840578F, -3.1891665F,
          -1.3701905F, -2.1845956F, -2.4046805F, -1.4101096F, -0.9660902F,
          -1.2806864F, -0.9002819F, -1.5331029F, -2.8771846F, -2.125022F,
          -2.4487F, -3.2077966F, -2.619512F, -1.4007155F, -0.7173147F,
          -1.5712653F, -1.8709127F, -0.55090046F, -2.2685454F, -2.067309F,
          -2.0717254F, -2.9653735F, -2.2840664F, -1.4370031F, -3.9096203F,
          -0.5037472F, -0.9842081F, -1.8504502F, -1.8437035F, 0.052863598F,
          -2.055992F, -0.89815784F, -1.9847763F, -2.3726153F, -0.94176364F,
          -1.959996F, -0.6607243F, -4.0607023F, -0.82435405F, -1.861419F,
          -2.6320672F, -0.7815877F, -1.8623569F, -1.1702077F, -0.5509962F,
          -1.7940485F, -2.8680496F, -2.539427F, -3.7427225F, -2.2183254F,
          -2.2210789F, -2.4963632F, -2.3335328F, -2.1878185F, -1.8610864F,
          -2.313472F, -4.0048585F, 0.09729767F, -2.2942383F, -1.9398134F,
          -1.8104F, -1.5986996F, -3.854023F, -1.6011385F, 0.20318389F,
          -4.455307F, -2.0712855F, -1.4183117F, -4.031319F, -1.936718F,
          -1.5762929F, -0.56389356F, -1.5235896F, -2.5474195F, -1.7500521F,
          -2.988073F, -2.83441F, -1.8848166F, -2.6632333F, -1.839572F,
          -1.2969332F, -3.7875133F, -2.35833F, -1.2547294F, -7.5861273F,
          -1.7332015F, -2.120811F, -2.1086807F, -0.62273145F, -2.0143406F,
          -2.2309713F, -1.8562031F, -3.754065F, -1.2495596F, -3.2017193F,
          -1.1628096F, -0.8055426F, -0.89632595F, -0.20890605F, -2.8936698F,
          -1.874037F, -4.7815175F, -2.1226811F, -4.0209055F, -1.4310932F,
          -2.6378288F, -2.6873405F, -3.506976F, -1.9341575F, -3.7603707F,
          -1.8737383F, -1.2379396F, -3.4144301F, -2.4190059F, -2.9167504F,
          -2.4652855F, -4.372814F, -2.1446536F, -0.865664F, -1.2538979F,
          -0.22219872F, -1.914723F, -2.0064135F, -3.1592371F, -3.051762F,
          -1.6346183F, -0.799283F, -2.6720145F, -3.1448743F, -1.4560478F,
          -3.2620704F, -2.5139139F, -2.3724794F, -1.2988601F, -1.2942073F,
          -3.3715549F, -3.4884715F, -0.9365084F, -1.9343116F, -3.2314768F,
          -3.4820092F, -2.4383974F, -0.43524623F, -2.1577091F, -3.1577702F,
          -3.0715866F, -1.6664981F, -2.9438393F, -1.7759285F, -0.96616876F,
          -2.8285153F, -2.1643581F, -2.531552F, -0.45622253F, -1.710146F,
          -1.9948901F, -1.7391958F, -1.0382272F, -4.381695F, -1.5049007F,
          -4.141487F, -1.9931787F, -2.2732372F, -3.8203886F, -1.7860504F,
          0.13059282F, -1.111485F, -3.5218575F, -2.3592057F, -0.029357195F,
          -0.026798487F, 0.1596992F, -2.6374393F, -1.9744592F, -2.194462F,
          -3.850877F, -1.0459089F, -3.2383585F, -2.8657424F, -2.9851997F,
          -4.452786F, -0.56761456F, -1.2886431F, -2.206286F, -3.1755629F,
          -2.5816817F, -3.9020514F, -2.9457037F, -1.081507F, -2.6869981F,
          -2.8010728F, -2.0854406F, -1.4568088F, -1.6155944F, -1.1394188F,
          -2.3772F, -3.852854F, -0.0841558F, -2.1090932F, -4.236253F,
          -4.1751366F, -3.5280945F, -2.286417F, -3.5280633F, -1.3624253F,
          -2.9022517F, -0.8766291F, -4.027419F, -2.6238499F, -1.9618475F,
          -2.2395265F, -4.5319815F, -3.7488492F, -2.2572699F, -1.5632992F,
          -2.3571758F, -0.7364657F, -0.24101448F, -4.5422926F, -1.4587874F,
          -0.46832788F, -3.4075453F, -1.5276321F, -0.18355417F, -2.1907487F,
          -0.73386F, -2.4315398F, -3.3483045F, -2.0395026F, -2.8767376F,
          -1.2826774F, -3.3535974F, -2.4655507F, -2.0466871F, -2.78343F,
          -1.0345807F, -1.6359487F, -2.1759639F, -2.1248436F, -2.1663516F,
          -3.8813212F, -0.49919772F, -2.4803796F, -2.0336454F, -3.3097236F,
          -2.2748637F, -1.4991055F, -0.06954968F, -1.7662257F, -1.340209F,
          -1.7001548F, -0.14922357F, -0.47843993F, -1.6467131F, -0.9614856F,
          -3.3843517F, -1.7283034F, -3.0769517F, -0.09149766F, -1.9381123F,
          -0.9436698F, -2.1537488F, -1.308941F, -4.477929F, -2.3788087F,
          0.11705232F, -2.6955428F, -2.6282332F, -2.5900922F, -1.5961106F,
          -1.0311646F, -0.98664176F, -2.538134F, -2.0569746F, -2.236915F,
          -4.766632F, -0.72167397F };

        static float s_Weights[4718592];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(s_Weights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_4066870.bin",
            4718592);
        }

        bufferInitialized = true;
        g_convolution(X[0], Z[0], s_Weights[0], biasReformatted[0]);
      }

      void h_conv2dDirectOptimizedColMajor(const float X[173056], float Z[86528])
      {
        static const float biasReformatted[512]{ 0.32995477F, 0.79067767F,
          0.10769232F, 0.04301413F, 0.8125676F, 0.10820143F, 0.8443281F,
          0.75829035F, 0.44270468F, 0.6973814F, 0.4134862F, 0.5037439F,
          0.58082837F, 0.369848F, -0.43773937F, 0.8658166F, 0.37410203F,
          0.5551373F, 0.37551025F, 0.87624663F, 0.6258272F, 0.7590513F,
          0.46973586F, 0.5340165F, 0.4866711F, 0.78412944F, 0.7840683F,
          0.22875421F, 0.5281035F, 0.79053617F, 0.38680947F, 0.99156076F,
          0.39728153F, 0.6608548F, 0.6456961F, 0.6195336F, 0.6497767F,
          0.5165586F, 0.6966327F, 0.06617481F, 0.1899694F, 0.73388827F,
          0.24363053F, 0.35831788F, 0.0851893F, 0.38269758F, 0.323352F,
          0.56858647F, 0.44904616F, 0.29027322F, 0.37167186F, 0.3176769F,
          0.59838575F, 0.3245517F, 0.21856453F, 0.80501616F, 0.48870978F,
          0.3642415F, 0.18336271F, 0.20901349F, 0.450602F, 0.31305274F,
          0.15434071F, 0.36058038F, 0.2405335F, 0.22307639F, 0.40890294F,
          0.473917F, 0.67702574F, 0.68454057F, 0.73071444F, 0.5050523F,
          0.75186694F, 0.43943483F, 0.71172816F, 0.22446477F, 0.1522306F,
          0.5211156F, 0.7274935F, 0.58399296F, 0.40591443F, 0.28192538F,
          0.42614165F, 0.14994922F, 0.3704821F, 0.7081233F, 0.14712462F,
          0.78279316F, 0.49028814F, 0.65189755F, 0.51767695F, 0.48689115F,
          0.49089065F, 0.27684945F, 0.46723306F, 0.9214178F, 0.38408226F,
          0.28736904F, 0.2561982F, 0.5348352F, 0.26511967F, 0.5415429F,
          0.40393317F, 0.6432378F, 0.64633197F, 0.066737026F, 0.49664775F,
          0.55047864F, 0.35143095F, 0.6123611F, 0.64192295F, 0.33042914F,
          0.3323565F, 0.65697515F, 0.5322095F, 0.34846357F, 0.2553702F,
          0.6496661F, 0.7438673F, 0.5408765F, 0.59073055F, 0.32917082F,
          0.16533521F, 0.32378554F, 0.4375677F, 0.4068453F, 0.24430451F,
          0.54634887F, 0.072532535F, 0.48225677F, 0.3352273F, 0.7782092F,
          0.5179533F, 0.22845356F, 0.40769196F, 0.38066107F, 0.40763932F,
          0.32268786F, 0.3503778F, 0.30551237F, 0.06391251F, 0.48091114F,
          0.7996798F, 0.6024604F, 0.80499566F, 0.6319773F, 0.40877637F,
          0.25594687F, 0.33026397F, 0.30055588F, 0.38827696F, 0.4691726F,
          0.24737476F, 0.3161494F, 0.55695105F, 0.1892814F, 0.52267903F,
          0.547773F, 0.6395553F, 0.6393168F, 0.5407384F, 0.33199933F,
          0.42454788F, 0.39778215F, 0.52393085F, 0.50833F, 0.44653156F,
          0.69458526F, 0.60434717F, 0.6609262F, 0.69698256F, 0.7245486F,
          0.5702849F, 0.5685271F, 0.1822782F, 0.70595336F, 0.55151105F,
          0.5261881F, 0.19201164F, 0.78196853F, 0.6381419F, 0.17364731F,
          0.29673707F, 0.7156941F, 0.40123743F, 0.6130972F, 0.70732594F,
          0.68361145F, 0.25683588F, 0.17355251F, 0.3154258F, 0.50493175F,
          0.39306492F, 0.5723072F, 0.5848594F, 0.14458026F, 0.68599373F,
          0.5620146F, 0.1892019F, 0.69168484F, 0.7892585F, 0.4040935F,
          0.8306626F, 0.85554034F, 0.4251305F, -0.47098705F, 0.26290935F,
          0.16551355F, 0.4058376F, 0.71810186F, 0.40816742F, 0.1898284F,
          0.576487F, 0.29184726F, 0.62266594F, 0.888451F, 0.623359F, 0.39904204F,
          0.3032707F, 0.51910496F, 0.1847233F, 0.30832887F, 0.4944799F,
          0.57684606F, 0.84579426F, 0.14906022F, 0.2891779F, 0.17899673F,
          0.63560385F, 0.21055077F, 0.45701775F, 0.4380638F, 0.44194698F,
          0.6755225F, 0.7625866F, 0.40737188F, 0.09486829F, 0.39585593F,
          0.6534697F, 1.206542F, 0.67193925F, 0.4088959F, 0.8526086F, 0.4698381F,
          0.52062315F, 0.38141996F, 0.58458894F, 0.5975644F, 0.43629935F,
          0.877578F, 0.61090755F, 0.59812456F, 0.3908654F, 0.16185217F,
          0.78489465F, 0.81882566F, 0.5527051F, 0.66148883F, 0.5704707F,
          0.5579502F, 0.7247655F, -0.10016337F, 0.6018892F, 0.47527483F,
          0.36442122F, 0.14909256F, 0.67912686F, 0.7822607F, 0.3523033F,
          0.41262922F, 0.49522904F, 0.07933341F, 0.6402914F, 0.6206962F,
          0.6655021F, 0.16109416F, 0.5163602F, 0.2929643F, 0.36498F, 0.64995325F,
          0.4963526F, 0.5745387F, 0.8046211F, 0.62333393F, 0.5704514F,
          0.6264572F, 0.7157603F, 0.2815073F, 0.46990818F, 0.65370715F,
          0.70917517F, 0.7048568F, 0.8938108F, 0.37313774F, 0.5401591F,
          0.53626215F, 0.3554026F, 0.5712819F, 0.3557023F, 0.31248218F,
          0.30866045F, 0.593931F, 0.05840224F, 0.47601682F, 0.6379083F,
          0.30528492F, 0.57258654F, 0.614734F, 0.6327101F, 0.73066914F,
          0.7480342F, 0.43753743F, 0.5702403F, 0.5286908F, 0.07009418F,
          0.7123708F, 0.52277386F, 0.5700907F, 0.42090735F, 0.4593851F,
          0.24684685F, 0.58106506F, 0.11062543F, 0.48762178F, 0.4578451F,
          0.63180983F, 0.62483495F, 0.26134646F, 0.13880911F, 0.5988218F,
          0.37533572F, 0.34929967F, 0.2604878F, 0.5719892F, 0.49731615F,
          0.16346571F, 0.54565793F, 0.52346075F, 0.77267414F, 0.44801793F,
          0.2972717F, 0.7866673F, 0.78008723F, 0.74705887F, 0.5423116F,
          0.96180403F, 0.8274488F, 0.45126495F, 0.6504869F, 0.53314453F,
          0.5223101F, 0.8691133F, 0.6219869F, 0.6534867F, 0.5591685F,
          0.42113143F, 0.92025423F, 0.83957565F, 0.37345698F, 0.7177134F,
          0.7925444F, 0.58739495F, 0.4996566F, 0.42365634F, 0.6265944F,
          0.084502675F, 0.65494245F, 0.4111788F, 0.24623927F, 0.057377987F,
          0.757316F, 0.38263494F, 0.3587072F, 0.35867542F, 0.61406237F,
          0.84785515F, 0.37989104F, 0.5257263F, 0.62162775F, 0.2480995F,
          0.7936722F, 0.69253534F, -0.16264822F, 0.45037717F, 0.31272605F,
          0.41343075F, 0.5840047F, 0.48647887F, 0.61317563F, 0.33861125F,
          0.6925299F, 0.5723155F, 0.54465526F, 0.65076613F, 0.33560473F,
          0.401356F, 0.36191463F, 0.61244863F, 0.8007139F, 0.4618091F,
          -0.00066812336F, 0.69595474F, 0.6417286F, 1.0766675F, 0.41641003F,
          0.7155857F, 0.14677706F, 0.43491682F, 0.48100424F, 0.8486817F,
          0.520069F, 0.58375514F, 0.38570982F, 0.7450143F, 0.62154317F,
          0.4717382F, 0.7299519F, 0.49377933F, 0.269718F, 0.46230453F,
          0.45678782F, 0.41235858F, 0.42238432F, 0.8901968F, 0.4225003F,
          0.7641781F, 0.53354967F, 0.34818593F, 0.315665F, 0.07782623F,
          0.20367357F, 0.45919195F, 0.56325614F, 0.5811453F, 0.7422172F,
          0.3330536F, 0.19774303F, 0.5443491F, 0.29543275F, 0.3499853F,
          0.08449462F, 0.41830593F, 0.31281382F, 0.5532001F, 0.8517419F,
          0.5907302F, 0.6727232F, 0.47809383F, 0.43582827F, 0.43733132F,
          0.4781399F, 0.6652304F, 0.24568659F, 0.38745517F, 0.303826F,
          0.5458253F, 0.48107558F, 0.7486967F, 0.34967697F, 0.49575546F,
          0.37352496F, 1.0111703F, 0.6936064F, 0.5319258F, 0.2715349F,
          0.8292653F, 0.5430071F, 0.57748306F, 0.16449334F, 0.40666884F,
          0.054207772F, 0.508098F, 0.70708746F, 0.3834471F, 0.5630052F, 0.10785F,
          0.22855264F, 0.09184119F, 0.95770895F, 0.28359598F, 0.7201919F,
          0.45746624F, 0.53247887F, 0.23428836F, 0.26926917F, 0.7435518F,
          0.468413F, 0.53498346F, 0.4520031F, -0.2670223F, 0.2960404F,
          0.55386686F, -0.069723055F, 0.24157019F, 0.06936553F, 0.5746465F,
          0.2989141F, 0.51592803F, 0.348845F, 0.4452988F, 0.26290905F,
          0.7541638F, 0.7154373F, 0.4546646F, 0.43493295F, 0.4841744F,
          0.15489957F, -0.09605253F, 0.52397645F, 0.43789315F, 0.21925229F,
          0.12172428F };

        static float s_Weights[4718592];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(s_Weights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_4066879.bin",
            4718592);
        }

        bufferInitialized = true;
        h_convolution(X[0], Z[0], s_Weights[0], biasReformatted[0]);
      }

      void i_conv2dDirectOptimizedColMajor(const float X[86528], float Z[71825])
      {
        static const float biasReformatted[425]{ -1.9480555F, -1.3651499F,
          -1.5816981F, -1.8956414F, -2.2176204F, 0.032695264F, 0.0087539265F,
          -0.053199247F, -0.030290887F, -0.062651485F, 0.06926264F,
          -0.048682258F, 0.02689746F, -0.002783803F, -0.06901134F, -0.19328642F,
          -0.17017925F, -0.022320237F, -0.21200578F, -0.09343791F, -0.17245945F,
          -0.07991317F, -0.15932593F, -0.047651663F, 0.0040927306F, 1.7952265F,
          1.4068031F, 1.0817411F, 0.5935224F, 0.98662746F, -0.023737276F,
          0.02372427F, 0.031785958F, 0.040488355F, -0.031625833F, 0.5847177F,
          0.2741172F, 0.08898675F, 0.19374795F, 0.09816373F, -0.06047286F,
          0.054378156F, 0.06398714F, 0.0380181F, 0.0073607187F, -0.13120006F,
          -0.019504556F, -0.016013872F, 0.06653744F, 0.036734972F, -0.19937003F,
          0.003872914F, 0.04346926F, 0.0009976857F, 0.02036076F, -0.19642223F,
          -0.007553246F, 0.012679886F, 0.07880813F, 0.058084927F, -0.10989624F,
          0.026964989F, 0.08754435F, 0.084883735F, 0.11520302F, 0.11055676F,
          0.122404195F, 0.07283209F, 0.13277449F, 0.13941345F, 0.013485825F,
          -0.0368117F, -0.065779164F, -0.024426397F, -0.052393746F, -0.1680876F,
          -0.13559085F, -0.056386366F, -0.07978364F, -0.09225575F, -0.14678366F,
          -0.15848988F, -0.09164223F, -0.08869468F, -0.088254005F, -0.20495005F,
          -0.137275F, -0.100795805F, -0.080440335F, -0.093972445F, -0.038162045F,
          0.094416276F, 0.044971216F, 0.10402372F, 0.052950967F, 0.11802386F,
          0.17731589F, 0.02051683F, 0.04424537F, 0.05068797F, -0.12054249F,
          0.006205003F, 0.0034110704F, 0.012736656F, 0.08686595F, 0.00742498F,
          0.0026619502F, 0.04339486F, -0.005972187F, 0.0008877425F, -0.0666161F,
          0.039349698F, 0.02187354F, -0.019185657F, -0.013346996F, 0.03700479F,
          0.00095415267F, -0.05951951F, -0.018044349F, -0.051564965F,
          -0.027963534F, 0.003141271F, -0.010344164F, -0.002613193F,
          -0.030260783F, -0.09467803F, -0.030136742F, -0.0013094437F,
          -0.033930063F, -0.023844173F, -0.16847888F, -0.13672349F,
          -0.077673495F, -0.1069015F, -0.08917882F, -0.16507068F, -0.10480974F,
          -0.07832843F, -0.06570297F, -0.09777497F, -0.23581511F, -0.10903256F,
          0.008109663F, -0.06260642F, -0.041843683F, -0.084360376F, -0.13698132F,
          -0.08472816F, -0.06033751F, -0.08256701F, 0.13418849F, 0.1869994F,
          0.08825725F, 0.07264055F, 0.043702126F, 0.11080524F, -0.039893005F,
          -0.06068905F, -0.048103053F, -0.05556773F, -0.04763768F,
          -0.0025368743F, 0.011402233F, -0.014757246F, -0.036717527F,
          -0.06304685F, 0.015112977F, 0.037503798F, 0.04309887F, 0.06553488F,
          -0.026206134F, -0.14394675F, -0.108751416F, -0.06763247F,
          -0.046155855F, -0.05571419F, -0.008184814F, -0.030943833F,
          -0.029328812F, -0.06515375F, -0.15447047F, -0.09375222F, -0.04646788F,
          -0.049404148F, -0.08020679F, 0.061491318F, -0.06794828F, -0.047048353F,
          -0.014745181F, -0.014044154F, 0.027802313F, 0.007995894F,
          0.0054162377F, 0.03294563F, 0.008499023F, -0.065474525F, -0.09469175F,
          -0.072603285F, -0.066252485F, -0.063522264F, -0.07369095F,
          -0.15996468F, -0.11393285F, -0.07612765F, -0.06775801F, 0.013437592F,
          -0.0031697922F, -0.0657453F, -0.049367294F, -0.059509218F,
          -0.020376533F, 0.031100577F, -0.00026798857F, 0.01225319F,
          -0.034833737F, -0.016853958F, -0.011487164F, -0.037512694F,
          -0.034037616F, -0.082199626F, 0.44264036F, 0.14575872F, 0.05011101F,
          0.07749548F, 0.030975755F, 0.006477755F, -0.00934993F, -0.032029167F,
          0.01207309F, -0.04470771F, 0.33609474F, 0.09865874F, 0.011006336F,
          -0.013518767F, 0.04112895F, -0.059069302F, -0.066795304F,
          -0.033203863F, -0.07568248F, -0.07219708F, 0.054297365F, -0.055836823F,
          -0.06706147F, -0.03705599F, -0.07929211F, 0.02106698F, -0.108261116F,
          -0.09385064F, -0.07475822F, -0.0772392F, 0.16863611F, 0.11259829F,
          0.028566904F, 0.07949237F, 0.04536587F, 0.098254845F, 0.04296648F,
          0.06365007F, 0.049177643F, 0.035135463F, -0.009455481F, -0.10073791F,
          -0.05654223F, -0.041749567F, -0.046015296F, -0.09768236F,
          0.0010743344F, -0.028173573F, -0.02869137F, -0.049414136F,
          -0.045091167F, -0.07902088F, -0.055636756F, -0.015397286F,
          -0.052152514F, -0.13928738F, -0.0968626F, -0.038221143F, -0.058692545F,
          -0.07633144F, 0.041612856F, -0.05584022F, -0.016334813F, -0.044311278F,
          -0.043920405F, -0.04873889F, -0.110621996F, -0.06653459F, -0.04431435F,
          -0.0704519F, -0.11877641F, -0.11551304F, -0.05265283F, -0.026820136F,
          -0.016470185F, 0.069674656F, -0.071521886F, -0.052955583F,
          0.0025495207F, -0.0038784903F, 0.011500213F, 0.076070115F,
          0.031900696F, 0.034313396F, 0.047408164F, 0.71038413F, 0.47069183F,
          0.23632357F, 0.14134237F, 0.046333905F, -0.18709716F, 0.013427913F,
          0.0536471F, 0.07595748F, -0.01735254F, 0.0006077765F, 0.09313471F,
          0.10223174F, -0.006706179F, 0.00062060787F, -0.17799775F, -0.07749699F,
          0.0033488977F, 0.08481059F, 0.18508002F, -0.062346328F, 0.05385306F,
          0.06761333F, 0.24584252F, 0.26047227F, -0.1507391F, -0.019166134F,
          0.001106411F, -0.044219486F, 0.005803481F, -0.06769435F,
          -0.0075949645F, -0.046704207F, -0.024451725F, 0.053211503F,
          -0.1046361F, 0.032354742F, 0.04319357F, -0.028303482F, 0.025617128F,
          -0.09459489F, -0.1769694F, -0.10384726F, -0.106223494F, -0.09446549F,
          0.091385975F, -0.10117537F, -0.060719565F, -0.0694706F, -0.07361901F,
          -0.14059028F, -0.11264806F, -0.06646977F, -0.016871091F, -0.042289086F,
          0.15778965F, -0.021842636F, -0.051749066F, -0.059011962F,
          -0.034805786F, -0.18926404F, -0.13611737F, -0.101440415F, -0.06269287F,
          -0.02632052F, -0.15350708F, -0.038293257F, 0.013854964F,
          -0.00086625986F, -0.0074965525F, -0.21669675F, -0.15774247F,
          -0.085288145F, -0.081450485F, -0.10134044F, -0.05807593F,
          -0.025647707F, -0.060792193F, 0.009079577F, -0.033848286F,
          -0.20325835F, -0.07136283F, 0.044948906F, -0.052025475F, 0.061705016F,
          0.354334F, 0.14401802F, 0.12125587F, 0.15007278F, 0.10771623F,
          0.016935503F, -0.06738074F, -0.062384374F, -0.07127104F, -0.067277394F,
          -0.023327062F, 0.02352032F, -0.039356757F, -0.074384764F,
          -0.016498499F, -0.10444278F, -0.047821403F, -0.024093442F,
          -0.059033226F, -0.06717279F, -0.05896447F, 0.059215907F, 0.031235766F,
          -0.043259945F, 0.028124489F, -0.23344113F, -0.1580798F, -0.089384764F,
          -0.076822974F, -0.065848365F, -0.08497436F, -0.116645195F,
          -0.060001373F, -0.07748261F, -0.0708429F };

        static float reformattedAndTruncatedWeights[217600];
        static bool bufferInitialized;
        if (!bufferInitialized) {
          readDnnConstants_real32_T(reformattedAndTruncatedWeights[0],
            "./codegen/lib/yolov2_detect//largeDnnConstants_3680689.bin", 217600);
        }

        bufferInitialized = true;
        i_convolution(X[0], Z[0], reformattedAndTruncatedWeights[0],
                      biasReformatted[0]);
      }
    }
  }
}

// End of code generation (conv2dDirectOptimizedColMajor.cpp)
