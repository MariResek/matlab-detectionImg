//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// main.cpp
//
// Code generation for function 'main'
//

/*************************************************************************/
/* This automatically generated example C++ main file shows how to call  */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

// Include files
#include "main.h"
#include "rt_nonfinite.h"
#include "yolov2_detect.h"
#include "yolov2_detect_initialize.h"
#include "yolov2_detect_terminate.h"
#include <cstring>

// Function Declarations
static void argInit_720x1280x3_uint8_T(unsigned char result[2764800]);

static unsigned char argInit_uint8_T();

// Function Definitions
static void argInit_720x1280x3_uint8_T(unsigned char result[2764800])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 720; idx0++) {
    for (int idx1{0}; idx1 < 1280; idx1++) {
      for (int idx2{0}; idx2 < 3; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 720 * idx1) + 921600 * idx2] = argInit_uint8_T();
      }
    }
  }
}

static unsigned char argInit_uint8_T()
{
  return 0U;
}

int main(int, char **)
{
  // Initialize the application.
  // You do not need to do this more than one time.
  yolov2_detect_initialize();
  // Invoke the entry-point functions.
  // You can call entry-point functions multiple times.
  main_yolov2_detect();
  // Terminate the application.
  // You do not need to do this more than one time.
  yolov2_detect_terminate();
  return 0;
}

void main_yolov2_detect()
{
  static double bboxes_data[3380];
  static unsigned char uv[2764800];
  float scores_data[845];
  unsigned int labels_data[845];
  int bboxes_size[2];
  int labels_size;
  int scores_size;
  // Initialize function 'yolov2_detect' input arguments.
  // Initialize function input argument 'in'.
  // Call the entry-point 'yolov2_detect'.
  argInit_720x1280x3_uint8_T(uv);
  yolov2_detect(uv, bboxes_data, bboxes_size, scores_data, &scores_size,
                labels_data, &labels_size);
}

// End of code generation (main.cpp)
