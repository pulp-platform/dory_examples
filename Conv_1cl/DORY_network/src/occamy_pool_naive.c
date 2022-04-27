/*
 * pulp_nn_conv_Ho_parallel.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "occamy_nn_utils.h"
#include "occamy_nn_kernels.h"
#include "printf.h"

void __attribute__ ((noinline)) occamy_pool_naive(
  kernel * kernel_i
) {
  ////// NAIVE CONVOLUTION ////////////
  int input_x_index, input_y_index, input_index, output_index;
  float max;
  //// Extension of input image with padding
  for (int z = 0; z < kernel_i->dim_out_y; z++)
  {
    for (int j = 0; j < kernel_i->dim_out_x; j++)
    {
      for (int i = 0; i < kernel_i->ch_out; i++)
      {
        max = 0;
        for (int n = 0; n < kernel_i->dim_kernel_x; n++)
        {
          for (int t = 0; t < kernel_i->dim_kernel_y; t++)
          {
            input_x_index = j * kernel_i->stride_x - kernel_i->padding_x_left + n;
            input_y_index = z * kernel_i->stride_y - kernel_i->padding_y_top + t;
            input_index = i + input_x_index * kernel_i->ch_in + input_y_index * kernel_i->dim_in_x * kernel_i->ch_in;
            if (input_x_index >= 0 && input_y_index >= 0 && input_x_index < kernel_i->dim_in_x && input_y_index < kernel_i->dim_in_y)
              if (kernel_i->pInBuffer[input_index] > max)
                max = kernel_i->pInBuffer[input_index];
          }
        }
        output_index = i + j * kernel_i->ch_out + z * kernel_i->ch_out * kernel_i->dim_out_x;
        kernel_i->pOutBuffer[output_index] = max;
      }
    }
  }
}