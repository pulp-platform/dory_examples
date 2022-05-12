/*
 * Copyright (C) 2020 ETH Zurich, University of Bologna and GreenWaves Technologies
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

/*
 * Authors:  Francesco Conti <fconti@iis.ee.ethz.ch>
 *           Gianna Paulin <pauling@iis.ee.ethz.ch>
 *           Renzo Andri <andrire@iis.ee.ethz.ch>
 * Main Test Program for the NE16
 */


#include <stdio.h>

#include "pulp_nnx.h"
#include "ne16_test.h"
#include "generated_ne16_tests.h"

#define L1_BUFFER_SIZE 35000
void *l1_mem_buffer;

void single_ne16_test(ne16_test_t test);

int run_test() {
  l1_mem_buffer = pmsis_l1_malloc(L1_BUFFER_SIZE);

  for (int i = 0; i < N_GENERATED_NE16_TESTS; i++) {
    printf("[Test %d] ", i + 1);
    single_ne16_test(ne16_tests[i]);
  }

  return 0;
}

void single_ne16_test(ne16_test_t test) {
  nnx_task_t nnx_task;

  nnx_weights_t nnx_weights = {
    .data = NULL,
    .height = test.fs,
    .width = test.fs,
    .depth = test.k_in,
    .n_weights = test.k_out,
    .bitwidth = test.qw,
    .offset_factor = -128,
    .offset_mode = weightOffsetModeLayerWise
  };

  nnx_feature_t nnx_input = {
    .data = NULL,
    .height = test.h_in,
    .width = test.w_in,
    .depth = test.k_in,
    .bitwidth = featureBitwidth8Bit
  };

  nnx_feature_t nnx_output = {
    .data = NULL,
    .height = test.h_in - (test.fs - 1),
    .width = test.w_in - (test.fs - 1),
    .depth = test.k_out,
    .bitwidth = featureBitwidth8Bit
  };

  const nnx_norm_t norm = {
    .mode  = normMode32Bit,
    .flag_bias  = FLAG_USED,
    .flag_shift = FLAG_UNUSED
  };

  const nnx_quant_t quant = {
    .shift_amount = 12,
    .mode = quantMode8Bit,
    .function = quantFunctionRelu,
    .flag_rounding = FLAG_UNUSED
  };

  nnx_task_init(&nnx_task);
  if (test.fs == 1) {
    nnx_conv_1x1(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output);
  } else {
    if (test.dw == 0) {
      nnx_conv_3x3(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output);
    } else {
      nnx_conv_3x3_dw(&nnx_task.cfg, nnx_weights, nnx_input, nnx_output);
    }
  }
  nnx_norm_quant(&nnx_task.cfg, norm, quant);

  unsigned int l1_mem = (unsigned int)l1_mem_buffer;

  nnx_task.infeat_ptr = l1_mem;
  l1_mem += test.h_in * test.w_in * test.k_in;
  nnx_task.weights_ptr = l1_mem;
  l1_mem += test.k_out * DIVNCEIL(test.k_in, NE16_INPUT_CHANNEL_THROUGHPUT) * test.qw * test.fs * test.fs * 2;
  nnx_task.scale_ptr = l1_mem;
  l1_mem += test.k_out * 4 /* activation data element size */;
  nnx_task.scale_bias_ptr = l1_mem;
  l1_mem += test.k_out * 4 /* activation data element size */;
  nnx_task.outfeat_ptr = l1_mem;

  nnx_soft_clear();
  nnx_acquire();
  nnx_offload(&nnx_task);
  nnx_commit();
  nnx_acquire();
  nnx_offload(&nnx_task);

  pi_perf_conf(1<<PI_PERF_CYCLES);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start();

  nnx_run_async();
  nnx_busywait();

  pi_perf_stop();

  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES) / 2; // divide by 2 since we ran 2 tests on the accelerator

  char dw_str[] = "depthwise ";

  if (test.dw == 0) dw_str[0] = 0;

  printf("Tile (%dx%dx%dx%d) with %s%dx%d convolution performance: %d cycles\n",
      test.k_out, test.k_in, test.h_in, test.w_in, dw_str, test.fs, test.fs, perf_cyc);

  nnx_soft_clear();
}