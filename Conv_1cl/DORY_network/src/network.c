/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
 *
 * Copyright (C) 2019-2020 University of Bologna
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
#include "mem_controller.h"
#include "network.h"
#include "dory.h"
#include "layerConv1.h"
#include "layerGemm2_last.h"
#include "layerConv0.h"
#include "snrt.h"
#include "printf.h"
#include "data.h"


// allocation of buffers with parameters needed by the network execution
const float * Weights_tensors[] = {
  Conv0_weights, Conv1_weights, Gemm2_weights
};

layer layer_i;
__attribute__((section(".data"))) int layers_pointers[3];
__attribute__((section(".data"))) char * Layers_name[3] = {"Conv", "Conv", "Gemm"};
__attribute__((section(".data"))) int L3_layers[3] = {0, 0, 0};
__attribute__((section(".data"))) int L3_input_layers[3] = {0, 0, 0};
__attribute__((section(".data"))) int L3_output_layers[3] = {0, 0, 0};
__attribute__((section(".data"))) int L3_weights_layers[3] = {0, 0, 0};
__attribute__((section(".data"))) int allocate_layer[3] = {1, 1, 1};
__attribute__((section(".data"))) int branch_input[3] = {0, 0, 0};
__attribute__((section(".data"))) int branch_output[3] = {0, 0, 0};
__attribute__((section(".data"))) int branch_change[3] = {0, 0, 0};
__attribute__((section(".data"))) int branch_last[3] = {0, 0, 0};
__attribute__((section(".data"))) int check_weights[3] = {344, -10059, -2619};
__attribute__((section(".data"))) int check_weights_dimension[3] = {128, 36864, 512};
__attribute__((section(".data"))) int cumulative_weights_dimension[3] = {0, 128, 36992};
__attribute__((section(".data"))) int check_activations[3] = {62827, 2501746, 5074622};
__attribute__((section(".data"))) int check_activations_dimension[3] = {1024, 65536, 64};
__attribute__((section(".data"))) int check_activations_dimension_L3_in[3] = {0, 0, 0};
__attribute__((section(".data"))) int check_activations_dimension_L3_out[3] = {0, 0, 0};
__attribute__((section(".data"))) int out_mult_vector[3] = {1, 1, 1};
__attribute__((section(".data"))) int out_shift_vector[3] = {5, 9, 0};
__attribute__((section(".data"))) int inmul1_vector[3] = {0, 0, 0};
__attribute__((section(".data"))) int inmul2_vector[3] = {0, 0, 0};
__attribute__((section(".data"))) int check_activations_out[3] = {2501746, 5074622, 0};
__attribute__((section(".data"))) int check_activations_out_dimension[3] = {65536, 65536, 32};
__attribute__((section(".data"))) int layer_with_weights[3] = {1, 1, 1};
__attribute__((section(".data"))) int NODEs_MACS[3] = {65536, 37748736, 512};

#ifdef VERBOSE
// check for input/output acitvation checksum
static void check_layer(char *output, int check_sum_true, int dim) {
  int checksum = 0;
  float *ptr = (float *) output;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else 
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}

#endif 


uint32_t L2_weights, L2_output, L2_input_add, L2_input;
__attribute__((section(".data"))) float L2_output_mem[1000000];
__attribute__((section(".data"))) float L2_input_add_mem[1000000];
__attribute__((section(".data"))) float L2_input_mem[1000000];
void network_run()
{   

/* 
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  uint16_t out_mult = 0;
  uint16_t out_shift = 0;
  uint16_t inmul1 = 0;
  uint16_t inmul2 = 0;
#ifdef VERBOSE
  if (snrt_cluster_compute_core_idx()==0)
    printf("I'm Core 0 from Occamy Cluster. Beginning the neural network computation\n");
#endif
/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */ 
/* ---------------------------------- */ 

  // perf measurement begin
  int cycle_network_execution = 0;
/* MAIN SECTION
  - for loop over all the layers of the network
  - double buffering using L3
  - check on layers to be executed from L3
  - residual check at the end of each layer
*/
/* ---------------------------------- */
/* -------- SECTION 2 BEGIN --------- */
/* ---------------------------------- */
  int j = 0;
  for(int i = 0; i < 3; i++)
  {
    if (layer_with_weights[i] == 1)
    {
      L2_weights = (uint32_t) Weights_tensors[j];
      j++;
    }
    if (i == 0)
    {
      L2_input = input;
      L2_output = L2_output_mem;
    }
    else if ( i % 2 == 0)
    {
      L2_input = L2_input_mem;
      L2_output = L2_output_mem;
    }
    else
    {
      L2_input = L2_output_mem;
      L2_output = L2_input_mem;
    }
#ifdef VERBOSE
    if(snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      if (i==0)
        check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      else if (branch_change[i-1]==0)
        check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      else
        printf("Switching branch, already checked activation\n");
    }
#endif  
    snrt_global_barrier();
    layer_i.L2_input = L2_input;
    layer_i.L2_input_add = L2_input_add;
    layer_i.L2_output = L2_output;
    layer_i.L2_weights = L2_weights;
    layer_i.l2_zeros = l2_zeros;
    layer_i.out_mult = out_mult_vector[i];
    layer_i.out_shift = out_shift_vector[i];
    layer_i.inmul1 = inmul1_vector[i];
    layer_i.inmul2 = inmul2_vector[i];
    switch (i)
    {
      case 0:
        break;
        benchmark_get_cycle();
        layerConv0(&layer_i);
        benchmark_get_cycle();
        break;
      case 1:
        benchmark_get_cycle();
        layerConv1(&layer_i);
        benchmark_get_cycle();
        break;
      case 2:
        break;
        benchmark_get_cycle();
        layerGemm2_last(&layer_i);
        benchmark_get_cycle();
        break;
    }
    snrt_global_barrier();

#ifdef VERBOSE
    if(snrt_cluster_compute_core_idx() == 0 && snrt_cluster_idx() == 0)
    {
      printf("Layer %s %d ended: \n", Layers_name[i], i);
      check_layer(L2_output, check_activations_out[i], check_activations_out_dimension[i]);
      
      if (i==100)
      {    
        check_layer_plus(L2_output,check_activations_out_dimension[i]);
      }
    }    
    snrt_global_barrier();
#endif 

  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */
#ifdef VERBOSE
  int cid = snrt_cluster_compute_core_idx();    
  int MACs = 37814784;
  if (cid == 0)
  {
    printf("[%d] : Total MACs: %d\n",cid,MACs ); 
  }
#endif
/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

