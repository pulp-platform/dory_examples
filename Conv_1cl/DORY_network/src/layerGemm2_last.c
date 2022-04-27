/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
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

#include "layerGemm2_last.h"

#define VERBOSE_PRINT(...) printf(__VA_ARGS__)


void layerGemm2_last(layer* layer_i) 
{

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  ////////////// VARIABLE DECLARATION AND INITIALIZATION //////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  unsigned int l2_x =         layer_i->L2_input;
  unsigned int l2_x_2 =       layer_i->L2_input_add;
  unsigned int l2_y =         layer_i->L2_output;
  unsigned int l2_W =         layer_i->L2_weights;
  unsigned int l2_zeros =     layer_i->l2_zeros;
  
  volatile kernel kernel_i;
  int CLUSTERS = 1;
  volatile DMA_copy DMA_copy_k, DMA_copy_lambda, DMA_copy_W, DMA_copy_x, DMA_copy_y, DMA_copy_p_top, DMA_copy_p_bottom, DMA_copy_p_left, DMA_copy_p_right, DMA_copy_bias;
  // Memory allocation
  float *memory_cluster = (float *)snrt_cluster_memory().start;
  unsigned int l1_buffer = (unsigned int) memory_cluster;
  volatile unsigned short x_length_nif_byte;
  // Weights Parameters
  volatile unsigned short  W_tile_size_nof, W_length_nif_byte, W_length_nif_byte_last;
  // Input Parameters for execution
  // Output Parameters for execution
  volatile int y_tile_size_nof, y_tile_size_h, y_tile_size_w, y_length_nof_byte;
  // Double Buffering parameters and states
  volatile int db_x, db_W, db_act, db_y, exec_db_x, exec_db_W, exec_db_act;
  int db_state_x=0, db_state_W=0, db_state_y=1;
  // tile loop indeces
  int iter, _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0, _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  //////////////// FIRST TILE TRANSFERING OF ALL TENSORS //////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  ////////////////////// BEGIN DMA DEDICATED SECTION //////////////////////
  if (snrt_is_dm_core())
  {
    x_length_nif_byte = 256;
    DMA_copy_k.hwc_to_chw = 0;
    DMA_copy_k.stride_2d = 0;
    DMA_copy_k.stride_1d = 0;
    DMA_copy_k.dir = 1;
    DMA_copy_k.dma_channel = NULL;

    DMA_copy_lambda.hwc_to_chw = 0;
    DMA_copy_lambda.stride_2d = 0;
    DMA_copy_lambda.stride_1d = 0;
    DMA_copy_lambda.dir = 1;
    DMA_copy_lambda.dma_channel = NULL;
    
    DMA_copy_p_top.ext = l2_zeros;
    DMA_copy_p_top.hwc_to_chw = 0;
    DMA_copy_p_top.stride_2d = 256;
    DMA_copy_p_top.stride_1d = 256;
    DMA_copy_p_top.dir = 1;
    DMA_copy_p_top.dma_channel = NULL;

    DMA_copy_p_bottom.ext = l2_zeros;
    DMA_copy_p_bottom.hwc_to_chw = 0;
    DMA_copy_p_bottom.stride_2d = 256;
    DMA_copy_p_bottom.stride_1d = 256;
    DMA_copy_p_bottom.dir = 1;
    DMA_copy_p_bottom.dma_channel = NULL;

    DMA_copy_p_left.ext = l2_zeros;
    DMA_copy_p_left.hwc_to_chw = 0;
    DMA_copy_p_left.stride_2d = 0;
    DMA_copy_p_left.stride_1d = 256;
    DMA_copy_p_left.dir = 1;
    DMA_copy_p_left.dma_channel = NULL;

    DMA_copy_p_right.ext = l2_zeros;
    DMA_copy_p_right.hwc_to_chw = 0;
    DMA_copy_p_right.stride_2d = 0;
    DMA_copy_p_right.stride_1d = 256;
    DMA_copy_p_right.dir = 1;
    DMA_copy_p_right.dma_channel = NULL;

    DMA_copy_x.hwc_to_chw = 0;
    DMA_copy_x.stride_2d = 256;
    DMA_copy_x.stride_1d = 256;
    DMA_copy_x.dir = 1;
    DMA_copy_x.dma_channel = NULL;
    
    DMA_copy_W.hwc_to_chw = 0;
    DMA_copy_W.stride_2d = 256;
    DMA_copy_W.stride_1d = 256;
    DMA_copy_W.dir = 1;
    DMA_copy_W.dma_channel = NULL;
    
    DMA_copy_y.hwc_to_chw = 0;
    DMA_copy_y.stride_2d = 32;
    DMA_copy_y.stride_1d = 32;
    DMA_copy_y.dir = 0;
    DMA_copy_y.dma_channel = NULL;

    if (0 > 0)
    {
      DMA_copy_p_top.loc = l1_buffer + (0 + 0);
      DMA_copy_p_top.number_of_2d_copies = 1;
      DMA_copy_p_top.number_of_1d_copies = 1;
      DMA_copy_p_top.length_1d_copy = 0*x_length_nif_byte*(0 + 0*(1==1) + 1);
      dory_dma_memcpy_async(DMA_copy_p_top);
    }

    if (1==1 && 0 > 0)
    {
      DMA_copy_p_bottom.loc = l1_buffer + (0 + 0) + x_length_nif_byte*1 + 0*x_length_nif_byte*(0 + 0*(1==1) + 1) + x_length_nif_byte*0*1 + x_length_nif_byte*0*1*(1==1);
      DMA_copy_p_bottom.number_of_2d_copies = 1;
      DMA_copy_p_bottom.number_of_1d_copies = 1;
      DMA_copy_p_bottom.length_1d_copy = 0*x_length_nif_byte*(0 + 0*(1==1) + 1);
      dory_dma_memcpy_async(DMA_copy_p_bottom);
    }

    if (0 > 0)
    {
      DMA_copy_p_left.loc = l1_buffer + (0 + 0) + 0*x_length_nif_byte*(0 + 0*(1==1) + 1);
      DMA_copy_p_left.number_of_2d_copies = 1;
      DMA_copy_p_left.number_of_1d_copies = 1;
      DMA_copy_p_left.length_1d_copy = 0*x_length_nif_byte;
      DMA_copy_p_left.stride_L1_1d = x_length_nif_byte*(0 + 0*(1==1) + 1);
      dory_dma_memcpy_async(DMA_copy_p_left);
    }

    if (1==1 && 0 > 0)
    {
      DMA_copy_p_right.loc = l1_buffer + (0 + 0) + 0*x_length_nif_byte*(0 + 0*(1==1) + 1) + x_length_nif_byte*(0 + 1);
      DMA_copy_p_right.number_of_2d_copies = 1;
      DMA_copy_p_right.number_of_1d_copies = 1;
      DMA_copy_p_right.length_1d_copy = 0*x_length_nif_byte;
      DMA_copy_p_right.stride_L1_1d = x_length_nif_byte*(0 + 0 + 1);
      dory_dma_memcpy_async(DMA_copy_p_right);
    }

    DMA_copy_x.ext = l2_x;
    DMA_copy_x.loc = l1_buffer + (0 + 0) + 0*x_length_nif_byte*(0 + 0*(1==1) + 1) + x_length_nif_byte*0;
    DMA_copy_x.number_of_2d_copies = 1;
    DMA_copy_x.number_of_1d_copies = 1;
    DMA_copy_x.length_1d_copy = x_length_nif_byte;
    DMA_copy_x.stride_L1_1d = DMA_copy_x.length_1d_copy;
    DMA_copy_x.stride_L1_2d = DMA_copy_x.length_1d_copy*DMA_copy_x.number_of_1d_copies + x_length_nif_byte*(0 + 0*(1==1));
    dory_dma_memcpy_async(DMA_copy_x);

    int cl = 0;
    W_length_nif_byte = ((cl+snrt_cluster_idx()) % CLUSTERS  == (1-1)) ? 256 : 256; 
    DMA_copy_W.ext = l2_W +2048*1*snrt_cluster_idx();
    DMA_copy_W.loc = l1_buffer + (320 + 0);
    DMA_copy_W.number_of_2d_copies = 8;
    DMA_copy_W.number_of_1d_copies = 1;
    DMA_copy_W.length_1d_copy = (int)W_length_nif_byte/1;
    DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
    DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy*DMA_copy_W.number_of_1d_copies;
    dory_dma_memcpy_async(DMA_copy_W);
  }
  ////////////////////// END DMA DEDICATED SECTION ////////////////////////
  dory_global_barrier();
  float sum = 0;

  int total_tiles = 1;
  total_tiles+= 0;


  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  ////////////////// LOOP OF COMPUTATION OVER ALL TILES ///////////////////
  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  for(iter=0; iter < total_tiles; iter++) 
  {
    _i_w_load += 1;
    if(_i_w_load==1) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==1) 
      {
        _i_h_load = 0;
        _i_nof_load += 1;
      }
    }

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 272 : 0;
    db_W = !db_state_W ? 2048 : 0;
    db_y = !db_state_y ? 32 : 0;
    exec_db_x = 0;
    exec_db_W = db_state_W ? 2048 : 0;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single pixel
    db_state_x = ! db_state_x;
    if (_i_nof_load!=_i_nof_exec)
    {
      db_state_W = ! db_state_W;
    }

    ///////////////////// BEGIN DMA DEDICATED SECTION ///////////////////////
    if (snrt_is_dm_core())
    {
      if(iter < (total_tiles-1) )
      {
        asm volatile("": : :"memory");
        W_tile_size_nof = (_i_nof_load+1 == 1) ? 8 : 8;

        // transfer of next weight tile if changed input or output channels
        if (_i_nof_load!=_i_nof_exec && (iter <  1 || snrt_cluster_idx() == (CLUSTERS-1)))
        {
          int cl = 0;
          W_length_nif_byte = ((cl+_i_nif_load+snrt_cluster_idx()) % CLUSTERS  == (1-1)) ? 256 : 256; 
          DMA_copy_W.ext = dory_get_tile_3d(l2_W,  _i_nof_load + snrt_cluster_idx() * 1, 0, 0, 8, 1*1, 64, 1*1, 64, 0,0,0,0,0,0, 32);
          DMA_copy_W.loc = l1_buffer + (320 + db_W);
          DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
          DMA_copy_W.length_1d_copy = (int) W_length_nif_byte;
          DMA_copy_W.stride_L1_1d = DMA_copy_W.length_1d_copy;
          DMA_copy_W.stride_L1_2d = DMA_copy_W.length_1d_copy * DMA_copy_W.number_of_1d_copies;
          dory_dma_memcpy_async(DMA_copy_W);
        }
      }
    }
    ////////////////////// END DMA DEDICATED SECTION ////////////////////////

    /////////////////// BEGIN COMPUTE DEDICATED SECTION /////////////////////
    if (snrt_is_compute_core())
    {
    // CREATING POINTERS TO TENSORS TO PASS TO THE KERNEL
      // SETTING PARAMETERS TO PASS TO THE KERNEL FUNCTION
      y_tile_size_h   = (_i_h_exec+1 == 1) ? 1 : 1;
      y_tile_size_w   = (_i_w_exec+1 == 1) ? 1 : 1;
      y_length_nof_byte = (_i_nof_exec+1 == 1)   ? 32 : 32;
      y_tile_size_nof = (_i_nof_exec+1 == 1) ? 8 : 8;
      asm volatile("": : :"memory");
      kernel_i.pInBuffer = (float *) (l1_buffer + (0 + exec_db_x));
      kernel_i.dim_in_x = (_i_w_exec+1 == 1) ? 1 : 1;
      kernel_i.dim_in_y = (_i_h_exec+1 == 1) ? 1 : 1;
      kernel_i.ch_in = 64;
      kernel_i.pWeight = (float *) (l1_buffer + (320 + exec_db_W));
      kernel_i.ch_out = y_tile_size_nof;
      kernel_i.dim_kernel_x = 1;
      kernel_i.dim_kernel_y = 1;
      kernel_i.padding_y_top = (_i_h_exec == 0) ? 0 : 0;
      kernel_i.padding_y_bottom = (_i_h_exec == 1-1) ? 0 : 0;
      kernel_i.padding_x_left = (_i_w_exec == 0) ? 0 : 0;
      kernel_i.padding_x_right = (_i_w_exec == 1-1) ? 0 : 0;
      kernel_i.stride_x = 1;
      kernel_i.stride_y = 1;
      kernel_i.bias = NULL;
      kernel_i.bias_shift = 0;
      kernel_i.out_shift = layer_i->out_shift;
      kernel_i.out_mult = 0;
      kernel_i.pOutBuffer = (float *) (l1_buffer + (280 + db_y));
      kernel_i.dim_out_x = y_tile_size_w;
      kernel_i.dim_out_y = y_tile_size_h;
      kernel_i.kappa = 0;
      kernel_i.lambda = 0;
      kernel_i.pIm2ColBuffer = (float *) (l1_buffer + 2392);
      kernel_i.flag_relu = 0 && (_i_nif_load==0);
      kernel_i.flag_batch_norm = 0 && (_i_nif_load==0);
      kernel_i.flag_y_accumulate_start = (_i_nif_exec==0);
      kernel_i.flag_y_accumulate_end = (_i_nif_load==0);
      kernel_i.memory_chan = NULL;
      //occamy_conv_naive
      //occamy_conv_opt_fp32
      //occamy_conv_chw_opt_fp32
      //occamy_conv_dw_opt_fp32

      if (iter <  1 || snrt_cluster_idx() == (CLUSTERS-1))
      {
	    occamy_conv_opt_fp32(&kernel_i);
      }
      else
      {
        snrt_cluster_hw_barrier();
      }
    }
    else
    {
      snrt_cluster_hw_barrier();
    }

    /////////////////// END COMPUTE DEDICATED SECTION /////////////////////
    dory_global_barrier();
	  /*
	  printf("Tile %d [h,w,c] indexes [%d,%d,%d] dimensions [%d, %d, %d] y: \n", iter, _i_h_exec, _i_w_exec, _i_nof_exec, y_tile_size_h, y_tile_size_w, y_tile_size_nof);
	  sum=0;
	  for (int i = 0; i < (y_tile_size_nof*y_tile_size_h*y_tile_size_w); i++)
	    sum+=*(y+i);
	  printf("%f ", sum);
	  printf("\n");
	  */  
    if (snrt_is_dm_core())
    {
        y_tile_size_h   = (_i_h_exec+1 == 1) ? 1 : 1;
        y_tile_size_w   = (_i_w_exec+1 == 1) ? 1 : 1;
        y_length_nof_byte = (_i_nof_exec+1 == 1)   ? 32 : 32;
	      if (iter <  1 || snrt_cluster_idx() == (CLUSTERS-1))
	      {
    	    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec + snrt_cluster_idx() * 1, 1, 1, 8, 1, 8, 0, 0, 0, 0, 0, 0, 32);
    	    DMA_copy_y.loc = l1_buffer + (280 + db_y);
    	    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    	    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    	    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    	    dory_dma_memcpy_async(DMA_copy_y); 
    	  }
    }
    // update prev iterators
      db_state_y = ! db_state_y;   
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }

  // wait for final write
  dory_global_barrier();
}
