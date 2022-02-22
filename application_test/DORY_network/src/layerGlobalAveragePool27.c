/*
 * pooling_layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
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
// first_layer                    0
// sdk                            gap_sdk
// number_of_clusters             1
// dma_parallelization            8-cores
// optional_type                  mixed-sw
// func_name                      layerGlobalAveragePool27
// flag_DW                        0
// optional                       GlobalAveragePool
// FLAG_BATCHNORM                 0
// has_bias                       1
// FLAG_RELU                      0
// test_location                  L3
// platform                       GAP8
// chip                           GAP8v3
// type                           char
// nof                            768
// factor                         1
// g                              1
// nif                            768
// conv_overlap1                  6
// conv_overlap2                  6
// padding_top                    0
// padding_bottom                 0
// padding_left                   0
// padding_right                  0
// stride                         1
// x_h                            7
// x_w                            7
// x_data_size_byte               4
// x_tile_size_nif                768
// x_tile_size_h                  7
// x_tile_size_w                  7
// x_tile_size_byte               18816
// x_tile_size_nif_byte           384
// x_stride_w_byte                2688
// x_stride_c_byte                384
// y_h                            1
// y_w                            1
// y_data_size_byte               4
// act_dim_bit                    0
// y_tile_size_nof                768
// y_tile_size_h                  1
// y_tile_size_w                  1
// y_tile_size_byte               384
// y_stride_w_byte                384
// y_stride_c_byte                384
// y_tile_size_nof_byte           384
// tile_dim_h                     1
// tile_dim_w                     1
// tile_dim_nof                   1
// tile_dim_nif                   1
// tile_n_in_last                 768
// fs1                            7
// fs2                            7
// W_data_size_byte               0
// W_tile_size_nof                768
// b_size_byte                    0
// W_tile_size_nif                768
// W_tile_size_nif_last           768
// W_tile_size_byte               0
// W_stride_nof_byte              0
// W_stride_hw_byte               0
// W_tile_nif_byte                0
// W_tile_nif_byte_last           0
// l2_off_bias                    0
// k_tile_size_byte               0
// lambda_tile_size_byte          0
// k_size_byte                    0
// lambda_size_byte               0
// l1_x_offset                    0
// l1_y_offset                    18824
// x_tile_size_nif_last           768
// x_tile_size_nif_byte_last      384
// x_tile_size_h_last             7
// x_tile_size_w_last             7
// y_tile_size_nof_last           768
// y_tile_size_h_last             1
// y_tile_size_w_last             1
// y_length_nof_byte_last         384


#include "layerGlobalAveragePool27.h"
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
void layerGlobalAveragePool27(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];
  unsigned int inmul1 = (unsigned int) real_arg[10];
  unsigned int inmul2 = (unsigned int) real_arg[11];
  unsigned int out_shift_in = (unsigned int) real_arg[12];
  int p_r, p_l, p_t, p_b;
  int last_nof_exec;
  int last_nif_exec;
  int last_h_exec;
  int last_w_exec;
  char *x;
  char *y;
  int x_tile_size_nif_exec;
  int x_tile_size_h_exec;
  int x_tile_size_w_exec;
  int y_tile_size_nof;
  int y_tile_size_h;
  int y_tile_size_w;
  int y_tile_size_byte;
  int y_length_h_px;
  int y_length_nof_byte;
  int db_x;
  int db_y;
  int exec_db_x;
  int exec_db_W;
 char *im2col;
  im2col = l1_buffer + 19240;
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_x, DMA_copy_y;
  // copy first tiles
  //l2_x has input activations

  DMA_copy_x.stride_2d = 2688;
  DMA_copy_x.stride_1d = 384;
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = 384;
  DMA_copy_y.stride_1d = 384;
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = dory_dma_channel;

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + 0) + 0;
  DMA_copy_x.number_of_2d_copies = 7;
  DMA_copy_x.number_of_1d_copies = 7;
  DMA_copy_x.length_1d_copy = 384;
  dory_dma_memcpy_async(DMA_copy_x);
  dory_dma_barrier(DMA_copy_x);
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

  // double buffering state
  int db_state_x=0;
  int db_state_y=1;
  int db_state_acc_out=1;
  int flag_first_ch_out;

  // last-tile flags
  int last_nof_load = (1 == 1) ? 1 : 0;
  int last_nif_load = (1 == 1) ? 1 : 0;
  int last_h_load = (1 == 1) ? 1 : 0;
  int last_w_load = (1 == 1) ? 1 : 0;
  int iter;
  // tile loop nest
  for(iter=0; iter<1*1*1; iter++) {
    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==1) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==1) 
      {
        _i_h_load = 0;
        _i_nif_load += 1;
        _i_nof_load += 1;
      }
    }
    if (_i_nof_exec==0)
      flag_first_ch_out = 1;
    else
      flag_first_ch_out = 0;
    // wait for x,W read
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_w_exec = last_w_load;
    last_nof_load = (_i_nof_load+1 == 1) ? 1 : 0;
    last_nif_load = (_i_nof_load+1 == 1) ? 1 : 0;
    last_h_load = (_i_h_load+1 == 1) ? 1 : 0;
    last_w_load = (_i_w_load+1 == 1) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 18816 : 0;
    db_y = !db_state_y ? 384 : 0;
    exec_db_x = 0;
    db_state_x = ! db_state_x;

    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single filter_out
    db_state_y = ! db_state_y;
    if(iter<1*1*1-1) 
    {
      y_tile_size_h   = (last_h_load)   ? 1 : 1;
      y_tile_size_w   = (last_w_load)   ? 1 : 1;
    }
    x = (char *) (l1_buffer + 0 + exec_db_x);
    y = (char *) (l1_buffer + 18824 + db_y);
   
    x_tile_size_nif_exec = (last_nif_exec) ? 768 : 768;
    x_tile_size_h_exec   = (last_h_exec)   ? 7 : 7;
    x_tile_size_w_exec   = (last_w_exec)   ? 7 : 7;
  
    y_tile_size_nof = (last_nof_exec) ? 768 : 768;
    y_tile_size_h   = (last_h_exec)   ? 1 : 1;
    y_tile_size_w   = (last_w_exec)   ? 1 : 1;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*4/8;
    y_length_nof_byte = (last_nof_exec)   ? 384 : 384;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = 0;
    if (_i_w_exec == 0)
      p_l = 0;
    if (_i_h_exec == 1-1)
      p_b = 0;
    if (_i_w_exec == 1-1)
      p_r = 0;
    dory_cores_barrier();
  
// aggiungere padding su tutti i lati, acc_out, and filter asymettric
    pulp_nn_avgpool_u4(
    x,
    x_tile_size_w_exec,
    x_tile_size_nif_exec,
    7,
    p_t,
    1,
    y_tile_size_w,
    y,
    im2col
    );
    dory_cores_barrier();
    dory_dma_barrier(DMA_copy_x);
    dory_dma_barrier(DMA_copy_y);
    // transfering of output to L2
    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 1, 1, 768, 1, 768, 0, 0, 0, 0, 0, 0, 4);
    DMA_copy_y.loc = (l1_buffer + 18824) + db_y;
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    dory_dma_memcpy_async(DMA_copy_y);
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }
  // wait for final write
  dory_dma_barrier(DMA_copy_y);
  dory_dma_deallocate(dory_dma_channel);
}
