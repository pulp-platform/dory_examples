  printf("y_buffer:");
for (int x_index = 0; x_index < 50; x_index ++)
{
  printf(" %d", *(y+x_index));
}
printf("\n");

printf("x_buffer:");
for (int x_index = 0; x_index < 50; x_index ++)
{
  printf(" %d", *(x+x_index));
}
printf("\n");

printf("W_buffer:");
for (int x_index = 0; x_index < 50; x_index ++)
{
  printf(" %d", *(W+x_index));
}
printf("\n");
if (rt_core_id()==0)
{
  printf("Arugments %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", x,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    W,
    y_tile_size_nof,
    3,
    3,
    p_t,
    p_b,
    p_l,
    p_r,
    1,
    1,
    NULL,
    0,
    out_shift,
    out_mult,
    y,
    y_tile_size_w,
    y_tile_size_h,
    k,
    lambda,
    im2col,
    1,
    1,
    &dma_evt);
}

if (rt_core_id()==0)
{
  printf("iter %d\n", iter);
printf("X summ: ");
int summ = 0;
for (int i=0; i<x_tile_size_w_exec*x_tile_size_h_exec*x_tile_size_nif_exec; i++ )
    summ+=*(x+i);
printf("%d\n", summ);
printf("W summ: ");
summ = 0;
for (int i=0; i<W_tile_size_byte; i++ )
    summ+=*(W+i);
printf("%d\n", summ);
printf("Y summ: ");
summ = 0;
for (int i=0; i<y_tile_size_byte; i++ )
    summ+=*(y+i);
printf("%d\n", summ);
}

    if (_i_w_exec == 7 ){
      printf("x_buffer:");
      for (int x_index = 0; x_index < 50; x_index ++)
      {
        printf(" %d", *(x+x_index));
      }
      printf("\n");
    }


      if (_i_w_exec == 7 ){
    printf("y_buffer:");
    for (int x_index = 0; x_index < 50; x_index ++)
    {
      printf(" %d", *(y+x_index));
    }
    printf("\n");
      if (i==4)
    {
      printf ("weights pixel0: %d, %d, %d, %d\n", *(pA3+0-4),*(pA3+1-4),*(pA3+2-4),*(pA3+3-4));
      printf ("act pixel0: %d, %d, %d, %d\n", *(pB+0-4),*(pB+1-4),*(pB+2-4),*(pB+3-4));
      printf("Sum pixel 0: %d, 1: %d\n", sum3,sum7);
    }
    printf("I'm in channel %d-%d\n",i*4,i*4+3);
    printf("Phi: %d Phi1: %d X: %d res: %d\n", phi, integer_image_phi, x, res);
        printf("\n\n\nPixel: [%d,%d]\n\n\n", i_out_y,i_out_x);
        
  rt_team_barrier();
  printf("X %d, W %d, y %d, k %d, lamb %d im2col %d\n", x,W,y,k,lambda,im2col );
  printf("X %d, W %d, y %x, k %x, lamb %d im2col %d\n", *x,*W,*y,*k,*lambda,*im2col );
   
//print 2-bits values
printf("Vector x: [");
for (int i=0;i<50;i++)
{
	printf("%d ", (*(x+i))&0x03);
	printf("%d ", (*(x+i)>>2)&0x03);
	printf("%d ", (*(x+i)>>4)&0x03);
	printf("%d ", (*(x+i)>>6)&0x03);
}
printf("]\n");

//print 4-bits values
printf("Vector x: [");
for (int i=0;i<50;i++)
{
	printf("%d ", (*(x+i))&0x0F);
	printf("%d ", (*(x+i)>>4)&0x0F);
}
printf("]\n");

//print 8-bits values
printf("Vector x: [");
for (int i=0;i<50;i++)
{
	printf("%d ", *(x+i));
}
printf("]\n");

//print lambda values
printf("Vector lambda: [");
for (int i=0;i<50;i++)
{
	printf("%d ", *(lambda+i));
}
printf("]\n");

//print k values
printf("Vector k: [");
for (int i=0;i<50;i++)
{
	printf("%d ", *(k+i));
}
printf("]\n");  


      printf("sum %d, k %d, lambda %d, out shift %d\n", sum, *k, *lambda, out_shift);
      printf("sum5 %d, k %d, lambda %d, out shift %d\n", sum5, *k, *lambda, out_shift);
      
      printf("sum %d, k %d, lambda %d, out shift %d\n", sum2, *k, *lambda, out_shift);
      printf("sum5 %d, k %d, lambda %d, out shift %d\n", sum6, *k, *lambda, out_shift);

      printf("sum %d, k %d, lambda %d, out shift %d\n", sum3, *k, *lambda, out_shift);
      printf("sum5 %d, k %d, lambda %d, out shift %d\n", sum7, *k, *lambda, out_shift);

      printf("sum %d, k %d, lambda %d, out shift %d\n", sum4, *k, *lambda, out_shift);
      printf("sum5 %d, k %d, lambda %d, out shift %d\n", sum8, *k, *lambda, out_shift);

      printf("sum after %d\n", sum);
      printf("sum2 after %d\n", sum2);
      printf("sum3 after %d\n", sum3);
      printf("sum4 after %d\n", sum4);
      printf("sum5 after %d\n", sum5);
      printf("sum6 after %d\n", sum6);
      printf("sum7 after %d\n", sum7);
      printf("sum8 after %d\n", sum8);

      
      printf("Sum0 =  , k = %ld, lambda = %ld, shift %d\n",  *k, *lambda, out_shift);
//      printf("Sum1 = %d , k = %ld, lambda = %ld, shift %d\n", sum2, *k, *lambda, out_shift);
//      printf("Sum2 = %d , k = %ld, lambda = %ld, shift %d\n", sum3, *k, *lambda, out_shift);
//      printf("Sum3 = %d , k = %ld, lambda = %ld, shift %d\n", sum4, *k, *lambda, out_shift);
//      printf("Sum4 = %d , k = %ld, lambda = %ld, shift %d\n", sum5, *k, *lambda, out_shift);
//      printf("Sum5 = %d , k = %ld, lambda = %ld, shift %d\n", sum6, *k, *lambda, out_shift);
//      printf("Sum6 = %d , k = %ld, lambda = %ld, shift %d\n", sum7, *k, *lambda, out_shift);
//      printf("Sum7 = %d , k = %ld, lambda = %ld, shift %d\n", sum8, *k, *lambda, out_shift);


//      printf("After quantization: \n");  
//      printf("Sum0 = %d , k = %d, lambda = %d, shift %d\n", sum, *k, *lambda, out_shift);
//      printf("Sum1 = %d , k = %d, lambda = %d, shift %d\n", sum2, *k, *lambda, out_shift);
//      printf("Sum2 = %d , k = %d, lambda = %d, shift %d\n", sum3, *k, *lambda, out_shift);
//      printf("Sum3 = %d , k = %d, lambda = %d, shift %d\n", sum4, *k, *lambda, out_shift);
//      printf("Sum4 = %d , k = %d, lambda = %d, shift %d\n", sum5, *k, *lambda, out_shift);
//      printf("Sum5 = %d , k = %d, lambda = %d, shift %d\n", sum6, *k, *lambda, out_shift);
//      printf("Sum6 = %d , k = %d, lambda = %d, shift %d\n", sum7, *k, *lambda, out_shift);
//      printf("Sum7 = %d , k = %d, lambda = %d, shift %d\n", sum8, *k, *lambda, out_shift);
