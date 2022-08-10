#include <iostream>
using namespace std;
#include <math.h>
#include "filter2d.h"

static DTYPE filter2d_kernel(DTYPE WB[3][3], DTYPE Kernel[3][3])
{
#pragma HLS PIPELINE
	int k;
	DTYPE out_pix;
	out_pix=0;
	for(int r=0;r<3;r++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=3

		for(int c=0;c<3;c++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
			out_pix += Kernel[r][c]*WB[r][c];
		}
	}

	return (DTYPE) out_pix;
}


void filter2d_accel(DTYPE* img_in, DTYPE* kernel, DTYPE* img_out, int rows, int cols)
{

#pragma HLS INTERFACE m_axi port=img_in  offset=slave depth=16384
#pragma HLS INTERFACE m_axi port=img_out offset=slave depth=15876
#pragma HLS INTERFACE m_axi port=kernel  offset=slave depth=9
#pragma HLS INTERFACE s_axilite port=rows  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
	DTYPE _filter;
	DTYPE LineBuffer[3][128];
#pragma HLS ARRAY_PARTITION variable=LineBuffer complete dim=1

	DTYPE WindowBuffer[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#pragma HLS ARRAY_PARTITION variable=WindowBuffer complete dim=0

	int row, col;
	DTYPE top, mid, btm;
	DTYPE filter_kernel[3][3];
#pragma HLS ARRAY_PARTITION variable=filter_kernel complete dim=0

	for(row = 0; row < 3; row++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
		for(col = 0; col < 3; col++)
		{
#pragma HLS pipeline
			filter_kernel[row][col] = *kernel++;
		}

	}


	for(col = 0; col < cols; col++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
//#pragma HLS pipeline
				LineBuffer[0][col] = *img_in++;
				LineBuffer[1][col];
		}

		int lb_r_i = 2;
		for(row = 2; row < rows; row++)
		{
#pragma HLS loop_flatten off
#pragma HLS LOOP_TRIPCOUNT min=1 max=126
			if(lb_r_i == 2)
			{
				top = 0; mid = 1; btm = 2;
			}
			else if(lb_r_i == 0)
			{
				top = 1; mid = 2; btm = 0;
			}
			else if(lb_r_i == 1)
			{
				top = 2; mid = 0; btm = 1;
			}

			WindowBuffer[top][0] = WindowBuffer[top][1] = 0;
			WindowBuffer[mid][0] = WindowBuffer[top][1] = 0;
			WindowBuffer[btm][0] = WindowBuffer[top][1] = 0;
			for(col = 2; col < cols; col++)
			{
#pragma HLS LOOP_TRIPCOUNT min=1 max=126
#pragma HLS pipeline
				if(row < rows)
				{
					LineBuffer[btm][col] =  img_in[row];
				}
				else

					LineBuffer[btm][col] = 0;

				WindowBuffer[0][2] = LineBuffer[top][col];
				WindowBuffer[1][2] = LineBuffer[mid][col];
				WindowBuffer[2][2] = LineBuffer[btm][col];
				_filter = filter2d_kernel(WindowBuffer, filter_kernel);
				WindowBuffer[0][0] = WindowBuffer[0][1];
				WindowBuffer[1][0] = WindowBuffer[1][1];
				WindowBuffer[2][0] = WindowBuffer[2][1];
				WindowBuffer[0][1] = WindowBuffer[0][2];
				WindowBuffer[1][1] = WindowBuffer[1][2];
				WindowBuffer[2][1] = WindowBuffer[2][2];

				*img_out++ = _filter;
			}
			lb_r_i++;
			if(lb_r_i == 3) lb_r_i = 0;
		}

}
