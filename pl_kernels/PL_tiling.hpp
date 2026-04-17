
#ifndef _PL_TILING_H_
#define _PL_TILING_H_

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>


// Single AIE tiling size
#define M 32
#define K 128
#define N 32


// Multiple AIEs parameters
#define X 10
#define Y 3
#define Z 10

// PL tiling parameters
#define U 4
#define V 2
#define W 4


const int data_width_PLIO = 128;

typedef ap_int<data_width_PLIO> data_t128;

typedef hls::stream<qdma_axis<data_width_PLIO, 0, 0, 0> > axi_stream;



const int depth_a = X*Y*U*V*(M*K/16);
const int depth_b = Y*Z*W*V*(K*N/16);
const int depth_c = X*Z*U*W*(M*N/4);


extern "C" {

void PL_tiling(	
	data_t128 *mem_ina,
	data_t128 *mem_inb,
	data_t128 *mem_outc,


	axi_stream &APL_out0,
	axi_stream &APL_out1,
	axi_stream &APL_out2,
	axi_stream &APL_out3,
	axi_stream &APL_out4,
	axi_stream &APL_out5,
	axi_stream &APL_out6,
	axi_stream &APL_out7,
	axi_stream &APL_out8,
	axi_stream &APL_out9,
	axi_stream &APL_out10,
	axi_stream &APL_out11,
	axi_stream &APL_out12,
	axi_stream &APL_out13,
	axi_stream &APL_out14,
	axi_stream &APL_out15,
	axi_stream &APL_out16,
	axi_stream &APL_out17,
	axi_stream &APL_out18,
	axi_stream &APL_out19,
	axi_stream &APL_out20,
	axi_stream &APL_out21,
	axi_stream &APL_out22,
	axi_stream &APL_out23,
	axi_stream &APL_out24,
	axi_stream &APL_out25,
	axi_stream &APL_out26,
	axi_stream &APL_out27,
	axi_stream &APL_out28,
	axi_stream &APL_out29,

	axi_stream &BPL_out0,
	axi_stream &BPL_out1,
	axi_stream &BPL_out2,
	axi_stream &BPL_out3,
	axi_stream &BPL_out4,
	axi_stream &BPL_out5,
	axi_stream &BPL_out6,
	axi_stream &BPL_out7,
	axi_stream &BPL_out8,
	axi_stream &BPL_out9,
	axi_stream &BPL_out10,
	axi_stream &BPL_out11,
	axi_stream &BPL_out12,
	axi_stream &BPL_out13,
	axi_stream &BPL_out14,
	axi_stream &BPL_out15,
	axi_stream &BPL_out16,
	axi_stream &BPL_out17,
	axi_stream &BPL_out18,
	axi_stream &BPL_out19,
	axi_stream &BPL_out20,
	axi_stream &BPL_out21,
	axi_stream &BPL_out22,
	axi_stream &BPL_out23,
	axi_stream &BPL_out24,
	axi_stream &BPL_out25,
	axi_stream &BPL_out26,
	axi_stream &BPL_out27,
	axi_stream &BPL_out28,
	axi_stream &BPL_out29,

	axi_stream &CPL_in0,
	axi_stream &CPL_in1,
	axi_stream &CPL_in2,
	axi_stream &CPL_in3,
	axi_stream &CPL_in4,
	axi_stream &CPL_in5,
	axi_stream &CPL_in6,
	axi_stream &CPL_in7,
	axi_stream &CPL_in8,
	axi_stream &CPL_in9,
	axi_stream &CPL_in10,
	axi_stream &CPL_in11,
	axi_stream &CPL_in12,
	axi_stream &CPL_in13,
	axi_stream &CPL_in14,
	axi_stream &CPL_in15,
	axi_stream &CPL_in16,
	axi_stream &CPL_in17,
	axi_stream &CPL_in18,
	axi_stream &CPL_in19,
	axi_stream &CPL_in20,
	axi_stream &CPL_in21,
	axi_stream &CPL_in22,
	axi_stream &CPL_in23,
	axi_stream &CPL_in24,
	axi_stream &CPL_in25,
	axi_stream &CPL_in26,
	axi_stream &CPL_in27,
	axi_stream &CPL_in28,
	axi_stream &CPL_in29,
	axi_stream &CPL_in30,
	axi_stream &CPL_in31,
	axi_stream &CPL_in32,
	axi_stream &CPL_in33,
	axi_stream &CPL_in34,
	axi_stream &CPL_in35,
	axi_stream &CPL_in36,
	axi_stream &CPL_in37,
	axi_stream &CPL_in38,
	axi_stream &CPL_in39,
	axi_stream &CPL_in40,
	axi_stream &CPL_in41,
	axi_stream &CPL_in42,
	axi_stream &CPL_in43,
	axi_stream &CPL_in44,
	axi_stream &CPL_in45,
	axi_stream &CPL_in46,
	axi_stream &CPL_in47,
	axi_stream &CPL_in48,
	axi_stream &CPL_in49,
	axi_stream &CPL_in50,
	axi_stream &CPL_in51,
	axi_stream &CPL_in52,
	axi_stream &CPL_in53,
	axi_stream &CPL_in54,
	axi_stream &CPL_in55,
	axi_stream &CPL_in56,
	axi_stream &CPL_in57,
	axi_stream &CPL_in58,
	axi_stream &CPL_in59,
	axi_stream &CPL_in60,
	axi_stream &CPL_in61,
	axi_stream &CPL_in62,
	axi_stream &CPL_in63,
	axi_stream &CPL_in64,
	axi_stream &CPL_in65,
	axi_stream &CPL_in66,
	axi_stream &CPL_in67,
	axi_stream &CPL_in68,
	axi_stream &CPL_in69,
	axi_stream &CPL_in70,
	axi_stream &CPL_in71,
	axi_stream &CPL_in72,
	axi_stream &CPL_in73,
	axi_stream &CPL_in74,
	axi_stream &CPL_in75,
	axi_stream &CPL_in76,
	axi_stream &CPL_in77,
	axi_stream &CPL_in78,
	axi_stream &CPL_in79,
	axi_stream &CPL_in80,
	axi_stream &CPL_in81,
	axi_stream &CPL_in82,
	axi_stream &CPL_in83,
	axi_stream &CPL_in84,
	axi_stream &CPL_in85,
	axi_stream &CPL_in86,
	axi_stream &CPL_in87,
	axi_stream &CPL_in88,
	axi_stream &CPL_in89,
	axi_stream &CPL_in90,
	axi_stream &CPL_in91,
	axi_stream &CPL_in92,
	axi_stream &CPL_in93,
	axi_stream &CPL_in94,
	axi_stream &CPL_in95,
	axi_stream &CPL_in96,
	axi_stream &CPL_in97,
	axi_stream &CPL_in98,
	axi_stream &CPL_in99

);
}


#endif  //_PL_TILING_H_

