
#include "PL_tiling.hpp"
#include <iostream>



void load_A(data_t128 *mem_ina, data_t128 A_buff[X*Y][U*V][(M*K/16)]){

	for (int u = 0; u < U; u++){
		for (int v = 0; v < V; v++){

			for (int x = 0; x < X; x++){
				for (int y = 0; y < Y; y++){

					for (int i = 0; i < (M*K/16); i++){
					#pragma HLS PIPELINE II=1

						int mem_ina_position = u*V*X*Y*(M*K/16) + v*X*Y*(M*K/16) + x*Y*(M*K/16) + y*(M*K/16) + i;

						A_buff[x*Y + y][u*V + v][i] = mem_ina[mem_ina_position];
					}

				}
			}

		}
	}
}


void load_B(data_t128 *mem_inb, data_t128 B_buff[Y*Z][W*V][(K*N/16)]){

	for (int w = 0; w < W; w++){
		for (int v = 0; v < V; v++){

			for (int z = 0; z < Z; z++){
				for (int y = 0; y < Y; y++){

					for (int i = 0; i < (K*N/16); i++){
					#pragma HLS PIPELINE II=1

						int mem_inb_position = w*V*Z*Y*(K*N/16) + v*Z*Y*(K*N/16) + z*Y*(K*N/16) + y*(K*N/16) + i;

						B_buff[z*Y + y][w*V + v][i] = mem_inb[mem_inb_position];

					}

				}
			}

		}
	}
}



void store_C(data_t128 C_buff[X*Z][U*W][(M*N/4)], data_t128 *mem_outc){

	for (int u = 0; u < U; u++){
		for (int w = 0; w < W; w++){

			for (int x = 0; x < X; x++){
				for (int z = 0; z < Z; z++){

					for (int i = 0; i < (M*N/4); i++){
					#pragma HLS PIPELINE II=1

						int mem_outc_position = u*W*X*Z*(M*N/4) + w*X*Z*(M*N/4) + x*Z*(M*N/4)+ z*(M*N/4) + i;

						mem_outc[mem_outc_position]= C_buff[x*Z + z][u*W + w][i];

					}

				}
			}

		}
	}

}


/*
 * This function implements the S2MM from AIE to output C_buffer,
 * as well as the required addition of the partial products.
 * When flag accumulate is 1, keeps accumulating data to the buffer,
 * otherwise just load on every new iteration as defined by parameters U,V,W
 *
 * Execution time: U*V*W*(M*N/4)
 */

void s2mm_add_C(axi_stream &stream_in, data_t128 C_buff[U*W][(M*N/4)], bool accumulate){


	qdma_axis<data_width_PLIO,0,0,0> stream_tmp;

	int accum_temp[4];
	#pragma HLS ARRAY_PARTITION variable = accum_temp complete

	for (int u = 0; u < U; u++){
		for (int w = 0; w < W; w++){

			// v = 0, and not accumulate => just load
			if (!accumulate){
				for (int i = 0; i < (M*N/4); i++){
				#pragma HLS PIPELINE II=1

					stream_tmp = stream_in.read();

					int C_position = u*W + w;

					C_buff[C_position][i] = stream_tmp.data;

				}
			}
			// v = 0, and accumulate => +=
			else {
				for (int i = 0; i < (M*N/4); i++){
				#pragma HLS PIPELINE II=1

					stream_tmp = stream_in.read();

					int C_position = u*W + w;

					for (int j = 0; j < 4; j++){
						accum_temp[j] = C_buff[C_position][i]((j+1)*32 - 1, j*32);
						accum_temp[j] += stream_tmp.data((j+1)*32 - 1, j*32);
						C_buff[C_position][i]((j+1)*32 - 1, j*32) = accum_temp[j];
					}

				}
			}

			// notice v starts at 1
			// for every iteration keep accumulating
			for (int v = 1; v < V; v++){

				for (int i = 0; i < (M*N/4); i++){
				#pragma HLS PIPELINE II=1

					stream_tmp = stream_in.read();

					int C_position = u*W + w;

					for (int j = 0; j < 4; j++){
						accum_temp[j] = C_buff[C_position][i]((j+1)*32 - 1, j*32);
						accum_temp[j] += stream_tmp.data((j+1)*32 - 1, j*32);
						C_buff[C_position][i]((j+1)*32 - 1, j*32) = accum_temp[j];
					}

				}
			}
		}
	}

}

/*
 * This function implements the mm2s logic from the A_buff to the stream input of the AIEs.
 *
 * Notice that A_buff is reused W times!
 *
 * Execution time: U*V*W*(M*K/16)
 */
void mm2s_A(data_t128 A_buff[U*V][(M*K/16)], axi_stream &stream_out){

	qdma_axis<data_width_PLIO,0,0,0> stream_tmp;


	for (int u = 0; u < U; u++){
		for (int w = 0; w < W; w++){
			for (int v = 0; v < V; v++){

				for (int i = 0; i < (M*K/16); i++){
				#pragma HLS PIPELINE II=1

					int A_position = u*V + v;

					stream_tmp.data = A_buff[A_position][i];

					// keep all data
					stream_tmp.keep = -1;

//					stream_tmp.keep_all();

					// tlast logic
					if (i == ((M*K/16) - 1)){
						stream_tmp.last = 1;
					}
					else {
						stream_tmp.last = 0;
					}

					stream_out.write(stream_tmp);

				}
			}
		}
	}

}


/*
 * This function implements the mm2s logic from the B_buff to the stream input of the AIEs.
 *
 * Notice that B_buff is reused U times!
 *
 * Execution time: U*V*W*(K*N/16)
 */
void mm2s_B(data_t128 B_buff[W*V][(K*N/16)], axi_stream &stream_out){

	qdma_axis<data_width_PLIO,0,0,0> stream_tmp;

	for (int u = 0; u < U; u++){
		for (int w = 0; w < W; w++){
			for (int v = 0; v < V; v++){

				for (int i = 0; i < (K*N/16); i++){
				#pragma HLS PIPELINE II=1

					int B_position = w*V + v;

					stream_tmp.data = B_buff[B_position][i];

					// keep all data
					stream_tmp.keep = -1;

//					stream_tmp.keep_all();

					// tlast logic
					if (i == ((K*N/16) - 1)){
						stream_tmp.last = 1;
					}
					else {
						stream_tmp.last = 0;
					}

					stream_out.write(stream_tmp);

				}

			}
		}
	}

}

          
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

){
// M_AXI ports
#pragma HLS INTERFACE m_axi port = mem_ina offset = slave depth = depth_a bundle = gmem_a
#pragma HLS INTERFACE m_axi port = mem_inb offset = slave depth = depth_b bundle = gmem_b
#pragma HLS INTERFACE m_axi port = mem_outc offset = slave depth = depth_c bundle = gmem_c
// AXIS ports
#pragma HLS INTERFACE axis port=APL_out0
#pragma HLS INTERFACE axis port=APL_out1
#pragma HLS INTERFACE axis port=APL_out2
#pragma HLS INTERFACE axis port=APL_out3
#pragma HLS INTERFACE axis port=APL_out4
#pragma HLS INTERFACE axis port=APL_out5
#pragma HLS INTERFACE axis port=APL_out6
#pragma HLS INTERFACE axis port=APL_out7
#pragma HLS INTERFACE axis port=APL_out8
#pragma HLS INTERFACE axis port=APL_out9
#pragma HLS INTERFACE axis port=APL_out10
#pragma HLS INTERFACE axis port=APL_out11
#pragma HLS INTERFACE axis port=APL_out12
#pragma HLS INTERFACE axis port=APL_out13
#pragma HLS INTERFACE axis port=APL_out14
#pragma HLS INTERFACE axis port=APL_out15
#pragma HLS INTERFACE axis port=APL_out16
#pragma HLS INTERFACE axis port=APL_out17
#pragma HLS INTERFACE axis port=APL_out18
#pragma HLS INTERFACE axis port=APL_out19
#pragma HLS INTERFACE axis port=APL_out20
#pragma HLS INTERFACE axis port=APL_out21
#pragma HLS INTERFACE axis port=APL_out22
#pragma HLS INTERFACE axis port=APL_out23
#pragma HLS INTERFACE axis port=APL_out24
#pragma HLS INTERFACE axis port=APL_out25
#pragma HLS INTERFACE axis port=APL_out26
#pragma HLS INTERFACE axis port=APL_out27
#pragma HLS INTERFACE axis port=APL_out28
#pragma HLS INTERFACE axis port=APL_out29

#pragma HLS INTERFACE axis port=BPL_out0
#pragma HLS INTERFACE axis port=BPL_out1
#pragma HLS INTERFACE axis port=BPL_out2
#pragma HLS INTERFACE axis port=BPL_out3
#pragma HLS INTERFACE axis port=BPL_out4
#pragma HLS INTERFACE axis port=BPL_out5
#pragma HLS INTERFACE axis port=BPL_out6
#pragma HLS INTERFACE axis port=BPL_out7
#pragma HLS INTERFACE axis port=BPL_out8
#pragma HLS INTERFACE axis port=BPL_out9
#pragma HLS INTERFACE axis port=BPL_out10
#pragma HLS INTERFACE axis port=BPL_out11
#pragma HLS INTERFACE axis port=BPL_out12
#pragma HLS INTERFACE axis port=BPL_out13
#pragma HLS INTERFACE axis port=BPL_out14
#pragma HLS INTERFACE axis port=BPL_out15
#pragma HLS INTERFACE axis port=BPL_out16
#pragma HLS INTERFACE axis port=BPL_out17
#pragma HLS INTERFACE axis port=BPL_out18
#pragma HLS INTERFACE axis port=BPL_out19
#pragma HLS INTERFACE axis port=BPL_out20
#pragma HLS INTERFACE axis port=BPL_out21
#pragma HLS INTERFACE axis port=BPL_out22
#pragma HLS INTERFACE axis port=BPL_out23
#pragma HLS INTERFACE axis port=BPL_out24
#pragma HLS INTERFACE axis port=BPL_out25
#pragma HLS INTERFACE axis port=BPL_out26
#pragma HLS INTERFACE axis port=BPL_out27
#pragma HLS INTERFACE axis port=BPL_out28
#pragma HLS INTERFACE axis port=BPL_out29

#pragma HLS INTERFACE axis port=CPL_in0
#pragma HLS INTERFACE axis port=CPL_in1
#pragma HLS INTERFACE axis port=CPL_in2
#pragma HLS INTERFACE axis port=CPL_in3
#pragma HLS INTERFACE axis port=CPL_in4
#pragma HLS INTERFACE axis port=CPL_in5
#pragma HLS INTERFACE axis port=CPL_in6
#pragma HLS INTERFACE axis port=CPL_in7
#pragma HLS INTERFACE axis port=CPL_in8
#pragma HLS INTERFACE axis port=CPL_in9
#pragma HLS INTERFACE axis port=CPL_in10
#pragma HLS INTERFACE axis port=CPL_in11
#pragma HLS INTERFACE axis port=CPL_in12
#pragma HLS INTERFACE axis port=CPL_in13
#pragma HLS INTERFACE axis port=CPL_in14
#pragma HLS INTERFACE axis port=CPL_in15
#pragma HLS INTERFACE axis port=CPL_in16
#pragma HLS INTERFACE axis port=CPL_in17
#pragma HLS INTERFACE axis port=CPL_in18
#pragma HLS INTERFACE axis port=CPL_in19
#pragma HLS INTERFACE axis port=CPL_in20
#pragma HLS INTERFACE axis port=CPL_in21
#pragma HLS INTERFACE axis port=CPL_in22
#pragma HLS INTERFACE axis port=CPL_in23
#pragma HLS INTERFACE axis port=CPL_in24
#pragma HLS INTERFACE axis port=CPL_in25
#pragma HLS INTERFACE axis port=CPL_in26
#pragma HLS INTERFACE axis port=CPL_in27
#pragma HLS INTERFACE axis port=CPL_in28
#pragma HLS INTERFACE axis port=CPL_in29
#pragma HLS INTERFACE axis port=CPL_in30
#pragma HLS INTERFACE axis port=CPL_in31
#pragma HLS INTERFACE axis port=CPL_in32
#pragma HLS INTERFACE axis port=CPL_in33
#pragma HLS INTERFACE axis port=CPL_in34
#pragma HLS INTERFACE axis port=CPL_in35
#pragma HLS INTERFACE axis port=CPL_in36
#pragma HLS INTERFACE axis port=CPL_in37
#pragma HLS INTERFACE axis port=CPL_in38
#pragma HLS INTERFACE axis port=CPL_in39
#pragma HLS INTERFACE axis port=CPL_in40
#pragma HLS INTERFACE axis port=CPL_in41
#pragma HLS INTERFACE axis port=CPL_in42
#pragma HLS INTERFACE axis port=CPL_in43
#pragma HLS INTERFACE axis port=CPL_in44
#pragma HLS INTERFACE axis port=CPL_in45
#pragma HLS INTERFACE axis port=CPL_in46
#pragma HLS INTERFACE axis port=CPL_in47
#pragma HLS INTERFACE axis port=CPL_in48
#pragma HLS INTERFACE axis port=CPL_in49
#pragma HLS INTERFACE axis port=CPL_in50
#pragma HLS INTERFACE axis port=CPL_in51
#pragma HLS INTERFACE axis port=CPL_in52
#pragma HLS INTERFACE axis port=CPL_in53
#pragma HLS INTERFACE axis port=CPL_in54
#pragma HLS INTERFACE axis port=CPL_in55
#pragma HLS INTERFACE axis port=CPL_in56
#pragma HLS INTERFACE axis port=CPL_in57
#pragma HLS INTERFACE axis port=CPL_in58
#pragma HLS INTERFACE axis port=CPL_in59
#pragma HLS INTERFACE axis port=CPL_in60
#pragma HLS INTERFACE axis port=CPL_in61
#pragma HLS INTERFACE axis port=CPL_in62
#pragma HLS INTERFACE axis port=CPL_in63
#pragma HLS INTERFACE axis port=CPL_in64
#pragma HLS INTERFACE axis port=CPL_in65
#pragma HLS INTERFACE axis port=CPL_in66
#pragma HLS INTERFACE axis port=CPL_in67
#pragma HLS INTERFACE axis port=CPL_in68
#pragma HLS INTERFACE axis port=CPL_in69
#pragma HLS INTERFACE axis port=CPL_in70
#pragma HLS INTERFACE axis port=CPL_in71
#pragma HLS INTERFACE axis port=CPL_in72
#pragma HLS INTERFACE axis port=CPL_in73
#pragma HLS INTERFACE axis port=CPL_in74
#pragma HLS INTERFACE axis port=CPL_in75
#pragma HLS INTERFACE axis port=CPL_in76
#pragma HLS INTERFACE axis port=CPL_in77
#pragma HLS INTERFACE axis port=CPL_in78
#pragma HLS INTERFACE axis port=CPL_in79
#pragma HLS INTERFACE axis port=CPL_in80
#pragma HLS INTERFACE axis port=CPL_in81
#pragma HLS INTERFACE axis port=CPL_in82
#pragma HLS INTERFACE axis port=CPL_in83
#pragma HLS INTERFACE axis port=CPL_in84
#pragma HLS INTERFACE axis port=CPL_in85
#pragma HLS INTERFACE axis port=CPL_in86
#pragma HLS INTERFACE axis port=CPL_in87
#pragma HLS INTERFACE axis port=CPL_in88
#pragma HLS INTERFACE axis port=CPL_in89
#pragma HLS INTERFACE axis port=CPL_in90
#pragma HLS INTERFACE axis port=CPL_in91
#pragma HLS INTERFACE axis port=CPL_in92
#pragma HLS INTERFACE axis port=CPL_in93
#pragma HLS INTERFACE axis port=CPL_in94
#pragma HLS INTERFACE axis port=CPL_in95
#pragma HLS INTERFACE axis port=CPL_in96
#pragma HLS INTERFACE axis port=CPL_in97
#pragma HLS INTERFACE axis port=CPL_in98
#pragma HLS INTERFACE axis port=CPL_in99


// S_AXILITE ports
#pragma HLS INTERFACE s_axilite port=mem_ina bundle=control
#pragma HLS INTERFACE s_axilite port=mem_inb bundle=control
#pragma HLS INTERFACE s_axilite port=mem_outc bundle=control
#pragma HLS interface s_axilite port=return bundle=control


          
	data_t128 A0_buff[X*Y][U*V][(M*K/16)];
    #pragma HLS BIND_STORAGE variable=A0_buff type=RAM_1P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = A0_buff dim = 1 complete

    data_t128 A1_buff[X*Y][U*V][(M*K/16)];
    #pragma HLS BIND_STORAGE variable=A1_buff type=RAM_1P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = A1_buff dim = 1 complete



	data_t128 B0_buff[Y*Z][W*V][(K*N/16)];
    #pragma HLS BIND_STORAGE variable=B0_buff type=RAM_1P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = B0_buff dim = 1 complete

	data_t128 B1_buff[Y*Z][W*V][(K*N/16)];
    #pragma HLS BIND_STORAGE variable=B1_buff type=RAM_1P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = B1_buff dim = 1 complete



	data_t128 C0_buff[X*Z][U*W][(M*N/4)];
    #pragma HLS BIND_STORAGE variable=C0_buff type=RAM_2P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = C0_buff dim = 1 complete

    data_t128 C1_buff[X*Z][U*W][(M*N/4)];
    #pragma HLS BIND_STORAGE variable=C1_buff type=RAM_2P impl=AUTO
    #pragma HLS ARRAY_PARTITION variable = C1_buff dim = 1 complete
    
	

    for (int i = 0; i < 10; i++){
		
		if (i % 2 == 0){



            // load_A
            load_A(mem_ina, A0_buff);
            // Load_B
            load_B(mem_inb, B0_buff);


			// send A to AIE
			mm2s_A(A1_buff[0], APL_out0);
			mm2s_A(A1_buff[1], APL_out1);
			mm2s_A(A1_buff[2], APL_out2);
			mm2s_A(A1_buff[3], APL_out3);
			mm2s_A(A1_buff[4], APL_out4);
			mm2s_A(A1_buff[5], APL_out5);
			mm2s_A(A1_buff[6], APL_out6);
			mm2s_A(A1_buff[7], APL_out7);
			mm2s_A(A1_buff[8], APL_out8);
			mm2s_A(A1_buff[9], APL_out9);
			mm2s_A(A1_buff[10], APL_out10);
			mm2s_A(A1_buff[11], APL_out11);
			mm2s_A(A1_buff[12], APL_out12);
			mm2s_A(A1_buff[13], APL_out13);
			mm2s_A(A1_buff[14], APL_out14);
			mm2s_A(A1_buff[15], APL_out15);
			mm2s_A(A1_buff[16], APL_out16);
			mm2s_A(A1_buff[17], APL_out17);
			mm2s_A(A1_buff[18], APL_out18);
			mm2s_A(A1_buff[19], APL_out19);
			mm2s_A(A1_buff[20], APL_out20);
			mm2s_A(A1_buff[21], APL_out21);
			mm2s_A(A1_buff[22], APL_out22);
			mm2s_A(A1_buff[23], APL_out23);
			mm2s_A(A1_buff[24], APL_out24);
			mm2s_A(A1_buff[25], APL_out25);
			mm2s_A(A1_buff[26], APL_out26);
			mm2s_A(A1_buff[27], APL_out27);
			mm2s_A(A1_buff[28], APL_out28);
			mm2s_A(A1_buff[29], APL_out29);


			// send B to AIE
			mm2s_B(B1_buff[0], BPL_out0);
			mm2s_B(B1_buff[1], BPL_out1);
			mm2s_B(B1_buff[2], BPL_out2);
			mm2s_B(B1_buff[3], BPL_out3);
			mm2s_B(B1_buff[4], BPL_out4);
			mm2s_B(B1_buff[5], BPL_out5);
			mm2s_B(B1_buff[6], BPL_out6);
			mm2s_B(B1_buff[7], BPL_out7);
			mm2s_B(B1_buff[8], BPL_out8);
			mm2s_B(B1_buff[9], BPL_out9);
			mm2s_B(B1_buff[10], BPL_out10);
			mm2s_B(B1_buff[11], BPL_out11);
			mm2s_B(B1_buff[12], BPL_out12);
			mm2s_B(B1_buff[13], BPL_out13);
			mm2s_B(B1_buff[14], BPL_out14);
			mm2s_B(B1_buff[15], BPL_out15);
			mm2s_B(B1_buff[16], BPL_out16);
			mm2s_B(B1_buff[17], BPL_out17);
			mm2s_B(B1_buff[18], BPL_out18);
			mm2s_B(B1_buff[19], BPL_out19);
			mm2s_B(B1_buff[20], BPL_out20);
			mm2s_B(B1_buff[21], BPL_out21);
			mm2s_B(B1_buff[22], BPL_out22);
			mm2s_B(B1_buff[23], BPL_out23);
			mm2s_B(B1_buff[24], BPL_out24);
			mm2s_B(B1_buff[25], BPL_out25);
			mm2s_B(B1_buff[26], BPL_out26);
			mm2s_B(B1_buff[27], BPL_out27);
			mm2s_B(B1_buff[28], BPL_out28);
			mm2s_B(B1_buff[29], BPL_out29);


			// get C from AIE and perform additions on PL
			s2mm_add_C(CPL_in0, C1_buff[0], 0);
			s2mm_add_C(CPL_in1, C1_buff[1], 0);
			s2mm_add_C(CPL_in2, C1_buff[2], 0);
			s2mm_add_C(CPL_in3, C1_buff[3], 0);
			s2mm_add_C(CPL_in4, C1_buff[4], 0);
			s2mm_add_C(CPL_in5, C1_buff[5], 0);
			s2mm_add_C(CPL_in6, C1_buff[6], 0);
			s2mm_add_C(CPL_in7, C1_buff[7], 0);
			s2mm_add_C(CPL_in8, C1_buff[8], 0);
			s2mm_add_C(CPL_in9, C1_buff[9], 0);
			s2mm_add_C(CPL_in10, C1_buff[10], 0);
			s2mm_add_C(CPL_in11, C1_buff[11], 0);
			s2mm_add_C(CPL_in12, C1_buff[12], 0);
			s2mm_add_C(CPL_in13, C1_buff[13], 0);
			s2mm_add_C(CPL_in14, C1_buff[14], 0);
			s2mm_add_C(CPL_in15, C1_buff[15], 0);
			s2mm_add_C(CPL_in16, C1_buff[16], 0);
			s2mm_add_C(CPL_in17, C1_buff[17], 0);
			s2mm_add_C(CPL_in18, C1_buff[18], 0);
			s2mm_add_C(CPL_in19, C1_buff[19], 0);
			s2mm_add_C(CPL_in20, C1_buff[20], 0);
			s2mm_add_C(CPL_in21, C1_buff[21], 0);
			s2mm_add_C(CPL_in22, C1_buff[22], 0);
			s2mm_add_C(CPL_in23, C1_buff[23], 0);
			s2mm_add_C(CPL_in24, C1_buff[24], 0);
			s2mm_add_C(CPL_in25, C1_buff[25], 0);
			s2mm_add_C(CPL_in26, C1_buff[26], 0);
			s2mm_add_C(CPL_in27, C1_buff[27], 0);
			s2mm_add_C(CPL_in28, C1_buff[28], 0);
			s2mm_add_C(CPL_in29, C1_buff[29], 0);
			s2mm_add_C(CPL_in30, C1_buff[30], 0);
			s2mm_add_C(CPL_in31, C1_buff[31], 0);
			s2mm_add_C(CPL_in32, C1_buff[32], 0);
			s2mm_add_C(CPL_in33, C1_buff[33], 0);
			s2mm_add_C(CPL_in34, C1_buff[34], 0);
			s2mm_add_C(CPL_in35, C1_buff[35], 0);
			s2mm_add_C(CPL_in36, C1_buff[36], 0);
			s2mm_add_C(CPL_in37, C1_buff[37], 0);
			s2mm_add_C(CPL_in38, C1_buff[38], 0);
			s2mm_add_C(CPL_in39, C1_buff[39], 0);
			s2mm_add_C(CPL_in40, C1_buff[40], 0);
			s2mm_add_C(CPL_in41, C1_buff[41], 0);
			s2mm_add_C(CPL_in42, C1_buff[42], 0);
			s2mm_add_C(CPL_in43, C1_buff[43], 0);
			s2mm_add_C(CPL_in44, C1_buff[44], 0);
			s2mm_add_C(CPL_in45, C1_buff[45], 0);
			s2mm_add_C(CPL_in46, C1_buff[46], 0);
			s2mm_add_C(CPL_in47, C1_buff[47], 0);
			s2mm_add_C(CPL_in48, C1_buff[48], 0);
			s2mm_add_C(CPL_in49, C1_buff[49], 0);
			s2mm_add_C(CPL_in50, C1_buff[50], 0);
			s2mm_add_C(CPL_in51, C1_buff[51], 0);
			s2mm_add_C(CPL_in52, C1_buff[52], 0);
			s2mm_add_C(CPL_in53, C1_buff[53], 0);
			s2mm_add_C(CPL_in54, C1_buff[54], 0);
			s2mm_add_C(CPL_in55, C1_buff[55], 0);
			s2mm_add_C(CPL_in56, C1_buff[56], 0);
			s2mm_add_C(CPL_in57, C1_buff[57], 0);
			s2mm_add_C(CPL_in58, C1_buff[58], 0);
			s2mm_add_C(CPL_in59, C1_buff[59], 0);
			s2mm_add_C(CPL_in60, C1_buff[60], 0);
			s2mm_add_C(CPL_in61, C1_buff[61], 0);
			s2mm_add_C(CPL_in62, C1_buff[62], 0);
			s2mm_add_C(CPL_in63, C1_buff[63], 0);
			s2mm_add_C(CPL_in64, C1_buff[64], 0);
			s2mm_add_C(CPL_in65, C1_buff[65], 0);
			s2mm_add_C(CPL_in66, C1_buff[66], 0);
			s2mm_add_C(CPL_in67, C1_buff[67], 0);
			s2mm_add_C(CPL_in68, C1_buff[68], 0);
			s2mm_add_C(CPL_in69, C1_buff[69], 0);
			s2mm_add_C(CPL_in70, C1_buff[70], 0);
			s2mm_add_C(CPL_in71, C1_buff[71], 0);
			s2mm_add_C(CPL_in72, C1_buff[72], 0);
			s2mm_add_C(CPL_in73, C1_buff[73], 0);
			s2mm_add_C(CPL_in74, C1_buff[74], 0);
			s2mm_add_C(CPL_in75, C1_buff[75], 0);
			s2mm_add_C(CPL_in76, C1_buff[76], 0);
			s2mm_add_C(CPL_in77, C1_buff[77], 0);
			s2mm_add_C(CPL_in78, C1_buff[78], 0);
			s2mm_add_C(CPL_in79, C1_buff[79], 0);
			s2mm_add_C(CPL_in80, C1_buff[80], 0);
			s2mm_add_C(CPL_in81, C1_buff[81], 0);
			s2mm_add_C(CPL_in82, C1_buff[82], 0);
			s2mm_add_C(CPL_in83, C1_buff[83], 0);
			s2mm_add_C(CPL_in84, C1_buff[84], 0);
			s2mm_add_C(CPL_in85, C1_buff[85], 0);
			s2mm_add_C(CPL_in86, C1_buff[86], 0);
			s2mm_add_C(CPL_in87, C1_buff[87], 0);
			s2mm_add_C(CPL_in88, C1_buff[88], 0);
			s2mm_add_C(CPL_in89, C1_buff[89], 0);
			s2mm_add_C(CPL_in90, C1_buff[90], 0);
			s2mm_add_C(CPL_in91, C1_buff[91], 0);
			s2mm_add_C(CPL_in92, C1_buff[92], 0);
			s2mm_add_C(CPL_in93, C1_buff[93], 0);
			s2mm_add_C(CPL_in94, C1_buff[94], 0);
			s2mm_add_C(CPL_in95, C1_buff[95], 0);
			s2mm_add_C(CPL_in96, C1_buff[96], 0);
			s2mm_add_C(CPL_in97, C1_buff[97], 0);
			s2mm_add_C(CPL_in98, C1_buff[98], 0);
			s2mm_add_C(CPL_in99, C1_buff[99], 0);
			// store_C
			store_C(C0_buff, mem_outc);

        }

		else {



            // load_A
            load_A(mem_ina, A1_buff);
            // Load_B
            load_B(mem_inb, B1_buff);


			// send A to AIE
			mm2s_A(A0_buff[0], APL_out0);
			mm2s_A(A0_buff[1], APL_out1);
			mm2s_A(A0_buff[2], APL_out2);
			mm2s_A(A0_buff[3], APL_out3);
			mm2s_A(A0_buff[4], APL_out4);
			mm2s_A(A0_buff[5], APL_out5);
			mm2s_A(A0_buff[6], APL_out6);
			mm2s_A(A0_buff[7], APL_out7);
			mm2s_A(A0_buff[8], APL_out8);
			mm2s_A(A0_buff[9], APL_out9);
			mm2s_A(A0_buff[10], APL_out10);
			mm2s_A(A0_buff[11], APL_out11);
			mm2s_A(A0_buff[12], APL_out12);
			mm2s_A(A0_buff[13], APL_out13);
			mm2s_A(A0_buff[14], APL_out14);
			mm2s_A(A0_buff[15], APL_out15);
			mm2s_A(A0_buff[16], APL_out16);
			mm2s_A(A0_buff[17], APL_out17);
			mm2s_A(A0_buff[18], APL_out18);
			mm2s_A(A0_buff[19], APL_out19);
			mm2s_A(A0_buff[20], APL_out20);
			mm2s_A(A0_buff[21], APL_out21);
			mm2s_A(A0_buff[22], APL_out22);
			mm2s_A(A0_buff[23], APL_out23);
			mm2s_A(A0_buff[24], APL_out24);
			mm2s_A(A0_buff[25], APL_out25);
			mm2s_A(A0_buff[26], APL_out26);
			mm2s_A(A0_buff[27], APL_out27);
			mm2s_A(A0_buff[28], APL_out28);
			mm2s_A(A0_buff[29], APL_out29);


			// send B to AIE
			mm2s_B(B0_buff[0], BPL_out0);
			mm2s_B(B0_buff[1], BPL_out1);
			mm2s_B(B0_buff[2], BPL_out2);
			mm2s_B(B0_buff[3], BPL_out3);
			mm2s_B(B0_buff[4], BPL_out4);
			mm2s_B(B0_buff[5], BPL_out5);
			mm2s_B(B0_buff[6], BPL_out6);
			mm2s_B(B0_buff[7], BPL_out7);
			mm2s_B(B0_buff[8], BPL_out8);
			mm2s_B(B0_buff[9], BPL_out9);
			mm2s_B(B0_buff[10], BPL_out10);
			mm2s_B(B0_buff[11], BPL_out11);
			mm2s_B(B0_buff[12], BPL_out12);
			mm2s_B(B0_buff[13], BPL_out13);
			mm2s_B(B0_buff[14], BPL_out14);
			mm2s_B(B0_buff[15], BPL_out15);
			mm2s_B(B0_buff[16], BPL_out16);
			mm2s_B(B0_buff[17], BPL_out17);
			mm2s_B(B0_buff[18], BPL_out18);
			mm2s_B(B0_buff[19], BPL_out19);
			mm2s_B(B0_buff[20], BPL_out20);
			mm2s_B(B0_buff[21], BPL_out21);
			mm2s_B(B0_buff[22], BPL_out22);
			mm2s_B(B0_buff[23], BPL_out23);
			mm2s_B(B0_buff[24], BPL_out24);
			mm2s_B(B0_buff[25], BPL_out25);
			mm2s_B(B0_buff[26], BPL_out26);
			mm2s_B(B0_buff[27], BPL_out27);
			mm2s_B(B0_buff[28], BPL_out28);
			mm2s_B(B0_buff[29], BPL_out29);


			// get C from AIE and perform additions on PL
			s2mm_add_C(CPL_in0, C0_buff[0], 0);
			s2mm_add_C(CPL_in1, C0_buff[1], 0);
			s2mm_add_C(CPL_in2, C0_buff[2], 0);
			s2mm_add_C(CPL_in3, C0_buff[3], 0);
			s2mm_add_C(CPL_in4, C0_buff[4], 0);
			s2mm_add_C(CPL_in5, C0_buff[5], 0);
			s2mm_add_C(CPL_in6, C0_buff[6], 0);
			s2mm_add_C(CPL_in7, C0_buff[7], 0);
			s2mm_add_C(CPL_in8, C0_buff[8], 0);
			s2mm_add_C(CPL_in9, C0_buff[9], 0);
			s2mm_add_C(CPL_in10, C0_buff[10], 0);
			s2mm_add_C(CPL_in11, C0_buff[11], 0);
			s2mm_add_C(CPL_in12, C0_buff[12], 0);
			s2mm_add_C(CPL_in13, C0_buff[13], 0);
			s2mm_add_C(CPL_in14, C0_buff[14], 0);
			s2mm_add_C(CPL_in15, C0_buff[15], 0);
			s2mm_add_C(CPL_in16, C0_buff[16], 0);
			s2mm_add_C(CPL_in17, C0_buff[17], 0);
			s2mm_add_C(CPL_in18, C0_buff[18], 0);
			s2mm_add_C(CPL_in19, C0_buff[19], 0);
			s2mm_add_C(CPL_in20, C0_buff[20], 0);
			s2mm_add_C(CPL_in21, C0_buff[21], 0);
			s2mm_add_C(CPL_in22, C0_buff[22], 0);
			s2mm_add_C(CPL_in23, C0_buff[23], 0);
			s2mm_add_C(CPL_in24, C0_buff[24], 0);
			s2mm_add_C(CPL_in25, C0_buff[25], 0);
			s2mm_add_C(CPL_in26, C0_buff[26], 0);
			s2mm_add_C(CPL_in27, C0_buff[27], 0);
			s2mm_add_C(CPL_in28, C0_buff[28], 0);
			s2mm_add_C(CPL_in29, C0_buff[29], 0);
			s2mm_add_C(CPL_in30, C0_buff[30], 0);
			s2mm_add_C(CPL_in31, C0_buff[31], 0);
			s2mm_add_C(CPL_in32, C0_buff[32], 0);
			s2mm_add_C(CPL_in33, C0_buff[33], 0);
			s2mm_add_C(CPL_in34, C0_buff[34], 0);
			s2mm_add_C(CPL_in35, C0_buff[35], 0);
			s2mm_add_C(CPL_in36, C0_buff[36], 0);
			s2mm_add_C(CPL_in37, C0_buff[37], 0);
			s2mm_add_C(CPL_in38, C0_buff[38], 0);
			s2mm_add_C(CPL_in39, C0_buff[39], 0);
			s2mm_add_C(CPL_in40, C0_buff[40], 0);
			s2mm_add_C(CPL_in41, C0_buff[41], 0);
			s2mm_add_C(CPL_in42, C0_buff[42], 0);
			s2mm_add_C(CPL_in43, C0_buff[43], 0);
			s2mm_add_C(CPL_in44, C0_buff[44], 0);
			s2mm_add_C(CPL_in45, C0_buff[45], 0);
			s2mm_add_C(CPL_in46, C0_buff[46], 0);
			s2mm_add_C(CPL_in47, C0_buff[47], 0);
			s2mm_add_C(CPL_in48, C0_buff[48], 0);
			s2mm_add_C(CPL_in49, C0_buff[49], 0);
			s2mm_add_C(CPL_in50, C0_buff[50], 0);
			s2mm_add_C(CPL_in51, C0_buff[51], 0);
			s2mm_add_C(CPL_in52, C0_buff[52], 0);
			s2mm_add_C(CPL_in53, C0_buff[53], 0);
			s2mm_add_C(CPL_in54, C0_buff[54], 0);
			s2mm_add_C(CPL_in55, C0_buff[55], 0);
			s2mm_add_C(CPL_in56, C0_buff[56], 0);
			s2mm_add_C(CPL_in57, C0_buff[57], 0);
			s2mm_add_C(CPL_in58, C0_buff[58], 0);
			s2mm_add_C(CPL_in59, C0_buff[59], 0);
			s2mm_add_C(CPL_in60, C0_buff[60], 0);
			s2mm_add_C(CPL_in61, C0_buff[61], 0);
			s2mm_add_C(CPL_in62, C0_buff[62], 0);
			s2mm_add_C(CPL_in63, C0_buff[63], 0);
			s2mm_add_C(CPL_in64, C0_buff[64], 0);
			s2mm_add_C(CPL_in65, C0_buff[65], 0);
			s2mm_add_C(CPL_in66, C0_buff[66], 0);
			s2mm_add_C(CPL_in67, C0_buff[67], 0);
			s2mm_add_C(CPL_in68, C0_buff[68], 0);
			s2mm_add_C(CPL_in69, C0_buff[69], 0);
			s2mm_add_C(CPL_in70, C0_buff[70], 0);
			s2mm_add_C(CPL_in71, C0_buff[71], 0);
			s2mm_add_C(CPL_in72, C0_buff[72], 0);
			s2mm_add_C(CPL_in73, C0_buff[73], 0);
			s2mm_add_C(CPL_in74, C0_buff[74], 0);
			s2mm_add_C(CPL_in75, C0_buff[75], 0);
			s2mm_add_C(CPL_in76, C0_buff[76], 0);
			s2mm_add_C(CPL_in77, C0_buff[77], 0);
			s2mm_add_C(CPL_in78, C0_buff[78], 0);
			s2mm_add_C(CPL_in79, C0_buff[79], 0);
			s2mm_add_C(CPL_in80, C0_buff[80], 0);
			s2mm_add_C(CPL_in81, C0_buff[81], 0);
			s2mm_add_C(CPL_in82, C0_buff[82], 0);
			s2mm_add_C(CPL_in83, C0_buff[83], 0);
			s2mm_add_C(CPL_in84, C0_buff[84], 0);
			s2mm_add_C(CPL_in85, C0_buff[85], 0);
			s2mm_add_C(CPL_in86, C0_buff[86], 0);
			s2mm_add_C(CPL_in87, C0_buff[87], 0);
			s2mm_add_C(CPL_in88, C0_buff[88], 0);
			s2mm_add_C(CPL_in89, C0_buff[89], 0);
			s2mm_add_C(CPL_in90, C0_buff[90], 0);
			s2mm_add_C(CPL_in91, C0_buff[91], 0);
			s2mm_add_C(CPL_in92, C0_buff[92], 0);
			s2mm_add_C(CPL_in93, C0_buff[93], 0);
			s2mm_add_C(CPL_in94, C0_buff[94], 0);
			s2mm_add_C(CPL_in95, C0_buff[95], 0);
			s2mm_add_C(CPL_in96, C0_buff[96], 0);
			s2mm_add_C(CPL_in97, C0_buff[97], 0);
			s2mm_add_C(CPL_in98, C0_buff[98], 0);
			s2mm_add_C(CPL_in99, C0_buff[99], 0);
			// store_C
			store_C(C1_buff, mem_outc);

		
	    }

    }
}

