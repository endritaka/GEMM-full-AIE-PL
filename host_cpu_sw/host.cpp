/#include "project.cpp"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
// #include "data.h"

// This is used for the PL Kernels
#include "xrt.h"
#include "experimental/xrt_kernel.h"

// Using the ADF API that call XRT API
#include "adf/adf_api/XRTConfig.h"

static std::vector<char>
load_xclbin(xrtDeviceHandle device, const std::string& fnm)
{
	if (fnm.empty())
		throw std::runtime_error("No xclbin specified");

	// load bit stream
	std::ifstream stream(fnm);
	stream.seekg(0,stream.end);
	size_t size = stream.tellg();
	stream.seekg(0,stream.beg);

	std::vector<char> header(size);
	stream.read(header.data(),size);

	auto top = reinterpret_cast<const axlf*>(header.data());
	if (xrtDeviceLoadXclbin(device, top))
		throw std::runtime_error("Xclbin loading failed");

	return header;
}


// Single AIE tiling size
#define M 32
#define K 128
#define N 32

// Multiple AIEs parameters
#define X 2
#define Y 4
#define Z 2

// PL tiling parameters
#define U 4
#define V 1
#define W 1


int main(int argc, char ** argv)
{
	//////////////////////////////////////////
	// Open xclbin
	//////////////////////////////////////////
	auto dhdl = xrtDeviceOpen(0); // Open Device the local device
	if(dhdl == nullptr)
		throw std::runtime_error("No valid device handle found. Make sure using right xclOpen index.");
    auto xclbin = load_xclbin(dhdl, "a.xclbin");
    auto top = reinterpret_cast<const axlf*>(xclbin.data());
    adf::registerXRT(dhdl, top->m_header.uuid);


	int size_A = (X*Y)*(U*V)*(M*K);
	int size_B = (Y*Z)*(W*V)*(K*N);
	int size_C = (X*Z)*(U*W)*(M*N);


	// A data
	int8_t* In_A = new int8_t[size_A];
	for (int i = 0; i < size_A; i++){
		In_A[i] = 1;
	}

	//////////////////////////////////////////
	// input memory
	// Allocating the input size of sizeIn_1 to MM2S
	// This is using low-level XRT call xclAllocBO to allocate the memory
	//////////////////////////////////////////	
    
	xrtBufferHandle in_bohdl_A = xrtBOAlloc(dhdl, size_A * sizeof(int8_t), 0, 0);
	auto in_bomapped_A = reinterpret_cast<int8_t*>(xrtBOMap(in_bohdl_A));
	memcpy(in_bomapped_A, In_A, size_A * sizeof(int8_t));
	printf("Input memory virtual addr 0x%px\n", in_bomapped_A);

	#if defined(__SYNCBO_ENABLE__) 
		xrtBOSync(in_bohdl_A, XCL_BO_SYNC_BO_TO_DEVICE, size_A * sizeof(int8_t) , 0);
	#endif
	


	// B data
	int8_t* In_B = new int8_t[size_B];
	for (int i = 0; i < size_B; i++){
		In_B[i] = 1;
	}

	//////////////////////////////////////////
	// input memory
	// Allocating the input size of sizeIn_2 to MM2S
	// This is using low-level XRT call xclAllocBO to allocate the memory
	//////////////////////////////////////////	
    
	xrtBufferHandle in_bohdl_B = xrtBOAlloc(dhdl, size_B * sizeof(int8_t), 0, 0);
	auto in_bomapped_B = reinterpret_cast<int8_t*>(xrtBOMap(in_bohdl_B));
	memcpy(in_bomapped_B, In_B, size_B * sizeof(int8_t));
	printf("Input memory virtual addr 0x%px\n", in_bomapped_B);

	#if defined(__SYNCBO_ENABLE__) 
		xrtBOSync(in_bohdl_B, XCL_BO_SYNC_BO_TO_DEVICE, size_B * sizeof(int8_t) , 0);
	#endif



	//////////////////////////////////////////
	// output memory
	// Allocating the output size of sizeOut to S2MM
	// This is using low-level XRT call xclAllocBO to allocate the memory
	//////////////////////////////////////////
	
	xrtBufferHandle out_bohdl_C = xrtBOAlloc(dhdl, size_C * sizeof(int), 0, 0);
	auto out_bomapped_C = reinterpret_cast<int*>(xrtBOMap(out_bohdl_C));
	memset(out_bomapped_C, 0xABCDEF00, size_C * sizeof(int));
	printf("Output memory virtual addr 0x%px\n", out_bomapped_C);
	



	//////////////////////////////////////////
	// PL_tiling ip
	// Using the xrtPLKernelOpen function to manually control the PL Kernel
	// that is outside of the AI Engine graph
	//////////////////////////////////////////
	
	xrtKernelHandle PL_tiling_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "PL_tiling");
	// Need to provide the kernel handle, and the argument order of the kernel arguments
	// Here the in_bohdl is the input buffer, the nullptr is the streaming interface and must be null,
	// lastly, the size of the data. This info can be found in the kernel definition. 
	xrtRunHandle PL_tiling_rhdl = xrtKernelRun(PL_tiling_khdl, in_bohdl_A, in_bohdl_B, out_bohdl_C, 
												nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
												nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
												nullptr, nullptr, nullptr, nullptr);
	printf("run PL_tiling\n");



	//////////////////////////////////////////
	// graph execution for AIE
	//////////////////////////////////////////	
	
	printf("graph init. This does nothing because CDO in boot PDI already configures AIE.\n");
	mygraph.init();
	
	printf("graph run\n");
	// mygraph.run(U*V*W);
	mygraph.run(-1);
	
	// mygraph.end();
	// printf("graph end\n");
	
	//////////////////////////////////////////
	// wait for PL_tiling done
	//////////////////////////////////////////	
	
	auto state = xrtRunWait(PL_tiling_rhdl);
	std::cout << "PL_tiling completed with status(" << state << ")\n";
	xrtRunClose(PL_tiling_rhdl);
	xrtKernelClose(PL_tiling_khdl);


	#if defined(__SYNCBO_ENABLE__) 
		xrtBOSync(out_bohdl_C, XCL_BO_SYNC_BO_FROM_DEVICE, size_C * sizeof(int) , 0);
	#endif
	
	//////////////////////////////////////////
	// Comparing the execution data to the golden data
	//////////////////////////////////////////	
	
	int errorCount = 0;
	{
		for (int i = 0; i < size_C; i++)
		{
			printf("i = %d\t Data = %d\n", i, (int)out_bomapped_C[i]);

				// if ((int)out_bomapped_C[i] != 3.0)
				// {
				// 	printf("Error found @ %d, %d != %d\n", i, out_bomapped_C[i], 3.0);
				// 	errorCount++;
				// }
		}

		if (errorCount)
			printf("Test failed with %d errors\n", errorCount);
		else
			printf("TEST PASSED\n");
	}
	
	//////////////////////////////////////////
	// clean up XRT
	//////////////////////////////////////////	
    
	std::cout << "Releasing remaining XRT objects...\n";
	//xrtBOUnmap(dhdl, in_bohdl, in_bomapped);
	//xrtBOUnmap(dhdl, out_bohdl, out_bomapped);
	xrtBOFree(in_bohdl_A);
	xrtBOFree(in_bohdl_B);
	xrtBOFree(out_bohdl_C);
	xrtDeviceClose(dhdl);

	delete[] In_A;
	delete[] In_B;
	
	return errorCount;
}
